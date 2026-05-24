"""
Hugging Face review collector.

This tool collects real Amazon reviews from Hugging Face.

Dataset used:
mteb/amazon_polarity

Important limitation:
This dataset contains real Amazon reviews, but it does not provide a product_name column.
So the collector searches inside the review text.

To avoid irrelevant reviews, the matching is strict:
- "book" must appear as a real word
- "iPhone 12" must contain both "iphone" and "12"
"""

import re
from typing import Dict, List

from datasets import load_dataset

from src.utils.errors import ReviewCollectionError
from src.utils.logger import log_event


def _normalize_text(text: str) -> str:
    """
    Normalize text for safer keyword matching.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_keywords(product_name: str) -> List[str]:
    """
    Extract meaningful keywords from product name.

    Example:
    "iPhone 12" -> ["iphone", "12"]
    "Wireless Headphones" -> ["wireless", "headphones"]
    """
    ignored_words = {
        "the", "a", "an", "and", "or", "for", "with",
        "new", "original"
    }

    normalized_product = _normalize_text(product_name)

    keywords = []

    for word in normalized_product.split():
        if word in ignored_words:
            continue

        if len(word) < 2 and not word.isdigit():
            continue

        keywords.append(word)

    return keywords


def _is_relevant_review(review_text: str, product_name: str, keywords: List[str]) -> bool:
    """
    Check if a review is relevant to the product.

    For one keyword:
    - at least that keyword must appear.

    For multiple keywords:
    - all keywords must appear.
    Example:
    "iPhone 12" requires both "iphone" and "12".
    """
    normalized_review = _normalize_text(review_text)
    normalized_product = _normalize_text(product_name)

    review_words = set(normalized_review.split())

    if not keywords:
        return False

    # Exact phrase match.
    if f" {normalized_product} " in f" {normalized_review} ":
        return True

    # Single keyword product.
    if len(keywords) == 1:
        return keywords[0] in review_words

    # Multi-keyword product: all important keywords must be present.
    return all(keyword in review_words for keyword in keywords)


def collect_reviews_from_huggingface(
    product_name: str,
    max_reviews: int = 20,
    max_scan: int = 30000
) -> List[Dict]:
    """
    Collect real Amazon reviews automatically from Hugging Face.

    If no relevant review is found, it raises an error instead of returning random reviews.
    """
    agent_name = "Collector Agent"

    try:
        log_event(
            agent=agent_name,
            action="huggingface_collection_started",
            status="started",
            input_data={
                "dataset": "mteb/amazon_polarity",
                "product_name": product_name,
                "max_reviews": max_reviews,
                "max_scan": max_scan
            }
        )

        if not product_name or not product_name.strip():
            raise ReviewCollectionError("Product name is empty.")

        keywords = _extract_keywords(product_name)

        if not keywords:
            raise ReviewCollectionError(
                "No useful keyword found in product name."
            )

        dataset = load_dataset(
            "mteb/amazon_polarity",
            split="train",
            streaming=True
        )

        matched_reviews = []

        for index, row in enumerate(dataset):
            if index >= max_scan:
                break

            review_text = str(row.get("text", "")).strip()

            if not review_text:
                continue

            if not _is_relevant_review(review_text, product_name, keywords):
                continue

            matched_reviews.append(
                {
                    "review_id": index,
                    "product": product_name.strip(),
                    "text": review_text,
                    "source": "huggingface_amazon_polarity"
                }
            )

            if len(matched_reviews) >= max_reviews:
                break

        if not matched_reviews:
            raise ReviewCollectionError(
                f"No relevant Hugging Face reviews found for '{product_name}'. "
                "Try a broader keyword like 'iphone', 'phone', 'book', 'headphones', "
                "or paste real reviews manually."
            )

        log_event(
            agent=agent_name,
            action="huggingface_collection_completed",
            status="success",
            output_data={
                "reviews_collected": len(matched_reviews),
                "keywords_used": keywords,
                "source": "huggingface_amazon_polarity"
            }
        )

        return matched_reviews

    except Exception as error:
        log_event(
            agent=agent_name,
            action="huggingface_collection_failed",
            status="error",
            error=str(error)
        )

        raise ReviewCollectionError(str(error))
