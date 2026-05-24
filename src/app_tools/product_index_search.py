"""
Smart product search engine for Amazon Reviews 2023 local SQLite index.

This module searches products from the local SQLite index.

Main goals:
- Understand product intent.
- Avoid accessories when the user searches for the main product.
- Boost accessories when the user explicitly asks for them.
- Support queries like:
  iPhone X
  iPhone X case
  iPhone 12 charger
  RayBan black
  Air Force shoes
"""

import math
import re
import sqlite3
from pathlib import Path
from typing import Dict, List


DB_PATH = Path("data/amazon2023_product_index.db")


ACCESSORY_TERMS = {
    "case", "cover", "protector", "screen", "charger", "charging",
    "cable", "cord", "adapter", "mount", "holder", "strap", "band",
    "skin", "wallet", "dock", "stand", "replacement", "lens", "lenses", "pouch", "holster", "bumper", "tpu", "hybrid", "grip", "bag", "organizer", "molle", "belt", "shell"
}

PHONE_BRANDS = {
    "iphone", "apple", "samsung", "galaxy", "pixel", "oneplus", "motorola"
}

PHONE_MODEL_TOKENS = {
    "x", "xr", "xs", "se"
}

IPHONE_VARIANTS = {
    "mini", "pro", "max", "plus"
}

PHONE_PRODUCT_TERMS = {
    "unlocked", "renewed", "smartphone", "phone", "64gb",
    "128gb", "256gb", "512gb", "5g"
}

FRENCH_TO_ENGLISH = {
    "chargeur": "charger",
    "coque": "case",
    "cable": "cable",
    "câble": "cable",
    "lunettes": "sunglasses",
    "noir": "black",
    "noire": "black",
    "chaussures": "shoes",
    "basket": "sneakers",
    "baskets": "sneakers",
    "casque": "headphones",
    "ecouteurs": "earbuds",
    "écouteurs": "earbuds",
}


def normalize_text(text: str) -> str:
    """
    Normalize text for search.
    """
    text = str(text).lower()

    for french_word, english_word in FRENCH_TO_ENGLISH.items():
        text = text.replace(french_word, english_word)

    text = text.replace("&", " and ")
    text = text.replace("-", " ")
    text = text.replace("_", " ")
    text = text.replace("/", " ")

    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    text = text.replace("rayban", "ray ban")

    return text


def tokenize(text: str) -> List[str]:
    """
    Convert text into clean tokens.
    """
    return normalize_text(text).split()


def extract_keywords(query: str) -> List[str]:
    """
    Extract useful keywords from query.
    """
    ignored_words = {
        "the", "a", "an", "and", "or", "for", "with",
        "new", "original", "official", "compatible"
    }

    keywords = []

    for word in tokenize(query):
        if word in ignored_words:
            continue

        # Keep important one-letter model tokens like iPhone X.
        if word in PHONE_MODEL_TOKENS:
            keywords.append(word)
            continue

        if len(word) < 2 and not word.isdigit():
            continue

        keywords.append(word)

    return keywords


def contains_keyword(title_tokens: List[str], keyword: str) -> bool:
    """
    Check if a keyword is really present.

    This avoids bad substring matches:
    - ray should not match gray
    - ban should not match band
    """
    if keyword in title_tokens:
        return True

    if len(keyword) <= 3:
        return False

    for token in title_tokens:
        if token.startswith(keyword):
            return True

    return False


def query_has_accessory_intent(keywords: List[str]) -> bool:
    """
    Detect if user explicitly searches for an accessory.
    """
    return any(keyword in ACCESSORY_TERMS for keyword in keywords)


def query_is_phone_main_product(keywords: List[str]) -> bool:
    """
    Detect if the query is probably about the phone itself.
    """
    has_phone_brand = any(keyword in PHONE_BRANDS for keyword in keywords)

    has_model_identifier = any(
        keyword.isdigit() or keyword in PHONE_MODEL_TOKENS
        for keyword in keywords
    )

    has_samsung_galaxy = "samsung" in keywords and "galaxy" in keywords

    return has_phone_brand and (has_model_identifier or has_samsung_galaxy)


def score_product(product: Dict, query: str, keywords: List[str]) -> float:
    """
    Compute a smart relevance score.
    """
    title = product["title"]
    title_normalized = normalize_text(title)
    title_tokens = tokenize(title)
    normalized_query = normalize_text(query)

    score = 0.0

    if normalized_query in title_normalized:
        score += 120

    for keyword in keywords:
        if contains_keyword(title_tokens, keyword):
            score += 25

    for i in range(len(keywords) - 1):
        phrase = f"{keywords[i]} {keywords[i + 1]}"
        if phrase in title_normalized:
            score += 35

    accessory_intent = query_has_accessory_intent(keywords)
    phone_main_product = query_is_phone_main_product(keywords)

    title_has_accessory = any(term in title_tokens for term in ACCESSORY_TERMS)

    # Important: if user searches "iPhone X" or "iPhone 12",
    # we strongly penalize accessories like cases, chargers, cables.
    if phone_main_product and not accessory_intent and title_has_accessory:
        score -= 500

    # If user searches "iPhone X case" or "iPhone 12 charger",
    # accessories are exactly what we want.
    if accessory_intent and title_has_accessory:
        score += 100

    # Boost real phone titles.
    if phone_main_product and not accessory_intent:
        if "apple" in title_tokens:
            score += 50

        if "iphone" in title_tokens:
            score += 50

        for term in PHONE_PRODUCT_TERMS:
            if term in title_tokens:
                score += 35

    # If query is "iPhone 12", avoid pushing "iPhone 12 Pro Max" too high.
    if "iphone" in keywords and not accessory_intent:
        query_variants = set(keywords).intersection(IPHONE_VARIANTS)
        title_variants = set(title_tokens).intersection(IPHONE_VARIANTS)

        unwanted_variants = title_variants - query_variants

        if unwanted_variants:
            score -= 40 * len(unwanted_variants)

    # Ray-Ban logic.
    if "ray" in keywords and "ban" in keywords:
        if "ray ban" in title_normalized:
            score += 160
        else:
            score -= 300

        if "sunglasses" in title_tokens or "glasses" in title_tokens:
            score += 80

    # Shoes logic.
    if "jordan" in keywords or ("air" in keywords and "force" in keywords):
        if product["category"] == "Clothing_Shoes_and_Jewelry":
            score += 80

        if "shoe" in title_normalized or "sneaker" in title_normalized:
            score += 100

    rating_number = product.get("rating_number") or 0

    try:
        rating_number = int(rating_number)
    except Exception:
        rating_number = 0

    score += min(math.log10(rating_number + 1) * 10, 45)

    average_rating = product.get("average_rating") or 0

    try:
        average_rating = float(average_rating)
    except Exception:
        average_rating = 0.0

    score += average_rating * 2

    return score


def load_candidates_from_db(keywords: List[str], candidate_limit: int = 50000) -> List[Dict]:
    """
    Load candidate products from SQLite using LIKE filtering.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Product index not found: {DB_PATH}. "
            "Run scripts/build_amazon2023_index.py first."
        )

    where_clause = " AND ".join(
        ["normalized_title LIKE ?" for _ in keywords]
    )

    params = [f"%{keyword}%" for keyword in keywords]
    params.append(candidate_limit)

    sql = f"""
        SELECT
            parent_asin,
            title,
            category,
            main_category,
            average_rating,
            rating_number
        FROM products
        WHERE {where_clause}
        LIMIT ?
    """

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    rows = connection.execute(sql, params).fetchall()
    connection.close()

    products = []

    for row in rows:
        products.append(
            {
                "parent_asin": row["parent_asin"],
                "title": row["title"],
                "category": row["category"],
                "main_category": row["main_category"],
                "average_rating": row["average_rating"],
                "rating_number": row["rating_number"],
            }
        )

    return products


def search_product_index(query: str, limit: int = 10) -> List[Dict]:
    """
    Search products in the local Amazon Reviews 2023 index.
    """
    keywords = extract_keywords(query)

    if not keywords:
        return []

    candidates = load_candidates_from_db(
        keywords=keywords,
        candidate_limit=50000
    )

    scored_products = []

    for product in candidates:
        title_tokens = tokenize(product["title"])

        if not all(
            contains_keyword(title_tokens, keyword)
            for keyword in keywords
        ):
            continue

        accessory_intent = query_has_accessory_intent(keywords)
        phone_main_product = query_is_phone_main_product(keywords)
        title_has_accessory = any(term in title_tokens for term in ACCESSORY_TERMS)

        # Strict rule:
        # If the user searches for the phone itself, remove accessories completely.
        # Example: "iPhone X" must not return pouch, case, holster, charger, etc.
        if phone_main_product and not accessory_intent and title_has_accessory:
            continue

        relevance_score = score_product(
            product=product,
            query=query,
            keywords=keywords
        )

        product["search_score"] = round(relevance_score, 2)

        if relevance_score > 0:
            scored_products.append(product)

    scored_products.sort(
        key=lambda item: item["search_score"],
        reverse=True
    )

    return scored_products[:limit]
