"""
Amazon Reviews 2023 collector using local SQLite product and review indexes.

Workflow:
1. Search products in the local SQLite product index.
2. Get the best product candidates.
3. Read local reviews from SQLite using parent_asin.
4. Return reviews instantly without scanning remote Amazon files.

This makes Streamlit much faster.
"""

import math
import sqlite3
from pathlib import Path
from typing import Dict, List

from src.app_tools.product_index_search import search_product_index
from src.utils.errors import ReviewCollectionError
from src.utils.logger import log_event


DB_PATH = Path("data/amazon2023_product_index.db")


def _read_reviews_from_sqlite(candidate: Dict, limit: int) -> List[Dict]:
    """
    Read local reviews for one product candidate using parent_asin.
    """
    if not DB_PATH.exists():
        raise ReviewCollectionError(
            f"Amazon index database not found: {DB_PATH}. "
            "Run scripts/build_amazon2023_index.py and scripts/build_amazon2023_reviews_index.py first."
        )

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    rows = connection.execute(
        """
        SELECT
            id,
            parent_asin,
            category,
            rating,
            review_title,
            text,
            verified_purchase,
            timestamp
        FROM reviews
        WHERE parent_asin = ?
        LIMIT ?
        """,
        (
            candidate["parent_asin"],
            limit
        )
    ).fetchall()

    connection.close()

    reviews = []

    for row in rows:
        reviews.append(
            {
                "review_id": row["id"],
                "product": candidate["title"],
                "matched_product_title": candidate["title"],
                "parent_asin": row["parent_asin"],
                "category": row["category"],
                "text": row["text"],
                "rating": row["rating"],
                "review_title": row["review_title"],
                "verified_purchase": row["verified_purchase"],
                "timestamp": row["timestamp"],
                "source": "amazon_reviews_2023_sqlite",
                "search_score": candidate.get("search_score"),
                "average_rating": candidate.get("average_rating"),
                "rating_number": candidate.get("rating_number"),
            }
        )

    return reviews


def _collect_reviews_for_candidates(
    candidates: List[Dict],
    max_reviews: int,
    max_products: int = 5
) -> List[Dict]:
    """
    Collect reviews from several matched products.

    This avoids taking all reviews from only one product listing.
    """
    selected_candidates = candidates[:max_products]

    if not selected_candidates:
        return []

    per_product_limit = max(
        1,
        math.ceil(max_reviews / len(selected_candidates))
    )

    reviews_by_product = []

    for candidate in selected_candidates:
        product_reviews = _read_reviews_from_sqlite(
            candidate=candidate,
            limit=per_product_limit
        )

        reviews_by_product.append(product_reviews)

    final_reviews = []

    max_length = max(
        [len(product_reviews) for product_reviews in reviews_by_product],
        default=0
    )

    for index in range(max_length):
        for product_reviews in reviews_by_product:
            if index < len(product_reviews):
                final_reviews.append(product_reviews[index])

            if len(final_reviews) >= max_reviews:
                return final_reviews

    return final_reviews


def collect_reviews_from_amazon2023(
    product_name: str,
    max_reviews: int = 10,
    max_candidates: int = 20
) -> List[Dict]:
    """
    Collect real product-level reviews from local Amazon Reviews 2023 SQLite indexes.
    """
    agent_name = "Collector Agent"

    try:
        log_event(
            agent=agent_name,
            action="amazon2023_sqlite_collection_started",
            status="started",
            input_data={
                "product_name": product_name,
                "max_reviews": max_reviews,
                "max_candidates": max_candidates
            }
        )

        if not product_name or not product_name.strip():
            raise ReviewCollectionError("Product name is empty.")

        candidates = search_product_index(
            query=product_name,
            limit=max_candidates
        )

        if not candidates:
            raise ReviewCollectionError(
                f"No product found in the local Amazon product index for '{product_name}'. "
                "Try another product name or rebuild the product index with more products."
            )

        reviews = _collect_reviews_for_candidates(
            candidates=candidates,
            max_reviews=max_reviews,
            max_products=5
        )

        if not reviews:
            top_candidates = [
                candidate["title"]
                for candidate in candidates[:5]
            ]

            raise ReviewCollectionError(
                "Products were found, but no local reviews were available. "
                "Run scripts/build_amazon2023_reviews_index.py for the needed categories. "
                f"Top candidates: {top_candidates}"
            )

        unique_products = list(
            dict.fromkeys(
                review["matched_product_title"]
                for review in reviews
            )
        )

        log_event(
            agent=agent_name,
            action="amazon2023_sqlite_collection_completed",
            status="success",
            output_data={
                "reviews_collected": len(reviews),
                "unique_products_used": len(unique_products),
                "matched_products": unique_products
            }
        )

        return reviews

    except Exception as error:
        log_event(
            agent=agent_name,
            action="amazon2023_sqlite_collection_failed",
            status="error",
            error=str(error)
        )

        raise ReviewCollectionError(str(error))
