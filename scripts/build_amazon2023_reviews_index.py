"""
Build local SQLite reviews index for Amazon Reviews 2023.

Goal:
Avoid scanning remote Amazon Reviews 2023 files every time Streamlit runs.

This script:
1. Reads the local product index from data/amazon2023_product_index.db
2. Gets indexed product parent_asin values
3. Streams Amazon Reviews 2023 review files once
4. Stores matching reviews locally in SQLite
"""

import argparse
import gzip
import json
import sqlite3
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable


DB_PATH = Path("data/amazon2023_product_index.db")


CATEGORY_REVIEW_URLS = {
    "Cell_Phones_and_Accessories": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Cell_Phones_and_Accessories.jsonl.gz",
    "Electronics": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz",
    "Clothing_Shoes_and_Jewelry": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Clothing_Shoes_and_Jewelry.jsonl.gz",
    "Home_and_Kitchen": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Home_and_Kitchen.jsonl.gz",
    "Beauty_and_Personal_Care": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Beauty_and_Personal_Care.jsonl.gz",
    "Sports_and_Outdoors": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Sports_and_Outdoors.jsonl.gz",
    "Automotive": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Automotive.jsonl.gz",
}


def open_jsonl_gz(url: str) -> Iterable[Dict]:
    """
    Stream a remote jsonl.gz file line by line.
    """
    with urllib.request.urlopen(url, timeout=120) as response:
        with gzip.GzipFile(fileobj=response) as gzip_file:
            for line in gzip_file:
                yield json.loads(line.decode("utf-8"))


def create_reviews_table(connection: sqlite3.Connection) -> None:
    """
    Create reviews table inside the existing SQLite database.
    """
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_asin TEXT NOT NULL,
            category TEXT NOT NULL,
            rating REAL,
            review_title TEXT,
            text TEXT NOT NULL,
            verified_purchase INTEGER,
            timestamp INTEGER,
            UNIQUE(parent_asin, text)
        )
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_reviews_parent_asin
        ON reviews(parent_asin)
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_reviews_category
        ON reviews(category)
        """
    )

    connection.commit()


def load_indexed_products(connection: sqlite3.Connection, category: str) -> set:
    """
    Load all parent_asin values already indexed in the products table.
    """
    rows = connection.execute(
        """
        SELECT parent_asin
        FROM products
        WHERE category = ?
        """,
        (category,)
    ).fetchall()

    return {row[0] for row in rows}


def load_existing_review_counts(connection: sqlite3.Connection, category: str) -> Dict[str, int]:
    """
    Load already indexed review counts per product.
    """
    rows = connection.execute(
        """
        SELECT parent_asin, COUNT(*)
        FROM reviews
        WHERE category = ?
        GROUP BY parent_asin
        """,
        (category,)
    ).fetchall()

    return {row[0]: row[1] for row in rows}


def insert_review(connection: sqlite3.Connection, category: str, review: Dict) -> None:
    """
    Insert one review into SQLite.
    """
    text = str(review.get("text", "")).strip()

    if not text:
        return

    verified_purchase = review.get("verified_purchase")

    if isinstance(verified_purchase, bool):
        verified_purchase = int(verified_purchase)

    connection.execute(
        """
        INSERT OR IGNORE INTO reviews (
            parent_asin,
            category,
            rating,
            review_title,
            text,
            verified_purchase,
            timestamp
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            review.get("parent_asin"),
            category,
            review.get("rating"),
            review.get("title"),
            text,
            verified_purchase,
            review.get("timestamp"),
        )
    )


def index_reviews_for_category(
    connection: sqlite3.Connection,
    category: str,
    max_scan_per_category: int,
    max_reviews_per_product: int
) -> None:
    """
    Index reviews for one category.
    """
    if category not in CATEGORY_REVIEW_URLS:
        print(f"Unsupported category: {category}")
        return

    product_asins = load_indexed_products(connection, category)

    if not product_asins:
        print(f"No products indexed for category: {category}")
        return

    review_counts = defaultdict(int)
    review_counts.update(load_existing_review_counts(connection, category))

    url = CATEGORY_REVIEW_URLS[category]

    print(f"\nIndexing reviews for category: {category}")
    print(f"Indexed products in this category: {len(product_asins)}")
    print(f"URL: {url}")

    inserted = 0
    scanned = 0

    for index, review in enumerate(open_jsonl_gz(url)):
        if index >= max_scan_per_category:
            break

        scanned += 1

        parent_asin = review.get("parent_asin")

        if parent_asin not in product_asins:
            continue

        if review_counts[parent_asin] >= max_reviews_per_product:
            continue

        insert_review(
            connection=connection,
            category=category,
            review=review
        )

        review_counts[parent_asin] += 1
        inserted += 1

        if inserted % 1000 == 0:
            connection.commit()
            print(f"  Scanned {scanned} reviews | Inserted {inserted} reviews")

        if scanned % 100000 == 0:
            print(f"  Progress: scanned {scanned} reviews...")

    connection.commit()

    print(f"Done category: {category}")
    print(f"Scanned reviews: {scanned}")
    print(f"Inserted reviews: {inserted}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--categories",
        nargs="+",
        default=["Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry"],
        help="Categories to index reviews for."
    )

    parser.add_argument(
        "--max-scan-per-category",
        type=int,
        default=1000000,
        help="Maximum reviews to scan per category."
    )

    parser.add_argument(
        "--max-reviews-per-product",
        type=int,
        default=20,
        help="Maximum reviews to store per product."
    )

    args = parser.parse_args()

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Product index not found: {DB_PATH}. "
            "Run scripts/build_amazon2023_index.py first."
        )

    connection = sqlite3.connect(DB_PATH)
    create_reviews_table(connection)

    for category in args.categories:
        index_reviews_for_category(
            connection=connection,
            category=category,
            max_scan_per_category=args.max_scan_per_category,
            max_reviews_per_product=args.max_reviews_per_product
        )

    connection.close()

    print("\nReview index completed.")
    print(f"Database: {DB_PATH}")


if __name__ == "__main__":
    main()
