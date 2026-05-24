"""
Build a local SQLite product index for Amazon Reviews 2023.

Why?
The full Amazon Reviews 2023 dataset is too large to load into memory.
So we only index product metadata:
- product title
- parent_asin
- category
- average rating
- number of ratings

Then the app can search this local index quickly.
"""

import argparse
import gzip
import json
import re
import sqlite3
import urllib.request
from pathlib import Path
from typing import Dict, Iterable


DB_PATH = Path("data/amazon2023_product_index.db")


CATEGORY_NAMES = [
    "Cell_Phones_and_Accessories",
    "Electronics",
    "Clothing_Shoes_and_Jewelry",
    "Home_and_Kitchen",
    "Beauty_and_Personal_Care",
    "Sports_and_Outdoors",
    "Automotive",
]


def normalize_text(text: str) -> str:
    """
    Normalize product titles for search.
    """
    text = str(text).lower()
    text = text.replace("&", " and ")
    text = text.replace("-", " ")
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Useful normalization for brands.
    text = text.replace("rayban", "ray ban")
    text = text.replace("ray ban", "ray ban")

    return text


def metadata_url(category_name: str) -> str:
    """
    Build metadata URL for a category.
    """
    return (
        "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/"
        f"meta_categories/meta_{category_name}.jsonl.gz"
    )


def open_jsonl_gz(url: str) -> Iterable[Dict]:
    """
    Stream a remote .jsonl.gz file line by line.
    This avoids loading the full file in memory.
    """
    with urllib.request.urlopen(url, timeout=120) as response:
        with gzip.GzipFile(fileobj=response) as gzip_file:
            for line in gzip_file:
                yield json.loads(line.decode("utf-8"))


def create_tables(connection: sqlite3.Connection) -> None:
    """
    Create product index table.
    """
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS products (
            parent_asin TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            normalized_title TEXT NOT NULL,
            category TEXT NOT NULL,
            main_category TEXT,
            average_rating REAL,
            rating_number INTEGER
        )
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_products_normalized_title
        ON products(normalized_title)
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_products_category
        ON products(category)
        """
    )

    connection.commit()


def insert_product(connection: sqlite3.Connection, product: Dict, category_name: str) -> None:
    """
    Insert one product metadata row into SQLite.
    """
    title = product.get("title")
    parent_asin = product.get("parent_asin")

    if not title or not parent_asin:
        return

    rating_number = product.get("rating_number") or 0
    average_rating = product.get("average_rating") or 0.0

    try:
        rating_number = int(rating_number)
    except Exception:
        rating_number = 0

    try:
        average_rating = float(average_rating)
    except Exception:
        average_rating = 0.0

    connection.execute(
        """
        INSERT OR REPLACE INTO products (
            parent_asin,
            title,
            normalized_title,
            category,
            main_category,
            average_rating,
            rating_number
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            parent_asin,
            title,
            normalize_text(title),
            category_name,
            product.get("main_category"),
            average_rating,
            rating_number,
        )
    )


def build_index(max_per_category: int) -> None:
    """
    Build the local product index.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)
    create_tables(connection)

    total_inserted = 0

    for category_name in CATEGORY_NAMES:
        url = metadata_url(category_name)

        print(f"\nIndexing category: {category_name}")
        print(f"URL: {url}")

        inserted_for_category = 0

        try:
            for index, product in enumerate(open_jsonl_gz(url)):
                if index >= max_per_category:
                    break

                insert_product(connection, product, category_name)
                inserted_for_category += 1
                total_inserted += 1

                if inserted_for_category % 5000 == 0:
                    connection.commit()
                    print(f"  Indexed {inserted_for_category} products...")

            connection.commit()
            print(f"Done: {inserted_for_category} products indexed for {category_name}")

        except Exception as error:
            print(f"ERROR while indexing {category_name}: {error}")

    connection.close()

    print("\nIndex build completed.")
    print(f"Database: {DB_PATH}")
    print(f"Total scanned products: {total_inserted}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-per-category",
        type=int,
        default=50000,
        help="Maximum metadata rows to scan per category."
    )

    args = parser.parse_args()

    build_index(max_per_category=args.max_per_category)


if __name__ == "__main__":
    main()
