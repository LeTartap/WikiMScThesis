#!/usr/bin/env python3
"""
Script to fetch Wikipedia article titles and page IDs based on grouped root categories.
Crawls each named category and its subcategories up to a given depth, tagging each article
with its supercategory and original category.
Outputs a CSV: supercategory,category,title,pageid.
Shows progress bars for both supercategories and their root categories.
"""
import csv
import requests
from typing import List, Dict, Set, Tuple
from tqdm import tqdm

# Configuration
API_URL = "https://en.wikipedia.org/w/api.php"
# Map of supercategories to lists of root categories (without 'Category:' prefix)
ROOT_CATEGORIES: Dict[str, List[str]] = {
    "Politics": [
        "Politics", "Political history", "Elections", "Political parties"
    ],
    "Science & Medicine": [
        "Science", "Medicine", "Biology", "Physics", "Chemistry"
    ],
    "History": [
        "History", "Military history", "History by country"
    ],
    "Technology": [
        "Technology", "Computing", "Engineering"
    ],
    "Popular Culture": [
        "Popular culture", "Music", "Television", "Film", "Video games"
    ]
}
MAX_DEPTH = 1  # Subcategory traversal depth
OUTPUT_CSV = "category_titles_by_group.csv"


def wiki_api_call(params: dict) -> dict:
    """
    Wrapper for GET requests to the MediaWiki API.
    """
    resp = requests.get(API_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_category_members(category: str,
                           member_type: str = "page",
                           cmcontinue: str = None) -> Tuple[List[dict], str]:
    """
    Fetch members of a category. member_type: 'page' or 'subcat'.
    Returns list of dicts and continuation token.
    """
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmtype": member_type,
        "cmlimit": "max"
    }
    if cmcontinue:
        params["cmcontinue"] = cmcontinue

    data = wiki_api_call(params)
    members = data.get("query", {}).get("categorymembers", [])
    cont = data.get("continue", {}).get("cmcontinue")
    return members, cont


def collect_titles(groups: Dict[str, List[str]], max_depth: int) -> Set[Tuple[str, str, str, int]]:
    """
    Recursively collect (supercategory, category, title, pageid).
    Displays progress for each supercategory and its root categories.
    """
    results: Set[Tuple[str, str, str, int]] = set()
    visited: Set[str] = set()

    def recurse(supercat: str, cat: str, depth: int):
        if depth > max_depth or cat in visited:
            return
        visited.add(cat)

        # Fetch pages in this category
        cont = None
        while True:
            pages, cont = fetch_category_members(cat, "page", cont)
            for p in pages:
                results.add((supercat, cat, p["title"], p.get("pageid", 0)))
            if not cont:
                break

        # Fetch subcategories and recurse
        cont = None
        while True:
            subcats, cont = fetch_category_members(cat, "subcat", cont)
            for sc in subcats:
                subcat_name = sc["title"].replace("Category:", "", 1)
                recurse(supercat, subcat_name, depth + 1)
            if not cont:
                break

    # Progress over supercategories
    for supercat, cats in tqdm(groups.items(), desc="Supercategories", unit="group"):
        # Progress over each root category within this supercategory
        for cat in tqdm(cats, desc=f"{supercat}", leave=False, unit="cat"):
            recurse(supercat, cat, 0)

    return results


def save_to_csv(records: Set[Tuple[str, str, str, int]], output_file: str) -> None:
    """
    Write records to CSV with header: supercategory,category,title,pageid
    """
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["supercategory", "category", "title", "pageid"])
        for supercat, cat, title, pid in sorted(records):
            writer.writerow([supercat, cat, title, pid])


def main():
    records = collect_titles(ROOT_CATEGORIES, MAX_DEPTH)
    save_to_csv(records, OUTPUT_CSV)
    print(f"Fetched {len(records)} articles across {len(ROOT_CATEGORIES)} groups (depth={MAX_DEPTH}).")


if __name__ == "__main__":
    main()

