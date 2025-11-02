# src/tools/fetch_services.py
from pathlib import Path
import json

# point to your API catalogs
CATALOG_WITH_QOS = Path("data/processed/api_repo.with_qos.jsonl")
CATALOG_NO_QOS = Path("data/processed/api_repo.no_qos.jsonl")


def iter_jsonl(path: Path):
    """Yield JSON objects line by line from a .jsonl file."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def fetch_services(category: str, offset: int = 0, limit: int = 50, with_qos: bool = True):
    """
    Return a list of service records from the local catalog for the given category.
    Each record is a JSON object from the JSONL file.
    """
    path = CATALOG_WITH_QOS if with_qos else CATALOG_NO_QOS
    if not path.exists():
        raise FileNotFoundError(f"Catalog not found: {path.resolve()}")
    # filter by category
    items = [r for r in iter_jsonl(path) if r.get("category") == category]
    # paginate
    return items[offset: offset + limit]
