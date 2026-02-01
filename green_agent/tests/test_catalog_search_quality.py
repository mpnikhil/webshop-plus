"""Integration checks for catalog search quality.

These tests use the real WebShop preview dataset to ensure common queries
return at least one product. If these fail, the underlying dataset or
search index is likely incomplete.
"""

from pathlib import Path

import pytest

from src.webshop_mcp.server import _parse_search_results
from src.webshop_wrapper import WebShopWrapper


WEBSHOP_DATA_FILE = (
    Path(__file__).parent.parent.parent / "webshop" / "data" / "items_shuffle_1000.json"
)
pytestmark = pytest.mark.skipif(
    not WEBSHOP_DATA_FILE.exists(),
    reason="WebShop data files not available (required for catalog search tests)",
)


@pytest.fixture(scope="module")
def wrapper():
    """Shared WebShop wrapper for search checks."""
    return WebShopWrapper(mode="preview")


@pytest.mark.parametrize(
    "query,min_results,required_terms",
    [
        ("dress shirt", 1, ["dress", "shirt"]),
        ("slim fit dress shirt", 1, ["slim", "dress", "shirt"]),
        ("loafers", 1, ["loafer", "loafers", "slip-on", "slip on"]),
        ("sulfate free shampoo", 1, ["shampoo", "sulfate free", "sulfate-free"]),
        ("floor lamp", 1, ["floor", "lamp"]),
    ],
)
def test_search_returns_results(wrapper, query, min_results, required_terms):
    """Ensure common preference-memory queries return relevant products."""
    wrapper.reset(goal_idx=0)
    result = wrapper.step(f"search[{query}]")
    products = _parse_search_results(result.observation)

    sample = ", ".join(p["name"][:40] for p in products[:3]) if products else "none"
    assert len(products) >= min_results, (
        f"Search for '{query}' returned {len(products)} products "
        f"(expected >= {min_results}). Sample: {sample}"
    )

    # Relevance check: at least one product name should contain a required term
    names = " | ".join(p["name"].lower() for p in products)
    has_required = any(term in names for term in required_terms)
    assert has_required, (
        f"Search for '{query}' returned products but none matched terms "
        f"{required_terms}. Sample: {sample}"
    )
