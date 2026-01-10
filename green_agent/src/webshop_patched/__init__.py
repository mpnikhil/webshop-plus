"""
Patched WebShop modules that don't require pyserini or spacy.

Uses BM25 for search and simpler text matching for rewards.
"""

from .engine import (
    load_products,
    map_action_to_html,
    parse_action,
    get_product_per_page,
    get_top_n_product_from_keywords,
    get_search_engine,
    init_search_engine,
    ACTION_TO_TEMPLATE,
    END_BUTTON,
    NEXT_PAGE,
    PREV_PAGE,
    BACK_TO_SEARCH,
    SEARCH_RETURN_N,
    DEFAULT_FILE_PATH,
)
from .goal import get_goals, get_reward
from .normalize import normalize_color

__all__ = [
    "load_products",
    "map_action_to_html",
    "parse_action",
    "get_product_per_page",
    "ACTION_TO_TEMPLATE",
    "END_BUTTON",
    "NEXT_PAGE",
    "PREV_PAGE",
    "BACK_TO_SEARCH",
    "SEARCH_RETURN_N",
    "get_goals",
    "get_reward",
    "normalize_color",
]
