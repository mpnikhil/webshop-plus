"""
WebShop Environment Wrapper

Provides a simplified interface to the Princeton WebShop environment.
Uses patched modules with BM25 search instead of pyserini/Lucene.
"""

import json
import random
import string
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from bs4 import BeautifulSoup
from bs4.element import Comment

from .webshop_patched import (
    load_products,
    map_action_to_html,
    parse_action,
    get_product_per_page,
    get_top_n_product_from_keywords,
    get_search_engine,
    ACTION_TO_TEMPLATE,
    END_BUTTON,
    NEXT_PAGE,
    PREV_PAGE,
    BACK_TO_SEARCH,
    SEARCH_RETURN_N,
    DEFAULT_FILE_PATH,
)
from .webshop_patched.goal import get_goals, get_reward


def _random_idx(cum_weights: list[float]) -> int:
    """Generate random index by sampling from cumulative weights."""
    import bisect
    pos = random.uniform(0, cum_weights[-1])
    idx = bisect.bisect(cum_weights, pos)
    idx = min(idx, len(cum_weights) - 2)
    return idx


@dataclass
class StepResult:
    """Result from a single environment step."""

    observation: str
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


def tag_visible(element) -> bool:
    """Filter for visible text elements in HTML."""
    ignore = {"style", "script", "head", "title", "meta", "[document]"}
    return element.parent.name not in ignore and not isinstance(element, Comment)


class WebShopWrapper:
    """
    Wrapper for the Princeton WebShop environment.

    Provides a simple interface for interacting with the simulated e-commerce environment.
    Uses BM25 for search instead of pyserini to avoid Java/Lucene dependency.

    Usage:
        wrapper = WebShopWrapper(mode="preview")
        observation = wrapper.reset()
        result = wrapper.step("search[shoes]")
        print(result.observation, result.reward, result.done)
    """

    def __init__(
        self,
        mode: str = "preview",
        observation_mode: str = "text",
        num_products: int | None = None,
        human_goals: bool = True,
    ):
        """
        Initialize the WebShop environment wrapper.

        Args:
            mode: "preview" (1000 products) or "full" (all products)
            observation_mode: "text" (simplified) or "html" (raw HTML)
            num_products: Override number of products to load
            human_goals: Whether to use human-generated goals
        """
        self.mode = mode
        self.observation_mode = observation_mode
        self.human_goals = human_goals

        # Determine number of products
        if num_products is not None:
            self.num_products = num_products
        elif mode == "preview":
            self.num_products = 1000
        else:
            self.num_products = None  # Load all

        # Load products and initialize search
        self._load_environment()

        # Session state
        self.session_id: str | None = None
        self.current_url: str | None = None
        self.page_source: str | None = None
        self.instruction_text: str | None = None
        self.user_sessions: dict[str, dict] = {}
        self.text_to_clickable: dict[str, Any] | None = None

        self.base_url = "http://127.0.0.1:3000"

    def _load_environment(self):
        """Load products, goals, and initialize search engine."""
        from rich import print as rprint
        rprint("[bold blue]Loading WebShop environment...[/bold blue]")

        # Load products (this also initializes the BM25 search engine)
        (
            self.all_products,
            self.product_item_dict,
            self.product_prices,
            self.attribute_to_asins,
        ) = load_products(
            filepath=DEFAULT_FILE_PATH,
            num_products=self.num_products,
            human_goals=self.human_goals,
        )

        # Get reference to search engine
        self.search_engine = get_search_engine()

        # Load goals
        self.goals = get_goals(
            self.all_products, self.product_prices, self.human_goals
        )

        # Shuffle goals deterministically
        random.seed(233)
        random.shuffle(self.goals)

        # Build cumulative weights for goal sampling
        self.weights = [goal["weight"] for goal in self.goals]
        self.cum_weights = [0] + list(np.cumsum(self.weights).tolist())

        rprint(
            f"[bold green]Loaded {len(self.all_products)} products and {len(self.goals)} goals[/bold green]"
        )

    def reset(self, session: str | int | None = None, goal_idx: int | None = None) -> str:
        """
        Reset the environment and start a new shopping session.

        Args:
            session: Optional session ID (string) or goal index (int)
            goal_idx: Specific goal index to use

        Returns:
            Initial observation string
        """
        # Generate or use provided session ID
        session_int = None
        if session is not None:
            if isinstance(session, int):
                session_int = session
                self.session_id = f"session_{session}"
            else:
                self.session_id = str(session)
        else:
            self.session_id = "".join(random.choices(string.ascii_lowercase, k=10))

        # Select goal
        if goal_idx is not None:
            idx = goal_idx
        elif session_int is not None:
            idx = session_int % len(self.goals)
        else:
            idx = _random_idx(self.cum_weights)

        goal = self.goals[idx]
        self.instruction_text = goal["instruction_text"]

        # Initialize session state
        self.user_sessions[self.session_id] = {
            "goal": goal,
            "done": False,
            "reward": 0.0,
            "keywords": None,
            "page": None,
            "asin": None,
            "asins": set(),
            "options": {},
            "actions": defaultdict(int),
        }

        # Render initial search page
        self.page_source = map_action_to_html(
            "start",
            session_id=self.session_id,
            instruction_text=self.instruction_text,
        )
        self.current_url = f"{self.base_url}/{self.session_id}"

        return self.observation

    def step(self, action: str) -> StepResult:
        """
        Take an action in the environment.

        Args:
            action: Action string, e.g., "search[shoes]" or "click[buy now]"

        Returns:
            StepResult with observation, reward, done flag, and info
        """
        if self.session_id is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Get available actions
        self.get_available_actions()

        # Parse action
        action_name, action_arg = parse_action(action)
        if action_arg is not None:
            action_arg = action_arg.lower()

        # Execute action
        if action_name == "search" and action_arg:
            reward, done = self._do_search(action_arg)
        elif (
            action_name == "click"
            and self.text_to_clickable
            and action_arg in self.text_to_clickable
            and action_arg != "search"
        ):
            reward, done = self._do_click(action_arg)
        else:
            # Invalid action - no change
            reward, done = 0.0, False

        session = self.user_sessions[self.session_id]
        return StepResult(
            observation=self.observation,
            reward=reward,
            done=done,
            info={
                "goal": session["goal"],
                "actions": dict(session["actions"]),
                "verbose_info": session.get("verbose_info"),
            },
        )

    def _do_search(self, keywords: str) -> tuple[float, bool]:
        """Execute a search action."""
        session = self.user_sessions[self.session_id]
        keyword_list = keywords.split()

        session["page"] = 1
        session["keywords"] = keyword_list
        session["actions"]["search"] += 1
        session["asin"] = None
        session["options"] = {}

        # Search using BM25
        top_products = get_top_n_product_from_keywords(
            keyword_list,
            self.search_engine,
            self.all_products,
            self.product_item_dict,
            self.attribute_to_asins,
        )
        products = get_product_per_page(top_products, 1)

        keywords_url_string = "+".join(keyword_list)
        self.current_url = (
            f"{self.base_url}/search_results/{self.session_id}/"
            f"{keywords_url_string}/1"
        )

        self.page_source = map_action_to_html(
            "search",
            session_id=self.session_id,
            products=products,
            keywords=keyword_list,
            page=1,
            total=len(top_products),
            instruction_text=session["goal"]["instruction_text"],
        )

        return 0.0, False

    def _do_click(self, clickable_name: str) -> tuple[float, bool]:
        """Execute a click action."""
        session = self.user_sessions[self.session_id]
        clickable = self.text_to_clickable[clickable_name]

        # Handle different click types
        if clickable_name.lower() == END_BUTTON.lower():
            return self._do_purchase()
        elif clickable_name.lower() == BACK_TO_SEARCH.lower():
            return self._do_back_to_search()
        elif clickable_name.lower() == NEXT_PAGE.lower():
            return self._do_next_page()
        elif clickable_name.lower() == PREV_PAGE.lower():
            return self._do_prev_page()
        elif clickable_name.lower() in [k.lower() for k in ACTION_TO_TEMPLATE]:
            return self._do_sub_page(clickable_name)
        else:
            return self._do_item_click(clickable_name, clickable)

    def _do_purchase(self) -> tuple[float, bool]:
        """Complete purchase and calculate reward."""
        session = self.user_sessions[self.session_id]

        if session["asin"] is None:
            return 0.0, False

        purchased_product = self.product_item_dict[session["asin"]]
        session["actions"]["purchase"] += 1
        price = self.product_prices.get(session["asin"])

        reward, info = get_reward(
            purchased_product,
            session["goal"],
            price=price,
            options=session["options"],
            verbose=True,
        )

        session["verbose_info"] = info
        session["done"] = True
        session["reward"] = reward

        self.current_url = (
            f"{self.base_url}/done/{self.session_id}/"
            f"{session['asin']}/{session['options']}"
        )

        self.page_source = map_action_to_html(
            f"click[{END_BUTTON}]",
            session_id=self.session_id,
            reward=reward,
            asin=session["asin"],
            options=session["options"],
            instruction_text=session["goal"]["instruction_text"],
        )

        return reward, True

    def _do_back_to_search(self) -> tuple[float, bool]:
        """Go back to search page."""
        session = self.user_sessions[self.session_id]

        self.page_source = map_action_to_html(
            "start",
            session_id=self.session_id,
            instruction_text=session["goal"]["instruction_text"],
        )
        self.current_url = f"{self.base_url}/{self.session_id}"

        session["keywords"] = None
        session["page"] = None
        session["asin"] = None
        session["options"] = {}

        return 0.0, False

    def _do_next_page(self) -> tuple[float, bool]:
        """Go to next page of search results."""
        session = self.user_sessions[self.session_id]

        if session["keywords"] is None:
            return 0.0, False

        session["page"] += 1
        top_products = get_top_n_product_from_keywords(
            session["keywords"],
            self.search_engine,
            self.all_products,
            self.product_item_dict,
            self.attribute_to_asins,
        )
        products = get_product_per_page(top_products, session["page"])

        keywords_url_string = "+".join(session["keywords"])
        self.current_url = (
            f"{self.base_url}/search_results/{self.session_id}/"
            f"{keywords_url_string}/{session['page']}"
        )

        self.page_source = map_action_to_html(
            "search",
            session_id=self.session_id,
            products=products,
            keywords=session["keywords"],
            page=session["page"],
            total=len(top_products),
            instruction_text=session["goal"]["instruction_text"],
        )

        return 0.0, False

    def _do_prev_page(self) -> tuple[float, bool]:
        """Go to previous page."""
        session = self.user_sessions[self.session_id]

        page_name = self._get_page_name()

        if page_name == "search_results" and session["page"] and session["page"] > 1:
            session["page"] -= 1
            top_products = get_top_n_product_from_keywords(
                session["keywords"],
                self.search_engine,
                self.all_products,
                self.product_item_dict,
                self.attribute_to_asins,
            )
            products = get_product_per_page(top_products, session["page"])

            keywords_url_string = "+".join(session["keywords"])
            self.current_url = (
                f"{self.base_url}/search_results/{self.session_id}/"
                f"{keywords_url_string}/{session['page']}"
            )

            self.page_source = map_action_to_html(
                "search",
                session_id=self.session_id,
                products=products,
                keywords=session["keywords"],
                page=session["page"],
                total=len(top_products),
                instruction_text=session["goal"]["instruction_text"],
            )
        elif page_name in ("item_page", "item_sub_page") and session["keywords"]:
            # Go back to search results
            top_products = get_top_n_product_from_keywords(
                session["keywords"],
                self.search_engine,
                self.all_products,
                self.product_item_dict,
                self.attribute_to_asins,
            )
            products = get_product_per_page(top_products, session["page"] or 1)

            keywords_url_string = "+".join(session["keywords"])
            self.current_url = (
                f"{self.base_url}/search_results/{self.session_id}/"
                f"{keywords_url_string}/{session['page']}"
            )

            self.page_source = map_action_to_html(
                "search",
                session_id=self.session_id,
                products=products,
                keywords=session["keywords"],
                page=session["page"] or 1,
                total=len(top_products),
                instruction_text=session["goal"]["instruction_text"],
            )

        return 0.0, False

    def _do_sub_page(self, clickable_name: str) -> tuple[float, bool]:
        """Navigate to product sub-page (description, features, reviews)."""
        session = self.user_sessions[self.session_id]

        if session["asin"] is None:
            return 0.0, False

        # Normalize clickable name
        for k in ACTION_TO_TEMPLATE:
            if clickable_name.lower() == k.lower():
                clickable_name = k
                break

        product_info = self.product_item_dict[session["asin"]]
        session["actions"][clickable_name] += 1

        keywords_url_string = "+".join(session["keywords"] or [])
        self.current_url = (
            f"{self.base_url}/item_sub_page/{self.session_id}/"
            f"{session['asin']}/{keywords_url_string}/{session['page']}/"
            f"{clickable_name}/{session['options']}"
        )

        self.page_source = map_action_to_html(
            f"click[{clickable_name}]",
            session_id=self.session_id,
            product_info=product_info,
            keywords=session["keywords"] or [],
            page=session["page"] or 1,
            asin=session["asin"],
            options=session["options"],
            instruction_text=session["goal"]["instruction_text"],
        )

        return 0.0, False

    def _do_item_click(self, clickable_name: str, clickable: Any) -> tuple[float, bool]:
        """Click on a product or option."""
        session = self.user_sessions[self.session_id]

        # Check if clicking a product link
        clickable_class = clickable.get("class") if hasattr(clickable, "get") else None
        if clickable_class and "product-link" in clickable_class:
            session["asin"] = clickable_name.upper()
            session["actions"]["asin"] += 1
            session["asins"].add(session["asin"])
        elif hasattr(clickable, "get") and clickable.get("name"):
            # Clicking an option
            clickable_key = clickable["name"].lower()
            session["options"][clickable_key] = clickable_name
            session["actions"]["options"] += 1

        if session["asin"] is None:
            return 0.0, False

        product_info = self.product_item_dict[session["asin"]]
        keywords_url_string = "+".join(session["keywords"] or [])
        option_string = json.dumps(session["options"])

        self.current_url = (
            f"{self.base_url}/item_page/{self.session_id}/"
            f"{session['asin']}/{keywords_url_string}/"
            f"{session['page']}/{option_string}"
        )

        self.page_source = map_action_to_html(
            "click",
            session_id=self.session_id,
            product_info=product_info,
            keywords=session["keywords"] or [],
            page=session["page"] or 1,
            asin=session["asin"],
            options=session["options"],
            instruction_text=session["goal"]["instruction_text"],
            show_attrs=False,
        )

        return 0.0, False

    def _get_page_name(self) -> str:
        """Determine current page type from URL."""
        if self.current_url is None:
            return ""
        page_names = ["search_results", "item_page", "item_sub_page", "done"]
        for page_name in page_names:
            if page_name in self.current_url:
                return page_name
        return ""

    @property
    def observation(self) -> str:
        """Get current observation in the configured mode."""
        if self.page_source is None:
            return ""

        if self.observation_mode == "html":
            return self.page_source
        elif self.observation_mode == "text":
            return self._convert_html_to_text(self.page_source)
        else:
            raise ValueError(f"Observation mode {self.observation_mode} not supported")

    def _convert_html_to_text(self, html: str) -> str:
        """Convert HTML to simplified text observation."""
        html_obj = BeautifulSoup(html, "html.parser")
        texts = html_obj.find_all(string=True)
        visible_texts = filter(tag_visible, texts)
        return " [SEP] ".join(t.strip() for t in visible_texts if t != "\n")

    def get_available_actions(self) -> dict:
        """Get available actions at current state."""
        if self.page_source is None:
            return {"has_search_bar": False, "clickables": []}

        html_obj = BeautifulSoup(self.page_source, "html.parser")

        search_bar = html_obj.find(id="search_input")
        has_search_bar = search_bar is not None

        buttons = html_obj.find_all(class_="btn")
        product_links = html_obj.find_all(class_="product-link")
        buying_options = html_obj.select('input[type="radio"]')

        self.text_to_clickable = {
            f"{b.get_text()}".lower(): b for b in buttons + product_links
        }
        for opt in buying_options:
            opt_value = opt.get("value")
            self.text_to_clickable[f"{opt_value}"] = opt

        return {
            "has_search_bar": has_search_bar,
            "clickables": list(self.text_to_clickable.keys()),
        }

    def get_goal(self) -> dict | None:
        """Get the current session's goal."""
        if self.session_id and self.session_id in self.user_sessions:
            return self.user_sessions[self.session_id].get("goal")
        return None

    def get_instruction(self) -> str | None:
        """Get the current instruction text."""
        return self.instruction_text

    def close(self):
        """Clean up resources."""
        pass
