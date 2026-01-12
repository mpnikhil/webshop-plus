"""
Shopping Agent for WebShop+ Purple Agent.

This module implements a baseline shopping agent that:
1. Parses task instructions to understand requirements
2. Generates search queries
3. Analyzes product listings
4. Makes purchase decisions based on constraints

The agent uses an LLM (via LiteLLM) to make decisions and maintains
conversation context within a shopping session.
"""

import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import structlog
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Role, TaskState, TextPart
from pydantic import BaseModel

from src.llm_client import LLMClient
from src.messenger import get_message_text

logger = structlog.get_logger()


# =============================================================================
# Enums and Constants
# =============================================================================


class AgentState(str, Enum):
    """States of the shopping agent."""

    IDLE = "idle"
    SEARCHING = "searching"
    BROWSING = "browsing"
    VIEWING_PRODUCT = "viewing_product"
    SELECTING_OPTIONS = "selecting_options"
    PURCHASING = "purchasing"
    COMPLETED = "completed"
    FAILED = "failed"


class ActionType(str, Enum):
    """Types of WebShop actions."""

    SEARCH = "search"
    CLICK = "click"


# =============================================================================
# Data Models
# =============================================================================


class TaskRequirements(BaseModel):
    """Parsed requirements from a task instruction."""

    product_type: str = ""
    budget: Optional[float] = None
    preferences: list[str] = []
    constraints: list[str] = []  # Things to avoid
    comparison_required: bool = False
    raw_instruction: str = ""


class ProductInfo(BaseModel):
    """Information about a product parsed from observations."""

    asin: Optional[str] = None
    name: str = ""
    price: Optional[float] = None
    options: list[str] = []
    attributes: dict[str, str] = {}
    description: str = ""


class SessionContext(BaseModel):
    """Context maintained during a shopping session."""

    task_id: str = ""
    requirements: Optional[TaskRequirements] = None
    state: AgentState = AgentState.IDLE
    action_history: list[str] = []
    observation_history: list[str] = []
    products_seen: list[ProductInfo] = []
    current_product: Optional[ProductInfo] = None
    selected_options: dict[str, str] = {}
    search_attempts: int = 0
    max_search_attempts: int = 3


# =============================================================================
# Prompt Templates
# =============================================================================


PARSE_TASK_PROMPT = """You are a shopping assistant. Parse the following shopping task instruction and extract:
1. The product type to search for
2. Budget constraint (if any)
3. Preferences (features wanted)
4. Constraints (things to avoid)
5. Whether comparison between products is required

Task instruction:
{instruction}

Respond in this exact format:
PRODUCT_TYPE: <the main product to search for>
BUDGET: <number or "none">
PREFERENCES: <comma-separated list or "none">
CONSTRAINTS: <comma-separated list or "none">
COMPARISON_REQUIRED: <yes or no>
SEARCH_QUERY: <suggested search query>"""


DECIDE_ACTION_PROMPT = """You are a shopping agent navigating an online store. Based on the current state, decide the next action.

Task: {task}

Current state: {state}
Previous actions: {actions}

Current observation:
{observation}

Requirements:
- Product: {product_type}
- Budget: {budget}
- Preferences: {preferences}
- Constraints: {constraints}

Available actions:
- search[query] - Search for products
- click[element] - Click on a product, option, or button

What should be the next action? Consider:
1. If on search results, click on a promising product
2. If on a product page, check if it meets requirements
3. If product is good, select options and buy
4. If product doesn't meet requirements, go back and try another

Respond with ONLY the action in the format: search[query] or click[element]
DO NOT include any explanation, just the action."""


SELECT_PRODUCT_PROMPT = """You are a shopping agent. Given these search results, select the best product that meets the requirements.

Requirements:
- Product: {product_type}
- Budget: {budget}
- Preferences: {preferences}
- Avoid: {constraints}

Search results:
{products}

Which product should I click on? Respond with ONLY the product identifier (ASIN or name) that I should click.
If no product meets the requirements, respond with "none"."""


PRODUCT_EVALUATION_PROMPT = """You are a shopping agent. Evaluate if this product meets the requirements.

Product:
{product_info}

Requirements:
- Budget: {budget}
- Preferences: {preferences}
- Avoid: {constraints}

Answer these questions:
1. Does it meet the budget? (yes/no)
2. Does it have the preferred features? (yes/no/partial)
3. Does it violate any constraints? (yes/no)
4. Should I buy this product? (yes/no)

Respond in this format:
MEETS_BUDGET: <yes or no>
HAS_PREFERENCES: <yes, no, or partial>
VIOLATES_CONSTRAINTS: <yes or no>
SHOULD_BUY: <yes or no>
REASON: <brief explanation>"""


# =============================================================================
# Shopping Agent
# =============================================================================


@dataclass
class ShopperAgent:
    """
    Baseline shopping agent for WebShop+.

    This agent uses an LLM to understand shopping tasks and decide actions.
    It maintains session context to track progress through a shopping task.

    Example:
        >>> agent = ShopperAgent()
        >>> action = agent.process_task_instruction("Find running shoes under $100")
        >>> print(action)  # "search[running shoes]"
        >>> action = agent.process_observation("Found 10 products: ...")
        >>> print(action)  # "click[B07XYZ123]"
    """

    llm_client: LLMClient = field(default_factory=LLMClient)
    context: SessionContext = field(default_factory=SessionContext)

    def reset(self, task_id: str = "") -> None:
        """Reset the agent for a new task.

        Args:
            task_id: Optional task ID for tracking.
        """
        self.context = SessionContext(task_id=task_id)
        logger.info("Agent reset", task_id=task_id)

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Process a message and update status via TaskUpdater.

        This method implements the agent-template Agent interface for SDK integration.

        IMPORTANT: Actions are returned as messages (not artifacts) because
        the green agent's parse_action_from_response() looks for actions in:
        1. result.history[].role=="agent" -> parts -> text
        2. result.message -> parts -> text

        Args:
            message: Incoming A2A SDK Message.
            updater: TaskUpdater for status/artifact updates.
        """
        text = get_message_text(message)
        logger.info("Agent.run() called", text_length=len(text))

        # Signal that we're working on the task
        await updater.update_status(
            state=TaskState.working,
            message=Message(
                messageId=str(uuid.uuid4()),
                role=Role.agent,
                parts=[TextPart(text="Processing...")]
            )
        )

        # Use agent state to determine message type:
        # - No requirements set = new task instruction
        # - Requirements set = observation from environment
        if self.context.requirements is None:
            # New task instruction - first message in a session
            action = self.process_task_instruction(text)
        else:
            # Ongoing session - this is an observation
            action = self.process_observation(text)

        logger.info("Agent.run() completed", action=action)

        # Return action as a message - green agent parses from messages, not artifacts
        await updater.complete(
            message=Message(
                messageId=str(uuid.uuid4()),
                role=Role.agent,
                parts=[TextPart(text=action)]
            )
        )

    def process_task_instruction(self, instruction: str) -> str:
        """Process a task instruction and return the first action.

        Args:
            instruction: The shopping task instruction.

        Returns:
            The first action to take (usually a search).
        """
        logger.info("Processing task instruction", instruction=instruction[:100])

        # Parse the task requirements
        self.context.requirements = self._parse_task(instruction)
        self.context.state = AgentState.SEARCHING
        self.context.observation_history.append(f"TASK: {instruction}")

        # Generate initial search query
        search_query = self._generate_search_query()
        action = f"search[{search_query}]"

        self.context.action_history.append(action)
        self.context.search_attempts += 1
        logger.info("Generated initial action", action=action)

        return action

    def process_observation(self, observation: str) -> str:
        """Process an observation and return the next action.

        Args:
            observation: The observation from WebShop (search results, product page, etc.)

        Returns:
            The next action to take.
        """
        logger.info("Processing observation", observation_length=len(observation))
        self.context.observation_history.append(observation)

        # Detect the type of observation
        obs_type = self._detect_observation_type(observation)
        logger.debug("Detected observation type", obs_type=obs_type)

        # Decide action based on observation type
        if obs_type == "search_results":
            action = self._handle_search_results(observation)
        elif obs_type == "product_page":
            action = self._handle_product_page(observation)
        elif obs_type == "purchase_complete":
            action = self._handle_purchase_complete(observation)
        elif obs_type == "error":
            action = self._handle_error(observation)
        else:
            # Use LLM to decide
            action = self._decide_action_with_llm(observation)

        self.context.action_history.append(action)
        logger.info("Decided action", action=action)

        return action

    def process_error(self, error_message: str) -> str:
        """Process an error notice and return a recovery action.

        Args:
            error_message: The error message.

        Returns:
            A recovery action.
        """
        logger.warning("Processing error", error=error_message)
        self.context.observation_history.append(f"ERROR: {error_message}")

        # Try to recover by searching again with a different query
        if self.context.search_attempts < self.context.max_search_attempts:
            self.context.search_attempts += 1
            query = self._generate_alternative_search_query()
            action = f"search[{query}]"
        else:
            # Give up
            action = "click[back to search]"
            self.context.state = AgentState.FAILED

        self.context.action_history.append(action)
        return action

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _parse_task(self, instruction: str) -> TaskRequirements:
        """Parse task instruction to extract requirements."""
        prompt = PARSE_TASK_PROMPT.format(instruction=instruction)

        try:
            response = self.llm_client.complete(
                [{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=512,
            )
            return self._parse_requirements_response(response, instruction)
        except Exception as e:
            logger.error("Failed to parse task with LLM", error=str(e))
            # Fallback to simple extraction
            return self._extract_requirements_simple(instruction)

    def _parse_requirements_response(self, response: str, instruction: str) -> TaskRequirements:
        """Parse the LLM response for task requirements."""
        req = TaskRequirements(raw_instruction=instruction)

        for line in response.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("PRODUCT_TYPE:"):
                req.product_type = line.split(":", 1)[1].strip()
            elif line.upper().startswith("BUDGET:"):
                budget_str = line.split(":", 1)[1].strip().lower()
                if budget_str != "none":
                    # Extract number from string like "$100" or "100"
                    match = re.search(r"[\d.]+", budget_str)
                    if match:
                        req.budget = float(match.group())
            elif line.upper().startswith("PREFERENCES:"):
                prefs = line.split(":", 1)[1].strip()
                if prefs.lower() != "none":
                    req.preferences = [p.strip() for p in prefs.split(",")]
            elif line.upper().startswith("CONSTRAINTS:"):
                cons = line.split(":", 1)[1].strip()
                if cons.lower() != "none":
                    req.constraints = [c.strip() for c in cons.split(",")]
            elif line.upper().startswith("COMPARISON_REQUIRED:"):
                req.comparison_required = "yes" in line.lower()

        # Ensure we have a product type
        if not req.product_type:
            req.product_type = instruction.split()[0:3]  # Take first 3 words
            req.product_type = " ".join(req.product_type) if isinstance(req.product_type, list) else str(req.product_type)

        logger.info("Parsed requirements", requirements=req.model_dump())
        return req

    def _extract_requirements_simple(self, instruction: str) -> TaskRequirements:
        """Simple fallback extraction without LLM."""
        req = TaskRequirements(raw_instruction=instruction)

        # Extract budget
        budget_match = re.search(r"\$(\d+(?:\.\d{2})?)", instruction)
        if budget_match:
            req.budget = float(budget_match.group(1))

        # Extract product type (first noun phrase)
        words = instruction.lower().split()
        for i, word in enumerate(words):
            if word in ["find", "buy", "get", "purchase", "search"]:
                # Take the next few words as product type
                req.product_type = " ".join(words[i + 1 : i + 4])
                break

        if not req.product_type:
            req.product_type = " ".join(words[:4])

        return req

    def _generate_search_query(self) -> str:
        """Generate a search query from requirements."""
        if not self.context.requirements:
            return "products"

        req = self.context.requirements
        query_parts = [req.product_type]

        # Add key preferences to search
        if req.preferences:
            query_parts.extend(req.preferences[:2])

        return " ".join(query_parts)

    def _generate_alternative_search_query(self) -> str:
        """Generate an alternative search query after a failure."""
        if not self.context.requirements:
            return "products"

        req = self.context.requirements
        # Try a simpler query
        return req.product_type

    def _detect_observation_type(self, observation: str) -> str:
        """Detect the type of observation."""
        obs_lower = observation.lower()

        if "search results" in obs_lower or "products found" in obs_lower or "[search]" in obs_lower:
            return "search_results"
        elif "product page" in obs_lower or "price:" in obs_lower or "options:" in obs_lower:
            return "product_page"
        elif "purchased" in obs_lower or "order placed" in obs_lower or "thank you" in obs_lower:
            return "purchase_complete"
        elif "error" in obs_lower or "not found" in obs_lower or "invalid" in obs_lower:
            return "error"
        else:
            return "unknown"

    def _handle_search_results(self, observation: str) -> str:
        """Handle search results observation."""
        self.context.state = AgentState.BROWSING

        # Parse products from observation
        products = self._parse_products_from_observation(observation)
        self.context.products_seen.extend(products)

        if not products:
            # No products found, try alternative search
            if self.context.search_attempts < self.context.max_search_attempts:
                self.context.search_attempts += 1
                query = self._generate_alternative_search_query()
                return f"search[{query}]"
            else:
                return "click[back to search]"

        # Select best product
        product_to_click = self._select_best_product(products, observation)
        if product_to_click:
            return f"click[{product_to_click}]"
        else:
            # No suitable product, try next page or alternative search
            return "click[next page]"

    def _handle_product_page(self, observation: str) -> str:
        """Handle product page observation."""
        self.context.state = AgentState.VIEWING_PRODUCT

        # Parse product info
        product = self._parse_product_info(observation)
        self.context.current_product = product

        # Evaluate if product meets requirements
        should_buy, reason = self._evaluate_product(product, observation)
        logger.info("Product evaluation", should_buy=should_buy, reason=reason)

        if should_buy:
            # Check if options need to be selected
            if self._needs_option_selection(observation):
                self.context.state = AgentState.SELECTING_OPTIONS
                option = self._select_option(observation)
                return f"click[{option}]"
            else:
                # Buy the product
                self.context.state = AgentState.PURCHASING
                return "click[buy now]"
        else:
            # Go back to search results
            return "click[back to search]"

    def _handle_purchase_complete(self, observation: str) -> str:
        """Handle purchase completion."""
        self.context.state = AgentState.COMPLETED
        logger.info("Purchase completed")
        return "click[done]"

    def _handle_error(self, observation: str) -> str:
        """Handle error observation."""
        # Try to recover
        if self.context.search_attempts < self.context.max_search_attempts:
            self.context.search_attempts += 1
            query = self._generate_alternative_search_query()
            return f"search[{query}]"
        else:
            self.context.state = AgentState.FAILED
            return "click[back to search]"

    def _decide_action_with_llm(self, observation: str) -> str:
        """Use LLM to decide the next action."""
        req = self.context.requirements or TaskRequirements()

        prompt = DECIDE_ACTION_PROMPT.format(
            task=req.raw_instruction,
            state=self.context.state.value,
            actions=", ".join(self.context.action_history[-5:]) if self.context.action_history else "none",
            observation=observation[:2000],  # Truncate long observations
            product_type=req.product_type,
            budget=f"${req.budget}" if req.budget else "no limit",
            preferences=", ".join(req.preferences) if req.preferences else "none",
            constraints=", ".join(req.constraints) if req.constraints else "none",
        )

        try:
            response = self.llm_client.complete(
                [{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=100,
            )
            action = self._extract_action_from_response(response)
            return action
        except Exception as e:
            logger.error("LLM decision failed", error=str(e))
            # Fallback
            return "click[back to search]"

    def _extract_action_from_response(self, response: str) -> str:
        """Extract action from LLM response."""
        response = response.strip()

        # Look for action pattern
        match = re.search(r"(search|click)\[([^\]]+)\]", response, re.IGNORECASE)
        if match:
            action_type = match.group(1).lower()
            action_arg = match.group(2)
            return f"{action_type}[{action_arg}]"

        # Try to interpret the response
        response_lower = response.lower()
        if "search" in response_lower:
            # Extract what to search for
            for word in response.split():
                if word.lower() not in ["search", "for", "the", "a", "an"]:
                    return f"search[{word}]"
        elif "click" in response_lower or "buy" in response_lower:
            return "click[buy now]"

        # Default fallback
        return "click[back to search]"

    def _parse_products_from_observation(self, observation: str) -> list[ProductInfo]:
        """Parse product list from search results observation."""
        products = []

        # Look for product patterns like "B07XXX - Product Name - $XX.XX"
        # Or numbered lists like "1. Product Name - $XX.XX"
        lines = observation.split("\n")

        for line in lines:
            product = ProductInfo()

            # Try to extract ASIN
            asin_match = re.search(r"\b(B[0-9A-Z]{7,9})\b", line)
            if asin_match:
                product.asin = asin_match.group(1)

            # Try to extract price
            price_match = re.search(r"\$(\d+(?:\.\d{2})?)", line)
            if price_match:
                product.price = float(price_match.group(1))

            # Use the line as name if we found useful info
            if product.asin or product.price:
                product.name = line.strip()
                products.append(product)

        return products

    def _parse_product_info(self, observation: str) -> ProductInfo:
        """Parse product information from product page observation."""
        product = ProductInfo()

        lines = observation.split("\n")
        for line in lines:
            line_lower = line.lower()

            # Extract ASIN
            asin_match = re.search(r"\b(B[0-9A-Z]{7,9})\b", line)
            if asin_match:
                product.asin = asin_match.group(1)

            # Extract price
            price_match = re.search(r"\$(\d+(?:\.\d{2})?)", line)
            if price_match:
                product.price = float(price_match.group(1))

            # Extract name (usually the first substantial line)
            if not product.name and len(line.strip()) > 10 and not line_lower.startswith(("price", "option", "description")):
                product.name = line.strip()

            # Extract options
            if "option" in line_lower or "size" in line_lower or "color" in line_lower:
                product.options.append(line.strip())

            # Build description
            if line.strip():
                product.description += line.strip() + " "

        return product

    def _select_best_product(self, products: list[ProductInfo], observation: str) -> Optional[str]:
        """Select the best product from the list."""
        req = self.context.requirements

        # Filter by budget first
        if req and req.budget:
            products = [p for p in products if p.price is None or p.price <= req.budget]

        if not products:
            return None

        # Use LLM to select if we have multiple options
        if len(products) > 1:
            return self._select_product_with_llm(products, observation)

        # Return the first product
        product = products[0]
        return product.asin or product.name

    def _select_product_with_llm(self, products: list[ProductInfo], observation: str) -> Optional[str]:
        """Use LLM to select the best product."""
        req = self.context.requirements or TaskRequirements()

        products_text = "\n".join(
            f"- {p.asin or 'N/A'}: {p.name} - ${p.price if p.price else 'N/A'}"
            for p in products[:10]  # Limit to top 10
        )

        prompt = SELECT_PRODUCT_PROMPT.format(
            product_type=req.product_type,
            budget=f"${req.budget}" if req.budget else "no limit",
            preferences=", ".join(req.preferences) if req.preferences else "any",
            constraints=", ".join(req.constraints) if req.constraints else "none",
            products=products_text,
        )

        try:
            response = self.llm_client.complete(
                [{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=100,
            )
            response = response.strip()

            if response.lower() == "none":
                return None

            # Try to match response to a product
            for product in products:
                if product.asin and product.asin in response:
                    return product.asin
                if product.name and any(word in response.lower() for word in product.name.lower().split()[:3]):
                    return product.asin or product.name

            # Return first product as fallback
            return products[0].asin or products[0].name

        except Exception as e:
            logger.error("LLM product selection failed", error=str(e))
            return products[0].asin if products else None

    def _evaluate_product(self, product: ProductInfo, observation: str) -> tuple[bool, str]:
        """Evaluate if a product meets requirements."""
        req = self.context.requirements
        if not req:
            return True, "No requirements specified"

        # Quick checks first
        if req.budget and product.price and product.price > req.budget:
            return False, f"Price ${product.price} exceeds budget ${req.budget}"

        # Check constraints
        obs_lower = observation.lower()
        for constraint in req.constraints:
            if constraint.lower() in obs_lower:
                return False, f"Product violates constraint: {constraint}"

        # Use LLM for more complex evaluation
        prompt = PRODUCT_EVALUATION_PROMPT.format(
            product_info=observation[:1500],
            budget=f"${req.budget}" if req.budget else "no limit",
            preferences=", ".join(req.preferences) if req.preferences else "any",
            constraints=", ".join(req.constraints) if req.constraints else "none",
        )

        try:
            response = self.llm_client.complete(
                [{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=256,
            )

            # Parse response
            should_buy = "SHOULD_BUY: yes" in response.lower() or "should_buy: yes" in response.lower()
            reason_match = re.search(r"REASON:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
            reason = reason_match.group(1).strip() if reason_match else "LLM evaluation"

            return should_buy, reason

        except Exception as e:
            logger.error("LLM evaluation failed", error=str(e))
            # Fallback: buy if within budget
            if req.budget and product.price:
                return product.price <= req.budget, "Fallback: within budget"
            return True, "Fallback: no budget constraint"

    def _needs_option_selection(self, observation: str) -> bool:
        """Check if product options need to be selected."""
        obs_lower = observation.lower()
        return any(keyword in obs_lower for keyword in ["select size", "select color", "choose option", "options:"])

    def _select_option(self, observation: str) -> str:
        """Select a product option."""
        # Look for option patterns
        # Try to find the first available option
        match = re.search(r"(?:size|color|option):\s*\[([^\]]+)\]", observation, re.IGNORECASE)
        if match:
            options = match.group(1).split(",")
            return options[0].strip()

        # Look for button-like options
        match = re.search(r"\[([^\]]+)\]", observation)
        if match:
            return match.group(1)

        # Default
        return "option 1"
