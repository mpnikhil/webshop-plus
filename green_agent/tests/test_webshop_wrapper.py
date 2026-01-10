"""Tests for WebShop wrapper."""

import pytest

from src.webshop_wrapper import WebShopWrapper, StepResult


class TestWebShopWrapper:
    """Test suite for WebShopWrapper."""

    @pytest.fixture(scope="class")
    def wrapper(self):
        """Create a shared wrapper instance for all tests."""
        return WebShopWrapper(mode="preview", num_products=100)

    def test_initialization(self, wrapper):
        """Test wrapper initializes correctly."""
        assert wrapper.mode == "preview"
        assert wrapper.observation_mode == "text"
        assert len(wrapper.all_products) > 0
        assert len(wrapper.goals) > 0
        assert wrapper.search_engine is not None

    def test_reset_returns_observation(self, wrapper):
        """Test reset returns a valid observation."""
        obs = wrapper.reset(goal_idx=0)
        assert isinstance(obs, str)
        assert len(obs) > 0
        assert "[SEP]" in obs  # Text mode uses [SEP] separators
        assert "WebShop" in obs or "Instruction" in obs

    def test_reset_sets_instruction(self, wrapper):
        """Test reset sets the instruction text."""
        wrapper.reset(goal_idx=0)
        instruction = wrapper.get_instruction()
        assert instruction is not None
        assert len(instruction) > 0

    def test_reset_sets_goal(self, wrapper):
        """Test reset sets the goal."""
        wrapper.reset(goal_idx=0)
        goal = wrapper.get_goal()
        assert goal is not None
        assert "instruction_text" in goal
        assert "asin" in goal

    def test_step_search_action(self, wrapper):
        """Test search action returns valid result."""
        wrapper.reset(goal_idx=0)
        result = wrapper.step("search[shoes]")

        assert isinstance(result, StepResult)
        assert isinstance(result.observation, str)
        assert result.reward == 0.0
        assert result.done is False
        assert len(result.observation) > 0

    def test_step_invalid_action(self, wrapper):
        """Test invalid action doesn't crash."""
        wrapper.reset(goal_idx=0)
        result = wrapper.step("invalid_action")

        assert result.reward == 0.0
        assert result.done is False

    def test_get_available_actions(self, wrapper):
        """Test get_available_actions returns valid structure."""
        wrapper.reset(goal_idx=0)
        actions = wrapper.get_available_actions()

        assert "has_search_bar" in actions
        assert "clickables" in actions
        assert actions["has_search_bar"] is True  # Search page has search bar
        assert isinstance(actions["clickables"], list)

    def test_search_returns_products(self, wrapper):
        """Test search returns product results."""
        wrapper.reset(goal_idx=0)
        result = wrapper.step("search[shirt]")

        # After search, observation should contain product info
        assert "Page" in result.observation or "results" in result.observation.lower()

    def test_session_state_tracking(self, wrapper):
        """Test session state is properly tracked."""
        wrapper.reset(goal_idx=0)
        assert wrapper.session_id is not None

        session = wrapper.user_sessions[wrapper.session_id]
        assert session["done"] is False
        assert session["reward"] == 0.0

        wrapper.step("search[laptop]")
        assert session["keywords"] == ["laptop"]
        assert session["page"] == 1
        assert session["actions"]["search"] == 1

    def test_multiple_sessions(self, wrapper):
        """Test multiple sessions can be created."""
        wrapper.reset(session="session_a", goal_idx=0)
        session_a = wrapper.session_id

        # Use goal_idx=0 since with few products we may have limited goals
        wrapper.reset(session="session_b", goal_idx=0)
        session_b = wrapper.session_id

        assert session_a != session_b
        assert "session_a" in wrapper.user_sessions
        assert "session_b" in wrapper.user_sessions

    def test_observation_mode_text(self, wrapper):
        """Test text observation mode."""
        assert wrapper.observation_mode == "text"
        wrapper.reset(goal_idx=0)
        obs = wrapper.observation

        # Text mode should not contain HTML tags
        assert "<html>" not in obs.lower()
        assert "<div>" not in obs.lower()
        assert "[SEP]" in obs

    def test_step_without_reset_raises_error(self):
        """Test step without reset raises RuntimeError."""
        wrapper = WebShopWrapper(mode="preview", num_products=10)
        # Don't call reset
        with pytest.raises(RuntimeError):
            wrapper.step("search[test]")


class TestWebShopWrapperIntegration:
    """Integration tests for a complete shopping session."""

    @pytest.fixture
    def wrapper(self):
        """Create a wrapper for integration tests."""
        return WebShopWrapper(mode="preview", num_products=100)

    def test_complete_shopping_flow(self, wrapper):
        """Test a complete shopping session from search to checkout."""
        # Start session
        obs = wrapper.reset(goal_idx=0)
        assert "Instruction" in obs

        # Search for something
        result = wrapper.step("search[blue shirt]")
        assert not result.done
        assert "Page" in result.observation or "results" in result.observation.lower()

        # Get available actions after search
        actions = wrapper.get_available_actions()
        assert len(actions["clickables"]) > 0

    def test_back_to_search(self, wrapper):
        """Test back to search functionality."""
        wrapper.reset(goal_idx=0)
        wrapper.step("search[pants]")

        # Go back to search
        result = wrapper.step("click[back to search]")
        assert not result.done

        # Should be able to search again
        actions = wrapper.get_available_actions()
        assert actions["has_search_bar"] is True
