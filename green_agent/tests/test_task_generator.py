"""
Tests for the TaskGenerator class.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.models import (
    BudgetConstrainedTask,
    ComparativeReasoningTask,
    Difficulty,
    ErrorRecoveryTask,
    NegativeConstraintTask,
    PreferenceMemoryTask,
    TaskType,
)
from src.task_generator import TaskGenerator


class TestTaskGeneratorInit:
    """Tests for TaskGenerator initialization."""

    def test_init_default_path(self):
        """Test initialization with default tasks directory."""
        tg = TaskGenerator()
        assert len(tg) == 80

    def test_init_custom_path(self):
        """Test initialization with custom tasks directory."""
        # Use the actual tasks directory
        tasks_dir = (
            Path(__file__).parent.parent / "data" / "tasks"
        )
        tg = TaskGenerator(str(tasks_dir))
        assert len(tg) == 80

    def test_init_nonexistent_directory(self):
        """Test initialization with non-existent directory."""
        with pytest.raises(FileNotFoundError, match="not found"):
            TaskGenerator("/nonexistent/path")

    def test_init_empty_directory(self):
        """Test initialization with empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="No JSON files"):
                TaskGenerator(tmpdir)

    def test_init_invalid_json_format(self):
        """Test initialization with invalid JSON structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a JSON object instead of array
            bad_file = Path(tmpdir) / "bad.json"
            bad_file.write_text('{"not": "an array"}')
            with pytest.raises(ValueError, match="Expected JSON array"):
                TaskGenerator(tmpdir)

    def test_init_duplicate_task_id(self):
        """Test initialization with duplicate task IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tasks = [
                {
                    "task_id": "dup_001",
                    "task_type": "budget_constrained",
                    "instruction": "Test 1",
                    "constraints": {
                        "budget": 50.0,
                        "required_items": [],
                    },
                },
                {
                    "task_id": "dup_001",  # Duplicate!
                    "task_type": "budget_constrained",
                    "instruction": "Test 2",
                    "constraints": {
                        "budget": 100.0,
                        "required_items": [],
                    },
                },
            ]
            task_file = Path(tmpdir) / "tasks.json"
            task_file.write_text(json.dumps(tasks))
            with pytest.raises(ValueError, match="Duplicate task ID"):
                TaskGenerator(tmpdir)


class TestGetTask:
    """Tests for get_task method."""

    @pytest.fixture
    def task_generator(self):
        return TaskGenerator()

    def test_get_existing_task(self, task_generator):
        """Test retrieving an existing task."""
        task = task_generator.get_task("budget_001")
        assert task.task_id == "budget_001"
        assert task.task_type == TaskType.BUDGET_CONSTRAINED

    def test_get_task_returns_correct_type(self, task_generator):
        """Test that get_task returns the correct task type."""
        budget_task = task_generator.get_task("budget_001")
        assert isinstance(budget_task, BudgetConstrainedTask)

        # Get a preference memory task
        memory_tasks = task_generator.get_tasks_by_type(TaskType.PREFERENCE_MEMORY)
        if memory_tasks:
            memory_task = task_generator.get_task(memory_tasks[0].task_id)
            assert isinstance(memory_task, PreferenceMemoryTask)

    def test_get_nonexistent_task(self, task_generator):
        """Test retrieving a non-existent task."""
        with pytest.raises(KeyError, match="Task not found"):
            task_generator.get_task("nonexistent_task")


class TestGetTasksByType:
    """Tests for get_tasks_by_type method."""

    @pytest.fixture
    def task_generator(self):
        return TaskGenerator()

    def test_get_budget_tasks(self, task_generator):
        """Test retrieving budget-constrained tasks."""
        tasks = task_generator.get_tasks_by_type(TaskType.BUDGET_CONSTRAINED)
        assert len(tasks) == 20
        for task in tasks:
            assert isinstance(task, BudgetConstrainedTask)

    def test_get_preference_memory_tasks(self, task_generator):
        """Test retrieving preference memory tasks."""
        tasks = task_generator.get_tasks_by_type(TaskType.PREFERENCE_MEMORY)
        assert len(tasks) == 15
        for task in tasks:
            assert isinstance(task, PreferenceMemoryTask)

    def test_get_negative_constraint_tasks(self, task_generator):
        """Test retrieving negative constraint tasks."""
        tasks = task_generator.get_tasks_by_type(TaskType.NEGATIVE_CONSTRAINT)
        assert len(tasks) == 20
        for task in tasks:
            assert isinstance(task, NegativeConstraintTask)

    def test_get_comparative_reasoning_tasks(self, task_generator):
        """Test retrieving comparative reasoning tasks."""
        tasks = task_generator.get_tasks_by_type(TaskType.COMPARATIVE_REASONING)
        assert len(tasks) == 15
        for task in tasks:
            assert isinstance(task, ComparativeReasoningTask)

    def test_get_error_recovery_tasks(self, task_generator):
        """Test retrieving error recovery tasks."""
        tasks = task_generator.get_tasks_by_type(TaskType.ERROR_RECOVERY)
        assert len(tasks) == 10
        for task in tasks:
            assert isinstance(task, ErrorRecoveryTask)

    def test_get_tasks_by_type_string(self, task_generator):
        """Test using string for task type."""
        tasks = task_generator.get_tasks_by_type("budget_constrained")
        assert len(tasks) == 20

    def test_get_tasks_invalid_type(self, task_generator):
        """Test with invalid task type."""
        with pytest.raises(ValueError, match="Invalid task type"):
            task_generator.get_tasks_by_type("invalid_type")

    def test_get_tasks_by_type_with_difficulty(self, task_generator):
        """Test filtering by type and difficulty."""
        # Get easy budget tasks
        tasks = task_generator.get_tasks_by_type(
            TaskType.BUDGET_CONSTRAINED,
            difficulty=Difficulty.EASY,
        )
        assert all(t.difficulty == Difficulty.EASY for t in tasks)
        assert all(t.task_type == TaskType.BUDGET_CONSTRAINED for t in tasks)

    def test_get_tasks_by_type_with_difficulty_string(self, task_generator):
        """Test using string for difficulty."""
        tasks = task_generator.get_tasks_by_type("budget_constrained", difficulty="easy")
        assert all(t.difficulty == Difficulty.EASY for t in tasks)

    def test_get_tasks_invalid_difficulty(self, task_generator):
        """Test with invalid difficulty."""
        with pytest.raises(ValueError, match="Invalid difficulty"):
            task_generator.get_tasks_by_type(TaskType.BUDGET_CONSTRAINED, difficulty="super_hard")


class TestGetAllTasks:
    """Tests for get_all_tasks method."""

    @pytest.fixture
    def task_generator(self):
        return TaskGenerator()

    def test_get_all_tasks(self, task_generator):
        """Test retrieving all tasks."""
        tasks = task_generator.get_all_tasks()
        assert len(tasks) == 80

    def test_get_all_tasks_sum_matches(self, task_generator):
        """Test that sum of all types equals total."""
        total = 0
        for task_type in TaskType:
            total += len(task_generator.get_tasks_by_type(task_type))
        assert total == 80

    def test_get_all_tasks_with_difficulty(self, task_generator):
        """Test filtering all tasks by difficulty."""
        easy_tasks = task_generator.get_all_tasks(difficulty=Difficulty.EASY)
        medium_tasks = task_generator.get_all_tasks(difficulty=Difficulty.MEDIUM)
        hard_tasks = task_generator.get_all_tasks(difficulty=Difficulty.HARD)

        # All should have correct difficulty
        assert all(t.difficulty == Difficulty.EASY for t in easy_tasks)
        assert all(t.difficulty == Difficulty.MEDIUM for t in medium_tasks)
        assert all(t.difficulty == Difficulty.HARD for t in hard_tasks)

        # Sum should equal total
        assert len(easy_tasks) + len(medium_tasks) + len(hard_tasks) == 80

    def test_get_all_tasks_invalid_difficulty(self, task_generator):
        """Test with invalid difficulty string."""
        with pytest.raises(ValueError, match="Invalid difficulty"):
            task_generator.get_all_tasks(difficulty="impossible")


class TestGetRandomTask:
    """Tests for get_random_task method."""

    @pytest.fixture
    def task_generator(self):
        return TaskGenerator()

    def test_get_random_task(self, task_generator):
        """Test getting a random task."""
        task = task_generator.get_random_task()
        assert task is not None
        assert task.task_id in task_generator

    def test_get_random_task_by_type(self, task_generator):
        """Test getting a random task of specific type."""
        task = task_generator.get_random_task(task_type=TaskType.BUDGET_CONSTRAINED)
        assert task.task_type == TaskType.BUDGET_CONSTRAINED

    def test_get_random_task_by_type_string(self, task_generator):
        """Test using string for task type."""
        task = task_generator.get_random_task(task_type="budget_constrained")
        assert task.task_type == TaskType.BUDGET_CONSTRAINED

    def test_get_random_task_by_difficulty(self, task_generator):
        """Test getting a random task of specific difficulty."""
        task = task_generator.get_random_task(difficulty=Difficulty.HARD)
        assert task.difficulty == Difficulty.HARD

    def test_get_random_task_by_type_and_difficulty(self, task_generator):
        """Test filtering by both type and difficulty."""
        task = task_generator.get_random_task(
            task_type=TaskType.BUDGET_CONSTRAINED,
            difficulty=Difficulty.EASY,
        )
        assert task.task_type == TaskType.BUDGET_CONSTRAINED
        assert task.difficulty == Difficulty.EASY

    def test_get_random_task_distribution(self, task_generator):
        """Test that random selection is roughly uniform."""
        # Get 100 random budget tasks and check we get variety
        task_ids = set()
        for _ in range(100):
            task = task_generator.get_random_task(task_type=TaskType.BUDGET_CONSTRAINED)
            task_ids.add(task.task_id)

        # With 20 budget tasks and 100 samples, we should see at least 5 different tasks
        assert len(task_ids) >= 5


class TestGetTaskCount:
    """Tests for get_task_count method."""

    @pytest.fixture
    def task_generator(self):
        return TaskGenerator()

    def test_total_count(self, task_generator):
        """Test total task count."""
        assert task_generator.get_task_count() == 80

    def test_count_by_type(self, task_generator):
        """Test count by task type."""
        assert task_generator.get_task_count(TaskType.BUDGET_CONSTRAINED) == 20
        assert task_generator.get_task_count(TaskType.PREFERENCE_MEMORY) == 15
        assert task_generator.get_task_count(TaskType.NEGATIVE_CONSTRAINT) == 20
        assert task_generator.get_task_count(TaskType.COMPARATIVE_REASONING) == 15
        assert task_generator.get_task_count(TaskType.ERROR_RECOVERY) == 10

    def test_count_by_difficulty(self, task_generator):
        """Test count by difficulty."""
        easy = task_generator.get_task_count(difficulty=Difficulty.EASY)
        medium = task_generator.get_task_count(difficulty=Difficulty.MEDIUM)
        hard = task_generator.get_task_count(difficulty=Difficulty.HARD)
        assert easy + medium + hard == 80


class TestGetTaskIds:
    """Tests for get_task_ids method."""

    @pytest.fixture
    def task_generator(self):
        return TaskGenerator()

    def test_get_all_task_ids(self, task_generator):
        """Test getting all task IDs."""
        ids = task_generator.get_task_ids()
        assert len(ids) == 80
        assert "budget_001" in ids

    def test_get_task_ids_by_type(self, task_generator):
        """Test getting task IDs by type."""
        ids = task_generator.get_task_ids(TaskType.BUDGET_CONSTRAINED)
        assert len(ids) == 20
        assert all(id.startswith("budget_") for id in ids)

    def test_get_task_ids_by_difficulty(self, task_generator):
        """Test getting task IDs by difficulty."""
        easy_ids = task_generator.get_task_ids(difficulty=Difficulty.EASY)
        assert len(easy_ids) > 0


class TestDunderMethods:
    """Tests for __len__ and __contains__."""

    @pytest.fixture
    def task_generator(self):
        return TaskGenerator()

    def test_len(self, task_generator):
        """Test __len__ method."""
        assert len(task_generator) == 80

    def test_contains_existing(self, task_generator):
        """Test __contains__ with existing task."""
        assert "budget_001" in task_generator

    def test_contains_nonexistent(self, task_generator):
        """Test __contains__ with non-existent task."""
        assert "nonexistent_task" not in task_generator


class TestTaskValidation:
    """Tests for task data validation."""

    @pytest.fixture
    def task_generator(self):
        return TaskGenerator()

    def test_budget_task_has_constraints(self, task_generator):
        """Test that budget tasks have valid constraints."""
        task = task_generator.get_task("budget_001")
        assert hasattr(task, "constraints")
        assert task.constraints.budget > 0
        assert isinstance(task.constraints.required_items, list)

    def test_all_tasks_have_required_fields(self, task_generator):
        """Test that all tasks have required base fields."""
        for task in task_generator.get_all_tasks():
            assert task.task_id
            assert task.task_type in TaskType
            assert task.instruction
            assert task.difficulty in Difficulty
            assert task.expected_actions > 0
            assert task.timeout_seconds > 0

    def test_task_ids_are_unique(self, task_generator):
        """Test that all task IDs are unique."""
        ids = task_generator.get_task_ids()
        assert len(ids) == len(set(ids))


class TestEdgeCases:
    """Tests for edge cases."""

    def test_custom_tasks_directory(self):
        """Test loading tasks from custom directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid task file
            tasks = [
                {
                    "task_id": "custom_001",
                    "task_type": "budget_constrained",
                    "instruction": "Custom task",
                    "constraints": {
                        "budget": 100.0,
                        "required_items": [
                            {"category": "test", "attributes": {}, "optional": False}
                        ],
                    },
                }
            ]
            task_file = Path(tmpdir) / "custom.json"
            task_file.write_text(json.dumps(tasks))

            tg = TaskGenerator(tmpdir)
            assert len(tg) == 1
            assert "custom_001" in tg

    def test_multiple_json_files(self):
        """Test loading from multiple JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two task files
            for i, suffix in enumerate(["a", "b"]):
                tasks = [
                    {
                        "task_id": f"multi_{suffix}_{j}",
                        "task_type": "budget_constrained",
                        "instruction": f"Task {suffix}{j}",
                        "constraints": {
                            "budget": 50.0,
                            "required_items": [],
                        },
                    }
                    for j in range(3)
                ]
                task_file = Path(tmpdir) / f"tasks_{suffix}.json"
                task_file.write_text(json.dumps(tasks))

            tg = TaskGenerator(tmpdir)
            assert len(tg) == 6  # 3 tasks * 2 files
