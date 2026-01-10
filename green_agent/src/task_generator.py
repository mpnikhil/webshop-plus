"""
Task generator for WebShop+ green agent.

This module provides the TaskGenerator class that loads, validates, and serves
tasks from the JSON task files.
"""

import json
import random
from pathlib import Path
from typing import Optional

from .models import Difficulty, Task, TaskType, parse_task


class TaskGenerator:
    """Loads and manages tasks from JSON files.

    The TaskGenerator loads all tasks from a directory containing JSON task files,
    validates them using Pydantic models, and provides methods to retrieve tasks
    by ID, type, or difficulty.

    Attributes:
        tasks_dir: Path to the directory containing task JSON files.
    """

    def __init__(self, tasks_dir: Optional[str] = None):
        """Initialize the TaskGenerator.

        Args:
            tasks_dir: Path to directory containing task JSON files.
                      Defaults to 'data/tasks/' relative to the green_agent package.
        """
        if tasks_dir is None:
            # Default to data/tasks/ relative to this module
            module_dir = Path(__file__).parent.parent
            self.tasks_dir = module_dir / "data" / "tasks"
        else:
            self.tasks_dir = Path(tasks_dir)

        # Task caches
        self._tasks_by_id: dict[str, Task] = {}
        self._tasks_by_type: dict[TaskType, list[Task]] = {t: [] for t in TaskType}
        self._all_tasks: list[Task] = []

        # Load tasks on initialization
        self._load_tasks()

    def _load_tasks(self) -> None:
        """Load and validate all tasks from JSON files."""
        if not self.tasks_dir.exists():
            raise FileNotFoundError(f"Tasks directory not found: {self.tasks_dir}")

        json_files = list(self.tasks_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in: {self.tasks_dir}")

        for json_file in json_files:
            with open(json_file, "r", encoding="utf-8") as f:
                tasks_data = json.load(f)

            if not isinstance(tasks_data, list):
                raise ValueError(
                    f"Expected JSON array in {json_file.name}, got {type(tasks_data).__name__}"
                )

            for task_data in tasks_data:
                task = parse_task(task_data)

                # Check for duplicate task IDs
                if task.task_id in self._tasks_by_id:
                    raise ValueError(f"Duplicate task ID: {task.task_id}")

                # Add to caches
                self._tasks_by_id[task.task_id] = task
                self._tasks_by_type[task.task_type].append(task)
                self._all_tasks.append(task)

    def get_task(self, task_id: str) -> Task:
        """Get a specific task by ID.

        Args:
            task_id: The unique task identifier.

        Returns:
            The task with the specified ID.

        Raises:
            KeyError: If task ID not found.
        """
        if task_id not in self._tasks_by_id:
            raise KeyError(f"Task not found: {task_id}")
        return self._tasks_by_id[task_id]

    def get_tasks_by_type(
        self,
        task_type: str | TaskType,
        difficulty: Optional[str | Difficulty] = None,
    ) -> list[Task]:
        """Get all tasks of a specific type.

        Args:
            task_type: The task type (string or TaskType enum).
            difficulty: Optional difficulty filter (string or Difficulty enum).

        Returns:
            List of tasks matching the criteria.

        Raises:
            ValueError: If task_type is invalid.
        """
        # Convert string to enum if needed
        if isinstance(task_type, str):
            try:
                task_type = TaskType(task_type)
            except ValueError:
                raise ValueError(f"Invalid task type: {task_type}")

        tasks = self._tasks_by_type[task_type]

        # Filter by difficulty if specified
        if difficulty is not None:
            if isinstance(difficulty, str):
                try:
                    difficulty = Difficulty(difficulty)
                except ValueError:
                    raise ValueError(f"Invalid difficulty: {difficulty}")
            tasks = [t for t in tasks if t.difficulty == difficulty]

        return tasks

    def get_all_tasks(
        self,
        difficulty: Optional[str | Difficulty] = None,
    ) -> list[Task]:
        """Get all tasks.

        Args:
            difficulty: Optional difficulty filter (string or Difficulty enum).

        Returns:
            List of all tasks, optionally filtered by difficulty.
        """
        tasks = self._all_tasks

        if difficulty is not None:
            if isinstance(difficulty, str):
                try:
                    difficulty = Difficulty(difficulty)
                except ValueError:
                    raise ValueError(f"Invalid difficulty: {difficulty}")
            tasks = [t for t in tasks if t.difficulty == difficulty]

        return tasks

    def get_random_task(
        self,
        task_type: Optional[str | TaskType] = None,
        difficulty: Optional[str | Difficulty] = None,
    ) -> Task:
        """Get a random task, optionally filtered by type and/or difficulty.

        Args:
            task_type: Optional task type filter.
            difficulty: Optional difficulty filter.

        Returns:
            A randomly selected task matching the criteria.

        Raises:
            ValueError: If no tasks match the criteria.
        """
        if task_type is not None:
            tasks = self.get_tasks_by_type(task_type, difficulty)
        else:
            tasks = self.get_all_tasks(difficulty)

        if not tasks:
            raise ValueError("No tasks match the specified criteria")

        return random.choice(tasks)

    def get_task_count(
        self,
        task_type: Optional[str | TaskType] = None,
        difficulty: Optional[str | Difficulty] = None,
    ) -> int:
        """Get the count of tasks matching criteria.

        Args:
            task_type: Optional task type filter.
            difficulty: Optional difficulty filter.

        Returns:
            Number of tasks matching the criteria.
        """
        if task_type is not None:
            return len(self.get_tasks_by_type(task_type, difficulty))
        return len(self.get_all_tasks(difficulty))

    def get_task_ids(
        self,
        task_type: Optional[str | TaskType] = None,
        difficulty: Optional[str | Difficulty] = None,
    ) -> list[str]:
        """Get list of task IDs matching criteria.

        Args:
            task_type: Optional task type filter.
            difficulty: Optional difficulty filter.

        Returns:
            List of task IDs matching the criteria.
        """
        if task_type is not None:
            tasks = self.get_tasks_by_type(task_type, difficulty)
        else:
            tasks = self.get_all_tasks(difficulty)

        return [t.task_id for t in tasks]

    def __len__(self) -> int:
        """Return total number of tasks."""
        return len(self._all_tasks)

    def __contains__(self, task_id: str) -> bool:
        """Check if a task ID exists."""
        return task_id in self._tasks_by_id
