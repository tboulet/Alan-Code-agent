"""Skill registry — central store for discovered skills.

The registry is created once at agent startup and held by AlanCodeAgent.
It's queried by the /skill command (repl.py) and the Skill tool.
"""

import logging

from alancode.skills.discovery import discover_skills
from alancode.skills.parser import SkillDefinition

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Central registry of all available skills."""

    def __init__(self, cwd: str):
        self._skills: dict[str, SkillDefinition] = {}
        self.reload(cwd)

    def reload(self, cwd: str) -> None:
        """Re-scan all skill sources and rebuild the index."""
        self._skills = discover_skills(cwd)
        logger.info("Skill registry loaded: %d skill(s)", len(self._skills))

    def get(self, name: str) -> SkillDefinition | None:
        """Look up a skill by name."""
        return self._skills.get(name)

    def list_all(self) -> list[SkillDefinition]:
        """Return all registered skills, sorted by name."""
        return sorted(self._skills.values(), key=lambda s: s.name)

    def expand(self, name: str, args: str = "") -> str | None:
        """Expand a skill's body template with arguments.

        Replaces ``$ARGUMENTS`` in the body with *args*.
        Returns None if the skill is not found.
        """
        skill = self.get(name)
        if skill is None:
            return None
        body = skill.body
        if args:
            body = body.replace("$ARGUMENTS", args)
        else:
            body = body.replace("$ARGUMENTS", "")
        return body

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills
