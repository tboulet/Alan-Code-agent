"""Skill discovery — scan directories for SKILL.md files.

Discovery sources (highest priority wins):
1. Project skills — .alan/skills/<name>/SKILL.md
2. User skills — ~/.alan/skills/<name>/SKILL.md
3. Built-in skills — Python dict in builtin.py
"""

import logging
from pathlib import Path

from alancode.skills.builtin import BUILTIN_SKILLS
from alancode.skills.parser import SkillDefinition, parse_skill_file

logger = logging.getLogger(__name__)

SKILL_FILENAME = "SKILL.md"


def _scan_skills_dir(skills_dir: Path) -> dict[str, SkillDefinition]:
    """Scan a skills directory for SKILL.md files.

    Expected layout: skills_dir/<name>/SKILL.md
    Returns {name: SkillDefinition}.
    """
    results: dict[str, SkillDefinition] = {}

    if not skills_dir.is_dir():
        return results

    for child in sorted(skills_dir.iterdir()):
        if not child.is_dir():
            continue
        skill_file = child / SKILL_FILENAME
        if not skill_file.is_file():
            continue

        skill = parse_skill_file(str(skill_file))
        if skill is None:
            continue

        # Use directory name as canonical name if frontmatter name differs
        dir_name = child.name
        if skill.name != dir_name:
            logger.debug(
                "Skill dir name %r differs from frontmatter name %r, using frontmatter",
                dir_name, skill.name,
            )

        if skill.name in results:
            logger.debug("Duplicate skill %r in %s, keeping first", skill.name, skills_dir)
        else:
            results[skill.name] = skill

    return results


def discover_skills(cwd: str) -> dict[str, SkillDefinition]:
    """Discover all available skills from all sources.

    Priority (highest wins): project > user > built-in.
    A project skill with the same name as a built-in replaces it.
    """
    # Start with built-in (lowest priority)
    skills: dict[str, SkillDefinition] = dict(BUILTIN_SKILLS)

    # User skills (~/.alan/skills/) override built-in
    user_skills_dir = Path.home() / ".alan" / "skills"
    user_skills = _scan_skills_dir(user_skills_dir)
    skills.update(user_skills)
    if user_skills:
        logger.info("Discovered %d user skill(s) from %s", len(user_skills), user_skills_dir)

    # Project skills (.alan/skills/) override everything
    project_skills_dir = Path(cwd) / ".alan" / "skills"
    project_skills = _scan_skills_dir(project_skills_dir)
    skills.update(project_skills)
    if project_skills:
        logger.info("Discovered %d project skill(s) from %s", len(project_skills), project_skills_dir)

    return skills
