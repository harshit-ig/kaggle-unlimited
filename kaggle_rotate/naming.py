"""Shared naming helpers.

Kaggle derives a kernel's slug from its title, so title and `id` must agree. Keeping
the slugifier in one place stops the pool and the account registry from drifting.
"""

from __future__ import annotations

import re

_NON_SLUG = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    return _NON_SLUG.sub("-", value.strip().lower()).strip("-")
