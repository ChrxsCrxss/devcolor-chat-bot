"""Strip markdown for plain-terminal display."""

from __future__ import annotations

import re

# **bold** or __bold__
_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
# *italic* or _italic_ (single; avoid greedy match across words)
_ITALIC = re.compile(r"(?<!\*)\*([^*]+?)\*(?!\*)|(?<!_)_([^_]+?)_(?!_)")
# `code`
_CODE = re.compile(r"`([^`]+)`")
# ## headers at line start
_HEADER = re.compile(r"^#{1,6}\s+", re.MULTILINE)


def plain_terminal_text(text: str) -> str:
    """Remove common markdown so Rich/terminal shows clean plain text."""
    out = text
    out = _HEADER.sub("", out)
    out = _CODE.sub(r"\1", out)

    for _ in range(3):
        prev = out
        out = _BOLD.sub(lambda m: m.group(1) or m.group(2) or "", out)
        out = _ITALIC.sub(lambda m: m.group(1) or m.group(2) or "", out)
        if out == prev:
            break

    # Stray asterisks from broken markdown
    out = re.sub(r"\*\*", "", out)
    return out
