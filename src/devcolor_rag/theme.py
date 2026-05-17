"""DevColor-inspired Rich theme."""

from rich.style import Style
from rich.theme import Theme

GOLD = "#F5A623"
PURPLE = "#7B5EA7"
MUTED = "#8A8A94"

DEVCOLOR_THEME = Theme(
    {
        "banner.title": f"bold {GOLD}",
        "banner.subtitle": "dim italic",
        "prompt": f"bold {GOLD}",
        "accent": GOLD,
        "secondary": PURPLE,
        "muted": f"dim {MUTED}",
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "table.header": f"bold {PURPLE}",
        "panel.border": GOLD,
        "panel.title": f"bold {GOLD}",
        "bot.prefix": f"bold {GOLD}",
        "bot.body": "white",
        "user.body": PURPLE,
        "user.tag": f"dim {MUTED}",
        "meta": f"dim {MUTED}",
    }
)

BOT_LABEL = "devColorBot > "
USER_TAG = "< user"
