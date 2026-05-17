"""Session configuration and /SET command handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from devcolor_rag.profiles import ProfileName, get_profile

STRICT_CONFIDENCE_THRESHOLD = 0.35


@dataclass
class SessionConfig:
    top_k: int = 5
    max_sources: int = 3
    strict: bool = False
    sources: bool = False
    debug: bool = False
    profile: ProfileName = "balanced"
    llm_model_override: str | None = None

    def summary(self) -> str:
        return (
            f"profile={self.profile} top-k={self.top_k} max-sources={self.max_sources} "
            f"strict={'on' if self.strict else 'off'} sources={'on' if self.sources else 'off'} "
            f"debug={'on' if self.debug else 'off'}"
        )


@dataclass(frozen=True)
class SettingDefinition:
    key: str
    aliases: tuple[str, ...]
    description: str
    allowed_display: str
    default: str
    getter: Callable[[SessionConfig], Any]
    setter: Callable[[SessionConfig, str], str | None]


def _parse_bool(value: str) -> bool:
    v = value.lower().strip()
    if v == "on":
        return True
    if v == "off":
        return False
    raise ValueError(f"'{value}' — use on or off")


def _set_int_field(cfg: SessionConfig, attr: str, value: str, lo: int, hi: int) -> None:
    try:
        n = int(value)
    except ValueError as exc:
        raise ValueError(f"'{value}' is not an integer") from exc
    if not lo <= n <= hi:
        raise ValueError(f"must be between {lo} and {hi}")
    setattr(cfg, attr, n)


SETTING_REGISTRY: list[SettingDefinition] = [
    SettingDefinition(
        key="top-k",
        aliases=("k",),
        description="Sentence chunks to retrieve before deduping",
        allowed_display="integer 1–10",
        default="5",
        getter=lambda c: c.top_k,
        setter=lambda c, v: _set_int_field(c, "top_k", v, 1, 10) or None,
    ),
    SettingDefinition(
        key="max-sources",
        aliases=(),
        description="Max unique FAQ entries passed to the LLM after dedupe",
        allowed_display="integer 1–10",
        default="3",
        getter=lambda c: c.max_sources,
        setter=lambda c, v: _set_int_field(c, "max_sources", v, 1, 10) or None,
    ),
    SettingDefinition(
        key="strict",
        aliases=(),
        description="Refuse to generate if retrieval confidence is too low",
        allowed_display="on, off",
        default="off",
        getter=lambda c: "on" if c.strict else "off",
        setter=lambda c, v: setattr(c, "strict", _parse_bool(v)) or None,
    ),
    SettingDefinition(
        key="sources",
        aliases=("show-sources",),
        description="Show the sources table after each answer",
        allowed_display="on, off",
        default="off",
        getter=lambda c: "on" if c.sources else "off",
        setter=lambda c, v: setattr(c, "sources", _parse_bool(v)) or None,
    ),
    SettingDefinition(
        key="debug",
        aliases=(),
        description="Print extra retrieval detail in the terminal",
        allowed_display="on, off",
        default="off",
        getter=lambda c: "on" if c.debug else "off",
        setter=lambda c, v: setattr(c, "debug", _parse_bool(v)) or None,
    ),
    SettingDefinition(
        key="profile",
        aliases=(),
        description="Hardware-friendly preset (embedder + Ollama model)",
        allowed_display="light, balanced, quality",
        default="balanced",
        getter=lambda c: c.profile,
        setter=lambda c, v: _set_profile(c, v),
    ),
]

_REGISTRY_BY_KEY: dict[str, SettingDefinition] = {}
for _defn in SETTING_REGISTRY:
    _REGISTRY_BY_KEY[_defn.key] = _defn
    for alias in _defn.aliases:
        _REGISTRY_BY_KEY[alias] = _defn


def _set_profile(cfg: SessionConfig, value: str) -> str | None:
    profile = get_profile(value)
    cfg.profile = profile.name  # type: ignore[assignment]
    return None


def resolve_setting(name: str) -> SettingDefinition | None:
    return _REGISTRY_BY_KEY.get(name.lower().strip())


def list_setting_keys() -> list[str]:
    return [d.key for d in SETTING_REGISTRY]


@dataclass
class SetResult:
    kind: str  # "list" | "help" | "applied" | "error"
    message: str = ""
    old_value: str | None = None
    new_value: str | None = None
    setting_key: str | None = None
    embedder_changed: bool = False


def parse_set_command(line: str, config: SessionConfig) -> SetResult | None:
    stripped = line.strip()
    if not stripped.upper().startswith("/SET"):
        return None

    rest = stripped[4:].strip()
    if not rest or rest.upper() == "HELP":
        return SetResult(kind="list")

    parts = rest.split(maxsplit=2)
    option = parts[0].lower()

    defn = resolve_setting(option)
    if defn is None:
        keys = ", ".join(list_setting_keys())
        return SetResult(
            kind="error",
            message=f"Unknown option `{option}`. Available: {keys}\nType /SET for full list.",
        )

    if len(parts) == 1:
        current = str(defn.getter(config))
        return SetResult(
            kind="help",
            message=(
                f"/SET {defn.key}\n"
                f"  Current:  {current}\n"
                f"  Allowed:  {defn.allowed_display}\n"
                f"  Default:  {defn.default}\n"
                f"  About:    {defn.description}\n"
                f"  Example:  /SET {defn.key} {defn.default}"
            ),
            setting_key=defn.key,
        )

    value = parts[1]
    old = str(defn.getter(config))
    old_profile_embedder = get_profile(config.profile).embedder

    try:
        defn.setter(config, value)
    except ValueError as exc:
        return SetResult(
            kind="error",
            message=(
                f"`{defn.key}` accepts: {defn.allowed_display}. "
                f"You entered: `{value}`.\n"
                f"Example: /SET {defn.key} {defn.default}\n"
                f"({exc})"
            ),
            setting_key=defn.key,
        )

    new = str(defn.getter(config))
    embedder_changed = False
    if defn.key == "profile":
        new_embedder = get_profile(config.profile).embedder
        embedder_changed = new_embedder != old_profile_embedder

    return SetResult(
        kind="applied",
        message=f"✓ {defn.key} → {new} (was {old})",
        old_value=old,
        new_value=new,
        setting_key=defn.key,
        embedder_changed=embedder_changed,
    )
