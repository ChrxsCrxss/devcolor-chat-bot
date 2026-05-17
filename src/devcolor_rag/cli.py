"""DevColor RAG CLI."""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

from devcolor_rag.config import (
    SETTING_REGISTRY,
    SessionConfig,
    list_setting_keys,
    parse_set_command,
)
from devcolor_rag.index import check_cache, get_or_build_index
from devcolor_rag.llm import check_ollama
from devcolor_rag.pipeline import RAGPipeline
from devcolor_rag.profiles import PROFILES, get_profile
from devcolor_rag.paths import CORPUS_PATH, PROJECT_ROOT
from devcolor_rag.retriever import RetrievalResult
from devcolor_rag.setup_wizard import run_setup, setup_needed
from devcolor_rag.spinner import run_with_spinner, spin
from devcolor_rag.theme import BOT_LABEL, DEVCOLOR_THEME, USER_TAG

HELP_TEXT = """
[bold]Commands[/bold]
  help, ?          Show this help
  exit, quit       Leave the chat
  (First time?)    Run [accent]devcolorbot setup[/accent] outside the REPL
  sources          Show sources from the last answer
  clear            Clear the screen
  doctor           Check environment and recommendations
  reindex          Rebuild the embedding cache
  reindex --wipe   Delete cache and rebuild

[bold]Settings[/bold] — type [accent]/SET[/accent] for the full options table
  /SET top-k 5           Chunks to retrieve (1–10)
  /SET max-sources 3     FAQ entries after dedupe (1–10)
  /SET strict on|off     Confidence gate (boolean)
  /SET sources on|off    Show sources table
  /SET debug on|off      Extra retrieval detail
  /SET profile balanced  light | balanced | quality
"""


def _project_root() -> Path:
    return PROJECT_ROOT


def prepare_environment(
    console: Console,
    profile_name: str,
    *,
    skip_setup: bool = False,
    skip_index: bool = False,
    force_reindex: bool = False,
    wipe: bool = False,
) -> None:
    """Auto setup: Ollama + model + vector index (first run only)."""
    profile = get_profile(profile_name)

    if not skip_setup and setup_needed(profile):
        if not run_setup(profile_name, console=console):
            console.print(
                "\n[error]Ollama is required for full answers.[/error]\n"
                "[meta]After installing, run:[/meta] [bold]devcolorbot setup[/bold]\n"
                "[meta]Offline demo only:[/meta] [bold]devcolorbot --echo[/bold]\n"
            )
            sys.exit(1)

    if skip_index:
        return

    cache_status = check_cache(CORPUS_PATH, profile, PROJECT_ROOT)
    if cache_status != "valid" or force_reindex:

        def _build_index() -> None:
            get_or_build_index(
                CORPUS_PATH,
                profile,
                PROJECT_ROOT,
                force_reindex=force_reindex or cache_status == "stale",
                wipe=wipe,
            )

        run_with_spinner(
            console,
            "Building FAQ knowledge index (first run may download embeddings)…",
            _build_index,
        )
        console.print("[success]✓ Index ready[/success]\n")
    else:
        console.print("[muted]Knowledge index ready (cached)[/muted]")


def _make_console(no_color: bool) -> Console:
    return Console(theme=DEVCOLOR_THEME, force_terminal=True, no_color=no_color)


def _print_banner(console: Console, config: SessionConfig) -> None:
    console.print()
    console.print("[banner.title]devColorBot[/banner.title] · [banner.subtitle]/dev/color FAQ[/banner.subtitle]")
    console.print(
        "[banner.subtitle]Empowering Black technologists — ask about programs, "
        "events, and community.[/banner.subtitle]"
    )
    console.print(f"[meta]{config.summary()}[/meta]")
    console.print("[meta]Type /SET to change settings · help for commands · exit to quit[/meta]")
    console.print()


def _print_set_table(console: Console, config: SessionConfig) -> None:
    table = Table(title="Settings — /SET <option> <value>", border_style="panel.border")
    table.add_column("Option", style="accent")
    table.add_column("Current", style="bold")
    table.add_column("Allowed", style="muted")
    table.add_column("Description")
    for defn in SETTING_REGISTRY:
        table.add_row(
            defn.key,
            str(defn.getter(config)),
            defn.allowed_display,
            defn.description,
        )
    console.print(table)
    console.print("[meta]Example: /SET top-k 5 · /SET strict on · /SET profile light[/meta]")


def _print_sources(console: Console, retrieval: RetrievalResult, *, debug: bool) -> None:
    table = Table(title="Sources", border_style="panel.border")
    table.add_column("#", style="muted", width=3)
    table.add_column("Score", justify="right", width=7)
    table.add_column("FAQ", width=4)
    table.add_column("Question", style="secondary")
    if debug:
        table.add_column("Snippet", style="muted", max_width=40)
    for chunk in retrieval.chunks:
        row = [
            str(chunk.rank),
            f"{chunk.score:.3f}",
            str(chunk.parent_id),
            chunk.parent_question[:60] + ("…" if len(chunk.parent_question) > 60 else ""),
        ]
        if debug:
            row.append(chunk.snippet)
        table.add_row(*row)
    console.print(table)


def _is_list_line(line: str) -> bool:
    stripped = line.lstrip()
    if stripped.startswith(("-", "•", "*", "–")):
        return True
    return bool(len(stripped) > 2 and stripped[0].isdigit() and stripped[1] in ".)")


def _print_user_message(console: Console, text: str) -> None:
    """Right-aligned user turn:  question...  < user"""
    console.print()
    width = max(console.width, 40)
    tag = f" {USER_TAG}"
    tag_len = len(USER_TAG) + 1
    wrap_width = max(20, width - tag_len - 2)

    lines = textwrap.wrap(text, width=wrap_width) or [text]
    for i, line in enumerate(lines):
        body = Text(line, style="user.body")
        if i == len(lines) - 1:
            body.append(tag, style="user.tag")
        console.print(body, justify="right")
    console.print()


def _emit_bot_line(
    console: Console,
    line: str,
    *,
    prefix: str,
    indent: str,
    first_line: bool,
) -> bool:
    if first_line:
        console.print(f"[bot.prefix]{prefix}[/bot.prefix][bot.body]{line}[/bot.body]")
        return False
    console.print(f"{indent}[bot.body]{line}[/bot.body]")
    return False


def _print_bot_message(console: Console, text: str) -> None:
    """Left-aligned bot turn: devColorBot > answer... (paragraph-aware)."""
    prefix = BOT_LABEL
    indent = " " * len(prefix)
    width = max(console.width, 40)
    wrap_width = max(20, width - len(prefix) - 2)
    list_wrap = max(20, wrap_width - 2)

    console.print()
    first_line = True
    paragraphs = text.split("\n\n")

    for para_idx, paragraph in enumerate(paragraphs):
        if para_idx > 0:
            console.print()

        for raw_line in paragraph.split("\n"):
            line = raw_line.rstrip()
            if not line:
                continue

            if _is_list_line(line):
                wrapped = textwrap.wrap(
                    line,
                    width=list_wrap,
                    subsequent_indent="  ",
                    initial_indent="",
                ) or [line]
            else:
                wrapped = textwrap.wrap(line, width=wrap_width) or [line]

            for wline in wrapped:
                first_line = _emit_bot_line(
                    console, wline, prefix=prefix, indent=indent, first_line=first_line
                )

    console.print()
    console.print()


def run_doctor(console: Console, config: SessionConfig) -> None:
    console.print("\n[bold secondary]Doctor[/bold secondary]\n")
    console.print(f"Python: {sys.version.split()[0]} on {sys.platform}")

    try:
        import psutil

        ram_gb = psutil.virtual_memory().total / (1024**3)
        console.print(f"RAM:    {ram_gb:.1f} GB total")
        if ram_gb < 8:
            console.print("[warning]→ Recommend profile: light[/warning]")
        else:
            console.print("[success]→ Recommend profile: balanced[/success]")
    except ImportError:
        console.print("RAM:    (install psutil for RAM check)")

    ok, models = check_ollama()
    if ok:
        console.print(f"Ollama: [success]reachable[/success] ({len(models)} models)")
        profile = get_profile(config.profile)
        if any(profile.llm_model in m for m in models):
            console.print(f"        [success]✓ {profile.llm_model} available[/success]")
        else:
            console.print(
                f"        [warning]✗ {profile.llm_model} not found — run: "
                f"ollama pull {profile.llm_model}[/warning]"
            )
    else:
        console.print(
            "[error]Ollama: not reachable[/error] — run [bold]devcolorbot setup[/bold]"
        )

    status = check_cache(CORPUS_PATH, get_profile(config.profile), _project_root())
    console.print(f"Cache:  {status}")
    console.print()


def run_reindex(
    console: Console,
    pipeline: RAGPipeline,
    config: SessionConfig,
    *,
    wipe: bool = False,
) -> RAGPipeline:
    profile = get_profile(config.profile)
    console.print("[accent]Rebuilding index…[/accent]")
    pipeline.reload_index(profile, wipe=wipe)
    console.print("[success]✓ Index rebuilt[/success]")
    return pipeline


def _handle_meta(
    line: str,
    console: Console,
    config: SessionConfig,
    pipeline: RAGPipeline,
) -> tuple[str, RAGPipeline | None]:
    """Returns ('continue'|'exit'|'handled'), possibly updated pipeline."""
    cmd = line.strip().lower()
    if cmd in ("exit", "quit"):
        return "exit", pipeline
    if cmd in ("help", "?"):
        console.print(HELP_TEXT)
        return "handled", pipeline
    if cmd == "clear":
        console.clear()
        _print_banner(console, config)
        return "handled", pipeline
    if cmd == "sources":
        if pipeline.last_retrieval:
            _print_sources(console, pipeline.last_retrieval, debug=config.debug)
        else:
            console.print("[muted]No query yet.[/muted]")
        return "handled", pipeline
    if cmd == "doctor":
        run_doctor(console, config)
        return "handled", pipeline
    if cmd.startswith("reindex"):
        wipe = "--wipe" in cmd.split()
        pipeline = run_reindex(console, pipeline, config, wipe=wipe)
        return "handled", pipeline

    set_result = parse_set_command(line, config)
    if set_result is not None:
        if set_result.kind == "list":
            _print_set_table(console, config)
        elif set_result.kind == "help":
            console.print(set_result.message)
        elif set_result.kind == "error":
            console.print(f"[error]{set_result.message}[/error]")
        elif set_result.kind == "applied":
            console.print(f"[success]{set_result.message}[/success]")
            if set_result.embedder_changed:
                console.print(
                    "[warning]Embedder changed — run [bold]reindex[/bold] to rebuild cache.[/warning]"
                )
            profile = get_profile(config.profile)
            pipeline.update_llm(profile)
        return "handled", pipeline

    return "continue", pipeline


def _process_query(
    query: str,
    console: Console,
    config: SessionConfig,
    pipeline: RAGPipeline,
    *,
    echo_user: bool = True,
) -> str:
    if echo_user:
        _print_user_message(console, query)
    with spin(console, "Thinking…"):
        result = pipeline.query(query, config)
    if config.sources or config.debug:
        _print_sources(console, result.retrieval, debug=config.debug)
    _print_bot_message(console, result.answer)
    if config.debug and result.log_paths:
        console.print(f"[meta]Log: {result.log_paths[0]}[/meta]")
    return result.answer


def run_repl(
    pipeline: RAGPipeline,
    config: SessionConfig,
    *,
    no_color: bool = False,
) -> None:
    console = _make_console(no_color)
    _print_banner(console, config)
    while True:
        try:
            line = console.input("[prompt]> [/prompt]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[muted]Goodbye![/muted]")
            break
        if not line:
            continue
        status, pipeline = _handle_meta(line, console, config, pipeline)
        if status == "exit":
            console.print("[muted]Goodbye![/muted]")
            break
        if status == "handled":
            continue
        _process_query(line, console, config, pipeline, echo_user=False)


def run_once(
    pipeline: RAGPipeline,
    config: SessionConfig,
    query: str,
    *,
    no_color: bool = False,
) -> str:
    console = _make_console(no_color)
    _print_banner(console, config)
    return _process_query(query, console, config, pipeline, echo_user=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="/dev/color FAQ RAG — local CLI chatbot",
    )
    p.add_argument("--profile", default="balanced", choices=list(PROFILES))
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--max-sources", type=int, default=3)
    p.add_argument("--strict-confidence", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument(
        "--show-sources",
        action="store_true",
        help="Show the sources table after each answer",
    )
    p.add_argument("--once", metavar="QUERY", help="Single query then exit")
    p.add_argument("--echo", action="store_true", help="Use EchoLLM (no Ollama)")
    p.add_argument("--reindex", action="store_true", help="Rebuild cache on startup")
    p.add_argument("--wipe", action="store_true", help="Wipe cache before reindex")
    p.add_argument("--llm-model", default=None)
    p.add_argument("--no-color", action="store_true")
    p.add_argument("--doctor", action="store_true", help="Run doctor and exit")
    p.add_argument(
        "--skip-setup",
        action="store_true",
        help="Do not offer first-time Ollama setup on launch",
    )
    return p


def _run_setup_command(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="devcolorbot setup",
        description="Install Ollama and download the recommended AI model (first-time only)",
    )
    parser.add_argument("--profile", default="balanced", choices=list(PROFILES))
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Non-interactive where possible (auto-confirm prompts)",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Only start Ollama and pull model; do not try to install Ollama",
    )
    args = parser.parse_args(argv)
    console = _make_console(no_color=False)
    ok = run_setup(
        args.profile,
        console=console,
        skip_install=args.skip_install,
    )
    sys.exit(0 if ok else 1)


def app_main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        _run_setup_command(sys.argv[2:])
        return

    args = build_parser().parse_args()
    config = SessionConfig(
        top_k=args.top_k,
        max_sources=args.max_sources,
        strict=args.strict_confidence,
        sources=args.show_sources,
        debug=args.debug,
        profile=args.profile,  # type: ignore[arg-type]
        llm_model_override=args.llm_model,
    )

    console = _make_console(args.no_color)
    console.print("[accent]devColorBot starting…[/accent]", highlight=False)

    if args.doctor:
        run_doctor(console, config)
        return

    if not CORPUS_PATH.exists():
        console.print(f"[error]Corpus not found: {CORPUS_PATH}[/error]")
        sys.exit(1)

    if not args.doctor:
        prepare_environment(
            console,
            config.profile,
            skip_setup=args.echo or args.skip_setup,
            force_reindex=args.reindex,
            wipe=args.wipe,
        )

    profile = get_profile(config.profile)

    def _load_pipeline() -> RAGPipeline:
        return RAGPipeline.from_profile(
            CORPUS_PATH,
            _project_root(),
            profile,
            llm_model_override=config.llm_model_override,
            force_reindex=False,
            wipe=False,
            use_echo=args.echo,
        )

    pipeline = run_with_spinner(console, "Starting chat session…", _load_pipeline)

    if args.once:
        run_once(pipeline, config, args.once, no_color=args.no_color)
        return

    run_repl(pipeline, config, no_color=args.no_color)


def main() -> None:
    from devcolor_rag.launcher import main as launcher_main

    launcher_main()


if __name__ == "__main__":
    main()
