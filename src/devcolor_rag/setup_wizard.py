"""First-time setup: Ollama install + recommended model download (CLI-only, no GUI)."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from devcolor_rag.llm import check_ollama
from devcolor_rag.profiles import Profile, get_profile
from devcolor_rag.theme import DEVCOLOR_THEME

OLLAMA_INSTALL_URL = "https://ollama.com/install.sh"

# Common install locations after install.sh
_OLLAMA_CANDIDATES = (
    "ollama",
    "/usr/local/bin/ollama",
    "/Applications/Ollama.app/Contents/Resources/ollama",
)


def ollama_binary() -> str | None:
    for candidate in _OLLAMA_CANDIDATES:
        if candidate == "ollama":
            found = shutil.which("ollama")
            if found:
                return found
        elif Path(candidate).is_file():
            return candidate
    return None


def model_available(model: str) -> bool:
    ok, models = check_ollama()
    if not ok:
        return False
    return any(model.split(":")[0] in m for m in models)


def setup_needed(profile: Profile) -> bool:
    if not ollama_binary():
        return True
    return not model_available(profile.llm_model)


def _run_install_script(console: Console) -> bool:
    """Download and run ollama.com/install.sh without launching the GUI."""
    if not shutil.which("curl"):
        return False

    console.print(
        f"[accent]Downloading Ollama from [link={OLLAMA_INSTALL_URL}]{OLLAMA_INSTALL_URL}[/link]…[/accent]"
    )
    console.print("[meta]This runs in the terminal only — the Ollama app window will not open.[/meta]")
    env = {**os.environ, "OLLAMA_NO_START": "1"}
    try:
        proc = subprocess.run(
            ["sh", "-c", f'curl -fsSL "{OLLAMA_INSTALL_URL}" | sh'],
            env=env,
            text=True,
        )
        if proc.returncode == 0 and ollama_binary():
            console.print("[success]✓ Ollama installed[/success]")
            return True
    except OSError as exc:
        console.print(f"[warning]{exc}[/warning]")
    return False


def _install_via_homebrew(console: Console) -> bool:
    brew = shutil.which("brew")
    if not brew:
        return False
    console.print("[accent]Installing Ollama via Homebrew…[/accent]")
    try:
        subprocess.run([brew, "install", "--cask", "ollama"], check=True, text=True)
        if ollama_binary():
            console.print("[success]✓ Ollama installed[/success]")
            return True
    except subprocess.CalledProcessError:
        console.print("[warning]Homebrew install did not complete.[/warning]")
    return False


def install_ollama(console: Console) -> bool:
    """Install Ollama CLI via URL — no browser, no menu-bar app launch."""
    if ollama_binary():
        console.print("[success]✓ Ollama CLI already installed[/success]")
        return True

    if _run_install_script(console):
        return True

    if platform.system() == "Darwin" and _install_via_homebrew(console):
        return True

    console.print(
        f"[error]Could not install Ollama automatically.[/error]\n"
        f"[meta]Run manually:  curl -fsSL {OLLAMA_INSTALL_URL} | sh[/meta]"
    )
    return ollama_binary() is not None


_serve_proc: subprocess.Popen[bytes] | None = None


def start_ollama_service(console: Console) -> bool:
    """Start `ollama serve` in the background — never open the Ollama.app window."""
    global _serve_proc

    if check_ollama()[0]:
        console.print("[success]✓ Ollama API already running[/success]")
        return True

    binary = ollama_binary()
    if not binary:
        console.print("[error]Ollama CLI not found after install.[/error]")
        return False

    console.print("[accent]Starting Ollama server in the background…[/accent]")
    _serve_proc = subprocess.Popen(
        [binary, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Waiting for API at localhost:11434…", total=None)
        for _ in range(90):
            if check_ollama()[0]:
                progress.update(task, description="Ollama is ready")
                console.print("[success]✓ Ollama server running[/success]")
                return True
            time.sleep(1)

    console.print(
        "[error]Ollama did not start in time.[/error] "
        "[meta]Try: ollama serve[/meta]"
    )
    return False


def pull_model(console: Console, model: str) -> bool:
    binary = ollama_binary()
    if not binary:
        return False

    console.print(
        f"[accent]Downloading model [bold]{model}[/bold] from Ollama "
        f"(first time only, ~2GB)…[/accent]"
    )
    console.print("[meta]Fetched over the network — please wait.[/meta]\n")
    try:
        proc = subprocess.Popen(
            [binary, "pull", model],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                console.print(f"[muted]{line}[/muted]")
        proc.wait()
        if proc.returncode != 0:
            console.print(f"[error]Failed to pull {model}[/error]")
            return False
    except OSError as exc:
        console.print(f"[error]{exc}[/error]")
        return False

    if model_available(model):
        console.print(f"[success]✓ Model ready: {model}[/success]")
        return True
    return False


def run_setup(
    profile_name: str = "balanced",
    *,
    console: Console | None = None,
    skip_install: bool = False,
) -> bool:
    """Full first-time setup. Returns True if devcolorbot can use the LLM."""
    console = console or Console(theme=DEVCOLOR_THEME)
    profile = get_profile(profile_name)
    model = profile.llm_model

    console.print()
    console.print("[banner.title]devColorBot — getting ready[/banner.title]")
    console.print("[meta]Setting up local AI (first time only)…[/meta]\n")

    if model_available(model) and check_ollama()[0]:
        console.print(f"[success]✓ Already set up ({model} is available)[/success]\n")
        return True

    if not skip_install:
        if not install_ollama(console):
            return False
    elif not ollama_binary():
        console.print("[error]Ollama not found.[/error]")
        return False

    if not start_ollama_service(console):
        return False

    if not model_available(model):
        if not pull_model(console, model):
            return False

    return True
