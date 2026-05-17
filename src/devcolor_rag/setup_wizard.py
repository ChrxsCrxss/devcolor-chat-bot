"""First-time setup: install Ollama + download the recommended model."""

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
from devcolor_rag.spinner import run_with_spinner
from devcolor_rag.theme import DEVCOLOR_THEME

OLLAMA_INSTALL_URL = "https://ollama.com/install.sh"
OLLAMA_WINDOWS_URL = "https://ollama.com/download/windows"

# Unix / macOS paths after install.sh or Homebrew
_UNIX_OLLAMA_PATHS = (
    "/usr/local/bin/ollama",
    "/usr/bin/ollama",
    "/Applications/Ollama.app/Contents/Resources/ollama",
)

_serve_proc: subprocess.Popen[bytes] | None = None
_resolved_binary: str | None = None


def _windows_ollama_paths() -> list[Path]:
    paths: list[Path] = []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        paths.append(Path(local) / "Programs" / "Ollama" / "ollama.exe")
    for env_key in ("ProgramFiles", "ProgramFiles(x86)"):
        root = os.environ.get(env_key)
        if root:
            paths.append(Path(root) / "Ollama" / "ollama.exe")
    return paths


def ollama_binary() -> str | None:
    """Return path to the Ollama CLI, searching PATH and known install locations."""
    global _resolved_binary
    if _resolved_binary and Path(_resolved_binary).is_file():
        return _resolved_binary

    found = shutil.which("ollama")
    if found:
        _resolved_binary = found
        return found

    for path in _UNIX_OLLAMA_PATHS:
        if Path(path).is_file():
            _resolved_binary = path
            return path

    for path in _windows_ollama_paths():
        if path.is_file():
            _resolved_binary = str(path)
            return _resolved_binary

    return None


def wait_for_ollama_binary(*, timeout_sec: int = 90) -> str | None:
    """Poll until the CLI appears (e.g. after winget or install.sh)."""
    for _ in range(timeout_sec):
        binary = ollama_binary()
        if binary:
            return binary
        time.sleep(1)
    return None


def model_available(model: str) -> bool:
    ok, models = check_ollama()
    if not ok:
        return False
    base = model.split(":")[0]
    return any(base in m for m in models)


def setup_needed(profile: Profile) -> bool:
    if not ollama_binary():
        return True
    if not check_ollama()[0]:
        return True
    return not model_available(profile.llm_model)


def _run_install_script(console: Console) -> bool:
    """Linux/macOS: ollama.com/install.sh (no GUI on macOS when OLLAMA_NO_START=1)."""
    if platform.system() == "Windows":
        return False
    if not shutil.which("curl") or not shutil.which("sh"):
        console.print("[warning]Need curl and sh to run the Ollama install script.[/warning]")
        return False

    console.print(
        f"[accent]Downloading Ollama from [link={OLLAMA_INSTALL_URL}]{OLLAMA_INSTALL_URL}[/link]…[/accent]"
    )
    console.print("[meta]Terminal install only — no browser required on macOS/Linux.[/meta]")
    env = {**os.environ, "OLLAMA_NO_START": "1"}
    try:
        proc = subprocess.run(
            ["sh", "-c", f'curl -fsSL "{OLLAMA_INSTALL_URL}" | sh'],
            env=env,
            text=True,
        )
        if proc.returncode == 0:
            binary = wait_for_ollama_binary(timeout_sec=60)
            if binary:
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
        binary = wait_for_ollama_binary(timeout_sec=60)
        if binary:
            console.print("[success]✓ Ollama installed[/success]")
            return True
    except subprocess.CalledProcessError:
        console.print("[warning]Homebrew install did not complete.[/warning]")
    return False


def _install_via_winget(console: Console) -> bool:
    winget = shutil.which("winget")
    if not winget:
        return False
    console.print("[accent]Installing Ollama via winget…[/accent]")
    try:
        proc = subprocess.run(
            [
                winget,
                "install",
                "-e",
                "--id",
                "Ollama.Ollama",
                "--accept-package-agreements",
                "--accept-source-agreements",
            ],
            text=True,
        )
        if proc.returncode == 0:
            binary = wait_for_ollama_binary(timeout_sec=120)
            if binary:
                console.print("[success]✓ Ollama installed[/success]")
                return True
    except OSError as exc:
        console.print(f"[warning]{exc}[/warning]")
    return False


def _print_manual_install_help(console: Console) -> None:
    system = platform.system()
    console.print("[error]Automatic Ollama install did not complete.[/error]")
    if system == "Windows":
        console.print(
            f"[meta]Install manually, then run [bold]devcolorbot setup[/bold] again:[/meta]\n"
            f"  winget install -e --id Ollama.Ollama\n"
            f"  — or download from [link={OLLAMA_WINDOWS_URL}]{OLLAMA_WINDOWS_URL}[/link]"
        )
    else:
        console.print(
            f"[meta]Install manually, then run [bold]devcolorbot setup[/bold] again:[/meta]\n"
            f"  curl -fsSL {OLLAMA_INSTALL_URL} | sh"
        )


def install_ollama(console: Console) -> bool:
    """Install Ollama for the current OS."""
    if ollama_binary():
        console.print("[success]✓ Ollama CLI already installed[/success]")
        return True

    system = platform.system()

    def _try_install() -> bool:
        if system == "Windows":
            return _install_via_winget(console)
        if system == "Darwin":
            if _run_install_script(console):
                return True
            return _install_via_homebrew(console)
        # Linux and other Unix
        return _run_install_script(console)

    if run_with_spinner(
        console,
        f"Installing Ollama ({system})…",
        _try_install,
    ):
        return True

    # Install may have succeeded but PATH not refreshed yet
    binary = wait_for_ollama_binary(timeout_sec=15)
    if binary:
        console.print("[success]✓ Ollama installed[/success]")
        return True

    _print_manual_install_help(console)
    return ollama_binary() is not None


def _popen_kwargs() -> dict:
    if platform.system() == "Windows":
        # Detached background process, no extra console window
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {"start_new_session": True}


def start_ollama_service(console: Console) -> bool:
    """Ensure the Ollama API is reachable at localhost:11434."""
    global _serve_proc

    if check_ollama()[0]:
        console.print("[success]✓ Ollama API already running[/success]")
        return True

    binary = ollama_binary()
    if not binary:
        console.print("[error]Ollama CLI not found after install.[/error]")
        return False

    # Windows/macOS installers often start the server automatically
    console.print("[accent]Waiting for Ollama API…[/accent]")
    with Progress(
        SpinnerColumn(style="accent"),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Checking localhost:11434…", total=None)
        for _ in range(30):
            if check_ollama()[0]:
                progress.update(task, description="Ollama is ready")
                console.print("[success]✓ Ollama server running[/success]")
                return True
            time.sleep(1)

    console.print("[accent]Starting Ollama server in the background…[/accent]")
    _serve_proc = subprocess.Popen(
        [binary, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **_popen_kwargs(),
    )

    with Progress(
        SpinnerColumn(style="accent"),
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
        "[error]Ollama did not start in time.[/error]\n"
        "[meta]On Windows, open the Ollama app once, or run: ollama serve[/meta]"
    )
    return False


def pull_model(console: Console, model: str) -> bool:
    binary = ollama_binary()
    if not binary:
        return False

    console.print(
        f"[accent]Downloading [bold]{model}[/bold] "
        f"(first time only, ~2GB over the network)…[/accent]"
    )
    try:
        proc = subprocess.Popen(
            [binary, "pull", model],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        with Progress(
            SpinnerColumn(style="accent"),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(f"Pulling {model}…", total=None)
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    progress.update(task, description=line[:72])
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
    max_attempts: int = 2,
) -> bool:
    """Install Ollama, start API, pull model. Returns True when the LLM is usable."""
    console = console or Console(theme=DEVCOLOR_THEME)
    profile = get_profile(profile_name)
    model = profile.llm_model

    console.print()
    console.print("[banner.title]devColorBot — setting up Ollama[/banner.title]")
    console.print("[meta]Local AI is required; installing if missing…[/meta]\n")

    if model_available(model) and check_ollama()[0]:
        console.print(f"[success]✓ Already set up ({model} is available)[/success]\n")
        return True

    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            console.print(f"[accent]Retrying setup (attempt {attempt}/{max_attempts})…[/accent]\n")

        if not skip_install:
            if not install_ollama(console):
                continue
        elif not ollama_binary():
            console.print("[error]Ollama not found. Run without --skip-install.[/error]")
            return False

        if not start_ollama_service(console):
            continue

        if not model_available(model):
            if not pull_model(console, model):
                continue

        return True

    _print_manual_install_help(console)
    return False
