"""Shared terminal spinners for long-running startup tasks."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeVar

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

T = TypeVar("T")


def make_progress(console: Console) -> Progress:
    return Progress(
        SpinnerColumn(style="accent"),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    )


@contextmanager
def spin(console: Console, message: str) -> Iterator[None]:
    """Show a spinner until the wrapped block finishes."""
    with make_progress(console) as progress:
        progress.add_task(message, total=None)
        yield


def run_with_spinner(console: Console, message: str, fn: Callable[[], T]) -> T:
    """Run callable while showing a spinner."""
    with spin(console, message):
        return fn()
