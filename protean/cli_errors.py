"""
Clean one-line error output instead of Python tracebacks.

Covers three sources of tracebacks:
  - uncaught exceptions in the main thread        (sys.excepthook)
  - uncaught exceptions in background threads     (threading.excepthook)
  - logger.exception(...) / asyncio task failures (logging.Formatter.formatException)

Set PROTEAN_TRACEBACK=1 to get full tracebacks back for debugging.
"""
from __future__ import annotations

import logging
import os
import sys
import threading


def _one_line(exc: BaseException) -> str:
    msg = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    return f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__


def install() -> None:
    if os.environ.get("PROTEAN_TRACEBACK"):
        return

    default_hook = sys.excepthook

    def excepthook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            default_hook(exc_type, exc, tb)
            return
        print(f"\nError: {_one_line(exc)}\n(set PROTEAN_TRACEBACK=1 for the full traceback)",
              file=sys.stderr, flush=True)

    def thread_hook(args):
        if args.exc_type is SystemExit:
            return
        name = args.thread.name if args.thread else "?"
        print(f"\nError in thread {name}: {_one_line(args.exc_value)}",
              file=sys.stderr, flush=True)

    def format_exception(self, ei):
        return _one_line(ei[1])

    sys.excepthook = excepthook
    threading.excepthook = thread_hook
    logging.Formatter.formatException = format_exception  # type: ignore[method-assign]
