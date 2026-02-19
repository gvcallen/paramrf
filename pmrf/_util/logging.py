import sys
import logging

class LevelFilteredLogger:
    def __init__(self, null_level=logging.WARNING):
        self.null_level = null_level

    def _should_suppress(self, level):
        return level < self.null_level

    def debug(self, msg, *args, **kwargs):
        if not self._should_suppress(logging.DEBUG):
            print(f"[DEBUG] {msg}", file=sys.stderr)

    def info(self, msg, *args, **kwargs):
        if not self._should_suppress(logging.INFO):
            print(f"[INFO] {msg}", file=sys.stderr)

    def warning(self, msg, *args, **kwargs):
        if not self._should_suppress(logging.WARNING):
            print(f"[WARNING] {msg}", file=sys.stderr)

    def error(self, msg, *args, **kwargs):
        if not self._should_suppress(logging.ERROR):
            print(f"[ERROR] {msg}", file=sys.stderr)

    def critical(self, msg, *args, **kwargs):
        if not self._should_suppress(logging.CRITICAL):
            print(f"[CRITICAL] {msg}", file=sys.stderr)

    def exception(self, msg, *args, exc_info=True, **kwargs):
        if not self._should_suppress(logging.ERROR):
            print(f"[EXCEPTION] {msg}", file=sys.stderr)