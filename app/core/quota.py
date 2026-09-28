"""Daily usage limits for the public demo.

Every visitor shares one Gemini API key, so each IP address gets a small
daily allowance. Counts live in memory and reset at midnight UTC (and
whenever the server restarts).
"""

from collections import defaultdict
from datetime import datetime, timezone


class DailyQuota:
    def __init__(self):
        self._day = ""
        self._used: defaultdict[str, int] = defaultdict(int)

    def _roll_over(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._day:
            self._day = today
            self._used.clear()

    def remaining(self, key: str, limit: int) -> int:
        self._roll_over()
        return max(0, limit - self._used[key])

    def spend(self, key: str, amount: int = 1) -> None:
        self._roll_over()
        self._used[key] += amount

    def clear(self) -> None:
        self._used.clear()
