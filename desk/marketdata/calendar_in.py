"""NSE/BSE trading calendar.

Wrong calendar means wrong bar alignment, and wrong bar alignment means a
backtest that is quietly off by a day in places. So this module does one
unusual thing deliberately: **it refuses to guess.**

Indian exchange holidays are not derivable. Republic Day, Independence Day,
Gandhi Jayanti and Christmas are fixed, but most of the list is lunar or
declared - Diwali, Holi, Eid, Muhurat trading - and NSE publishes it annually.
A calendar that hardcodes a half-remembered list is worse than no calendar,
because it fails silently on exactly the dates that matter.

So: holidays are LOADED from a file you supply, and every query for a year with
no loaded holidays raises `CalendarNotLoaded` rather than falling back to
weekends-only. Fail closed. See `docs/data/holidays.md` for the format and the
NSE source.

Session times and settlement ARE hardcoded, because they are stable, published
and rarely change:
  - pre-open       09:00 - 09:08 IST  (order collection + matching)
  - continuous     09:15 - 15:30 IST
  - closing sess.  15:40 - 16:00 IST
  - equity settle  T+1
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path

IST_OFFSET_MINUTES = 330          # UTC+05:30, no DST in India

PRE_OPEN_START = time(9, 0)
PRE_OPEN_END = time(9, 8)
SESSION_OPEN = time(9, 15)
SESSION_CLOSE = time(15, 30)
CLOSING_SESSION_START = time(15, 40)
CLOSING_SESSION_END = time(16, 0)

SETTLEMENT_DAYS = 1               # T+1 for Indian cash equities


class CalendarError(Exception):
    """Base for every calendar failure."""


class CalendarNotLoaded(CalendarError):
    """A date was queried for a year with no holiday data.

    Deliberately fatal. The alternative - assuming weekends-only - produces a
    calendar that is right about 95% of days and wrong about the rest, which is
    the worst possible failure mode for a backtest.
    """


@dataclass(frozen=True, slots=True)
class Holiday:
    day: date
    name: str
    exchange: str = "NSE"


@dataclass(frozen=True, slots=True)
class SpecialSession:
    """Muhurat and any other off-schedule session.

    A trading day that is ALSO a holiday - the market is closed for its normal
    session and opens for roughly an hour in the evening. Anything that assumes
    a special session has normal hours will misalign its bars.
    """
    day: date
    name: str
    start: time
    end: time
    exchange: str = "NSE"


@dataclass(slots=True)
class TradingCalendar:
    """Holidays and special sessions for one exchange, loaded from file."""

    exchange: str = "NSE"
    _holidays: dict[date, Holiday] = field(default_factory=dict)
    _specials: dict[date, SpecialSession] = field(default_factory=dict)
    _loaded_years: set[int] = field(default_factory=set)

    # ------------------------------------------------------------ loading ---

    @classmethod
    def from_file(cls, path: str | Path, exchange: str = "NSE") -> "TradingCalendar":
        """Load from the JSON shape documented in docs/data/holidays.md."""
        p = Path(path)
        if not p.exists():
            raise CalendarError(
                f"holiday file not found: {p}. Download the NSE holiday list "
                f"and save it in the documented format - this module will not "
                f"guess Indian exchange holidays."
            )
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CalendarError(f"{p} is not valid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise CalendarError(
                f"{p} must contain a JSON object, got {type(raw).__name__}"
            )
        declared = raw.get("exchange")
        if declared and declared != exchange:
            # NSE and BSE holiday lists are usually identical and occasionally
            # are not. Silently tagging one exchange's dates with the other's
            # name is a mislabelling nobody would notice until a settlement
            # date came out wrong.
            raise CalendarError(
                f"{p} declares exchange {declared!r} but was loaded as "
                f"{exchange!r}. Pass exchange={declared!r}, or use the right file."
            )
        cal = cls(exchange=declared or exchange)
        cal.load(raw)
        return cal

    def load(self, payload: dict) -> None:
        """Merge a payload. Callable repeatedly, one year at a time.

        Shape::

            {"exchange": "NSE",
             "years": [2026],
             "holidays":  [{"date": "2026-01-26", "name": "Republic Day"}],
             "special_sessions": [{"date": "2026-11-08", "name": "Muhurat",
                                   "start": "18:15", "end": "19:15"}]}
        """
        if not isinstance(payload, dict):
            raise CalendarError(
                f"payload must be a JSON object, got {type(payload).__name__}"
            )

        years = payload.get("years")
        if not years:
            raise CalendarError(
                "payload must declare which years it covers, so the calendar "
                "knows the difference between 'no holidays that year' and "
                "'that year was never loaded'"
            )

        holiday_rows = payload.get("holidays", [])
        special_rows = payload.get("special_sessions", [])
        if not isinstance(holiday_rows, list):
            raise CalendarError(
                f"'holidays' must be a list, got {type(holiday_rows).__name__}")
        if not isinstance(special_rows, list):
            raise CalendarError(
                f"'special_sessions' must be a list, got {type(special_rows).__name__}")

        # Every malformed-SHAPE problem below (a missing key, a non-numeric
        # year, a row that isn't a dict) is converted to CalendarError rather
        # than left as a raw KeyError/TypeError/ValueError. This module's
        # entire discipline is failing closed with a NAMED reason rather than
        # guessing - a config that crashes /health with an unhandled 500
        # instead of a clean 503 breaks that discipline just as much as
        # silently accepting bad data would.
        try:
            # Parse into locals first and commit only once everything
            # succeeds. A bad date halfway down the list would otherwise
            # leave the calendar holding the entries parsed before it, while
            # `_loaded_years` stays empty - so queries fail closed, but a
            # corrected reload merges into dirty state.
            years_i = {int(y) for y in years}
            holidays: dict[date, Holiday] = {}
            specials: dict[date, SpecialSession] = {}

            for h in holiday_rows:
                d = _parse_date(h["date"])
                if d.year not in years_i:
                    raise CalendarError(
                        f"holiday {d} is outside the declared years "
                        f"{sorted(years_i)}, so it would be stored but never "
                        f"reachable"
                    )
                holidays[d] = Holiday(d, h.get("name", ""), self.exchange)

            for s in special_rows:
                d = _parse_date(s["date"])
                if d.year not in years_i:
                    raise CalendarError(
                        f"special session {d} is outside the declared years "
                        f"{sorted(years_i)}"
                    )
                specials[d] = SpecialSession(
                    d, s.get("name", ""), _parse_time(s["start"]),
                    _parse_time(s["end"]), self.exchange,
                )
        except CalendarError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise CalendarError(f"malformed calendar payload: {exc}") from exc

        self._holidays.update(holidays)
        self._specials.update(specials)
        self._loaded_years.update(years_i)

    @property
    def loaded_years(self) -> list[int]:
        return sorted(self._loaded_years)

    def _require(self, d: date) -> None:
        if d.year not in self._loaded_years:
            raise CalendarNotLoaded(
                f"{d} is in {d.year}, which has no holiday data loaded "
                f"(loaded: {self.loaded_years or 'none'}). Refusing to guess."
            )

    # ------------------------------------------------------------ queries ---

    def is_weekend(self, d: date) -> bool:
        return d.weekday() >= 5           # Sat=5, Sun=6

    def is_holiday(self, d: date) -> bool:
        self._require(d)
        return d in self._holidays

    def is_trading_day(self, d: date) -> bool:
        """A normal, full trading session.

        A special-session day is NOT one of these - the regular session is shut
        and only the evening window opens. This must agree with `is_open_at`,
        which already excludes it; two public methods disagreeing about the
        same day is how bar alignment and session counts silently diverge.
        """
        self._require(d)
        return (
            not self.is_weekend(d)
            and d not in self._holidays
            and d not in self._specials
        )

    def special_session(self, d: date) -> SpecialSession | None:
        self._require(d)
        return self._specials.get(d)

    def holiday_name(self, d: date) -> str | None:
        self._require(d)
        h = self._holidays.get(d)
        return h.name if h else None

    # --------------------------------------------------------- navigation ---

    def next_trading_day(self, d: date, *, max_lookahead: int = 30) -> date:
        cur = d
        for _ in range(max_lookahead):
            cur += timedelta(days=1)
            if self.is_trading_day(cur):
                return cur
        raise CalendarError(
            f"no trading day within {max_lookahead} days after {d} - the "
            f"holiday data is probably wrong"
        )

    def previous_trading_day(self, d: date, *, max_lookback: int = 30) -> date:
        cur = d
        for _ in range(max_lookback):
            cur -= timedelta(days=1)
            if self.is_trading_day(cur):
                return cur
        raise CalendarError(
            f"no trading day within {max_lookback} days before {d} - the "
            f"holiday data is probably wrong"
        )

    def sessions_between(self, start: date, end: date) -> list[date]:
        """Every full trading session in [start, end], inclusive."""
        if end < start:
            raise CalendarError(f"end {end} is before start {start}")
        out, cur = [], start
        while cur <= end:
            if self.is_trading_day(cur):
                out.append(cur)
            cur += timedelta(days=1)
        return out

    def session_count(self, start: date, end: date) -> int:
        return len(self.sessions_between(start, end))

    def settlement_date(self, trade_date: date) -> date:
        """T+1. When the shares actually arrive, which is when you could sell
        them in the delivery segment."""
        self._require(trade_date)
        if not self.is_trading_day(trade_date):
            raise CalendarError(f"{trade_date} is not a trading day")
        d = trade_date
        for _ in range(SETTLEMENT_DAYS):
            d = self.next_trading_day(d)
        return d

    def is_open_at(self, when: datetime) -> bool:
        """Naive-IST datetime -> is the continuous session running?

        Special sessions are handled explicitly: on a Muhurat day the regular
        session is closed and only the special window is open.
        """
        d = when.date()
        self._require(d)
        special = self._specials.get(d)
        if special is not None:
            return special.start <= when.time() <= special.end
        if not self.is_trading_day(d):
            return False
        return SESSION_OPEN <= when.time() <= SESSION_CLOSE


def _parse_date(s: str) -> date:
    try:
        return date.fromisoformat(s)
    except ValueError as exc:
        raise CalendarError(f"bad date {s!r}, expected YYYY-MM-DD") from exc


def _parse_time(s: str) -> time:
    try:
        return time.fromisoformat(s)
    except ValueError as exc:
        raise CalendarError(f"bad time {s!r}, expected HH:MM") from exc
