# Exchange holiday data

`desk/marketdata/calendar_in.py` will not guess Indian exchange holidays. It
loads them from a file you supply, and raises `CalendarNotLoaded` for any year
that has not been loaded.

## Why it works this way

Only four NSE holidays are derivable from the date alone — Republic Day
(26 Jan), Independence Day (15 Aug), Gandhi Jayanti (2 Oct) and Christmas
(25 Dec). Everything else is lunar or declared: Holi, Eid, Diwali, Muhurat
trading, Guru Nanak Jayanti, and occasional unscheduled closures.

A hardcoded list assembled from memory would be right about most days and wrong
about the ones that matter, and it would fail **silently** — a backtest would
simply align bars to a day the market was shut. Refusing to answer is the
correct failure mode here.

## Where to get it

NSE publishes the list annually:

- **Trading holidays** — <https://www.nseindia.com/resources/exchange-communication-holidays>
- **Clearing holidays** — same page, separate table. These differ from trading
  holidays and matter for settlement dates.
- BSE publishes its own list; they are almost always identical, but load them
  separately if you trade BSE-only scrips.

Muhurat trading (Diwali evening) is announced by circular each year, usually a
few weeks ahead.

## Format

One JSON file per exchange, covering any number of years. `years` is required
and is not decoration — it is how the calendar distinguishes *"no holidays that
year"* from *"that year was never loaded"*.

```json
{
  "exchange": "NSE",
  "years": [2026],
  "holidays": [
    { "date": "2026-01-26", "name": "Republic Day" },
    { "date": "2026-08-15", "name": "Independence Day" }
  ],
  "special_sessions": [
    { "date": "2026-11-08", "name": "Muhurat Trading",
      "start": "18:15", "end": "19:15" }
  ]
}
```

Dates are `YYYY-MM-DD`, times are `HH:MM` in IST.

## Loading it

```python
from desk.marketdata.calendar_in import TradingCalendar

cal = TradingCalendar.from_file("configs/holidays_nse_2026.json")
cal.is_trading_day(date(2026, 1, 26))     # False
cal.settlement_date(date(2026, 1, 27))    # T+1
```

## A note on special sessions

A Muhurat day is **not** a trading day. The regular session is closed and the
market opens for roughly an hour in the evening. `is_trading_day()` returns
`False` and `special_session()` returns the window — anything that treats it as
a normal session will misalign its bars by a full day.
