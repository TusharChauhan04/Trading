"""The F&O segment parser, and the two ways its prices lie.

Both are measured facts about NSE's F&O bhavcopy, not hypotheticals:

  1. 46% of option contracts have NO VOLUME on a given day, and the file still
     reports a `ClsPric` for every one of them - a settlement NSE computed, not
     a price anyone paid.
  2. Even among traded contracts, ~2-6% violate basic no-arbitrage bounds,
     because two legs' closing prices come from different moments of the day.
     A spread built from them is not a price anyone could have paid.

A backtest that ignores either finds free money. These tests exist so the
filters cannot be removed quietly.
"""

from __future__ import annotations

import io
import zipfile

import pandas as pd
import pytest

from desk.marketdata.fo import (
    INSTRUMENT_TYPES, coherence_report, parse_fo_bhavcopy,
)

HEADER = (
    "TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,"
    "XpryDt,FininstrmActlXpryDt,StrkPric,OptnTp,FinInstrmNm,OpnPric,HghPric,"
    "LwPric,ClsPric,LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,"
    "ChngInOpnIntrst,TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd,SsnId,"
    "NewBrdLotQty,Rmks,Rsvd1,Rsvd2,Rsvd3,Rsvd4"
)


def _row(sym="ACME", strike="100.00", opt="CE", close="12.00", vol="500",
         oi="10000", under="105.00", lot="250", expiry="2026-10-27",
         last=None, settle=None, instr="STO", trad="2026-09-29"):
    last = close if last is None else last
    settle = close if settle is None else settle
    return (f"{trad},{trad},FO,NSE,{instr},1,,{sym},,{expiry},{expiry},"
            f"{strike},{opt},{sym}OPT,{close},{close},{close},{close},"
            f"{last},{close},{under},{settle},{oi},0,{vol},1000.0,5,F1,"
            f"{lot},,,,,")


def _csv(*rows: str) -> str:
    return "\n".join([HEADER, *rows]) + "\n"


def _zip(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("BhavCopy_NSE_FO_0_0_0_20260929_F_0000.csv", text)
    return buf.getvalue()


# -- the tradeability filter, which is the headline ------------------------

def test_untraded_contracts_are_dropped_by_default() -> None:
    """THE test. An untraded contract's close is a settlement NSE computed.
    Reading it as a price is how a backtest buys at a number nobody paid."""
    text = _csv(
        _row(strike="100.00", close="12.00", vol="500"),
        _row(strike="415.00", close="30.00", vol="0", oi="0"),   # the real case
    )
    day = parse_fo_bhavcopy(text)
    assert len(day.contracts) == 1
    assert day.dropped_untraded == 1
    assert float(day.contracts["strike"].iloc[0]) == 100.0
    assert any("zero volume" in n for n in day.notes)


def test_turning_the_filter_off_is_flagged_loudly() -> None:
    """Measuring the segment's shape is a legitimate reason to keep untraded
    rows. Feeding them to a backtest is not, so the state must be visible."""
    text = _csv(_row(vol="0", oi="0"))
    day = parse_fo_bhavcopy(text, tradeable_only=False)
    assert len(day.contracts) == 1
    assert day.dropped_untraded == 0
    assert any("FILTER OFF" in n.upper() for n in day.notes)


def test_traded_share_reports_how_much_of_the_file_was_real() -> None:
    text = _csv(*[_row(strike=f"{100 + i}.00", vol="500") for i in range(3)],
                *[_row(strike=f"{200 + i}.00", vol="0", oi="0")
                  for i in range(7)])
    day = parse_fo_bhavcopy(text)
    assert day.rows_in_file == 10
    assert day.traded_share == pytest.approx(0.3)


def test_a_minimum_open_interest_gate_is_available_and_off_by_default() -> None:
    """A contract can print one lot and still be untradeable at size. The
    floor depends on the position size being tested, so inventing one would
    hide the choice."""
    text = _csv(_row(strike="100.00", vol="5", oi="100"),
                _row(strike="110.00", vol="900", oi="50000"))
    assert len(parse_fo_bhavcopy(text).contracts) == 2
    tight = parse_fo_bhavcopy(text, min_open_interest=1000)
    assert len(tight.contracts) == 1
    assert float(tight.contracts["strike"].iloc[0]) == 110.0


# -- arbitrage coherence, the subtler trap ---------------------------------

def test_coherence_report_catches_a_call_below_intrinsic() -> None:
    """A call must cost at least spot - strike. The ADANIPORTS case: 1800/1820
    calls priced at 6.70 with the spot at 1821.96 is 13.30 of free money."""
    text = _csv(
        _row(strike="1800.00", close="6.00", under="1821.96"),   # intrinsic 21.96
        _row(strike="1900.00", close="5.00", under="1821.96"),   # fine
    )
    day = parse_fo_bhavcopy(text)
    rep = coherence_report(day.options, price="close")
    assert rep["below_intrinsic"] == pytest.approx(0.5)


def test_coherence_report_catches_non_monotonic_strikes() -> None:
    """A higher-strike call must not cost MORE than a lower-strike one."""
    text = _csv(
        _row(strike="100.00", close="5.00", under="90.00"),
        _row(strike="110.00", close="9.00", under="90.00"),   # dearer, wrong
    )
    day = parse_fo_bhavcopy(text)
    rep = coherence_report(day.options, price="close")
    assert rep["non_monotonic"] == pytest.approx(1.0)
    assert rep["below_intrinsic"] == pytest.approx(0.0)


def test_drop_incoherent_removes_below_intrinsic_calls() -> None:
    text = _csv(
        _row(strike="1800.00", close="6.00", under="1821.96"),
        _row(strike="1900.00", close="5.00", under="1821.96"),
    )
    kept = parse_fo_bhavcopy(text, drop_incoherent=True)
    assert len(kept.contracts) == 1
    assert float(kept.contracts["strike"].iloc[0]) == 1900.0
    assert any("below intrinsic" in n for n in kept.notes)


def test_coherence_on_an_empty_chain_is_nan_not_zero() -> None:
    """0.0 would read as "perfectly coherent", which is the opposite of
    "nothing to check"."""
    rep = coherence_report(pd.DataFrame(
        columns=["option_type", "strike", "close", "underlying", "symbol",
                 "expiry"]))
    assert rep["rows"] == 0.0
    assert rep["below_intrinsic"] != rep["below_intrinsic"]       # NaN


# -- the schema details that decide capital --------------------------------

def test_lot_size_is_read_per_row_not_assumed() -> None:
    """NSE revises lot sizes, and they differ per underlying. Hard-coding one
    is how a backtest sizes positions it could never have taken."""
    text = _csv(_row(sym="AAA", lot="250"), _row(sym="BBB", lot="1200"))
    day = parse_fo_bhavcopy(text)
    got = dict(zip(day.contracts["symbol"], day.contracts["lot_size"]))
    assert got == {"AAA": 250, "BBB": 1200}


def test_a_future_has_no_strike_and_it_is_nan_not_zero() -> None:
    """0.0 would sort a future among the deepest in-the-money options."""
    text = _csv(_row(instr="STF", opt="", strike="", close="1800.00"))
    day = parse_fo_bhavcopy(text, tradeable_only=True)
    assert len(day.futures) == 1
    assert pd.isna(day.futures["strike"].iloc[0])
    assert day.options.empty


def test_the_three_price_columns_are_all_kept() -> None:
    """ClsPric, LastPric and SttlmPric differ materially on a thin contract
    and no single one is correct, so the caller chooses."""
    text = _csv(_row(close="12.00", last="11.50", settle="12.25"))
    row = parse_fo_bhavcopy(text).contracts.iloc[0]
    assert float(row["close"]) == 12.00
    assert float(row["last"]) == 11.50
    assert float(row["settle"]) == 12.25


def test_a_zip_and_a_plain_csv_both_parse() -> None:
    text = _csv(_row())
    assert len(parse_fo_bhavcopy(text).contracts) == 1
    assert len(parse_fo_bhavcopy(_zip(text)).contracts) == 1


def test_a_schema_change_raises_rather_than_producing_a_wrong_frame() -> None:
    """If NSE renames a column, failing loudly beats silently losing it."""
    with pytest.raises(ValueError, match="missing"):
        parse_fo_bhavcopy("TradDt,TckrSymb\n2026-09-29,ACME\n")


def test_a_zip_without_a_csv_member_raises() -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("readme.txt", "nope")
    with pytest.raises(ValueError, match="no .csv member"):
        parse_fo_bhavcopy(buf.getvalue())


def test_instrument_types_are_documented() -> None:
    assert set(INSTRUMENT_TYPES) == {"STO", "IDO", "STF", "IDF"}
