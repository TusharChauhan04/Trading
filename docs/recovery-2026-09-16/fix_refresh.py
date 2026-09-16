"""Repair the two things the replay could not: the stray except-block a bulk
sed injected into refresh_actions, and the failure accounting that the
follow-up edits (which no longer matched) were carrying."""
import io

p = r"C:\Users\TUSHAR\Trading-RESTORED\desk\marketdata\refresh.py"
s = io.open(p, encoding="utf-8").read()

# 1. The stray block: a bulk patch aimed at main()'s handler landed inside
#    refresh_actions, leaving a dedented body under `except CalendarError`.
broken = '''        try:
            payload = session.fetch_corporate_actions(base)
        except CalendarError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("The existing calendar was left untouched.", file=sys.stderr)
        return 1
    except SourceError as exc:
            print(f"{base:14} FAILED  {exc}")
            continue
'''
fixed = '''        try:
            payload = session.fetch_corporate_actions(base)
        except SourceError as exc:
            print(f"{base:14} FAILED  {exc}")
            failed += 1
            continue
'''
assert broken in s, "stray except block not found"
s = s.replace(broken, fixed)

# 2. A symbol that could not be fetched must make the command EXIT NON-ZERO.
#    A refresh that silently reports success while half the symbols failed is
#    how a scheduled job goes green for weeks on stale data.
s = s.replace(
    '''    out_dir.mkdir(parents=True, exist_ok=True)
    total_unparsed = 0
''',
    '''    out_dir.mkdir(parents=True, exist_ok=True)
    total_unparsed = 0
    failed = 0
''')
s = s.replace(
    '''        except SymbolError as exc:
            print(f"{raw_symbol[:30]:14} SKIPPED  {exc}")
            continue
''',
    '''        except SymbolError as exc:
            print(f"{raw_symbol[:30]:14} SKIPPED  {exc}")
            failed += 1
            continue
''')

# 3. Every line printed REVIEW is a price-affecting action that could not be
#    reduced to one factor. "no usable ex-date" belongs in that set: an action
#    with no date cannot be applied point-in-time at all.
s = s.replace(
    '''        for u in unparsed:
            if "cannot be reduced" in u.reason:''',
    '''        for u in unparsed:
            if any(k in u.reason for k in _REVIEW_WORTHY):''')

s = s.replace(
    '''def refresh_actions(session: NseSession, symbols: list[str], out_dir: Path) -> int:''',
    '''#: Reasons that mean a HUMAN must look, not that a retry might help. An
#: action with no usable ex-date cannot be applied point-in-time at all, so it
#: belongs here beside the ones that cannot be reduced to a single factor.
_REVIEW_WORTHY = ("cannot be reduced", "no usable ex-date")


def refresh_actions(session: NseSession, symbols: list[str], out_dir: Path) -> int:''')

s = s.replace(
    '''              f"show a gap the quality gate reports as a missing action.")
    return 0''',
    '''              f"show a gap the quality gate reports as a missing action.")
    if failed:
        print(f"\\n{failed} symbol(s) failed. Exiting non-zero so a scheduled "
              f"run does not report success.")
    return 1 if failed else 0''')

io.open(p, "w", encoding="utf-8").write(s)
print("repaired")
