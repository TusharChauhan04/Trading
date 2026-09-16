"""Re-apply the _rejection_counts extraction. It was made by a review
subagent, so it never appeared in this session's transcript and the replay
could not reproduce it."""
import io

p = r"C:\Users\TUSHAR\Trading-RESTORED\desk\scanner\stage4.py"
s = io.open(p, encoding="utf-8").read()

dup = '''        counts: dict[str, int] = {}
        for s in self.rejected:
            for r in s.reasons:
                counts[r.value] = counts.get(r.value, 0) + 1
'''
assert s.count(dup) == 2, f"expected 2 duplicate loops, found {s.count(dup)}"

# no_trade_reason()'s copy
s = s.replace(dup + '''        top = ", ".join(f"{k} ({v})" for k, v in
                        sorted(counts.items(), key=lambda kv: -kv[1])[:3])''',
'''        counts = self._rejection_counts()
        top = ", ".join(f"{k} ({v})" for k, v in
                        sorted(counts.items(), key=lambda kv: -kv[1])[:3])''')

# summary()'s copy
s = s.replace(dup + '''        for reason, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"  -{n:5} rejected: {reason}")''',
'''        for reason, n in sorted(self._rejection_counts().items(),
                                key=lambda kv: -kv[1]):
            lines.append(f"  -{n:5} rejected: {reason}")''')

# The shared helper, placed just above no_trade_reason.
s = s.replace('''    def no_trade_reason(self) -> str | None:''',
'''    def _rejection_counts(self) -> dict[str, int]:
        """Every rejection reason across `rejected`, tallied once. Shared by
        `no_trade_reason` and `summary` so the two tellings of the same day
        cannot drift apart."""
        counts: dict[str, int] = {}
        for s in self.rejected:
            for r in s.reasons:
                counts[r.value] = counts.get(r.value, 0) + 1
        return counts

    def no_trade_reason(self) -> str | None:''')

assert "_rejection_counts" in s and s.count(dup) == 0
io.open(p, "w", encoding="utf-8").write(s)
print("stage4 repaired")
