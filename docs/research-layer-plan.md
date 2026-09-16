# Research layer — audit of the proposed design, and the plan

Covers fundamentals, news and events: master prompt sections 10, 11 and 12.
Written 2026-09-16 in response to the four-tier fundamentals / three-tier news
proposal. **Everything below was verified against live endpoints before being
written** — no capability is claimed here that was not actually called.

---

## 1. Verdict on the proposal

**The structure is right. The source list needs cutting roughly in half, and
the reason is better than "those ones cost money."**

The proposal's core insight — *four kinds of evidence rather than one website
doing everything*, with official disclosure weighted highest — is correct and
is what the plan below builds. Two things change:

1. **NSE's own free API already provides more than tiers 2–4 combined.**
   Verified live, six endpoints, all working (§2). This was not obvious and it
   removes most of the dependency question.
2. **The paid/scraped sources are not just expensive — they are
   backtest-poison.** That is the finding that matters, and it is explained in
   §4. It is a correctness argument, not a budget one.

---

## 2. What was verified working — live, today

One request each, throttled, using the existing authenticated NSE session:

| endpoint | rows returned | the field that matters |
| --- | --- | --- |
| `corporate-announcements` | 3,345 for RELIANCE | `an_dt` = `16-Sep-2026 17:45:33` |
| `corporate-board-meetings` | 20 | `bm_date`, `bm_purpose` |
| `corporates-financial-results` | 130 | `broadCastDate`, **`xbrl` link** |
| `corporate-share-holdings-master` | 22 | `broadcastDate`, promoter/employee-trust holdings |
| `corporates-pit` (insider/SAST) | 20 | `acqName`, `acqMode`, `afterAcqSharesPer` |
| `event-calendar` | 65 | upcoming results dates |

And the XBRL behind a results row was fetched and parsed **with stdlib
`xml.etree` alone** — no library, no key:

```
RevenueFromOperations               1282600000000.00
OtherIncome                           32140000000.00
Income                              1314740000000.00
Expenses                            1198770000000.00
ProfitBeforeExceptionalItemsAndTax   115970000000.00
ProfitBeforeTax                      115970000000.00
CurrentTax                            24830000000.00
```

92 distinct tags per filing. That is real fundamentals — revenue, margins,
tax, EPS — from the exchange itself.

**So the answer to "where do fundamentals come from" is: the same place the
price data comes from.** No new provider, no key, no subscription, no scraping.

---

## 3. Source-by-source ruling

### Keep — free, official, sanctioned

| source | role | status |
| --- | --- | --- |
| **NSE** | filings, results+XBRL, announcements, board meetings, shareholding, insider deals, event calendar | ✅ already authenticated, all verified |
| **BSE** | cross-check, and BSE-only scrips | ✅ same shape, not yet wired |
| **Moneycontrol / ET / Business Standard** | Indian market & company news | ✅ **via RSS only** — see below |

RSS matters here. Those outlets publish RSS feeds, and a feed is *published
specifically to be read by machines* — that is sanctioned access. Scraping
their HTML article pages is not, and the distinction is not a technicality:
one is an invitation, the other is against their terms.

### Drop — paid

| source | why |
| --- | --- |
| **Bloomberg** | Terminal product, roughly $24–30k/yr. There is no free tier. |
| **Reuters** | The real API is Refinitiv/LSEG, paid. The public website is copyrighted and its terms forbid automated collection. |
| **Tijori** | Paid product. |

You said free only, so these are out. Worth knowing what is actually lost:
**macro context** — RBI, crude, rupee, global rates. That gap is real and §7
covers how to fill it without them.

### Drop for correctness, not price — Screener.in

Screener is genuinely excellent, and I would still use it *by hand*. But it
cannot go in the automated pipeline, for two independent reasons:

1. **No official API.** Automated collection is a terms question, and you
   should not be finding out the answer through a ban.
2. **It has no point-in-time semantics** — and this is the disqualifying one.
   Screener shows what a company's financials look like **now**, including
   restatements. It cannot tell you what was *knowable on 15 March 2024*. Feed
   that into a backtest and every result is contaminated by information that
   did not exist at the time. §4.

NSE's XBRL does not have this problem, because `broadCastDate` is the moment
the market learned it.

### Ask you — Trendlyne

The MCP tooling you describe is real and the field list is genuinely close to
what our agents want. Two questions before I plan around it (§8): **do you
have a paid subscription**, and does its MCP expose a *filing date* per
figure? Without the second, it inherits Screener's problem.

---

## 4. The finding that should drive the design

> **Only exchange-disclosed data carries a timestamp of when the market
> actually learned it. Everything else is a snapshot of the present wearing
> the costume of history.**

Master prompt section 26 already requires this — *"fundamentals lagged to
actual FILING date, not period-end"* — but it was written as a rule to
remember. It is really a **source-selection constraint**: it rules out most
aggregators before you evaluate their data quality at all.

Consequence for the plan: NSE/BSE are not merely Tier 1 by authority. They
are the only tier that can be **backtested**. Everything else is
decision-support for a human reading today's screen, and must be tagged as
such so it can never silently enter a backtest.

---

## 5. Two problems in the proposed design

### 5a. Three news sources carrying one story

Tier 3 is Moneycontrol + ET + Business Standard. All three republish the same
PTI/wire copy. A result covered by all three is **one** piece of evidence
appearing three times — and a naive "3 sources agree" scorer reads that as
confirmation.

This is section 37's ensemble-bias failure applied to news, and your own
section 11 already demanded the fix: distinguish **NEW** from **DUPLICATE**.
The plan handles it at ingest (§6, R5) — cluster by content similarity, keep
the earliest, and let the cluster carry one vote weighted by its *best*
source tier, not by how many outlets picked it up.

### 5b. The diagram runs research on everything

The proposed flow sends market data and news through normalisation into
research agents. Our funnel deliberately does not: expensive work happens
**only** after the universe is narrowed.

Running fundamental and news agents across 1,598 Stage-0 survivors is exactly
the cost explosion the staged funnel exists to prevent — the difference
between roughly ₹1,500/month and ₹40,000–80,000/month.

The reconciliation is a split, and it is a clean one:

- **Cheap, deterministic, whole-universe** → Stage 2. Fundamental *filters*
  computed from already-stored XBRL. No per-call cost, so they can run on
  everything. (debt/equity ceiling, margin trend, earnings recency, promoter
  pledge flag)
- **Expensive, narrative, shortlist-only** → Stage 3. Reading announcements,
  framing bull/bear, surfacing conflicts. ~8 names.
- **Event proximity** → Stage 4, as a hard gate. Already designed:
  `RegimeState.EventProximity` exists and is documented as *the only dimension
  that can force NO TRADE on its own*. The event calendar endpoint is what
  finally lets it read real data.

---

## 6. The plan

Each phase ends the way every phase on this project has: tests pinned against
real captured payloads, then the six review agents.

### R1 — NSE research source *(no new dependency)*
Extend `desk/marketdata/sources/nse.py` with the six verified endpoints, same
fetch/parse split, parsers tested against captured fixtures.
New: `desk/research/` for the parsed shapes — `Filing`, `Announcement`,
`BoardMeeting`, `ShareholdingSnapshot`, `InsiderDeal`, `CorporateEvent`.
**Every one carries `disclosed_at` as a required field.** Not optional — a
record that cannot say when the market learned it cannot be used point-in-time,
and making it required means that is impossible to forget.

### R2 — XBRL parser + fundamentals store
Parse filings into a normalised fact set. Store under `desk/store/` beside the
bars, same discipline: one file per filing, point-in-time by construction,
`as_of` filtering that never opens a file dated after it.
Refuses to guess: a tag it does not recognise is reported, never dropped.

### R3 — Fundamental filters at Stage 2
Deterministic, whole-universe, from stored facts. Declares in `unavailable`
exactly which names had no filing on file — "no data" and "fails the filter"
must never collapse into the same outcome.

### R4 — Event gate at Stage 4
Wire `event-calendar` into `EventProximity`. Results within the holding window
force NO TRADE. This closes a gate that has been designed and inert since the
regime vector was built.

### R5 — News via RSS, with deduplication
`desk/research/news.py`. Tier 1 = NSE/BSE announcements (highest weight,
timestamped, backtestable). Tier 3 = Moneycontrol/ET/BS RSS.
Dedupe by normalised-title + content hash, cluster near-duplicates, keep
earliest, one vote per cluster. Classify each item **before-market /
during-market / after-hours / historical** off its real timestamp — section
11's requirement, and now trivially satisfiable because `an_dt` carries a
clock time.

### R6 — BSE cross-check
Second official source. Disagreement between NSE and BSE on the same filing is
a data-quality signal, which is precisely what `quality.py`'s cross-source
check was built for and has never had a second source to use.

### R7 — Stage 3
The only stage that costs money. Blocked on your decisions in §8.

**Order matters:** R1→R2 unblocks R3 and R4; R4 is the highest
safety-per-hour in the list (it stops trades into an earnings print, today).
R5 is independent and can slot anywhere. R7 last.

---

## 7. The macro gap, stated honestly

Dropping Reuters and Bloomberg costs real coverage: RBI policy, crude, the
rupee, global rates, geopolitics. Nothing free fully replaces it.

Partial, free substitutes, in order of usefulness:
- **RBI's own site** — policy dates and decisions, official and free
- **NSE India VIX + sectoral indices** — the regime engine's actual inputs,
  already reachable through our session
- **FII/DII daily activity** — NSE publishes it
- **Business-press RSS** for the narrative layer

That covers *scheduled, structural* macro well and *breaking* macro poorly.
My recommendation: accept the gap, and make the system state it rather than
paper over it — `market_risks` on the daily plan should say plainly that
breaking macro is not monitored. A known blind spot you can see is safer than
one you cannot.

---

## 8. What I need from you

**Blocking — R7 only:**
1. **Stage 3 LLM**: which model, and a hard monthly ceiling in ₹. The funnel's
   entire economic argument is that only this stage costs per-call, and that
   is unverifiable until spend is actually metered. I would wire a cost meter
   before the first call.

**Blocking — Trendlyne only:**
2. Do you have a **paid Trendlyne subscription**? If yes, does its MCP give a
   *filing date* per figure? If no filing date, it is Screener's problem again
   and I would use it for human research, not the pipeline.

**Confirmations, non-blocking:**
3. Confirm **zero paid spend** on data — I will assume this and build without
   it unless you say otherwise.
4. **Screener**: I recommend manual use only, not automated. Say if you
   disagree and I will put the terms question to you properly first.
5. **BSE priority** — mainly matters if you intend to trade BSE-only scrips.

**Nothing else is blocked.** R1 through R6 need no key, no subscription and no
decision from you, and I can start immediately.

---

## 9. What this does *not* fix

Being straight about the limits of this plan:

- It does not make predictions accurate. It widens the evidence base and makes
  the reasoning auditable. Whether a signal *wins* is measured by the decision
  journal, which still does not exist.
- Fundamental filters do not make a strategy profitable. They remove a class
  of obviously-bad candidate.
- XBRL coverage is not universal — smaller scrips file late or in odd formats.
  Coverage will be reported per scan, not assumed.
- Announcement *text* is unstructured. R1–R6 give structured metadata and the
  PDF link; actually reading the PDF is Stage 3's job, and costs money.

---

## 10. Measured cost of a full-universe pass (added after R2)

Numbers, not estimates. Measured on a live run of 2 symbols (RELIANCE +
IRCTC) through all five endpoints plus XBRL documents, then projected to the
1,598 symbols Stage 0 survives.

| run | requests | at 1.0s throttle |
| --- | --- | --- |
| first full backfill, with XBRL | ~92,700 | **~26 h** |
| all five endpoints, `--no-xbrl` | ~8,000 | ~2.2 h |
| incremental run, after the backfill | ~8,000 + new filings only | ~2.2 h |
| daily top-up (`--kinds filings,announcements`) | ~3,200 | ~0.9 h |

Disk: ~1.5 MB/symbol, so **~2.5 GB** for the universe. That is dominated by
announcement history — RELIANCE alone has 3,345 announcements going back
years.

**The XBRL documents are the cost**, roughly 53 per symbol against 5 endpoint
calls. Two things follow, and both are now implemented:

- **An incremental run never refetches an XBRL it already parsed.** A
  historical filing's numbers do not change. Without this, every run paid the
  full 26 hours; with it, a re-run costs one document per genuinely new
  filing.
- **Resume is per KIND, not per symbol.** Getting this wrong was a silent
  data-loss bug: `--kinds insider` marked the symbol done, and a later full
  run skipped it entirely, never fetching filings or announcements while the
  operator believed the data was complete.

### How to actually run the backfill

The first pass is a multi-day job, and it is designed to be interrupted:

```powershell
# Stage it. Every run resumes; nothing is refetched.
python -m desk.marketdata.refresh research <SYMBOLS> --no-xbrl     # ~2h, metadata
python -m desk.marketdata.refresh research <SYMBOLS>               # then documents
```

Do the `--no-xbrl` pass first: it gets every filing, announcement, board
meeting, shareholding and insider record for the whole universe in about two
hours, which is enough for the event gate (R4) and for most of R5. The
documents — the part that takes a day — are only needed for R3's fundamental
filters, and can fill in behind.

**Not yet solved:** nothing bounds how far back a first fetch goes. A
`--since` option would let a new symbol pull only recent history instead of
all 3,345 announcements. Worth adding before the real backfill runs.
