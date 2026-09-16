"""Exchange-disclosed research: filings, announcements, events.

Every record here carries when the market learned it. See models.py.
"""

from desk.research.models import (
    Announcement,
    BoardMeeting,
    CorporateEvent,
    Filing,
    InsiderDeal,
    ResultPeriod,
    ShareholdingSnapshot,
    Undated,
)

__all__ = [
    "Announcement", "BoardMeeting", "CorporateEvent", "Filing",
    "InsiderDeal", "ResultPeriod", "ShareholdingSnapshot", "Undated",
]

# The store and the XBRL parser, exported at package level for symmetry with
# desk.store - callers should not have to reach into a submodule for the main
# entry points. Imported at the END because store.py imports models.py.
from desk.research.store import (            # noqa: E402
    FilingStore,
    FilingStoreError,
    StoredFiling,
    at_open,
)
from desk.research.xbrl import (             # noqa: E402
    FinancialFacts,
    XbrlError,
    parse_xbrl,
)

__all__ += [
    "FilingStore", "FilingStoreError", "FinancialFacts", "StoredFiling",
    "XbrlError", "at_open", "parse_xbrl",
]
