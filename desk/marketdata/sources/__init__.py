"""Data sources. Each one fetches raw bytes and parses them into desk types.

The split between `fetch` and `parse` is deliberate and load-bearing: parsing
is a pure function over captured bytes, so it is testable offline against real
recorded payloads, while fetching is the only part that needs the network.
Every parser in here is tested against fixtures captured from the live source.
"""
