#!/bin/bash

# ── Install dependencies ──────────────────────────────────────────────
echo "=== Checking dependencies ==="
python3 -m pip install matplotlib -q 2>/dev/null || python3 -m pip install matplotlib --break-system-packages -q

# ── Clean previous results ────────────────────────────────────────────
rm -f *_enhanced.txt *_sweep_gillett.txt *_enhanced.png *_sweep_gillett.png comparison_results.txt

# ── Run Gillett & Miller Sweep ────────────────────────────────────────
echo "=== Running Gillett Sweep ==="
python3 cvrp_sweep_gillett.py

# ── Run Enhanced Sweep ────────────────────────────────────────────────
echo "=== Running Enhanced Sweep ==="
python3 cvrp_enhanced.py

# ── Compare results ───────────────────────────────────────────────────
echo "=== Comparing Results ==="
python3 extract.py

echo "All done."