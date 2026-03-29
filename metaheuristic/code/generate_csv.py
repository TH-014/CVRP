"""
generate_csv.py
===============
Reads optimal solutions from the dataset .sol files and the GA solver
output files, then writes a comparative results.csv.

Output CSV columns
------------------
  testcase_name       | instance name (e.g. A-n32-k5)
  dataset             | which benchmark set: A | E | Golden
  instance_size       | number of customers (excluding depot)
  num_of_vehicles     | number of routes in the optimal solution
  optimal_distance    | cost from the .sol file
  original_ga_distance| cost found by Original GA
  original_ga_gap     | gap % = (original - optimal) / optimal * 100
  improved_ga_distance| cost found by Improved GA
  improved_ga_gap     | gap % = (improved - optimal) / optimal * 100

Usage
-----
  cd metaheuristic/code
  python generate_csv.py
"""

import os
import re
import csv
from pathlib import Path

# ── Path configuration ────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    'A':      os.path.join(SCRIPT_DIR, '..', '..', 'A',      'A'),
    'E':      os.path.join(SCRIPT_DIR, '..', '..', 'E',      'E'),
    'Golden': os.path.join(SCRIPT_DIR, '..', '..', 'Golden', 'Golden'),
}

OUT1_DIR = os.path.join(SCRIPT_DIR, '..', 'outputs', 'original_ga')
OUT2_DIR = os.path.join(SCRIPT_DIR, '..', 'outputs', 'improved_ga')
OUTPUT_CSV = os.path.join(SCRIPT_DIR, '..', 'outputs', 'results.csv')


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_sol_file(filepath):
    """Read a .sol file; return (instance_size, num_vehicles, optimal_distance)."""
    with open(filepath, 'r') as f:
        content = f.read()

    routes = re.findall(r'Route #\d+:', content)
    num_vehicles = len(routes)

    cost_match = re.search(r'Cost\s+(\d+(?:\.\d+)?)', content)
    optimal_distance = float(cost_match.group(1)) if cost_match else None

    node_pattern = r'Route #\d+:\s*(.*?)(?:\n|$)'
    all_nodes = []
    for m in re.finditer(node_pattern, content):
        all_nodes.extend(m.group(1).strip().split())
    instance_size = len(set(all_nodes))

    return instance_size, num_vehicles, optimal_distance


def read_ga_solution(filepath):
    """Extract total distance from a GA _solution.txt file."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        content = f.read()
    m = re.search(r'Total distance:\s*(\d+(?:\.\d+)?)', content)
    return float(m.group(1)) if m else None


def calculate_gap(optimal, solution):
    if optimal is None or solution is None or optimal == 0:
        return None
    return round((solution - optimal) / optimal * 100, 2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    results = []

    for ds_name, ds_dir in DATASETS.items():
        ds_dir = os.path.normpath(ds_dir)
        if not os.path.isdir(ds_dir):
            print(f"  [!] Dataset directory not found: {ds_dir}")
            continue

        sol_files = sorted(f for f in os.listdir(ds_dir) if f.endswith('.sol'))
        if not sol_files:
            print(f"  [!] No .sol files in {ds_dir}")
            continue

        print(f"\n{'='*70}")
        print(f"  Dataset: {ds_name}  ({len(sol_files)} instance(s))")
        print(f"{'='*70}")

        for sol_file in sol_files:
            name = Path(sol_file).stem
            print(f"\n  Processing: {name}")

            sol_path = os.path.join(ds_dir, sol_file)
            try:
                inst_size, num_veh, optimal = read_sol_file(sol_path)
            except Exception as e:
                print(f"    ERROR reading .sol: {e}"); continue

            print(f"    Instance size   : {inst_size}")
            print(f"    Vehicles        : {num_veh}")
            print(f"    Optimal distance: {optimal}")

            # Original GA output
            orig_path = os.path.join(OUT1_DIR, f"{name}_solution.txt")
            orig_dist = read_ga_solution(orig_path)
            if orig_dist is not None:
                print(f"    Original GA dist: {orig_dist}  [OK]")
            else:
                print(f"    Original GA     : not found ({orig_path})")

            # Improved GA output
            impr_path = os.path.join(OUT2_DIR, f"{name}_solution.txt")
            impr_dist = read_ga_solution(impr_path)
            if impr_dist is not None:
                print(f"    Improved GA dist: {impr_dist}  [OK]")
            else:
                print(f"    Improved GA     : not found ({impr_path})")

            orig_gap = calculate_gap(optimal, orig_dist)
            impr_gap = calculate_gap(optimal, impr_dist)

            results.append({
                'testcase_name':        name,
                'dataset':              ds_name,
                'instance_size':        inst_size,
                'num_of_vehicles':      num_veh,
                'optimal_distance':     optimal if optimal is not None else '',
                'original_ga_distance': orig_dist if orig_dist is not None else '',
                'original_ga_gap':      orig_gap  if orig_gap  is not None else '',
                'improved_ga_distance': impr_dist if impr_dist is not None else '',
                'improved_ga_gap':      impr_gap  if impr_gap  is not None else '',
            })

    if not results:
        print("\nNo results to write."); return

    results.sort(key=lambda x: (x['dataset'], x['testcase_name']))

    fieldnames = [
        'testcase_name', 'dataset', 'instance_size', 'num_of_vehicles',
        'optimal_distance',
        'original_ga_distance', 'original_ga_gap',
        'improved_ga_distance', 'improved_ga_gap',
    ]

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*70}")
    print(f"  Results written to: {os.path.normpath(OUTPUT_CSV)}")
    print(f"  Total instances   : {len(results)}")

    # ── Summary statistics ────────────────────────────────────────────────────
    orig_found  = sum(1 for r in results if r['original_ga_distance'] != '')
    impr_found  = sum(1 for r in results if r['improved_ga_distance'] != '')
    orig_gaps   = [r['original_ga_gap'] for r in results
                   if isinstance(r['original_ga_gap'], float)]
    impr_gaps   = [r['improved_ga_gap'] for r in results
                   if isinstance(r['improved_ga_gap'], float)]

    print(f"\n  Original GA solutions found  : {orig_found}/{len(results)}")
    print(f"  Improved GA solutions found  : {impr_found}/{len(results)}")
    if orig_gaps:
        print(f"  Original GA avg gap          : {sum(orig_gaps)/len(orig_gaps):+.2f}%")
    if impr_gaps:
        print(f"  Improved GA avg gap          : {sum(impr_gaps)/len(impr_gaps):+.2f}%")

    # Per-dataset summary
    print(f"\n  {'Dataset':<10} {'Algo':<16} {'Instances':>10} {'Avg Gap %':>10}")
    print(f"  {'-'*10} {'-'*16} {'-'*10} {'-'*10}")
    for ds in DATASETS:
        rows = [r for r in results if r['dataset'] == ds]
        for label, key in [('Original GA', 'original_ga_gap'),
                            ('Improved GA', 'improved_ga_gap')]:
            gaps = [r[key] for r in rows if isinstance(r[key], float)]
            if gaps:
                avg = sum(gaps) / len(gaps)
                print(f"  {ds:<10} {label:<16} {len(gaps):>10} {avg:>+10.2f}")

    # ── Preview ───────────────────────────────────────────────────────────────
    print(f"\n  Preview (first 5 rows):")
    print(f"  {'-'*70}")
    with open(OUTPUT_CSV, 'r') as f:
        for i, line in enumerate(f):
            if i >= 6: break
            print(f"  {line.strip()}")


if __name__ == '__main__':
    main()
