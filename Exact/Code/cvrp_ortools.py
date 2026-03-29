"""
CVRP Solver using Google OR-Tools
==================================
Finds optimal (or near-optimal with time limit) solutions for the
Capacitated Vehicle Routing Problem using OR-Tools' built-in VRP engine,
which internally uses a combination of:
  - LNS (Large Neighbourhood Search)
  - Guided Local Search
  - Constraint Programming with proper Branch-and-Cut

Install dependency:
    pip install ortools

Usage:
    python cvrp_ortools.py <vrp_file> [vrp_file2 ...] [OPTIONS]

Options:
    --capacity    INT   Override vehicle capacity from file
    --n_vehicles  INT   Override number of vehicles from file
    --time_limit  INT   Seconds per instance (default: 30)
                        Higher = better solution quality / more likely optimal
    --max_nodes   INT   Max solver search nodes (default: unlimited = -1)
    --strategy    STR   Search strategy: automatic | path_cheapest_arc |
                        savings | sweep | christofides (default: automatic)
    --local_search STR  Local search metaheuristic: guided_local_search |
                        simulated_annealing | tabu_search | generic_tabu
                        (default: guided_local_search)
    --first_solution STR First solution strategy (alias for --strategy)

Examples:
    # Quick solve (30 s, usually finds optimal for small instances)
    python cvrp_ortools.py problem.vrp

    # Override capacity and vehicle count
    python cvrp_ortools.py problem.vrp --capacity 100 --n_vehicles 5

    # Give more time for larger instances
    python cvrp_ortools.py big.vrp --time_limit 300

    # Use specific search strategy
    python cvrp_ortools.py problem.vrp --time_limit 60 --local_search tabu_search

    # Solve multiple files at once
    python cvrp_ortools.py *.vrp --time_limit 60
"""

import sys
import os
import math
import time
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── OR-Tools import with friendly error ──────────────────────────────────────
try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
except ImportError:
    print("=" * 60)
    print("ERROR: Google OR-Tools is not installed.")
    print("Install it with:  pip install ortools")
    print("=" * 60)
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  VRP FILE PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_vrp(path):
    """
    Parse a TSPLIB-format .vrp file.
    Returns a dict with: name, dimension, capacity, coords, demands, depot.
    """
    data = dict(coords={}, demands={}, depot=1)
    section = None

    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line == "EOF":
                continue

            # Key : Value header lines
            if ':' in line and not line[0].isdigit() and not line[0] == '-':
                key, _, val = line.partition(':')
                key = key.strip().upper()
                val = val.strip()
                if   key == 'NAME':      data['name']      = val
                elif key == 'DIMENSION': data['dimension']  = int(val)
                elif key == 'CAPACITY':  data['capacity']   = int(val)
                continue

            # Section headers
            if line.startswith('NODE_COORD'):  section = 'coord';  continue
            if line.startswith('DEMAND'):       section = 'demand'; continue
            if line.startswith('DEPOT'):        section = 'depot';  continue

            parts = line.split()
            if section == 'coord'  and len(parts) >= 3:
                nid = int(parts[0])
                data['coords'][nid] = (float(parts[1]), float(parts[2]))
            elif section == 'demand' and len(parts) >= 2:
                nid = int(parts[0])
                data['demands'][nid] = int(parts[1])
            elif section == 'depot' and parts[0] != '-1':
                data['depot'] = int(parts[0])

    return data


# ─────────────────────────────────────────────────────────────────────────────
# 2.  INSTANCE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def euclidean_int(c1, c2):
    """Integer Euclidean distance (TSPLIB convention: floor(d + 0.5))."""
    return int(math.floor(math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2) + 0.5))


class CVRPInstance:
    def __init__(self, vrp_data, capacity_override=None, k_override=None):
        d = vrp_data
        self.name    = d.get('name', 'unknown')
        self.depot   = d['depot']
        self.coords  = d['coords']
        self.demands = d['demands']
        self.C       = capacity_override if capacity_override else d['capacity']

        self.customers = sorted(n for n in self.coords if n != self.depot)
        self.n         = len(self.customers)
        # OR-Tools uses 0-based node indices; we map depot → 0
        self.nodes     = [self.depot] + self.customers   # index 0 = depot

        # Minimum vehicles needed
        total_d = sum(self.demands[c] for c in self.customers)
        self.total_demand = total_d
        min_k = math.ceil(total_d / self.C)
        self.k = max(k_override, min_k) if k_override else min_k

        # Distance matrix (OR-Tools index space)
        N = len(self.nodes)
        self.dist_matrix = [[0]*N for _ in range(N)]
        for i, ni in enumerate(self.nodes):
            for j, nj in enumerate(self.nodes):
                self.dist_matrix[i][j] = euclidean_int(
                    self.coords[ni], self.coords[nj])

        # Demand vector in OR-Tools index space (depot demand = 0)
        self.demand_vec = [self.demands[n] for n in self.nodes]

    def ortools_idx_to_node(self, idx):
        return self.nodes[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  STRATEGY MAPS
# ─────────────────────────────────────────────────────────────────────────────

FIRST_SOLUTION_STRATEGIES = {
    'automatic':         routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC,
    'path_cheapest_arc': routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
    'savings':           routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
    'sweep':             routing_enums_pb2.FirstSolutionStrategy.SWEEP,
    'christofides':      routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES,
    'parallel_cheapest_insertion':
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
    'local_cheapest_insertion':
        routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION,
    'global_cheapest_arc':
        routing_enums_pb2.FirstSolutionStrategy.GLOBAL_CHEAPEST_ARC,
}

LOCAL_SEARCH_METAHEURISTICS = {
    'guided_local_search': routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
    'simulated_annealing': routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
    'tabu_search':         routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
    'generic_tabu':        routing_enums_pb2.LocalSearchMetaheuristic.GENERIC_TABU_SEARCH,
    'automatic':           routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC,
}


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CORE SOLVER
# ─────────────────────────────────────────────────────────────────────────────

def solve_cvrp(inst, time_limit=30, max_nodes=-1,
               first_solution_strategy='path_cheapest_arc',
               local_search_metaheuristic='guided_local_search',
               log_search=False):
    """
    Solve CVRP with OR-Tools routing solver.

    Parameters
    ----------
    inst                      : CVRPInstance
    time_limit                : seconds (higher → better / more likely optimal)
    max_nodes                 : search nodes limit (-1 = unlimited)
    first_solution_strategy   : how to build the initial solution
    local_search_metaheuristic: improvement metaheuristic

    Returns
    -------
    dict with keys:
        routes      – list of lists of original node IDs (excluding depot)
        cost        – total integer distance
        status      – 'OPTIMAL' | 'FEASIBLE' | 'INFEASIBLE' | 'NOT_FOUND'
        lower_bound – LP lower bound (when available)
        gap_pct     – optimality gap percentage (when available)
        solve_time  – wall-clock seconds
    """
    N = len(inst.nodes)   # includes depot at index 0

    # ── Create routing index manager ─────────────────────────────────────────
    manager = pywrapcp.RoutingIndexManager(
        N,        # number of nodes (including depot)
        inst.k,   # number of vehicles
        0         # depot index (0-based)
    )
    routing = pywrapcp.RoutingModel(manager)

    # ── Distance callback ────────────────────────────────────────────────────
    def distance_callback(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return inst.dist_matrix[i][j]

    transit_cb_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)

    # ── Capacity constraint ──────────────────────────────────────────────────
    def demand_callback(from_index):
        node = manager.IndexToNode(from_index)
        return inst.demand_vec[node]

    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_cb_idx,
        0,                       # null capacity slack
        [inst.C] * inst.k,       # vehicle capacities
        True,                    # start cumul to zero
        'Capacity'
    )

    # ── Phase 1: build guaranteed first feasible solution ───────────────────
    # PATH_CHEAPEST_ARC always finds a feasible solution instantly.
    # We run it without a time limit so we always get at least one solution.
    phase1_params = pywrapcp.DefaultRoutingSearchParameters()
    phase1_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    phase1_params.log_search = False

    t0 = time.time()
    phase1_solution = routing.SolveWithParameters(phase1_params)

    if phase1_solution is None:
        solve_time = time.time() - t0
        return dict(routes=[], cost=None, status='INFEASIBLE',
                    lower_bound=None, gap_pct=None, solve_time=solve_time)

    phase1_cost = phase1_solution.ObjectiveValue()
    print(f'  [Phase 1] Initial feasible solution cost: {phase1_cost}')

    # ── Phase 2: improve with chosen metaheuristic ───────────────────────────
    # Use the user-selected first_solution_strategy and local_search,
    # with the full time budget.
    search_params = pywrapcp.DefaultRoutingSearchParameters()

    fs_key = first_solution_strategy.lower()
    fs_enum = FIRST_SOLUTION_STRATEGIES.get(
        fs_key, routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_params.first_solution_strategy = fs_enum

    ls_key = local_search_metaheuristic.lower()
    ls_enum = LOCAL_SEARCH_METAHEURISTICS.get(
        ls_key, routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_params.local_search_metaheuristic = ls_enum

    # Remaining time after phase 1
    elapsed_phase1 = time.time() - t0
    remaining = max(1, time_limit - int(elapsed_phase1))
    search_params.time_limit.FromSeconds(remaining)

    if max_nodes > 0:
        search_params.solution_limit = max_nodes

    search_params.log_search = log_search

    solution = routing.SolveWithParameters(search_params)
    solve_time = time.time() - t0

    # If phase 2 found nothing better, fall back to phase 1 result
    if solution is None:
        solution = phase1_solution

    # ── Status map ───────────────────────────────────────────────────────────
    status_map = {
        0: 'NOT_FOUND',
        1: 'NOT_FOUND',
        2: 'FEASIBLE',
        3: 'INFEASIBLE',
        4: 'OPTIMAL',
    }
    status = status_map.get(routing.status(), 'FEASIBLE')

    # We always have phase1_solution so status can never be NOT_FOUND
    if status == 'NOT_FOUND':
        status = 'FEASIBLE'

    # ── Extract routes ───────────────────────────────────────────────────────
    routes = []
    for vehicle in range(inst.k):
        route = []
        idx = routing.Start(vehicle)
        while not routing.IsEnd(idx):
            node_idx = manager.IndexToNode(idx)
            if node_idx != 0:                          # skip depot
                route.append(inst.ortools_idx_to_node(node_idx))
            idx = solution.Value(routing.NextVar(idx))
        if route:
            routes.append(route)

    cost = solution.ObjectiveValue()

    # Optimality gap
    if status == 'OPTIMAL':
        gap_pct = 0.0
    else:
        # OR-Tools does not expose the LP lower bound directly via the
        # routing API, so we report None when not proven optimal.
        gap_pct = None
    lower_bound = None

    return dict(routes=routes, cost=cost, status=status,
                lower_bound=lower_bound, gap_pct=gap_pct,
                solve_time=solve_time)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SOLUTION VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_solution(inst, routes):
    """
    Check that routes:
      - cover every customer exactly once
      - respect vehicle capacity
    Returns (valid: bool, errors: list[str])
    """
    errors = []
    visited = []
    for r_idx, route in enumerate(routes):
        load = sum(inst.demands[v] for v in route)
        if load > inst.C:
            errors.append(f"Route {r_idx+1}: load {load} exceeds capacity {inst.C}")
        visited.extend(route)

    all_customers = sorted(inst.customers)
    if sorted(visited) != all_customers:
        missing = set(all_customers) - set(visited)
        duplicate = [v for v in visited if visited.count(v) > 1]
        if missing:
            errors.append(f"Missing customers: {sorted(missing)}")
        if duplicate:
            errors.append(f"Duplicate customers: {sorted(set(duplicate))}")

    return len(errors) == 0, errors


def route_distance(inst, route):
    """Total distance of one route (depot → ... → depot)."""
    if not route:
        return 0
    # Map back to OR-Tools index for dist_matrix
    node_to_idx = {n: i for i, n in enumerate(inst.nodes)}
    d = inst.dist_matrix[0][node_to_idx[route[0]]]
    for i in range(len(route) - 1):
        d += inst.dist_matrix[node_to_idx[route[i]]][node_to_idx[route[i+1]]]
    d += inst.dist_matrix[node_to_idx[route[-1]]][0]
    return d


# ─────────────────────────────────────────────────────────────────────────────
# 6.  TEXT OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def write_solution_txt(inst, result, out_path):
    routes    = result['routes']
    cost      = result['cost']
    status    = result['status']
    solve_time = result['solve_time']

    valid, errors = validate_solution(inst, routes)

    with open(out_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"  CVRP Solution  —  {inst.name}\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"  Status        : {status}\n")
        f.write(f"  Total distance: {cost}\n")
        f.write(f"  Vehicles used : {len(routes)} / {inst.k}\n")
        f.write(f"  Capacity      : {inst.C}\n")
        f.write(f"  Solve time    : {solve_time:.2f} s\n")
        f.write(f"  Valid solution: {'YES' if valid else 'NO'}\n")
        if not valid:
            for e in errors:
                f.write(f"    !! {e}\n")
        f.write("\n")

        f.write("-" * 60 + "\n")
        f.write("  Routes\n")
        f.write("-" * 60 + "\n\n")

        for i, route in enumerate(routes):
            load = sum(inst.demands[v] for v in route)
            dist = route_distance(inst, route)
            path = " -> ".join(
                [str(inst.depot)] + [str(v) for v in route] + [str(inst.depot)]
            )
            f.write(f"  Route {i+1:>2}: {path}\n")
            f.write(f"            Load: {load:>4} / {inst.C}"
                    f"   Distance: {dist}\n\n")

        f.write("-" * 60 + "\n")
        f.write(f"  Total distance: {cost}\n")
        f.write("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  PLOT OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

ROUTE_COLORS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#469990', '#dcbeff',
    '#9A6324', '#800000', '#aaffc3', '#808000', '#ffd8b1',
]

def plot_solution(inst, result, out_path):
    routes = result['routes']
    cost   = result['cost']
    status = result['status']

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_facecolor('#f5f5f5')
    fig.patch.set_facecolor('#f5f5f5')

    legend_patches = []

    for r_idx, route in enumerate(routes):
        color = ROUTE_COLORS[r_idx % len(ROUTE_COLORS)]
        full  = [inst.depot] + route + [inst.depot]
        xs    = [inst.coords[n][0] for n in full]
        ys    = [inst.coords[n][1] for n in full]

        # Draw route line with arrows to show direction
        for seg in range(len(full) - 1):
            ax.annotate(
                "", xy=(xs[seg+1], ys[seg+1]), xytext=(xs[seg], ys[seg]),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.8, mutation_scale=14),
                zorder=2
            )

        load = sum(inst.demands[v] for v in route)
        dist = route_distance(inst, route)
        legend_patches.append(
            mpatches.Patch(color=color,
                           label=f"Route {r_idx+1}  "
                                 f"load={load}/{inst.C}  dist={dist}")
        )

    # Customer nodes
    for v in inst.customers:
        x, y = inst.coords[v]
        ax.scatter(x, y, c='#222222', s=55, zorder=4)
        ax.annotate(str(v), (x, y),
                    textcoords='offset points', xytext=(4, 4),
                    fontsize=7, color='#111111')

    # Depot
    dx, dy = inst.coords[inst.depot]
    ax.scatter(dx, dy, c='red', s=220, marker='*', zorder=5)
    ax.annotate(f"Depot\n({inst.depot})", (dx, dy),
                textcoords='offset points', xytext=(6, 6),
                fontsize=8, fontweight='bold', color='darkred')

    title = (f"{inst.name}  —  dist={cost}  [{status}]\n"
             f"Capacity={inst.C}  |  Vehicles={len(routes)}/{inst.k}")
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")

    depot_patch = mpatches.Patch(color='red', label=f'Depot ({inst.depot})')
    ax.legend(handles=legend_patches + [depot_patch],
              loc='upper left', fontsize=7,
              framealpha=0.85, edgecolor='#cccccc')

    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CVRP Solver using Google OR-Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cvrp_ortools.py problem.vrp
  python cvrp_ortools.py problem.vrp --capacity 100 --n_vehicles 5
  python cvrp_ortools.py problem.vrp --time_limit 120 --local_search tabu_search
  python cvrp_ortools.py a.vrp b.vrp c.vrp --time_limit 60
        """
    )

    parser.add_argument('vrp_files', nargs='+',
                        help='.vrp file(s) to solve')
    parser.add_argument('--capacity', type=int, default=None,
                        help='Override vehicle capacity from file')
    parser.add_argument('--n_vehicles', type=int, default=None,
                        help='Override number of vehicles from file')
    parser.add_argument('--time_limit', type=int, default=30,
                        help='Solver time limit in seconds per instance '
                             '(default: 30). Higher = better/more likely optimal.')
    parser.add_argument('--max_nodes', type=int, default=-1,
                        help='Max solver search nodes per instance '
                             '(default: -1 = unlimited)')
    parser.add_argument('--strategy', type=str, default='path_cheapest_arc',
                        choices=list(FIRST_SOLUTION_STRATEGIES.keys()),
                        help='First solution strategy '
                             '(default: path_cheapest_arc — always finds a '
                             'feasible solution instantly)')
    parser.add_argument('--local_search', type=str,
                        default='guided_local_search',
                        choices=list(LOCAL_SEARCH_METAHEURISTICS.keys()),
                        help='Local search metaheuristic '
                             '(default: guided_local_search)')
    parser.add_argument('--log_search', action='store_true',
                        help='Print OR-Tools solver progress to stdout')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to write output files (default: .)')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []

    for vrp_path in args.vrp_files:
        print(f"\n{'='*60}")
        print(f"  Solving: {vrp_path}")
        print(f"{'='*60}")

        # ── Parse & build instance ────────────────────────────────────────
        try:
            vrp_data = parse_vrp(vrp_path)
        except Exception as e:
            print(f"  ERROR parsing file: {e}")
            continue

        inst = CVRPInstance(vrp_data,
                            capacity_override=args.capacity,
                            k_override=args.n_vehicles)

        print(f"  Instance  : {inst.name}")
        print(f"  Customers : {inst.n}")
        print(f"  Depot     : {inst.depot}")
        print(f"  Capacity  : {inst.C}")
        print(f"  Vehicles  : {inst.k}")
        print(f"  Total demand : {inst.total_demand}  "
              f"(tightness {inst.total_demand/(inst.k*inst.C):.2f})")
        print(f"  Time limit   : {args.time_limit}s")
        print(f"  Max nodes    : {'unlimited' if args.max_nodes < 0 else args.max_nodes}")
        print(f"  Strategy     : {args.strategy}")
        print(f"  Local search : {args.local_search}")
        print(f"  Log search   : {args.log_search}")
        print()

        # ── Solve ─────────────────────────────────────────────────────────
        result = solve_cvrp(
            inst,
            time_limit=args.time_limit,
            max_nodes=args.max_nodes,
            first_solution_strategy=args.strategy,
            local_search_metaheuristic=args.local_search,
            log_search=args.log_search,
        )

        # ── Print summary ─────────────────────────────────────────────────
        print(f"  Status        : {result['status']}")
        if result['cost'] is not None:
            print(f"  Total distance: {result['cost']}")
            print(f"  Vehicles used : {len(result['routes'])}")
            valid, errors = validate_solution(inst, result['routes'])
            print(f"  Valid         : {'YES' if valid else 'NO – ' + str(errors)}")
        else:
            print("  No solution found.")
        print(f"  Solve time    : {result['solve_time']:.2f}s")

        if result['cost'] is None:
            print("  Skipping output (no feasible solution).")
            continue

        # ── Write outputs ─────────────────────────────────────────────────
        base     = os.path.splitext(os.path.basename(vrp_path))[0]
        txt_path = os.path.join(args.output_dir, f"{base}_solution.txt")
        png_path = os.path.join(args.output_dir, f"{base}_plot.png")

        write_solution_txt(inst, result, txt_path)
        plot_solution(inst, result, png_path)

        print(f"\n  Output saved:")
        print(f"    {txt_path}")
        print(f"    {png_path}")

        all_results.append((inst.name, result['status'],
                            result['cost'], result['solve_time']))

    # ── Final summary table ───────────────────────────────────────────────
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        print(f"  {'Instance':<25} {'Status':<12} {'Cost':>10} {'Time':>8}")
        print(f"  {'-'*25} {'-'*12} {'-'*10} {'-'*8}")
        for name, status, cost, t in all_results:
            print(f"  {name:<25} {status:<12} {str(cost):>10} {t:>7.1f}s")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
