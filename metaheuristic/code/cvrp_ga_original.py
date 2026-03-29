"""
CVRP Solver — Original Genetic Algorithm
=========================================
Solves the Capacitated Vehicle Routing Problem using a standard
Genetic Algorithm.

Algorithm details:
  Representation  : Giant-tour permutation of all customer nodes.
  Decoding        : Greedy sequential split — greedy-fills routes
                    until capacity is exceeded, then starts a new route.
  Selection       : Binary tournament selection.
  Crossover       : Order Crossover (OX1) — preserves relative order.
  Mutation        : Swap mutation OR inversion mutation (50/50 chance).
  Survivor sel.   : Elitism — top elite_size individuals survive unchanged.

Usage:
    python cvrp_ga_original.py <vrp_file> [vrp_file2 ...] [OPTIONS]

Options:
    --capacity    INT   Override vehicle capacity from file
    --n_vehicles  INT   Override number of vehicles from file
    --time_limit  INT   Wall-clock seconds per instance (default: 120)
    --pop_size    INT   GA population size (default: 100)
    --generations INT   Number of GA generations (default: 300)
    --seed        INT   Random seed for reproducibility (default: 42)
    --output_dir  PATH  Directory to write outputs (default: ../outputs/original_ga)

Examples:
    python cvrp_ga_original.py problem.vrp
    python cvrp_ga_original.py ../../A/A/*.vrp --output_dir ../outputs/original_ga
    python cvrp_ga_original.py problem.vrp --pop_size 150 --generations 500
"""

import sys
import os
import math
import time
import random
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─────────────────────────────────────────────────────────────────────────────
# 1.  VRP FILE PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_vrp(path):
    """
    Parse a TSPLIB-format .vrp file.
    Returns a dict with: name, dimension, capacity, coords, demands, depot,
                         edge_weight_type, distance_matrix (EXPLICIT only).
    """
    data = dict(coords={}, demands={}, depot=1,
                edge_weight_type='EUC_2D', edge_weight_format=None,
                distance_matrix=None)
    section = None
    matrix_data = []

    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line == "EOF":
                continue

            # Key : Value header lines
            if ':' in line and section != 'EDGE_WEIGHT':
                first_char = line[0]
                if not first_char.isdigit() and first_char != '-':
                    key, _, val = line.partition(':')
                    key = key.strip().upper()
                    val = val.strip()
                    if   key == 'NAME':               data['name']               = val
                    elif key == 'DIMENSION':          data['dimension']          = int(val)
                    elif key == 'CAPACITY':           data['capacity']           = int(val)
                    elif key == 'EDGE_WEIGHT_TYPE':   data['edge_weight_type']   = val
                    elif key == 'EDGE_WEIGHT_FORMAT': data['edge_weight_format'] = val
                    continue

            if line.startswith('NODE_COORD'):   section = 'coord';  continue
            if line.startswith('DEMAND'):        section = 'demand'; continue
            if line.startswith('DEPOT'):         section = 'depot';  continue
            if line.startswith('EDGE_WEIGHT_SECTION'): section = 'EDGE_WEIGHT'; continue

            parts = line.split()
            if section == 'coord' and len(parts) >= 3:
                nid = int(parts[0])
                data['coords'][nid] = (float(parts[1]), float(parts[2]))
            elif section == 'demand' and len(parts) >= 2:
                nid = int(parts[0])
                data['demands'][nid] = int(parts[1])
            elif section == 'depot' and parts[0] != '-1':
                data['depot'] = int(parts[0])
            elif section == 'EDGE_WEIGHT':
                try:
                    matrix_data.extend(int(x) for x in parts)
                except ValueError:
                    pass

    # Build explicit distance matrix
    if data['edge_weight_type'] == 'EXPLICIT' and matrix_data:
        n = data['dimension']
        dist = [[0] * (n + 1) for _ in range(n + 1)]
        fmt = data.get('edge_weight_format', '')
        if fmt == 'LOWER_ROW':
            idx = 0
            for i in range(2, n + 1):
                for j in range(1, i):
                    dist[i][j] = matrix_data[idx]
                    dist[j][i] = matrix_data[idx]
                    idx += 1
        data['distance_matrix'] = dist

    return data


# ─────────────────────────────────────────────────────────────────────────────
# 2.  INSTANCE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def euclidean_int(c1, c2):
    """Integer Euclidean distance — TSPLIB convention: floor(d + 0.5)."""
    return int(math.floor(math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2) + 0.5))


class CVRPInstance:
    def __init__(self, vrp_data, capacity_override=None, k_override=None):
        d = vrp_data
        self.name    = d.get('name', 'unknown')
        self.depot   = d['depot']
        self.coords  = d['coords']
        self.demands = d['demands']
        self.C       = capacity_override if capacity_override else d['capacity']
        self.ew_type = d['edge_weight_type']

        self.customers = sorted(n for n in self.demands if n != self.depot)
        self.n         = len(self.customers)

        total_d = sum(self.demands[c] for c in self.customers)
        self.total_demand = total_d
        min_k = math.ceil(total_d / self.C)
        self.k = max(k_override, min_k) if k_override else min_k

        n_nodes = d['dimension']

        # Build distance matrix
        if self.ew_type == 'EXPLICIT':
            self.dist = d['distance_matrix']
        else:
            self.dist = [[0] * (n_nodes + 1) for _ in range(n_nodes + 1)]
            for i in range(1, n_nodes + 1):
                for j in range(1, n_nodes + 1):
                    if i != j:
                        self.dist[i][j] = euclidean_int(self.coords[i], self.coords[j])

    def route_distance(self, route):
        if not route:
            return 0
        d = self.dist[self.depot][route[0]]
        for i in range(len(route) - 1):
            d += self.dist[route[i]][route[i + 1]]
        d += self.dist[route[-1]][self.depot]
        return d

    def total_distance(self, routes):
        return sum(self.route_distance(r) for r in routes)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  GENETIC ALGORITHM — ORIGINAL
# ─────────────────────────────────────────────────────────────────────────────

class OriginalGA:
    """
    Standard GA for CVRP — no local search, purely random initialisation.

    Parameters
    ----------
    inst            : CVRPInstance
    pop_size        : population size
    generations     : number of generations
    crossover_rate  : probability of applying OX1 crossover
    mutation_rate   : probability of mutating each offspring
    tournament_size : k for tournament selection
    elite_size      : number of elite individuals copied each generation
    seed            : random seed
    time_limit      : wall-clock time limit in seconds (None = no limit)
    """
    def __init__(self, inst, pop_size=100, generations=300,
                 crossover_rate=0.9, mutation_rate=0.2,
                 tournament_size=3, elite_size=5,
                 seed=42, time_limit=None):
        self.inst = inst
        self.pop_size        = pop_size
        self.generations     = generations
        self.crossover_rate  = crossover_rate
        self.mutation_rate   = mutation_rate
        self.tournament_size = tournament_size
        self.elite_size      = elite_size
        self.time_limit      = time_limit
        random.seed(seed)

        self.best_cost     = float('inf')
        self.best_routes   = None
        self.history       = []   # best cost per generation
        self.elapsed_time  = 0.0

    # ── Decoding ────────────────────────────────────────────────────────────
    def decode(self, chromosome):
        """Giant-tour → capacity-feasible routes."""
        routes, route, load = [], [], 0
        for c in chromosome:
            d = self.inst.demands[c]
            if load + d <= self.inst.C:
                route.append(c); load += d
            else:
                if route: routes.append(route)
                route, load = [c], d
        if route: routes.append(route)
        return routes

    def fitness(self, chrom):
        return self.inst.total_distance(self.decode(chrom))

    # ── Selection ───────────────────────────────────────────────────────────
    def tournament(self, pop, fits):
        idx = random.sample(range(len(pop)), self.tournament_size)
        return pop[min(idx, key=lambda i: fits[i])][:]

    # ── Crossover: OX1 ──────────────────────────────────────────────────────
    def ox1(self, p1, p2):
        n = len(p1)
        a, b = sorted(random.sample(range(n), 2))
        def make_child(parent_seg, other):
            child = [None] * n
            child[a:b+1] = parent_seg[a:b+1]
            seg_set = set(child[a:b+1])
            remaining = [x for x in other if x not in seg_set]
            pos = (b + 1) % n
            for val in remaining:
                child[pos] = val
                pos = (pos + 1) % n
            return child
        return make_child(p1, p2), make_child(p2, p1)

    # ── Mutation ────────────────────────────────────────────────────────────
    def mutate(self, c):
        if random.random() >= self.mutation_rate:
            return c[:]
        c = c[:]
        i, j = random.sample(range(len(c)), 2)
        if random.random() < 0.5:           # swap mutation
            c[i], c[j] = c[j], c[i]
        else:                               # inversion mutation
            lo, hi = min(i, j), max(i, j)
            c[lo:hi+1] = c[lo:hi+1][::-1]
        return c

    # ── Population init ─────────────────────────────────────────────────────
    def init_pop(self):
        pop = []
        for _ in range(self.pop_size):
            chrom = self.inst.customers[:]
            random.shuffle(chrom)
            pop.append(chrom)
        return pop

    # ── Main loop ────────────────────────────────────────────────────────────
    def run(self):
        pop  = self.init_pop()
        fits = [self.fitness(c) for c in pop]
        t0   = time.time()

        for _ in range(self.generations):
            if self.time_limit and (time.time() - t0) > self.time_limit:
                break

            bi = min(range(len(pop)), key=lambda i: fits[i])
            if fits[bi] < self.best_cost:
                self.best_cost   = fits[bi]
                self.best_routes = self.decode(pop[bi])
            self.history.append(self.best_cost)

            # Elitism
            order    = sorted(range(len(pop)), key=lambda i: fits[i])
            new_pop  = [pop[i][:] for i in order[:self.elite_size]]
            new_fits = [fits[i]   for i in order[:self.elite_size]]

            # Fill rest
            while len(new_pop) < self.pop_size:
                p1 = self.tournament(pop, fits)
                p2 = self.tournament(pop, fits)
                if random.random() < self.crossover_rate:
                    c1, c2 = self.ox1(p1, p2)
                else:
                    c1, c2 = p1[:], p2[:]
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                new_pop.append(c1); new_fits.append(self.fitness(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2); new_fits.append(self.fitness(c2))

            pop, fits = new_pop, new_fits

        # Final best
        bi = min(range(len(pop)), key=lambda i: fits[i])
        if fits[bi] < self.best_cost:
            self.best_cost   = fits[bi]
            self.best_routes = self.decode(pop[bi])

        self.elapsed_time = time.time() - t0
        return self.best_routes, self.best_cost


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SOLUTION VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_solution(inst, routes):
    errors, visited = [], []
    for idx, route in enumerate(routes):
        load = sum(inst.demands[v] for v in route)
        if load > inst.C:
            errors.append(f"Route {idx+1}: load {load} exceeds capacity {inst.C}")
        visited.extend(route)
    if sorted(visited) != sorted(inst.customers):
        missing   = set(inst.customers) - set(visited)
        duplicate = [v for v in visited if visited.count(v) > 1]
        if missing:   errors.append(f"Missing customers: {sorted(missing)}")
        if duplicate: errors.append(f"Duplicate customers: {sorted(set(duplicate))}")
    return len(errors) == 0, errors


# ─────────────────────────────────────────────────────────────────────────────
# 5.  TEXT OUTPUT  (identical format to cvrp_ortools.py)
# ─────────────────────────────────────────────────────────────────────────────

def write_solution_txt(inst, routes, cost, solve_time, params, out_path):
    valid, errors = validate_solution(inst, routes)

    with open(out_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"  CVRP Solution  \u2014  {inst.name}\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"  Algorithm     : Original Genetic Algorithm\n")
        f.write(f"  Status        : FEASIBLE\n")
        f.write(f"  Total distance: {cost}\n")
        f.write(f"  Vehicles used : {len(routes)} / {inst.k}\n")
        f.write(f"  Capacity      : {inst.C}\n")
        f.write(f"  Solve time    : {solve_time:.2f} s\n")
        f.write(f"  Population    : {params['pop_size']}\n")
        f.write(f"  Generations   : {params['generations']}\n")
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
            dist = inst.route_distance(route)
            path = " -> ".join([str(inst.depot)] + [str(v) for v in route] + [str(inst.depot)])
            f.write(f"  Route {i+1:>2}: {path}\n")
            f.write(f"            Load: {load:>4} / {inst.C}   Distance: {dist}\n\n")

        f.write("-" * 60 + "\n")
        f.write(f"  Total distance: {cost}\n")
        f.write("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PLOT OUTPUT  (identical style to cvrp_ortools.py)
# ─────────────────────────────────────────────────────────────────────────────

ROUTE_COLORS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#469990', '#dcbeff',
    '#9A6324', '#800000', '#aaffc3', '#808000', '#ffd8b1',
]

def plot_solution(inst, routes, cost, out_path):
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_facecolor('#f5f5f5')
    fig.patch.set_facecolor('#f5f5f5')

    legend_patches = []

    for r_idx, route in enumerate(routes):
        color = ROUTE_COLORS[r_idx % len(ROUTE_COLORS)]
        full  = [inst.depot] + route + [inst.depot]
        xs    = [inst.coords[n][0] for n in full]
        ys    = [inst.coords[n][1] for n in full]

        for seg in range(len(full) - 1):
            ax.annotate(
                "", xy=(xs[seg+1], ys[seg+1]), xytext=(xs[seg], ys[seg]),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.8, mutation_scale=14),
                zorder=2
            )

        load = sum(inst.demands[v] for v in route)
        dist = inst.route_distance(route)
        legend_patches.append(
            mpatches.Patch(color=color,
                           label=f"Route {r_idx+1}  load={load}/{inst.C}  dist={dist}")
        )

    for v in inst.customers:
        x, y = inst.coords[v]
        ax.scatter(x, y, c='#222222', s=55, zorder=4)
        ax.annotate(str(v), (x, y),
                    textcoords='offset points', xytext=(4, 4),
                    fontsize=7, color='#111111')

    dx, dy = inst.coords[inst.depot]
    ax.scatter(dx, dy, c='red', s=220, marker='*', zorder=5)
    ax.annotate(f"Depot\n({inst.depot})", (dx, dy),
                textcoords='offset points', xytext=(6, 6),
                fontsize=8, fontweight='bold', color='darkred')

    title = (f"{inst.name}  \u2014  dist={cost}  [FEASIBLE]\n"
             f"Capacity={inst.C}  |  Vehicles={len(routes)}/{inst.k}")
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")

    depot_patch = mpatches.Patch(color='red', label=f'Depot ({inst.depot})')
    ax.legend(handles=legend_patches + [depot_patch],
              loc='upper left', fontsize=7, framealpha=0.85, edgecolor='#cccccc')

    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CVRP Solver — Original Genetic Algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('vrp_files', nargs='+', help='.vrp file(s) to solve')
    parser.add_argument('--capacity',    type=int,   default=None)
    parser.add_argument('--n_vehicles',  type=int,   default=None)
    parser.add_argument('--time_limit',  type=int,   default=120,
                        help='Wall-clock seconds per instance (default: 120)')
    parser.add_argument('--pop_size',    type=int,   default=100)
    parser.add_argument('--generations', type=int,   default=300)
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--output_dir',  type=str,   default='../outputs/original_ga')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []

    for vrp_path in args.vrp_files:
        print(f"\n{'='*60}")
        print(f"  Solving: {vrp_path}")
        print(f"{'='*60}")

        try:
            vrp_data = parse_vrp(vrp_path)
        except Exception as e:
            print(f"  ERROR parsing file: {e}"); continue

        inst = CVRPInstance(vrp_data,
                            capacity_override=args.capacity,
                            k_override=args.n_vehicles)

        print(f"  Instance  : {inst.name}")
        print(f"  Customers : {inst.n}")
        print(f"  Depot     : {inst.depot}")
        print(f"  Capacity  : {inst.C}  |  Vehicles: {inst.k}")
        print(f"  Algorithm : Original GA  "
              f"(pop={args.pop_size}, gen={args.generations}, seed={args.seed})")
        print(f"  Time limit: {args.time_limit}s")

        ga = OriginalGA(
            inst,
            pop_size=args.pop_size,
            generations=args.generations,
            time_limit=args.time_limit,
            seed=args.seed,
        )
        routes, cost = ga.run()
        valid, errors = validate_solution(inst, routes)

        print(f"  Status        : FEASIBLE")
        print(f"  Total distance: {cost}")
        print(f"  Vehicles used : {len(routes)}")
        print(f"  Valid         : {'YES' if valid else 'NO — ' + str(errors)}")
        print(f"  Solve time    : {ga.elapsed_time:.2f}s")

        base     = os.path.splitext(os.path.basename(vrp_path))[0]
        txt_path = os.path.join(args.output_dir, f"{base}_solution.txt")
        png_path = os.path.join(args.output_dir, f"{base}_plot.png")

        params = {'pop_size': args.pop_size, 'generations': args.generations}
        write_solution_txt(inst, routes, cost, ga.elapsed_time, params, txt_path)

        if inst.ew_type == 'EUC_2D':
            plot_solution(inst, routes, cost, png_path)
            print(f"\n  Output saved:")
            print(f"    {txt_path}")
            print(f"    {png_path}")
        else:
            print(f"\n  Output saved:")
            print(f"    {txt_path}")
            print(f"    (no plot — EXPLICIT distance type, no coordinates)")

        all_results.append((inst.name, 'FEASIBLE', cost, ga.elapsed_time))

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
