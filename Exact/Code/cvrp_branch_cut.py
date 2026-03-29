"""
Branch and Cut Algorithm for the Capacitated Vehicle Routing Problem (CVRP)
===========================================================================
Based on: Augerat et al. (1995) "Computational results with a Branch and Cut
           Code for the Capacitated Vehicle Routing Problem"

Usage:
    python cvrp_branch_cut.py <vrp_file> [--capacity C] [--n_vehicles K]

The algorithm implements:
  - Two-index vehicle-flow LP relaxation
  - Capacity constraints (generalized subtour elimination)  [Sec 3]
  - Greedy-shrinking separation heuristic                   [Sec 3]
  - Comb-inequality separation (simplified)                 [Sec 4.2]
  - Nearest-neighbour + 2-opt initial upper bound
  - Branch-and-bound with set-branching (strategy S4/S5)    [Sec 5]
  - LIFO node selection

The LP is solved with a custom revised-simplex / scipy linprog wrapper.
"""

import sys
import os
import math
import copy
import argparse
import itertools
import time
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import linprog

# ─────────────────────────────────────────────────────────────────────────────
# 1.  VRP FILE PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_vrp(path):
    """Return dict with keys: name, dimension, capacity, coords, demands, depot"""
    data = dict(coords={}, demands={}, depot=1)
    section = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line == "EOF":
                continue
            if ':' in line and not line[0].isdigit():
                key, _, val = line.partition(':')
                key = key.strip().upper()
                val = val.strip()
                if key == 'NAME':
                    data['name'] = val # type: ignore
                elif key == 'DIMENSION':
                    data['dimension'] = int(val)
                elif key == 'CAPACITY':
                    data['capacity'] = int(val)
                continue
            if line.startswith('NODE_COORD'):
                section = 'coord'; continue
            if line.startswith('DEMAND'):
                section = 'demand'; continue
            if line.startswith('DEPOT'):
                section = 'depot'; continue
            parts = line.split()
            if section == 'coord':
                nid, x, y = int(parts[0]), float(parts[1]), float(parts[2])
                data['coords'][nid] = (x, y) # type: ignore
            elif section == 'demand':
                nid, d = int(parts[0]), int(parts[1])
                data['demands'][nid] = d # type: ignore
            elif section == 'depot':
                if parts[0] != '-1':
                    data['depot'] = int(parts[0])
    return data

# ─────────────────────────────────────────────────────────────────────────────
# 2.  DISTANCE & INDEXING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def euclidean(c1, c2):
    return math.floor(math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2) + 0.5)

class CVRPInstance:
    def __init__(self, vrp_data, capacity_override=None, k_override=None):
        d = vrp_data
        self.name = d.get('name', 'unknown')
        n_total = d['dimension']
        self.depot = d['depot']
        self.coords = d['coords']
        self.demands = d['demands']
        self.C = capacity_override if capacity_override else d['capacity']

        # customers = all nodes except depot
        self.customers = sorted(n for n in self.coords if n != self.depot)
        self.n = len(self.customers)               # number of customers
        self.nodes = [self.depot] + self.customers  # node list, depot first

        # vehicle count: override > ceil(total_demand / capacity)
        total_d = sum(self.demands[c] for c in self.customers)
        min_k = math.ceil(total_d / self.C)
        if k_override:
            self.k = max(k_override, min_k)
        else:
            self.k = min_k

        # Pre-compute distance matrix (integer, as in paper)
        self.dist = {}
        for i in self.nodes:
            for j in self.nodes:
                self.dist[(i,j)] = euclidean(self.coords[i], self.coords[j])

        # edge list (undirected): pairs (i,j) with i < j over ALL nodes
        self.edges = []
        self.edge_idx = {}
        idx = 0
        for a in range(len(self.nodes)):
            for b in range(a+1, len(self.nodes)):
                i, j = self.nodes[a], self.nodes[b]
                self.edges.append((i, j))
                self.edge_idx[(i,j)] = idx
                self.edge_idx[(j,i)] = idx
                idx += 1
        self.m = len(self.edges)   # number of edge variables

    def edge(self, i, j):
        """Return canonical edge index for nodes i,j."""
        return self.edge_idx[(i,j)]

    def cost_vec(self):
        return np.array([self.dist[(i,j)] for i,j in self.edges], dtype=float)

    def demand(self, S):
        return sum(self.demands[v] for v in S if v != self.depot)

    def r(self, S):
        """Lower bound on number of vehicles needed for customer set S."""
        return math.ceil(self.demand(S) / self.C)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  LP RELAXATION (scipy linprog)
# ─────────────────────────────────────────────────────────────────────────────

def build_lp(inst, extra_cuts):
    """
    Build LP: min c^T x
    Subject to:
      - depot degree = 2k
      - each customer degree = 2
      - capacity constraints (extra_cuts)
      - 0 <= x_e <= 1  (customer-customer edges)
      - 0 <= x_e <= 2  (depot edges)

    Returns: (c, A_eq, b_eq, A_ub, b_ub, bounds)
    """
    inst_ = inst
    m = inst_.m
    c_obj = inst_.cost_vec()

    eq_rows, eq_rhs = [], []

    # Depot degree constraint: sum_{e in delta({depot})} x_e = 2k
    depot_row = np.zeros(m)
    for e_idx, (i,j) in enumerate(inst_.edges):
        if i == inst_.depot or j == inst_.depot:
            depot_row[e_idx] = 1.0
    eq_rows.append(depot_row)
    eq_rhs.append(2 * inst_.k)

    # Customer degree constraints: sum_{e in delta({v})} x_e = 2  for each v
    for v in inst_.customers:
        row = np.zeros(m)
        for e_idx, (i,j) in enumerate(inst_.edges):
            if i == v or j == v:
                row[e_idx] = 1.0
        eq_rows.append(row)
        eq_rhs.append(2.0)

    A_eq = np.array(eq_rows)
    b_eq = np.array(eq_rhs)

    # Inequality constraints from cuts:  x(delta(S)) >= 2*r(S)
    ub_rows, ub_rhs = [], []
    for S, rhs in extra_cuts:
        # -x(delta(S)) <= -rhs  (i.e. x(delta(S)) >= rhs)
        row = np.zeros(m)
        S_set = set(S)
        for e_idx, (i,j) in enumerate(inst_.edges):
            in_i = i in S_set
            in_j = j in S_set
            if in_i != in_j:   # edge crosses the cut
                row[e_idx] = -1.0
        ub_rows.append(row)
        ub_rhs.append(-float(rhs))

    if ub_rows:
        A_ub = np.array(ub_rows)
        b_ub = np.array(ub_rhs)
    else:
        A_ub = None
        b_ub = None

    # Variable bounds
    bounds = []
    for i,j in inst_.edges:
        if i == inst_.depot or j == inst_.depot:
            bounds.append((0.0, 2.0))
        else:
            bounds.append((0.0, 1.0))

    return c_obj, A_eq, b_eq, A_ub, b_ub, bounds

def solve_lp(inst, extra_cuts, fixed_zero=None, fixed_one=None):
    """
    Solve LP relaxation.
    fixed_zero / fixed_one: lists of edge indices forced to 0 or 1.
    Returns (obj_val, x_vec) or None if infeasible.
    """
    c_obj, A_eq, b_eq, A_ub, b_ub, bounds = build_lp(inst, extra_cuts)

    # Apply branching fixings
    bounds = list(bounds)
    if fixed_zero:
        for ei in fixed_zero:
            lo, hi = bounds[ei]
            bounds[ei] = (lo, min(hi, 0.0))
    if fixed_one:
        for ei in fixed_one:
            lo, hi = bounds[ei]
            bounds[ei] = (max(lo, 1.0), hi)

    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub,
                      A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds,
                      method='highs',
                      options={'disp': False, 'time_limit': 30.0})
        if res.status in (0,):
            return res.fun, res.x
        return None
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# 4.  SEPARATION HEURISTICS
# ─────────────────────────────────────────────────────────────────────────────

def coboundary_value(x_vec, inst, S):
    """Compute x(delta(S)) for a set S of nodes."""
    S_set = set(S)
    total = 0.0
    for e_idx, (i,j) in enumerate(inst.edges):
        if (i in S_set) != (j in S_set):
            total += x_vec[e_idx]
    return total

def find_violated_capacity_cuts(x_vec, inst, existing_cuts):
    """
    Greedy-shrinking separation heuristic (Section 3 of the paper).
    Returns list of (S_tuple, rhs) pairs that are violated.
    """
    existing_sets = set(frozenset(S) for S,_ in existing_cuts)
    violated = []

    customers = inst.customers

    def check_and_add(S_list):
        S = [v for v in S_list if v != inst.depot]
        if not S:
            return
        fs = frozenset(S)
        if fs in existing_sets:
            return
        r_S = inst.r(S)
        if r_S == 0:
            return
        cob = coboundary_value(x_vec, inst, S)
        rhs = 2 * r_S
        if cob < rhs - 1e-6:
            existing_sets.add(fs)
            violated.append((tuple(sorted(S)), rhs))

    # Strategy 1: each single customer
    for v in customers:
        check_and_add([v])

    # Strategy 6: greedy shrinking starting from random subsets not including depot
    # Try each customer as seed, greedily add node that maximises cut decrease
    for seed in customers:
        S = [seed]
        S_set = set(S)
        while True:
            best_gain = -1e9
            best_v = None
            for v in customers:
                if v in S_set:
                    continue
                # compute x((S : v))
                gain = 0.0
                for e_idx, (i,j) in enumerate(inst.edges):
                    if (i == v and j in S_set) or (j == v and i in S_set):
                        gain += x_vec[e_idx]
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
            if best_v is None:
                break
            new_S = S + [best_v]
            # stop if adding would exceed a capacity multiple badly
            if inst.demand(new_S) > inst.k * inst.C:
                break
            S = new_S
            S_set.add(best_v)
            check_and_add(S)

    # Also check complement sets
    for S_tuple, rhs in list(violated):
        S_set = set(S_tuple)
        complement = [v for v in customers if v not in S_set]
        check_and_add(complement)

    # Fractional capacity: check sets where demand is just above p*C
    # (sets whose demand slightly exceeds a multiple of C are good candidates)
    for p in range(1, inst.k + 1):
        target = p * inst.C
        # build a greedy set approaching this target
        cands = sorted(customers, key=lambda v: inst.demands[v], reverse=True)
        S = []
        total_d = 0
        for v in cands:
            if total_d + inst.demands[v] <= target + 0.33 * inst.C:
                S.append(v)
                total_d += inst.demands[v]
                if total_d > target:
                    check_and_add(S)

    return violated

def find_comb_cuts(x_vec, inst, existing_cuts):
    """
    Simplified comb separation: look for 3-toothed combs.
    A comb (H; T1,T2,T3) violates (2.12): x(δ(H)) + Σ x(δ(Ti)) ≥ 3s+1 = 10
    We identify handle H as a dense cluster, then find teeth.
    """
    existing_sets = set(frozenset(S) for S,_ in existing_cuts)
    violated = []
    customers = inst.customers

    # Build adjacency by edge value
    adj = defaultdict(float)
    for e_idx, (i,j) in enumerate(inst.edges):
        if i != inst.depot and j != inst.depot:
            v = x_vec[e_idx]
            if v > 0.1:
                adj[(i,j)] = v
                adj[(j,i)] = v

    # Find handle candidates: biconnected-style clusters via high-value edges
    def cluster_around(seed, threshold=0.3):
        """Collect nodes highly connected to seed."""
        H = {seed}
        changed = True
        while changed:
            changed = False
            for v in customers:
                if v in H:
                    continue
                conn = sum(adj.get((v,u), 0) for u in H)
                if conn >= threshold * len(H):
                    H.add(v)
                    changed = True
        return H

    seen_handles = set()
    for seed in customers[:min(len(customers), 10)]:
        H = cluster_around(seed)
        if len(H) < 2:
            continue
        fH = frozenset(H)
        if fH in seen_handles:
            continue
        seen_handles.add(fH)

        # Find teeth: sets Ti with Ti∩H≠∅ and Ti\H≠∅
        teeth_candidates = []
        for v in H:
            for u in customers:
                if u not in H:
                    T = frozenset([v, u])
                    teeth_candidates.append(T)

        # Pick 3 best teeth by their cut value
        def teeth_val(T):
            T_list = list(T)
            return coboundary_value(x_vec, inst, T_list)

        teeth_candidates = sorted(set(teeth_candidates), key=teeth_val)[:10]

        # Try all combinations of 3 teeth
        for combo in itertools.combinations(teeth_candidates, 3):
            # check pairwise disjoint
            union = set()
            ok = True
            for T in combo:
                if T & union:
                    ok = False
                    break
                union |= T
            if not ok:
                continue

            H_list = list(H)
            cob_H = coboundary_value(x_vec, inst, H_list)
            cob_T = sum(coboundary_value(x_vec, inst, list(T)) for T in combo)
            lhs = cob_H + cob_T
            rhs = 10.0  # 3*s+1 with s=3

            if lhs < rhs - 1e-6:
                # Add capacity cut for H and each tooth
                for S in [H_list] + [list(T) for T in combo]:
                    fs = frozenset(S)
                    if fs not in existing_sets:
                        r_S = inst.r(S)
                        if r_S > 0:
                            cob = coboundary_value(x_vec, inst, S)
                            if cob < 2*r_S - 1e-6:
                                existing_sets.add(fs)
                                violated.append((tuple(sorted(S)), 2*r_S))

    return violated

# ─────────────────────────────────────────────────────────────────────────────
# 5.  CUTTING PLANE LOOP
# ─────────────────────────────────────────────────────────────────────────────

def cutting_plane(inst, fixed_zero=None, fixed_one=None, max_iter=60):
    """
    Run the cutting plane algorithm.
    Returns (obj, x_vec, cuts) or None if infeasible.
    """
    cuts = []
    result = solve_lp(inst, cuts, fixed_zero, fixed_one)
    if result is None:
        return None

    obj, x_vec = result

    for iteration in range(max_iter):
        new_cuts = find_violated_capacity_cuts(x_vec, inst, cuts)
        new_cuts += find_comb_cuts(x_vec, inst, cuts)

        if not new_cuts:
            break

        cuts.extend(new_cuts)

        result = solve_lp(inst, cuts, fixed_zero, fixed_one)
        if result is None:
            return None
        new_obj, x_vec = result
        if abs(new_obj - obj) < 1e-8 and iteration > 5:
            break
        obj = new_obj

    return obj, x_vec, cuts

# ─────────────────────────────────────────────────────────────────────────────
# 6.  HEURISTIC SOLUTION (nearest-neighbour + 2-opt)
# ─────────────────────────────────────────────────────────────────────────────

def nearest_neighbour(inst):
    """Build initial routes using nearest-neighbour heuristic."""
    unvisited = set(inst.customers)
    routes = []
    for _ in range(inst.k):
        if not unvisited:
            break
        route = []
        cap_left = inst.C
        current = inst.depot
        while True:
            best = None
            best_d = float('inf')
            for v in unvisited:
                if inst.demands[v] <= cap_left:
                    d = inst.dist[(current, v)]
                    if d < best_d:
                        best_d = d
                        best = v
            if best is None:
                break
            route.append(best)
            cap_left -= inst.demands[best]
            unvisited.remove(best)
            current = best
        if route:
            routes.append(route)

    # assign leftovers to first route that can fit
    for v in unvisited:
        placed = False
        for route in routes:
            load = sum(inst.demands[u] for u in route)
            if load + inst.demands[v] <= inst.C:
                route.append(v)
                placed = True
                break
        if not placed:
            routes.append([v])

    return routes

def route_cost(inst, route):
    if not route:
        return 0
    cost = inst.dist[(inst.depot, route[0])]
    for i in range(len(route)-1):
        cost += inst.dist[(route[i], route[i+1])]
    cost += inst.dist[(route[-1], inst.depot)]
    return cost

def two_opt(inst, route):
    """2-opt improvement for a single route."""
    improved = True
    while improved:
        improved = False
        for i in range(len(route)-1):
            for j in range(i+2, len(route)):
                new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                if route_cost(inst, new_route) < route_cost(inst, route) - 1e-6:
                    route = new_route
                    improved = True
    return route

def total_cost(inst, routes):
    return sum(route_cost(inst, r) for r in routes)

def heuristic_solution(inst):
    routes = nearest_neighbour(inst)
    routes = [two_opt(inst, r) for r in routes]
    return routes, total_cost(inst, routes)

# ─────────────────────────────────────────────────────────────────────────────
# 7.  EXTRACT INTEGER SOLUTION FROM LP
# ─────────────────────────────────────────────────────────────────────────────

def extract_routes_from_x(x_vec, inst):
    """
    Try to extract routes from an (approximately) integer LP solution.
    Build adjacency and follow cycles.
    """
    adj = defaultdict(list)
    for e_idx, (i,j) in enumerate(inst.edges):
        v = x_vec[e_idx]
        if v > 0.5:
            times = 2 if v > 1.5 else 1
            for _ in range(times):
                adj[i].append(j)
                adj[j].append(i)

    routes = []
    visited_depot_edges = 0
    used = defaultdict(int)

    # Follow each route starting from depot
    max_routes = inst.k * 2
    for _ in range(max_routes):
        if not adj[inst.depot]:
            break
        start = adj[inst.depot][0]
        route = []
        prev = inst.depot
        current = start

        # Remove the depot -> start edge
        adj[inst.depot].remove(start)
        adj[start].remove(inst.depot)

        iters = 0
        while current != inst.depot and iters < inst.n + 5:
            iters += 1
            route.append(current)
            # find next node (not going back to prev unless forced)
            nexts = adj[current][:]
            next_node = None
            for nxt in nexts:
                if nxt != prev or len(nexts) == 1:
                    next_node = nxt
                    break
            if next_node is None:
                break
            adj[current].remove(next_node)
            adj[next_node].remove(current)
            prev = current
            current = next_node

        if route:
            routes.append(route)

    return routes

def is_integer_solution(x_vec, tol=1e-4):
    """Check if LP solution is integer."""
    for v in x_vec:
        if tol < v < 1.0 - tol and not (1.0 + tol > v > 1.0 - tol):
            return False
    return True

def solution_is_feasible(routes, inst):
    """Check that routes cover all customers exactly once and respect capacity."""
    all_visited = []
    for r in routes:
        if sum(inst.demands[v] for v in r) > inst.C + 1e-6:
            return False
        all_visited.extend(r)
    return sorted(all_visited) == sorted(inst.customers)

# ─────────────────────────────────────────────────────────────────────────────
# 8.  BRANCHING
# ─────────────────────────────────────────────────────────────────────────────

def select_branch_edge(x_vec, inst):
    """
    Select fractional edge for branching.
    Prefer edges close to 0.5 with high cost (strategy S4/S5 inspired).
    """
    best_score = -1
    best_ei = None
    for ei, (i,j) in enumerate(inst.edges):
        v = x_vec[ei]
        frac = abs(v - round(v))
        if frac > 1e-4:
            # score: closeness to 0.5 * edge_cost
            score = (1.0 - abs(v - 0.5) * 2) * inst.dist[(i,j)]
            if score > best_score:
                best_score = score
                best_ei = ei
    return best_ei

# ─────────────────────────────────────────────────────────────────────────────
# 9.  BRANCH AND CUT
# ─────────────────────────────────────────────────────────────────────────────

class BnBNode:
    def __init__(self, fixed_zero, fixed_one, lb):
        self.fixed_zero = list(fixed_zero)
        self.fixed_one = list(fixed_one)
        self.lb = lb

def branch_and_cut(inst, time_limit=300, max_nodes=200):
    """
    Full Branch-and-Cut.
    Args:
        inst       : CVRPInstance to solve
        time_limit : wall-clock seconds before stopping (default 300)
        max_nodes  : maximum B&B nodes to explore (default 200)
    Returns (best_cost, best_routes).
    """
    t_start = time.time()

    # Initial upper bound from heuristic
    best_routes, best_cost = heuristic_solution(inst)
    print(f"  Heuristic upper bound: {best_cost:.1f}")

    # Root LP
    print("  Solving root LP relaxation...")
    root = cutting_plane(inst, max_iter=60)
    if root is None:
        print("  Root LP infeasible – returning heuristic solution")
        return best_cost, best_routes

    root_lb, root_x, root_cuts = root
    print(f"  Root LP lower bound:   {root_lb:.2f}")
    print(f"  Gap: {100*(best_cost - root_lb)/best_cost:.2f}%")
    print(f"  B&B limits — time_limit: {time_limit}s  |  max_nodes: {max_nodes}")

    # Check if root is already integer
    if is_integer_solution(root_x):
        routes = extract_routes_from_x(root_x, inst)
        if solution_is_feasible(routes, inst):
            c = total_cost(inst, routes)
            if c < best_cost:
                best_cost = c
                best_routes = routes
        return best_cost, best_routes

    # Branch-and-bound stack (LIFO)
    stack = [BnBNode([], [], root_lb)]
    nodes_explored = 0

    while stack and nodes_explored < max_nodes:
        if time.time() - t_start > time_limit:
            print(f"  Time limit reached after {nodes_explored} nodes")
            break

        node = stack.pop()  # LIFO
        nodes_explored += 1

        if node.lb >= best_cost - 1e-6:
            continue  # prune

        result = cutting_plane(inst, node.fixed_zero, node.fixed_one, max_iter=40)
        if result is None:
            continue

        lb, x_vec, cuts = result

        if lb >= best_cost - 1e-6:
            continue  # prune

        if is_integer_solution(x_vec):
            routes = extract_routes_from_x(x_vec, inst)
            if routes and solution_is_feasible(routes, inst):
                c = total_cost(inst, routes)
                if c < best_cost:
                    best_cost = c
                    best_routes = routes
                    print(f"  Node {nodes_explored}: new best = {best_cost:.1f}")
            continue

        # Branch on a fractional edge
        ei = select_branch_edge(x_vec, inst)
        if ei is None:
            continue

        child0 = BnBNode(node.fixed_zero + [ei], node.fixed_one, lb)
        child1 = BnBNode(node.fixed_zero, node.fixed_one + [ei], lb)

        # Push higher-lb child first (so lower-lb is explored first, LIFO)
        stack.append(child0)
        stack.append(child1)

    print(f"  B&B explored {nodes_explored} nodes. Best cost: {best_cost:.1f}")
    return best_cost, best_routes

# ─────────────────────────────────────────────────────────────────────────────
# 10. OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

COLORS = [
    '#e6194b','#3cb44b','#4363d8','#f58231','#911eb4',
    '#42d4f4','#f032e6','#bfef45','#fabed4','#469990',
    '#dcbeff','#9A6324','#fffac8','#800000','#aaffc3',
]

def write_solution_txt(inst, routes, cost, out_path):
    with open(out_path, 'w') as f:
        f.write(f"Instance: {inst.name}\n")
        f.write(f"Capacity: {inst.C}  |  Vehicles used: {len(routes)}\n")
        f.write(f"Total distance: {cost:.0f}\n\n")
        for i, route in enumerate(routes):
            load = sum(inst.demands[v] for v in route)
            rc = route_cost(inst, route)
            path_str = " -> ".join(
                [str(inst.depot)] + [str(v) for v in route] + [str(inst.depot)]
            )
            f.write(f"Route {i+1}: {path_str}\n")
            f.write(f"  Load: {load}/{inst.C}   Distance: {rc:.0f}\n\n")

def plot_solution(inst, routes, cost, out_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#f8f9fa')

    # Draw routes
    legend_patches = []
    for r_idx, route in enumerate(routes):
        color = COLORS[r_idx % len(COLORS)]
        full = [inst.depot] + route + [inst.depot]
        xs = [inst.coords[n][0] for n in full]
        ys = [inst.coords[n][1] for n in full]
        ax.plot(xs, ys, '-', color=color, linewidth=1.8, alpha=0.85, zorder=2)
        load = sum(inst.demands[v] for v in route)
        rc = route_cost(inst, route)
        legend_patches.append(
            mpatches.Patch(color=color,
                           label=f"Route {r_idx+1}  load={load}  dist={rc:.0f}")
        )

    # Draw customers
    for v in inst.customers:
        x, y = inst.coords[v]
        ax.scatter(x, y, c='#333333', s=60, zorder=4)
        ax.annotate(str(v), (x, y), textcoords='offset points',
                    xytext=(4, 4), fontsize=7, color='#222222')

    # Draw depot
    dx, dy = inst.coords[inst.depot]
    ax.scatter(dx, dy, c='red', s=200, marker='*', zorder=5, label='Depot')
    ax.annotate(f"Depot ({inst.depot})", (dx, dy),
                textcoords='offset points', xytext=(5, 5),
                fontsize=8, fontweight='bold', color='red')

    ax.set_title(f"{inst.name}  –  total dist={cost:.0f}  "
                 f"(C={inst.C}, k={inst.k})", fontsize=11, fontweight='bold')
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.legend(handles=legend_patches + [mpatches.Patch(color='red', label='Depot')],
              loc='upper left', fontsize=7, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 11. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Branch-and-Cut CVRP Solver (Augerat et al. 1995)")
    parser.add_argument('vrp_files', nargs='+', help='.vrp file(s) to solve')
    parser.add_argument('--capacity', type=int, default=None,
                        help='Override vehicle capacity')
    parser.add_argument('--n_vehicles', type=int, default=None,
                        help='Override number of vehicles')
    parser.add_argument('--time_limit', type=int, default=300,
                        help='Wall-clock seconds allowed per instance (default: 300)')
    parser.add_argument('--max_nodes', type=int, default=200,
                        help='Maximum branch-and-bound nodes per instance (default: 200)')
    args = parser.parse_args()

    os.makedirs('../outputs/b&c', exist_ok=True)

    for vrp_path in args.vrp_files:
        print(f"\n{'='*60}")
        print(f"Solving: {vrp_path}")
        print(f"{'='*60}")

        vrp_data = parse_vrp(vrp_path)
        inst = CVRPInstance(vrp_data,
                            capacity_override=args.capacity,
                            k_override=args.n_vehicles)

        print(f"Instance: {inst.name}")
        print(f"Customers: {inst.n}  |  Depot: {inst.depot}")
        print(f"Capacity: {inst.C}  |  Vehicles: {inst.k}")
        total_d = sum(inst.demands[c] for c in inst.customers)
        print(f"Total demand: {total_d}  |  Tightness: {total_d/(inst.k*inst.C):.2f}")

        t0 = time.time()
        best_cost, best_routes = branch_and_cut(inst,
                                                time_limit=args.time_limit,
                                                max_nodes=args.max_nodes)
        elapsed = time.time() - t0

        print(f"\nFinal solution:")
        print(f"  Cost        : {best_cost:.0f}")
        print(f"  Routes used : {len(best_routes)}")
        print(f"  Time        : {elapsed:.1f}s")

        base = os.path.splitext(os.path.basename(vrp_path))[0]
        txt_path = f"../outputs/b&c/{base}_solution.txt"
        png_path = f"../outputs/b&c/{base}_plot.png"

        write_solution_txt(inst, best_routes, best_cost, txt_path)
        plot_solution(inst, best_routes, best_cost, png_path)

        print(f"  Saved: {txt_path}")
        print(f"  Saved: {png_path}")

if __name__ == '__main__':
    main()
