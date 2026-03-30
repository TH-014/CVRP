"""
CVRP — Enhanced Sweep  (Python)

Algorithms (1-to-1 translation of cvrp_enhanced.cpp):
  1. Multi-start sweep      (K=8 starting angles)
  2. 2-opt                  (intra-route)
  3. Or-opt 1/2/3           (intra-route, first-improvement)
  4. Inter-route relocate   (max 20 passes)
  5. Inter-route swap       (max 20 passes)

Output per instance:
  *_enhanced.txt   — cost report
  *_enhanced.png   — route plot
"""

import os, math, time, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ── Node ──────────────────────────────────────────────────────────────

class Node:
    __slots__ = ("id", "x", "y", "demand", "angle")
    def __init__(self, id_=0, x=0.0, y=0.0, demand=0):
        self.id     = id_
        self.x      = x
        self.y      = y
        self.demand = demand
        self.angle  = 0.0


# ── Geometry ──────────────────────────────────────────────────────────

def dist(a: Node, b: Node) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)

def polar_angle(depot: Node, p: Node) -> float:
    a = math.atan2(p.y - depot.y, p.x - depot.x)
    return a + 2 * math.pi if a < 0 else a


# ── Cost helpers ──────────────────────────────────────────────────────

def route_cost(route: list, nodes: list) -> float:
    return sum(dist(nodes[route[i-1]], nodes[route[i]])
               for i in range(1, len(route)))

def total_cost(routes: list, nodes: list) -> float:
    return sum(route_cost(r, nodes) for r in routes)

def route_demand(route: list, nodes: list) -> int:
    # route[0] and route[-1] are depot (id=1)
    return sum(nodes[route[i]].demand for i in range(1, len(route) - 1))


# ── Nearest-neighbour construction ────────────────────────────────────

def nn_route(cluster: list, nodes: list) -> list:
    remaining = list(cluster)
    route = [1]
    cur = 1
    while remaining:
        best_d, best_v, best_i = 1e18, -1, -1
        for i, v in enumerate(remaining):
            d = dist(nodes[cur], nodes[v])
            if d < best_d:
                best_d, best_v, best_i = d, v, i
        route.append(best_v)
        cur = best_v
        remaining.pop(best_i)
    route.append(1)
    return route


# ── 2-opt ─────────────────────────────────────────────────────────────

def two_opt(route: list, nodes: list) -> list:
    n = len(route)
    imp = True
    while imp:
        imp = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                gain = (dist(nodes[route[i-1]], nodes[route[i]])
                      + dist(nodes[route[j]],   nodes[route[j+1]])
                      - dist(nodes[route[i-1]], nodes[route[j]])
                      - dist(nodes[route[i]],   nodes[route[j+1]]))
                if gain > 1e-9:
                    route[i:j+1] = route[i:j+1][::-1]
                    imp = True
    return route


# ── Or-opt (chain length = length) ────────────────────────────────────
# First-improvement: breaks on the first gain found, then restarts.

def or_opt(route: list, nodes: list, length: int) -> list:
    imp = True
    while imp:
        imp = False
        n = len(route)
        for i in range(1, n - length):          # chain start
            if imp:
                break
            e = i + length - 1                  # chain end (inclusive)
            if e >= n - 1:
                break

            rem_gain = (dist(nodes[route[i-1]], nodes[route[i]])
                      + dist(nodes[route[e]],   nodes[route[e+1]])
                      - dist(nodes[route[i-1]], nodes[route[e+1]]))

            for j in range(n - 1):
                if imp:
                    break
                if i - 1 <= j <= e:
                    continue

                # forward insertion
                ins_fwd = (dist(nodes[route[j]],  nodes[route[i]])
                         + dist(nodes[route[e]],  nodes[route[j+1]])
                         - dist(nodes[route[j]],  nodes[route[j+1]]))
                if rem_gain - ins_fwd > 1e-9:
                    chain = route[i:e+1]
                    rest  = route[:i] + route[e+1:]
                    ins   = j if j < i else j - length
                    route = rest[:ins+1] + chain + rest[ins+1:]
                    imp   = True
                    break

                # reversed insertion
                ins_rev = (dist(nodes[route[j]],  nodes[route[e]])
                         + dist(nodes[route[i]],  nodes[route[j+1]])
                         - dist(nodes[route[j]],  nodes[route[j+1]]))
                if rem_gain - ins_rev > 1e-9:
                    chain = route[i:e+1][::-1]
                    rest  = route[:i] + route[e+1:]
                    ins   = j if j < i else j - length
                    route = rest[:ins+1] + chain + rest[ins+1:]
                    imp   = True
                    break
    return route


# ── Intra-route LS ────────────────────────────────────────────────────

def intra_ls(route: list, nodes: list) -> list:
    prev = float("inf")
    while True:
        prev = route_cost(route, nodes)
        route = two_opt(route, nodes)
        route = or_opt(route, nodes, 1)
        route = or_opt(route, nodes, 2)
        route = or_opt(route, nodes, 3)
        if route_cost(route, nodes) >= prev - 1e-9:
            break
    return route


# ── Inter-route relocate ──────────────────────────────────────────────

def inter_relocate(routes: list, nodes: list, cap: int) -> bool:
    any_imp = False
    R = len(routes)
    for a in range(R):
        pi = 1
        while pi + 1 < len(routes[a]):
            cu = routes[a][pi]
            rem_gain = (dist(nodes[routes[a][pi-1]], nodes[cu])
                      + dist(nodes[cu],              nodes[routes[a][pi+1]])
                      - dist(nodes[routes[a][pi-1]], nodes[routes[a][pi+1]]))
            best_gain, best_b, best_pos = 1e-9, -1, -1
            for b in range(R):
                if b == a:
                    continue
                if route_demand(routes[b], nodes) + nodes[cu].demand > cap:
                    continue
                for q in range(len(routes[b]) - 1):
                    ins = (dist(nodes[routes[b][q]],   nodes[cu])
                         + dist(nodes[cu],             nodes[routes[b][q+1]])
                         - dist(nodes[routes[b][q]],   nodes[routes[b][q+1]]))
                    gain = rem_gain - ins
                    if gain > best_gain:
                        best_gain, best_b, best_pos = gain, b, q + 1
            if best_b != -1:
                routes[a].pop(pi)
                routes[best_b].insert(best_pos, cu)
                any_imp = True
                # don't advance pi
            else:
                pi += 1

    # remove empty routes (only depot–depot)
    routes[:] = [r for r in routes if len(r) > 2]
    return any_imp


# ── Inter-route swap ──────────────────────────────────────────────────

def inter_swap(routes: list, nodes: list, cap: int) -> bool:
    any_imp = False
    R = len(routes)
    for a in range(R):
        for pi in range(1, len(routes[a]) - 1):
            for b in range(a + 1, R):
                for qi in range(1, len(routes[b]) - 1):
                    cu, cv = routes[a][pi], routes[b][qi]
                    if (route_demand(routes[a], nodes)
                            - nodes[cu].demand + nodes[cv].demand > cap):
                        continue
                    if (route_demand(routes[b], nodes)
                            - nodes[cv].demand + nodes[cu].demand > cap):
                        continue
                    delta = (
                        dist(nodes[routes[a][pi-1]], nodes[cv])
                      + dist(nodes[cv],              nodes[routes[a][pi+1]])
                      + dist(nodes[routes[b][qi-1]], nodes[cu])
                      + dist(nodes[cu],              nodes[routes[b][qi+1]])
                      - dist(nodes[routes[a][pi-1]], nodes[cu])
                      - dist(nodes[cu],              nodes[routes[a][pi+1]])
                      - dist(nodes[routes[b][qi-1]], nodes[cv])
                      - dist(nodes[cv],              nodes[routes[b][qi+1]])
                    )
                    if delta < -1e-9:
                        routes[a][pi], routes[b][qi] = cv, cu
                        any_imp = True
    return any_imp


# ── Parse .vrp ────────────────────────────────────────────────────────

def parse_vrp(filename: str):
    nodes     = {}
    dimension = 0
    capacity  = 0
    section   = None

    with open(filename) as f:
        for raw in f:
            line = raw.strip()
            if "DIMENSION" in line and ":" in line:
                dimension = int(line.split(":")[1])
            elif "CAPACITY" in line and ":" in line:
                capacity = int(line.split(":")[1])
            elif line == "NODE_COORD_SECTION":
                section = "coord"
            elif line == "DEMAND_SECTION":
                section = "demand"
            elif line in ("DEPOT_SECTION", "EOF"):
                section = None
            elif section == "coord":
                parts = line.split()
                if len(parts) >= 3:
                    nid = int(parts[0])
                    nodes[nid] = Node(nid, float(parts[1]), float(parts[2]))
            elif section == "demand":
                parts = line.split()
                if len(parts) >= 2:
                    nid, dem = int(parts[0]), int(parts[1])
                    if nid in nodes:
                        nodes[nid].demand = dem

    return nodes, dimension, capacity


# ── Solve ─────────────────────────────────────────────────────────────

def solve(nodes: dict, dimension: int, capacity: int):
    depot = nodes[1]

    customers = []
    for i in range(2, dimension + 1):
        nodes[i].angle = polar_angle(depot, nodes[i])
        customers.append(nodes[i])

    K = 8
    best_routes = None
    best_cost   = float("inf")

    for k in range(K):
        start_angle = k * (2.0 * math.pi / K)

        order = []
        for c in customers:
            a = c.angle - start_angle
            if a < 0:
                a += 2 * math.pi
            order.append((a, c.id))
        order.sort()

        # Greedy clustering
        clusters, cur, load = [], [], 0
        for _, cid in order:
            dem = nodes[cid].demand
            if load + dem <= capacity:
                cur.append(cid)
                load += dem
            else:
                if cur:
                    clusters.append(cur)
                cur, load = [cid], dem
        if cur:
            clusters.append(cur)

        # NN + intra-LS
        routes = []
        for cl in clusters:
            r = nn_route(cl, nodes)
            r = intra_ls(r, nodes)
            routes.append(r)

        c = total_cost(routes, nodes)
        if c < best_cost:
            best_cost   = c
            best_routes = [list(r) for r in routes]

    # Inter-route optimisation (max 20 passes)
    for _ in range(20):
        r1 = inter_relocate(best_routes, nodes, capacity)
        r2 = inter_swap    (best_routes, nodes, capacity)
        if not r1 and not r2:
            break
        # quick touch-up
        for i in range(len(best_routes)):
            best_routes[i] = two_opt(best_routes[i], nodes)
            best_routes[i] = or_opt(best_routes[i], nodes, 1)

    return best_routes


# ── Plot ──────────────────────────────────────────────────────────────

def plot_routes(routes, nodes, capacity, title, out_png):
    fig, ax = plt.subplots(figsize=(9, 7))
    colors  = cm.tab10.colors
    depot   = nodes[1]

    for idx, route in enumerate(routes):
        col   = colors[idx % len(colors)]
        dem   = route_demand(route, nodes)
        label = f"Route {idx+1}  (dem {dem}/{capacity})"

        xs = [nodes[n].x for n in route]
        ys = [nodes[n].y for n in route]
        ax.plot(xs, ys, "-o", color=col, linewidth=1.4,
                markersize=5, label=label)

        for nid in route[1:-1]:
            ax.annotate(str(nid),
                        (nodes[nid].x, nodes[nid].y),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=7, color=col)

    ax.plot(depot.x, depot.y, "ks", markersize=10, zorder=5, label="Depot")
    ax.annotate("Depot", (depot.x, depot.y),
                textcoords="offset points", xytext=(6, 4),
                fontsize=8, fontweight="bold")

    tc = total_cost(routes, nodes)
    ax.set_title(f"{title}\nRoutes: {len(routes)}   Total Cost: {tc:.2f}",
                 fontsize=11)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# ── Write result txt ──────────────────────────────────────────────────

def write_result(routes, nodes, capacity, elapsed, filename, out_txt, dimension):
    K = 8
    tc = total_cost(routes, nodes)
    with open(out_txt, "w") as f:
        f.write("========== CVRP ENHANCED SWEEP ==========\n")
        f.write(f"Instance      : {filename}\n")
        f.write(f"Customers     : {dimension - 1}\n")
        f.write(f"Vehicle Cap   : {capacity}\n")
        f.write(f"Multi-starts  : {K}\n")
        f.write(f"Routes        : {len(routes)}\n")
        f.write(f"Total Cost    : {tc:.4f}\n")
        f.write(f"Exec Time (s) : {elapsed:.6f}\n")
        f.write("=========================================\n\n")
        for i, r in enumerate(routes):
            dem = route_demand(r, nodes)
            rc  = route_cost(r, nodes)
            f.write(f"Route {i+1}  [demand={dem}/{capacity}, dist={rc:.2f}]:\n  ")
            f.write(" ".join(map(str, r)))
            f.write("\n\n")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    vrp_files = sorted(glob.glob("*.vrp"))
    if not vrp_files:
        print("No .vrp files found in current directory.")
        return

    for vrp_path in vrp_files:
        base = vrp_path[:-4]
        print(f"Running {vrp_path} ...")

        nodes, dimension, capacity = parse_vrp(vrp_path)

        t0      = time.perf_counter()
        routes  = solve(nodes, dimension, capacity)
        elapsed = time.perf_counter() - t0

        tc      = total_cost(routes, nodes)
        out_txt = base + "_enhanced.txt"
        out_png = base + "_enhanced.png"

        write_result(routes, nodes, capacity, elapsed, vrp_path, out_txt, dimension)
        plot_routes(routes, nodes, capacity, title=base, out_png=out_png)

        print(f"  Routes    : {len(routes)}")
        print(f"  Cost      : {tc:.2f}")
        print(f"  Time      : {elapsed:.4f}s")
        print(f"  TXT  →  {out_txt}")
        print(f"  PNG  →  {out_png}\n")

    print("All instances done.")


if __name__ == "__main__":
    main()