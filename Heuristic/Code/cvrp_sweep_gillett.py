"""
CVRP — Gillett & Miller Sweep  (Python)
Reads every *.vrp in the current directory, solves it,
writes *_sweep_gillett.txt  and  *_sweep_gillett.png
"""

import os, math, time, glob
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ── Data structures ───────────────────────────────────────────────────

class Node:
    __slots__ = ("id", "x", "y", "demand", "angle")
    def __init__(self, id_=0, x=0.0, y=0.0, demand=0):
        self.id     = id_
        self.x      = x
        self.y      = y
        self.demand = demand
        self.angle  = 0.0


# ── Geometry ──────────────────────────────────────────────────────────

def euclidean(a: Node, b: Node) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)

def polar_angle(depot: Node, p: Node) -> float:
    a = math.atan2(p.y - depot.y, p.x - depot.x)
    return a + 2 * math.pi if a < 0 else a


# ── Cost ──────────────────────────────────────────────────────────────

def route_cost(route: list, nodes: list) -> float:
    return sum(euclidean(nodes[route[i-1]], nodes[route[i]])
               for i in range(1, len(route)))


# ── Nearest-neighbour TSP ─────────────────────────────────────────────

def nearest_neighbour(cluster: list, nodes: list) -> list:
    remaining = list(cluster)
    route = [1]
    cur = 1
    while remaining:
        best_d, best_v, best_i = 1e18, -1, -1
        for i, v in enumerate(remaining):
            d = euclidean(nodes[cur], nodes[v])
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
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                before = (euclidean(nodes[route[i-1]], nodes[route[i]]) +
                          euclidean(nodes[route[j]],   nodes[route[j+1]]))
                after  = (euclidean(nodes[route[i-1]], nodes[route[j]]) +
                          euclidean(nodes[route[i]],   nodes[route[j+1]]))
                if after < before - 1e-9:
                    route[i:j+1] = route[i:j+1][::-1]
                    improved = True
    return route


# ── Parse .vrp ────────────────────────────────────────────────────────

def parse_vrp(filename: str):
    nodes     = {}
    dimension = 0
    capacity  = 0
    section   = None

    with open(filename) as f:
        for raw in f:
            line = raw.strip()
            if "DIMENSION" in line:
                dimension = int(line.split(":")[1])
            elif "CAPACITY" in line:
                capacity = int(line.split(":")[1])
            elif line == "NODE_COORD_SECTION":
                section = "coord"
            elif line == "DEMAND_SECTION":
                section = "demand"
            elif line in ("DEPOT_SECTION", "EOF"):
                section = None
            elif section == "coord":
                parts = line.split()
                nid = int(parts[0])
                nodes[nid] = Node(nid, float(parts[1]), float(parts[2]))
            elif section == "demand":
                parts = line.split()
                nid, dem = int(parts[0]), int(parts[1])
                if nid in nodes:
                    nodes[nid].demand = dem

    return nodes, dimension, capacity


# ── Solve one instance ────────────────────────────────────────────────

def solve(nodes: dict, dimension: int, capacity: int):
    depot = nodes[1]

    # Polar angles
    customers = []
    for i in range(2, dimension + 1):
        nodes[i].angle = polar_angle(depot, nodes[i])
        customers.append(nodes[i])

    # Sort by angle
    customers.sort(key=lambda c: c.angle)

    # Cluster
    clusters, cur, load = [], [], 0
    for c in customers:
        if load + c.demand <= capacity:
            cur.append(c.id);  load += c.demand
        else:
            if cur: clusters.append(cur)
            cur, load = [c.id], c.demand
    if cur: clusters.append(cur)

    # Build routes
    routes, route_demands = [], []
    for cl in clusters:
        r = nearest_neighbour(cl, nodes)
        r = two_opt(r, nodes)
        routes.append(r)
        route_demands.append(sum(nodes[i].demand for i in cl))

    return routes, route_demands


# ── Plot ──────────────────────────────────────────────────────────────

def plot_routes(routes, nodes, capacity, route_demands, title, out_png):
    fig, ax = plt.subplots(figsize=(9, 7))
    colors  = cm.tab10.colors

    depot = nodes[1]

    for idx, route in enumerate(routes):
        col   = colors[idx % len(colors)]
        label = f"Route {idx+1}  (dem {route_demands[idx]}/{capacity})"

        xs = [nodes[n].x for n in route]
        ys = [nodes[n].y for n in route]

        ax.plot(xs, ys, "-o", color=col, linewidth=1.4,
                markersize=5, label=label)

        # Annotate customer ids (skip depot = 1)
        for nid in route[1:-1]:
            ax.annotate(str(nid),
                        (nodes[nid].x, nodes[nid].y),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=7, color=col)

    # Depot
    ax.plot(depot.x, depot.y, "ks", markersize=10, zorder=5, label="Depot")
    ax.annotate("Depot", (depot.x, depot.y),
                textcoords="offset points", xytext=(6, 4),
                fontsize=8, fontweight="bold")

    total = sum(route_cost(r, nodes) for r in routes)
    ax.set_title(f"{title}\nRoutes: {len(routes)}   Total Cost: {total:.2f}",
                 fontsize=11)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# ── Write result txt ──────────────────────────────────────────────────

def write_result(routes, route_demands, nodes, capacity, elapsed,
                 filename, out_txt, dimension):
    total = sum(route_cost(r, nodes) for r in routes)
    with open(out_txt, "w") as f:
        f.write("========== GILLETT & MILLER SWEEP CVRP ==========\n")
        f.write(f"Instance       : {filename}\n")
        f.write(f"Customers      : {dimension - 1}\n")
        f.write(f"Vehicle Cap    : {capacity}\n")
        f.write(f"Routes         : {len(routes)}\n")
        f.write(f"Total Cost     : {total:.4f}\n")
        f.write(f"Exec Time (s)  : {elapsed:.6f}\n")
        f.write("==================================================\n\n")
        for i, r in enumerate(routes):
            f.write(f"Route {i+1}  "
                    f"[demand={route_demands[i]}/{capacity}, "
                    f"dist={route_cost(r, nodes):.4f}]:\n  ")
            f.write(" ".join(map(str, r)))
            f.write("\n\n")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    vrp_files = sorted(glob.glob("*.vrp"))
    if not vrp_files:
        print("No .vrp files found in current directory.")
        return

    for vrp_path in vrp_files:
        base = vrp_path[:-4]          # strip .vrp
        print(f"Running {vrp_path} ...")

        nodes, dimension, capacity = parse_vrp(vrp_path)

        t0 = time.perf_counter()
        routes, route_demands = solve(nodes, dimension, capacity)
        elapsed = time.perf_counter() - t0

        total = sum(route_cost(r, nodes) for r in routes)

        out_txt = base + "_sweep_gillett.txt"
        out_png = base + "_sweep_gillett.png"

        write_result(routes, route_demands, nodes, capacity,
                     elapsed, vrp_path, out_txt, dimension)
        plot_routes(routes, nodes, capacity, route_demands,
                    title=base, out_png=out_png)

        print(f"  Routes    : {len(routes)}")
        print(f"  Cost      : {total:.2f}")
        print(f"  Time      : {elapsed:.4f}s")
        print(f"  TXT  →  {out_txt}")
        print(f"  PNG  →  {out_png}\n")

    print("All instances done.")


if __name__ == "__main__":
    main()