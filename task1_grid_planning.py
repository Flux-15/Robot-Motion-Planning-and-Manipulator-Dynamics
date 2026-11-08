"""
Dijkstra vs A* on a 30x30 occupancy grid (8-connected, no corner cutting),
with line-of-sight path smoothing.

Run:  python grid_planner.py
Outputs: results_task1.json, dijkstra.png, astar.png, astar_smoothed.png, comparison.png + table in console
"""
import heapq
import json
import math
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ----------------------------------------------------------------- problem
N = 30
START, GOAL = (2, 2), (27, 27)
OBSTACLES = [  # (x_min, x_max, y_min, y_max), inclusive
    (8, 10, 0, 18),   # R1
    (15, 17, 10, 29), # R2
    (22, 24, 0, 20),  # R3
]
SQRT2 = math.sqrt(2.0)

grid = np.zeros((N, N), dtype=bool)  # grid[y, x] = True -> obstacle
for x0, x1, y0, y1 in OBSTACLES:
    grid[y0:y1 + 1, x0:x1 + 1] = True


def free(x, y):
    return 0 <= x < N and 0 <= y < N and not grid[y, x]


MOVES = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
         (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2)]


def neighbours(node):
    x, y = node
    for dx, dy, c in MOVES:
        nx, ny = x + dx, y + dy
        if not free(nx, ny):
            continue
        # no corner cutting: both orthogonal neighbours must be free
        if dx != 0 and dy != 0 and not (free(x + dx, y) and free(x, y + dy)):
            continue
        yield (nx, ny), c


def euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ----------------------------------------------------------------- planners
def search(start, goal, use_heuristic):
    """Dijkstra (use_heuristic=False) or A* (True).
    Returns path, cost, expanded-node list (in expansion order)."""
    h = (lambda n: euclid(n, goal)) if use_heuristic else (lambda n: 0.0)
    g = {start: 0.0}
    parent = {start: None}
    closed = set()
    expanded = []
    counter = 0
    pq = [(h(start), 0, start)]
    while pq:
        _, _, node = heapq.heappop(pq)
        if node in closed:
            continue
        closed.add(node)
        expanded.append(node)
        if node == goal:
            break
        for nb, c in neighbours(node):
            ng = g[node] + c
            if nb not in g or ng < g[nb] - 1e-12:
                g[nb] = ng
                parent[nb] = node
                counter += 1
                heapq.heappush(pq, (ng + h(nb), counter, nb))
    if goal not in parent:
        return None, math.inf, expanded
    path, n = [], goal
    while n is not None:
        path.append(n)
        n = parent[n]
    return path[::-1], g[goal], expanded


def dijkstra(start, goal):
    return search(start, goal, False)


def astar(start, goal):
    return search(start, goal, True)


# ----------------------------------------------------------------- metrics
def path_length(path):
    return sum(euclid(path[i], path[i + 1]) for i in range(len(path) - 1))


def turn_metrics(path):
    """Returns (number of turns, total absolute heading change in rad)."""
    headings = [math.atan2(path[i + 1][1] - path[i][1], path[i + 1][0] - path[i][0])
                for i in range(len(path) - 1)]
    turns, total = 0, 0.0
    for a, b in zip(headings[:-1], headings[1:]):
        d = (b - a + math.pi) % (2 * math.pi) - math.pi  # wrap to [-pi, pi]
        if abs(d) > 1e-9:
            turns += 1
            total += abs(d)
    return turns, total


def mean_runtime(fn, runs=20):
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(START, GOAL)
        ts.append(time.perf_counter() - t0)
    return 1000.0 * float(np.mean(ts))  # ms


# ----------------------------------------------------------------- smoothing
def line_of_sight(a, b, step=0.01):
    """True if the straight segment a->b stays in free cells.
    Cell (i, j) occupies [i-0.5, i+0.5] x [j-0.5, j+0.5]. The segment is
    sampled densely; every sample's cell (and the cells touched within a
    small margin) must be free, which also blocks squeezing through
    diagonal gaps between obstacle corners."""
    (x0, y0), (x1, y1) = a, b
    L = math.hypot(x1 - x0, y1 - y0)
    n = max(2, int(L / step))
    eps = 1e-6
    for k in range(n + 1):
        t = k / n
        x, y = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
        # round half away from the cell boundary conservatively: check both sides
        for cx in {math.floor(x + 0.5 - eps), math.floor(x + 0.5 + eps)}:
            for cy in {math.floor(y + 0.5 - eps), math.floor(y + 0.5 + eps)}:
                if not free(cx, cy):
                    return False
    return True


def smooth_path(path):
    """Greedy line-of-sight shortcutting: from the current anchor, jump to the
    farthest path vertex that is directly visible."""
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not line_of_sight(path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


# ----------------------------------------------------------------- plotting
def plot_result(title, path, expanded, fname, smooth=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = ListedColormap(["white", "black"])
    ax.imshow(grid, origin="lower", cmap=cmap, extent=(-0.5, N - 0.5, -0.5, N - 0.5))
    if expanded:
        ex = np.array(expanded)
        ax.scatter(ex[:, 0], ex[:, 1], marker="s", s=38, c="lightskyblue",
                   edgecolors="none", label=f"Expanded ({len(expanded)})")
    if path:
        p = np.array(path)
        ax.plot(p[:, 0], p[:, 1], "r-", lw=2.2, label="Path")
    if smooth:
        s = np.array(smooth)
        ax.plot(s[:, 0], s[:, 1], "g-", lw=2.5, marker="o", ms=4, label="Smoothed path")
    ax.plot(*START, "go", ms=11, mec="k", label="Start (2,2)")
    ax.plot(*GOAL, "m*", ms=17, mec="k", label="Goal (27,27)")
    ax.set_xticks(range(0, N, 2))
    ax.set_yticks(range(0, N, 2))
    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(which="minor", color="lightgrey", lw=0.4)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------- main
def main():
    d_path, d_cost, d_exp = dijkstra(START, GOAL)
    a_path, a_cost, a_exp = astar(START, GOAL)
    assert d_path and a_path, "No path found"

    d_time = mean_runtime(dijkstra)
    a_time = mean_runtime(astar)
    d_turns, d_head = turn_metrics(d_path)
    a_turns, a_head = turn_metrics(a_path)

    # (b) plots
    plot_result("Dijkstra", d_path, d_exp, "dijkstra.png")
    plot_result("A* (Euclidean heuristic)", a_path, a_exp, "astar.png")

    # (c) comparison table
    print("\n(c) Comparison")
    print(f"{'Metric':<32}{'Dijkstra':>12}{'A*':>12}")
    print("-" * 56)
    print(f"{'Path cost':<32}{d_cost:>12.4f}{a_cost:>12.4f}")
    print(f"{'Nodes expanded':<32}{len(d_exp):>12d}{len(a_exp):>12d}")
    print(f"{'Mean time of 20 runs (ms)':<32}{d_time:>12.3f}{a_time:>12.3f}")
    print(f"{'Number of turns':<32}{d_turns:>12d}{a_turns:>12d}")
    print(f"{'Total heading change (rad)':<32}{d_head:>12.4f}{a_head:>12.4f}")

    # (d) smoothing
    s_path = smooth_path(a_path)
    s_len = path_length(s_path)
    s_turns, s_head = turn_metrics(s_path)
    plot_result("A* path with line-of-sight smoothing", a_path, a_exp,
                "astar_smoothed.png", smooth=s_path)

    print("\n(d) Smoothed A* path")
    print(f"Waypoints            : {s_path}")
    print(f"Length               : {s_len:.4f}  (raw A*: {path_length(a_path):.4f})")
    print(f"Number of turns      : {s_turns}")
    print(f"Total heading change : {s_head:.4f} rad")

    json.dump({
        "dijkstra": {"cost": d_cost, "expanded": len(d_exp), "time_ms": d_time,
                     "turns": d_turns, "heading": d_head},
        "astar": {"cost": a_cost, "expanded": len(a_exp), "time_ms": a_time,
                  "turns": a_turns, "heading": a_head},
        "smooth": {"length": s_len, "turns": s_turns, "heading": s_head,
                   "waypoints": s_path, "raw_length": path_length(a_path)},
    }, open("results_task1.json", "w"), indent=1)

    # side-by-side summary figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.3))
    for ax, (t, p, e, s) in zip(axes, [
        (f"Dijkstra ({len(d_exp)} expanded)", d_path, d_exp, None),
        (f"A* ({len(a_exp)} expanded)", a_path, a_exp, None),
        ("A* + smoothing", a_path, [], s_path)]):
        ax.imshow(grid, origin="lower", cmap=ListedColormap(["white", "black"]),
                  extent=(-0.5, N - 0.5, -0.5, N - 0.5))
        if e:
            ex = np.array(e)
            ax.scatter(ex[:, 0], ex[:, 1], marker="s", s=14, c="lightskyblue", edgecolors="none")
        pp = np.array(p)
        ax.plot(pp[:, 0], pp[:, 1], "r-", lw=1.8)
        if s:
            ss = np.array(s)
            ax.plot(ss[:, 0], ss[:, 1], "g-", lw=2.5, marker="o", ms=4)
        ax.plot(*START, "go", ms=9, mec="k")
        ax.plot(*GOAL, "m*", ms=14, mec="k")
        ax.set_title(t)
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig("comparison.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
