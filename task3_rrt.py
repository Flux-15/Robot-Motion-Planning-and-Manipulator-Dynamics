"""
Task 3: RRT for a point robot among circular obstacles (100 x 100 workspace).
Parameters: step 5, goal bias 5 %, goal tolerance 5, max 5000 iterations.
Own implementation: no library planners.
Outputs: results_task3.json, task3_rrt_tree.png, task3_rrt_path.png
"""
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

START, GOAL = (5.0, 5.0), (95.0, 95.0)
WORLD = 100.0
OBSTACLES = [(30, 30, 10), (60, 70, 12), (70, 30, 8), (20, 75, 9), (50, 50, 7)]  # (cx, cy, r)
STEP, GOAL_BIAS, GOAL_TOL, MAX_ITER = 5.0, 0.05, 5.0, 5000
SEED = 1  # assumption: fixed seed for a reproducible figure


# (a) collision checking -------------------------------------------------------
def segment_hits_circle(p, q, c, r):
    """True if segment p-q intersects the disc with centre c and radius r.
    Uses the distance from the centre to the closest point of the segment."""
    px, py = p
    dx, dy = q[0] - px, q[1] - py
    seg2 = dx * dx + dy * dy
    if seg2 == 0.0:
        t = 0.0
    else:
        t = ((c[0] - px) * dx + (c[1] - py) * dy) / seg2
        t = max(0.0, min(1.0, t))          # clamp to the segment
    cx, cy = px + t * dx, py + t * dy      # closest point on the segment
    return math.hypot(c[0] - cx, c[1] - cy) <= r


def segment_free(p, q, obstacles=OBSTACLES):
    """True if the segment p-q is collision free w.r.t. all circular obstacles."""
    return not any(segment_hits_circle(p, q, (ox, oy), r) for ox, oy, r in obstacles)


def point_free(p):
    return segment_free(p, p)


# (b) RRT ------------------------------------------------------------------------
def rrt(rng):
    nodes = [START]
    parent = [-1]
    samples = []
    for it in range(1, MAX_ITER + 1):
        # sample with 5 % goal bias
        if rng.random() < GOAL_BIAS:
            s = GOAL
        else:
            s = (rng.uniform(0, WORLD), rng.uniform(0, WORLD))
        samples.append(s)
        # nearest node (linear search)
        d2 = [(n[0] - s[0]) ** 2 + (n[1] - s[1]) ** 2 for n in nodes]
        i_near = int(np.argmin(d2))
        near = nodes[i_near]
        dist = math.sqrt(d2[i_near])
        if dist == 0.0:
            continue
        # steer: move at most STEP towards the sample
        k = min(1.0, STEP / dist)
        new = (near[0] + k * (s[0] - near[0]), near[1] + k * (s[1] - near[1]))
        if not segment_free(near, new):
            continue
        nodes.append(new)
        parent.append(i_near)
        # goal test: within tolerance and the final connection is collision free
        if math.hypot(new[0] - GOAL[0], new[1] - GOAL[1]) <= GOAL_TOL and segment_free(new, GOAL):
            nodes.append(GOAL)
            parent.append(len(nodes) - 2)
            path, i = [], len(nodes) - 1
            while i != -1:
                path.append(nodes[i])
                i = parent[i]
            return path[::-1], nodes, parent, samples, it
    return None, nodes, parent, samples, MAX_ITER


def path_len(path):
    return sum(math.dist(path[i], path[i + 1]) for i in range(len(path) - 1))


def draw_scene(ax):
    for ox, oy, r in OBSTACLES:
        ax.add_patch(Circle((ox, oy), r, fc="dimgray", ec="k", alpha=0.85))
    ax.plot(*START, "go", ms=11, mec="k", label="Start (5, 5)", zorder=5)
    ax.plot(*GOAL, "m*", ms=17, mec="k", label="Goal (95, 95)", zorder=5)
    ax.add_patch(Circle(GOAL, GOAL_TOL, fc="none", ec="m", ls=":", lw=1))
    ax.set_xlim(0, WORLD)
    ax.set_ylim(0, WORLD)
    ax.set_aspect("equal")
    ax.set_xlabel("x [units]")
    ax.set_ylabel("y [units]")


def main():
    assert point_free(START) and point_free(GOAL)
    rng = np.random.default_rng(SEED)
    path, nodes, parent, samples, iters = rrt(rng)
    assert path is not None, "RRT failed within the iteration limit"
    length = path_len(path)
    straight = math.dist(START, GOAL)
    print(f"Seed {SEED}: iterations {iters}, nodes {len(nodes)}, path length {length:.2f}, "
          f"straight-line {straight:.2f}, ratio {length / straight:.3f}, waypoints {len(path)}")

    # (c-i) samples and tree
    fig, ax = plt.subplots(figsize=(8, 8))
    sm = np.array(samples)
    ax.scatter(sm[:, 0], sm[:, 1], s=4, c="lightgray", label=f"Random samples ({len(samples)})")
    for i in range(1, len(nodes)):
        a, b = nodes[parent[i]], nodes[i]
        ax.plot([a[0], b[0]], [a[1], b[1]], "-", color="tab:blue", lw=0.8)
    ax.plot([], [], "-", color="tab:blue", label=f"RRT tree ({len(nodes)} nodes)")
    draw_scene(ax)
    ax.set_title("RRT: random samples and expanding tree")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig("task3_rrt_tree.png", dpi=150)
    plt.close(fig)

    # (c-ii) final path
    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(1, len(nodes)):
        a, b = nodes[parent[i]], nodes[i]
        ax.plot([a[0], b[0]], [a[1], b[1]], "-", color="lightsteelblue", lw=0.6)
    ax.plot([], [], "-", color="lightsteelblue", label="RRT tree")
    p = np.array(path)
    ax.plot(p[:, 0], p[:, 1], "r-o", lw=2.5, ms=4, label=f"Final path (length {length:.1f})")
    draw_scene(ax)
    ax.set_title("RRT: final collision-free path")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig("task3_rrt_path.png", dpi=150)
    plt.close(fig)

    # extra: repeatability over 30 random seeds
    stats = []
    for sd in range(100, 130):
        pth, nd, _, _, it = rrt(np.random.default_rng(sd))
        if pth:
            stats.append((it, len(nd), path_len(pth)))
    st = np.array(stats)
    print(f"30 seeds: success {len(st)}/30, iterations mean {st[:, 0].mean():.0f}, "
          f"nodes mean {st[:, 1].mean():.0f}, length mean {st[:, 2].mean():.1f} "
          f"(min {st[:, 2].min():.1f}, max {st[:, 2].max():.1f})")

    json.dump({"iters": iters, "nodes": len(nodes), "length": length, "straight": straight,
               "waypoints": len(path), "path": [list(map(float, q)) for q in path],
               "n_seeds": 30, "success": int(len(st)),
               "it_mean": float(st[:, 0].mean()), "nodes_mean": float(st[:, 1].mean()),
               "len_mean": float(st[:, 2].mean()), "len_min": float(st[:, 2].min()),
               "len_max": float(st[:, 2].max()), "seed": SEED},
              open("results_task3.json", "w"), indent=1)


if __name__ == "__main__":
    main()
