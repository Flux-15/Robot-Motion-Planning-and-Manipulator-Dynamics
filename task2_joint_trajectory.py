"""
Task 2: cubic joint-space trajectory of a 2-link planar manipulator.
q(t) = a0 + a1 t + a2 t^2 + a3 t^3 with q(0)=q0, q(T)=qf, qd(0)=qd(T)=0
  a0 = q0, a1 = 0, a2 = 3(qf-q0)/T^2, a3 = -2(qf-q0)/T^3
Outputs: results_task2.json, task2_joints.png, task2_ee_time.png, task2_ee_path.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L1, L2 = 1.0, 0.8                       # link lengths [m]
Q0 = np.radians([30.0, 45.0])           # start (theta1, theta2) [rad]
QF = np.radians([60.0, 90.0])           # end   (theta1, theta2) [rad]
T, DT = 3.0, 0.01                       # total time, sampling time [s]


def cubic_coeffs(q0, qf, T):
    """Return [a0, a1, a2, a3] for one joint (rest-to-rest)."""
    d = qf - q0
    return np.array([q0, 0.0, 3.0 * d / T**2, -2.0 * d / T**3])


def cubic_eval(a, t):
    """Position, velocity, acceleration of the cubic with coefficients a."""
    q = a[0] + a[1] * t + a[2] * t**2 + a[3] * t**3
    qd = a[1] + 2 * a[2] * t + 3 * a[3] * t**2
    qdd = 2 * a[2] + 6 * a[3] * t
    return q, qd, qdd


def fk(q1, q2):
    """Forward kinematics: positions of elbow and end-effector."""
    xe, ye = L1 * np.cos(q1), L1 * np.sin(q1)
    x = xe + L2 * np.cos(q1 + q2)
    y = ye + L2 * np.sin(q1 + q2)
    return xe, ye, x, y


def trajectory():
    """Cubic trajectory of both joints on the sampled time grid."""
    t = np.arange(0.0, T + DT / 2, DT)
    coeffs = [cubic_coeffs(Q0[i], QF[i], T) for i in range(2)]
    q = np.zeros((2, t.size))
    qd = np.zeros_like(q)
    qdd = np.zeros_like(q)
    for i in range(2):
        q[i], qd[i], qdd[i] = cubic_eval(coeffs[i], t)
    return t, q, qd, qdd, coeffs


def main():
    t, q, qd, qdd, coeffs = trajectory()
    deg = np.degrees

    # (a) coefficients (angles in degrees)
    print("(a) Cubic coefficients [deg, deg/s, deg/s^2, deg/s^3]")
    for i, a in enumerate(coeffs):
        print(f"  joint {i + 1}: a0={deg(a[0]):.4f}  a1={deg(a[1]):.4f}  "
              f"a2={deg(a[2]):.4f}  a3={deg(a[3]):.4f}")

    # (b) peaks
    peaks = []
    for i in range(2):
        k_v = np.argmax(np.abs(qd[i]))
        k_a = np.argmax(np.abs(qdd[i]))
        peaks.append({"joint": i + 1,
                      "vel": float(deg(qd[i][k_v])), "t_vel": float(t[k_v]),
                      "acc": float(deg(qdd[i][k_a])), "t_acc": float(t[k_a]),
                      "acc_end": float(deg(qdd[i][-1]))})
        print(f"  joint {i + 1}: peak vel {peaks[-1]['vel']:.2f} deg/s at t={t[k_v]:.2f} s, "
              f"peak |acc| {abs(peaks[-1]['acc']):.2f} deg/s^2 at t={t[k_a]:.2f} s")

    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    labels = [r"$\theta_1$", r"$\theta_2$"]
    cols = ["tab:blue", "tab:red"]
    for i in range(2):
        axs[0].plot(t, deg(q[i]), color=cols[i], label=labels[i])
        axs[1].plot(t, deg(qd[i]), color=cols[i], label=labels[i])
        axs[2].plot(t, deg(qdd[i]), color=cols[i], label=labels[i])
        # mark peak values
        k_v = np.argmax(np.abs(qd[i]))
        axs[1].plot(t[k_v], deg(qd[i][k_v]), "o", color=cols[i], mec="k", ms=8)
        axs[1].annotate(f"{deg(qd[i][k_v]):.1f} deg/s", (t[k_v], deg(qd[i][k_v])),
                        textcoords="offset points", xytext=(8, 6), fontsize=9)
        for kk in (0, t.size - 1):
            axs[2].plot(t[kk], deg(qdd[i][kk]), "o", color=cols[i], mec="k", ms=8)
            axs[2].annotate(f"{deg(qdd[i][kk]):.1f}", (t[kk], deg(qdd[i][kk])),
                            textcoords="offset points",
                            xytext=(8 if kk == 0 else -40, 6 if kk == 0 else 6), fontsize=9)
        axs[0].plot([0, T], deg([q[i][0], q[i][-1]]), "o", color=cols[i], mec="k", ms=6)
    axs[0].set_ylabel("Joint angle [deg]")
    axs[1].set_ylabel("Joint velocity [deg/s]")
    axs[2].set_ylabel(r"Joint acceleration [deg/s$^2$]")
    axs[2].set_xlabel("Time [s]")
    axs[0].set_title("Cubic joint trajectories (peak values marked)")
    for ax in axs:
        ax.grid(alpha=0.4)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig("task2_joints.png", dpi=150)
    plt.close(fig)

    # (c) forward kinematics
    xe, ye, x, y = fk(q[0], q[1])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, x, label="x(t)")
    ax.plot(t, y, label="y(t)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("End-effector coordinate [m]")
    ax.set_title("End-effector coordinates vs time")
    ax.grid(alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig("task2_ee_time.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x, y, "k--", lw=2, label="End-effector path")
    ax.plot(x[0], y[0], "go", ms=11, mec="k", label=f"Start ({x[0]:.3f}, {y[0]:.3f}) m")
    ax.plot(x[-1], y[-1], "r*", ms=16, mec="k", label=f"End ({x[-1]:.3f}, {y[-1]:.3f}) m")
    snaps = [0.0, T / 4, T / 2, 3 * T / 4, T]
    cm = plt.cm.viridis(np.linspace(0, 0.9, len(snaps)))
    for c, ts in zip(cm, snaps):
        k = int(round(ts / DT))
        ax.plot([0, xe[k], x[k]], [0, ye[k], y[k]], "-o", color=c, lw=2.5, ms=6,
                label=f"Arm at t = {ts:.2f} s")
    ax.plot(0, 0, "ks", ms=9, label="Base")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Arm configurations and end-effector path in the x-y plane")
    ax.grid(alpha=0.4)
    ax.set_xlim(-0.4, 1.9)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig("task2_ee_path.png", dpi=150)
    plt.close(fig)

    ee = {"start": [float(x[0]), float(y[0])], "end": [float(x[-1]), float(y[-1])],
          "x_range": [float(x.min()), float(x.max())],
          "y_range": [float(y.min()), float(y.max())]}
    print("(c) Start", ee["start"], "End", ee["end"])

    json.dump({"coeffs_deg": [[float(deg(c)) for c in a] for a in coeffs],
               "peaks": peaks, "ee": ee, "n_samples": int(t.size)},
              open("results_task2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
