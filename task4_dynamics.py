"""
Task 4: dynamics of a 2-link planar manipulator in the vertical plane.
Uniform slender rods, COM at mid-link, I = m l^2 / 12, gravity along -y, no friction.

Equations of motion (Lagrange):  M(q) qdd + c(q, qd) + G(q) = tau
  M11 = m1 lc1^2 + I1 + m2 (l1^2 + lc2^2 + 2 l1 lc2 cos q2) + I2
  M12 = m2 (lc2^2 + l1 lc2 cos q2) + I2,   M22 = m2 lc2^2 + I2
  h   = m2 l1 lc2 sin q2
  c   = [-h (2 qd1 qd2 + qd2^2),  h qd1^2]
  G1  = (m1 lc1 + m2 l1) g cos q1 + m2 lc2 g cos(q1+q2),  G2 = m2 lc2 g cos(q1+q2)

Outputs: results_task4.json, task4_free_motion.png, task4_energy.png, task4_torques.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp          # RK45 = Python equivalent of ode45

from task2_joint_trajectory import Q0, QF, T, cubic_coeffs, cubic_eval

L1, L2 = 1.0, 0.8
M1, M2 = 2.0, 1.5
G = 9.81
LC1, LC2 = L1 / 2, L2 / 2
I1, I2 = M1 * L1**2 / 12, M2 * L2**2 / 12


def mass_matrix(q):
    c2 = np.cos(q[1])
    m11 = M1 * LC1**2 + I1 + M2 * (L1**2 + LC2**2 + 2 * L1 * LC2 * c2) + I2
    m12 = M2 * (LC2**2 + L1 * LC2 * c2) + I2
    m22 = M2 * LC2**2 + I2
    return np.array([[m11, m12], [m12, m22]])


def coriolis(q, qd):
    h = M2 * L1 * LC2 * np.sin(q[1])
    return np.array([-h * (2 * qd[0] * qd[1] + qd[1] ** 2), h * qd[0] ** 2])


def gravity(q):
    g1 = (M1 * LC1 + M2 * L1) * G * np.cos(q[0]) + M2 * LC2 * G * np.cos(q[0] + q[1])
    g2 = M2 * LC2 * G * np.cos(q[0] + q[1])
    return np.array([g1, g2])


def inverse_dynamics(q, qd, qdd):
    return mass_matrix(q) @ qdd + coriolis(q, qd) + gravity(q)


def forward_dynamics(q, qd, tau):
    return np.linalg.solve(mass_matrix(q), tau - coriolis(q, qd) - gravity(q))


def energy(q, qd):
    kin = 0.5 * qd @ mass_matrix(q) @ qd
    # potential energy, reference y = 0 at the shoulder
    pot = M1 * G * LC1 * np.sin(q[0]) + M2 * G * (L1 * np.sin(q[0]) + LC2 * np.sin(q[0] + q[1]))
    return kin, pot


def fk(q):
    xe, ye = L1 * np.cos(q[0]), L1 * np.sin(q[0])
    return xe, ye, xe + L2 * np.cos(q[0] + q[1]), ye + L2 * np.sin(q[0] + q[1])


def main():
    out = {}

    # ---------------- (a) forward dynamics, tau = 0, from rest at q = (0, 0) -------------
    def rhs(t, s):
        q, qd = s[:2], s[2:]
        return np.concatenate([qd, forward_dynamics(q, qd, np.zeros(2))])

    t_eval = np.arange(0.0, 5.0 + 1e-9, 0.01)
    sol = solve_ivp(rhs, (0, 5), [0, 0, 0, 0], method="RK45", t_eval=t_eval,
                    rtol=1e-10, atol=1e-10)
    t, q, qd = sol.t, sol.y[:2], sol.y[2:]
    E = np.array([sum(energy(q[:, k], qd[:, k])) for k in range(t.size)])
    KE = np.array([energy(q[:, k], qd[:, k])[0] for k in range(t.size)])
    PE = np.array([energy(q[:, k], qd[:, k])[1] for k in range(t.size)])
    drift = float(np.max(np.abs(E - E[0])))
    print(f"(a) E0 = {E[0]:.4f} J, max |E-E0| = {drift:.2e} J, "
          f"theta1 range [{np.degrees(q[0]).min():.1f}, {np.degrees(q[0]).max():.1f}] deg, "
          f"theta2 range [{np.degrees(q[1]).min():.1f}, {np.degrees(q[1]).max():.1f}] deg")
    out["free"] = {"E0": float(E[0]), "drift": drift,
                   "th1": [float(np.degrees(q[0]).min()), float(np.degrees(q[0]).max())],
                   "th2": [float(np.degrees(q[1]).min()), float(np.degrees(q[1]).max())],
                   "KE_max": float(KE.max()), "PE_min": float(PE.min())}

    xe, ye, x, y = np.array([fk(q[:, k]) for k in range(t.size)]).T
    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25])
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
    axs[0].plot(t, np.degrees(q[0]), label=r"$\theta_1$")
    axs[0].plot(t, np.degrees(q[1]), label=r"$\theta_2$")
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("Joint angle [deg]")
    axs[0].set_title("Free motion (tau = 0): joint angles")
    axs[0].grid(alpha=0.4)
    axs[0].legend()
    axs[1].plot(t, x, label="x")
    axs[1].plot(t, y, label="y")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("End-effector coordinate [m]")
    axs[1].set_title("End-effector coordinates")
    axs[1].grid(alpha=0.4)
    axs[1].legend()
    axs[2].plot(x, y, "b-", lw=1.2, label="End-effector path")
    axs[2].plot(x[0], y[0], "go", mec="k", ms=9, label="Start (1.8, 0) m")
    axs[2].plot(x[-1], y[-1], "r*", mec="k", ms=13, label="End (t = 5 s)")
    axs[2].plot(0, 0, "ks", label="Base")
    axs[2].set_aspect("equal")
    axs[2].set_xlabel("x [m]")
    axs[2].set_ylabel("y [m]")
    axs[2].set_title("End-effector path (x-y plane)")
    axs[2].grid(alpha=0.4)
    axs[2].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig("task4_free_motion.png", dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axs[0].plot(t, KE, label="Kinetic")
    axs[0].plot(t, PE, label="Potential")
    axs[0].plot(t, E, "k--", label="Total")
    axs[0].set_ylabel("Energy [J]")
    axs[0].set_title("Energy during free motion")
    axs[0].grid(alpha=0.4)
    axs[0].legend()
    axs[1].plot(t, E - E[0], "k")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("E(t) - E(0) [J]")
    axs[1].set_title("Total-energy error (conservation check)")
    axs[1].grid(alpha=0.4)
    axs[1].legend(["Total energy error"])
    fig.tight_layout()
    fig.savefig("task4_energy.png", dpi=150)
    plt.close(fig)

    # ---------------- (c) inverse dynamics along the Task 2 cubic trajectory -------------
    tt = np.arange(0.0, T + 0.005, 0.01)
    coef = [cubic_coeffs(Q0[i], QF[i], T) for i in range(2)]
    qc = np.zeros((2, tt.size)); qdc = np.zeros_like(qc); qddc = np.zeros_like(qc)
    for i in range(2):
        qc[i], qdc[i], qddc[i] = cubic_eval(coef[i], tt)
    tau = np.array([inverse_dynamics(qc[:, k], qdc[:, k], qddc[:, k]) for k in range(tt.size)]).T
    peaks = []
    for i in range(2):
        k = int(np.argmax(np.abs(tau[i])))
        peaks.append({"joint": i + 1, "peak": float(tau[i][k]), "t": float(tt[k]),
                      "t0": float(tau[i][0]), "tT": float(tau[i][-1])})
        print(f"(c) tau{i + 1}: peak {tau[i][k]:.2f} N m at t = {tt[k]:.2f} s "
              f"(tau(0) = {tau[i][0]:.2f}, tau(T) = {tau[i][-1]:.2f})")
    out["tau_peaks"] = peaks

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, c in enumerate(["tab:blue", "tab:red"]):
        ax.plot(tt, tau[i], color=c, label=rf"$\tau_{i + 1}$")
        k = int(np.argmax(np.abs(tau[i])))
        ax.plot(tt[k], tau[i][k], "o", color=c, mec="k", ms=8)
        ax.annotate(f"{tau[i][k]:.1f} N m", (tt[k], tau[i][k]), textcoords="offset points",
                    xytext=(8, 6), fontsize=9)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Joint torque [N m]")
    ax.set_title("Inverse dynamics: torques for the Task 2 cubic trajectory (peaks marked)")
    ax.grid(alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig("task4_torques.png", dpi=150)
    plt.close(fig)

    # verification: apply tau(t) in forward dynamics and compare with the reference
    def rhs_tau(t, s):
        qr = np.array([cubic_eval(coef[i], t)[0] for i in range(2)])
        qdr = np.array([cubic_eval(coef[i], t)[1] for i in range(2)])
        qddr = np.array([cubic_eval(coef[i], t)[2] for i in range(2)])
        return np.concatenate([s[2:], forward_dynamics(s[:2], s[2:], inverse_dynamics(qr, qdr, qddr))])

    v = solve_ivp(rhs_tau, (0, T), [*Q0, 0, 0], method="RK45", t_eval=tt, rtol=1e-10, atol=1e-10)
    err = float(np.degrees(np.max(np.abs(v.y[:2] - qc))))
    print(f"(c) check: max joint error forward(tau) vs reference = {err:.2e} deg")
    out["check_err_deg"] = err

    json.dump(out, open("results_task4.json", "w"), indent=1)


if __name__ == "__main__":
    main()
