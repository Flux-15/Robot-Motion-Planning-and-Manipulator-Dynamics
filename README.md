# Robot Motion Planning and Manipulator Dynamics
A Python-based robotics simulation project exploring classical motion-planning algorithms, joint-space trajectory generation, collision avoidance, forward kinematics, and manipulator dynamics.
The project implements and compares different planning approaches while studying the motion and dynamic behaviour of a 2-DOF planar robotic manipulator.

---

## Overview
This repository contains four simulation tasks covering fundamental concepts in **robot motion planning and manipulator dynamics**.
The simulations are implemented in Python using numerical computation, algorithmic planning, and scientific visualization.

### Project Modules
| Task | Topic | Implementation |
|---|---|---|
| 01 | Grid-Based Motion Planning | Dijkstra vs A* |
| 02 | Joint-Space Trajectory Planning | Cubic Polynomial Trajectory |
| 03 | Sampling-Based Motion Planning | Rapidly-exploring Random Tree (RRT) |
| 04 | Manipulator Dynamics | Forward and Inverse Dynamics |

---

## 1. Grid-Based Motion Planning
**File:** `task1_grid_planning.py`
Implements and compares Dijkstra's algorithm and A* search for finding collision-free paths in a 2D occupancy grid.
**Key features:**
- 30 × 30 occupancy grid.
- 8-connected movement.
- Obstacle avoidance with corner-cutting prevention.
- Dijkstra's shortest-path algorithm.
- A* search using a Euclidean distance heuristic.
- Line-of-sight path smoothing.
- Performance comparison based on path cost, expanded nodes, execution time, and path turns.
**Generated outputs:**
- `dijkstra.png`
- `astar.png`
- `astar_smoothed.png`
- `comparison.png`
- `results_task1.json`

---

## 2. Joint-Space Trajectory Planning
**File:** `task2_joint_trajectory.py`
Generates a smooth joint-space trajectory for a 2-link planar robotic manipulator using cubic polynomial interpolation.
The trajectory satisfies the following boundary conditions:
- Initial and final joint positions.
- Zero initial joint velocity.
- Zero final joint velocity.
**Cubic trajectory equation:**
\[
q(t)=a_0+a_1t+a_2t^2+a_3t^3
\]
For rest-to-rest motion:
\[
a_0=q_0,\quad a_1=0
\]
\[
a_2=\frac{3(q_f-q_0)}{T^2}
\]
\[
a_3=-\frac{2(q_f-q_0)}{T^3}
\]
**Manipulator parameters:**
- Link 1 length: 1.0 m
- Link 2 length: 0.8 m
- Initial joint configuration: (30°, 45°)
- Final joint configuration: (60°, 90°)
- Trajectory duration: 3 seconds
- Sampling interval: 0.01 seconds
**Key features:**
- Cubic polynomial trajectory generation.
- Joint position, velocity, and acceleration analysis.
- Forward kinematics.
- End-effector trajectory visualization.
- Robotic arm configuration at different time intervals.
**Generated outputs:**
- `task2_joints.png`
- `task2_ee_time.png`
- `task2_ee_path.png`
- `results_task2.json`
---

## 3. Sampling-Based Motion Planning (RRT)
**File:** `task3_rrt.py`
Implements the Rapidly-exploring Random Tree (RRT) algorithm for collision-free path planning in a continuous 2D workspace.
Unlike grid-based planning, RRT explores the configuration space using random sampling and incremental tree expansion.
**Simulation parameters:**
- Workspace: 100 × 100 units
- Start position: (5, 5)
- Goal position: (95, 95)
- Step size: 5 units
- Goal bias: 5%
- Goal tolerance: 5 units
- Maximum iterations: 5000
- Circular obstacles with different positions and radii.
**Key features:**
- Random configuration sampling.
- Nearest-neighbour selection.
- Incremental tree expansion.
- Collision detection using line-segment and circle intersection.
- Goal-biased sampling.
- Path reconstruction.
- Repeatability analysis across 30 random seeds.
The RRT algorithm is implemented from scratch without relying on external motion-planning libraries.
**Generated outputs:**
- `task3_rrt_tree.png`
- `task3_rrt_path.png`
- `results_task3.json`
---

## 4. Manipulator Dynamics
**File:** `task4_dynamics.py`
Studies the dynamic behaviour of a 2-link planar robotic manipulator using the Lagrangian formulation of robot dynamics.
The equations of motion are represented as:
\[
M(q)\ddot q+C(q,\dot q)+G(q)=\tau
\]
Where:
- \(M(q)\) represents the inertia matrix.
- \(C(q,\dot q)\) represents Coriolis and centrifugal effects.
- \(G(q)\) represents gravitational forces.
- \(\tau\) represents applied joint torques.
**Manipulator parameters:**
| Parameter | Value |
|---|---:|
| Link 1 length | 1.0 m |
| Link 2 length | 0.8 m |
| Link 1 mass | 2.0 kg |
| Link 2 mass | 1.5 kg |
| Gravity | 9.81 m/s² |
| Centre of mass | Midpoint of each link |
**Key features:**
- Lagrangian dynamic modelling.
- Mass/inertia matrix computation.
- Coriolis and centrifugal force calculation.
- Gravitational torque calculation.
- Forward dynamics simulation.
- Inverse dynamics and joint torque estimation.
- Kinetic, potential, and total energy analysis.
- Energy conservation verification.
- Numerical integration using the RK45 solver.
- Verification of inverse dynamics through forward simulation.
**Generated outputs:**
- `task4_free_motion.png`
- `task4_energy.png`
- `task4_torques.png`
- `results_task4.json`
---

## Repository Structure
```text
Robot-Motion-Planning-and-Manipulator-Dynamics/
│
├── Simulation_Problems.pdf
├── Simulation_Problems_Solution.pdf
│
├── task1_grid_planning.py
├── task2_joint_trajectory.py
├── task3_rrt.py
├── task4_dynamics.py
│
└── README.md
```
The simulation scripts automatically generate their corresponding plots and JSON result files when executed.
---
## Technologies Used
| Technology | Purpose |
|---|---|
| Python | Core programming language |
| NumPy | Numerical computation |
| Matplotlib | Data visualization and plotting |
| SciPy | Numerical integration and differential equation solving |
| JSON | Simulation result storage |
---

## Installation and Setup
### 1. Clone the repository
```bash
git clone https://github.com/Flux-15/Robot-Motion-Planning-and-Manipulator-Dynamics.git
cd Robot-Motion-Planning-and-Manipulator-Dynamics
```
### 2. Create a virtual environment (Recommended)
```bash
python -m venv venv
```
Activate it:
**Windows**
```bash
venv\Scripts\activate
```
**Linux / macOS**
```bash
source venv/bin/activate
```
### 3. Install dependencies
```bash
pip install numpy matplotlib scipy
```
---

## Running the Simulations
Run each task independently:
**Task 1 – Grid-Based Planning**
```bash
python task1_grid_planning.py
```
**Task 2 – Joint Trajectory Planning**
```bash
python task2_joint_trajectory.py
```
**Task 3 – RRT Motion Planning**
```bash
python task3_rrt.py
```
**Task 4 – Manipulator Dynamics**
```bash
python task4_dynamics.py
```
**Important:** Task 4 imports trajectory parameters and polynomial functions from `task2_joint_trajectory.py`. Keep both scripts in the same directory.
After execution, the generated plots are saved in the project directory, and numerical results are exported as JSON files.
---
## Learning Outcomes
This project explores several important concepts in robotics:
- Understanding graph-based and sampling-based motion planning.
- Comparing Dijkstra and A* search algorithms.
- Generating smooth joint-space trajectories.
- Understanding forward kinematics of planar manipulators.
- Implementing collision detection and obstacle avoidance.
- Deriving and implementing robot equations of motion.
- Analysing joint torques and manipulator energy.
- Validating mathematical models through numerical simulation.
---

## References and Supporting Material
The repository includes two supporting documents:
- `Simulation_Problems.pdf` – Simulation problem statements.
- `Simulation_Problems_Solution.pdf` – Corresponding solutions and reference material.
---

## Author
**Om Patel (FLφX)**
Electronics and Communication Engineering | Robotics & Embedded Systems
- GitHub: [@Flux-15](https://github.com/Flux-15)
---
*Exploring robotics through mathematics, algorithms, and simulation.*

