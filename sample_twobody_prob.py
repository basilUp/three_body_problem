import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initial Conditions
r0 = np.array([42000, 0, 0])
v0 = np.array([0, 3.0807, 0])

# Constants
mu = 398600.4418    # Earth Gravitational Parameter in km^3/s^2
dt = 1.0            # seconds
M = 5.972e+24       # Earth mass in kg
G = 6.67430e-20     # Gravitational constant in km^3/(kg*s^2)

# Equations of Motion
# r' = v
# v' = (-mu / |r|^3) * r

# Number of time steps
num_steps = 5 * 24 * 60 * 60    # 5 days in seconds

# Lists to store earth and satellite positions
earth_pos = []
sat_pos = []

# Function to compute acceleration
def acceleration(r, M):
    r_norm = np.linalg.norm(r)
    return -G * M * r / r_norm ** 3

# Function to compute one Runge-Kutta integration step
def runge_kutta(r, v, dt, M):
    t1_v = acceleration(r, M)
    t1_r = v
    t2_v = acceleration(r + 0.5 * dt * t1_r, M)
    t2_r = v + 0.5 * dt * t1_v
    t3_v = acceleration(r + 0.5 * dt * t2_r, M)
    t3_r = v + 0.5 * dt * t2_v
    t4_v = acceleration(r + dt * t3_r, M)
    t4_r = v + dt * t3_v

    new_r = r + (dt / 6.0) * (t1_r + 2 * t2_r + 2 * t3_r + t4_r)
    new_v = v + (dt / 6.0) * (t1_v + 2 * t2_v + 2 * t3_v + t4_v)

    return new_r, new_v

# Function to predict trajectory
def predict_traj(r0, v0, dt, N):
    traj = [(r0, v0)]
    for _ in range(N):
        r, v = traj[-1]
        new_r, new_v = runge_kutta(r, v, dt, M)
        traj.append((new_r, new_v))
    return traj

# PREDICT TRAJECTORY
trajectory = predict_traj(r0, v0, dt, num_steps)

# Function to plot trajectory
def animate_traj(i):
    plt.cla()
    # x and y coord for each time step
    r, _ = trajectory[i]
    x, y = r[0], r[1]

    # Plot Earth
    earth_radius = 6371  # Earth's radius in km
    earth_circle = plt.Circle((0, 0), earth_radius, color='blue', alpha=0.5)
    plt.gcf().gca().add_artist(earth_circle)

    # Plot satellite trajectory
    plt.plot(x, y, 'ro', label='Satellite Position')
    plt.xlabel('X Position (km)')
    plt.ylabel('Y Position (km)')
    plt.title(f'Satellite Trajectory (Step {i})')
    plt.legend()
    #plt.axis('equal')
    plt.xlim(-50000, 50000)
    plt.ylim(-50000, 50000)
    plt.grid(True)

# Function to plot trajectory
def plot_traj(trajectory):
    # x and y coord for each time step
    x = [r[0] for r, _ in trajectory]
    y = [r[1] for r, _ in trajectory]

    # Plot Earth
    earth_radius = 6371  # Earth's radius in km
    earth_circle = plt.Circle((0, 0), earth_radius, color='blue', alpha=0.5)
    plt.gcf().gca().add_artist(earth_circle)

    # Plot satellite trajectory
    plt.plot(x, y, '-o', label='Satellite Trajectory')
    plt.xlabel('X Position (km)')
    plt.ylabel('Y Position (km)')
    plt.title('Satellite Trajectory')
    plt.legend()
    # plt.axis('equal')
    plt.xlim(-50000, 50000)
    plt.ylim(-50000, 50000)
    plt.grid(True)
    plt.show()

# ANIMATE TRAJECTORY
# fig, ax = plt.subplots()
# ani = FuncAnimation(fig, animate_traj, frames=len(trajectory), interval=1, repeat=False)
# plt.show()

# PLOT TRAJECTORY
plot_traj(trajectory)

# PRINT TRAJECTORY and PLOT
# for i, (r, v) in enumerate(trajectory):
#     print(f"Time step {i}:")
#     print("Position:", r)
#     print("Velocity:", v)
#     print("------------------")





