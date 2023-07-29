import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Initial Conditions
r0 = np.array([42000, 0, 0])
v0 = np.array([0, 3.0807, 0])

# Constants
mu = 398600.4418    # Earth Gravitational Parameter in km^3/s^2
dt = 1.0            # seconds
M = 5.972e+24       # Earth mass in kg
G = 6.67430e-20     # Gravitational constant in km^3/(kg*s^2)
Cd = 1000.0            # Drag coefficient
A = 10.0            # Cross-sectional area in m^2
const_atm_density = 5  # kg/m^3

# Number of time steps
num_steps = 5 * 24 * 60 * 60    # 5 days in seconds

# Function to Compute Acceleration
def acceleration(r, M):
    r_norm = np.linalg.norm(r)
    return -G * M * r / r_norm ** 3

# Function to compute one Runge-Kutta integration step
def runge_kutta_w_drag(r, v, dt, M, Cd, A):
    t1_v = acceleration(r, M) - 0.5 * const_atm_density * Cd * A * np.linalg.norm(v) * v / M
    t1_r = v
    t2_v = acceleration(r + 0.5 * dt * t1_r, M) - 0.5 * const_atm_density * Cd * A * np.linalg.norm(v + 0.5 * dt * t1_v) * (v + 0.5 * dt * t1_v) / M
    t2_r = v + 0.5 * dt * t1_v
    t3_v = acceleration(r + 0.5 * dt * t2_r, M) - 0.5 * const_atm_density * Cd * A * np.linalg.norm(v + 0.5 * dt * t2_v) * (v + 0.5 * dt * t2_v) / M
    t3_r = v + 0.5 * dt * t2_v
    t4_v = acceleration(r + dt * t3_r, M) - 0.5 * const_atm_density * Cd * A * np.linalg.norm(v + dt * t3_v) * (v + dt * t3_v) / M
    t4_r = v + dt * t3_v

    new_r = r + (dt / 6.0) * (t1_r + 2 * t2_r + 2 * t3_r + t4_r)
    new_v = v + (dt / 6.0) * (t1_v + 2 * t2_v + 2 * t3_v + t4_v)

    return new_r, new_v

# Function to predict trajectory
def predict_traj_w_drag(r0, v0, dt, N, Cd, A):
    trajectory = [(r0, v0)]
    for _ in range(N):
        r, v = trajectory[-1]
        new_r, new_v = runge_kutta_w_drag(r, v, dt, M, Cd, A)
        trajectory.append((new_r, new_v))
    return trajectory

# PREDICT TRAJECTORY
trajectory_w_drag = predict_traj_w_drag(r0, v0, dt, num_steps, Cd, A)

# Function to create 3D animation
def animate_3d_traj_w_drag(i):
    ax.clear()
    # Plot the Earth as a sphere
    earth_radius = 6371  # Earth's radius in km
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = earth_radius * np.outer(np.cos(u), np.sin(v))
    y = earth_radius * np.outer(np.sin(u), np.sin(v))
    z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='blue', alpha=0.5)

    # Extract x, y, and z coordinates for the current time step
    r, _ = trajectory_w_drag[i]
    x_coords, y_coords, z_coords = r[0], r[1], r[2]

    # Plot the satellite position
    ax.plot([x_coords], [y_coords], [z_coords], 'ro', label='Satellite Position')
    ax.set_xlabel('X Position (km)')
    ax.set_ylabel('Y Position (km)')
    ax.set_zlabel('Z Position (km)')
    ax.set_title(f'Satellite Trajectory under Gravity + Atmospheric Drag')
    ax.axes.set_xlim(-60000, 60000)
    ax.axes.set_ylim(-50000, 50000)
    ax.axes.set_zlim(-40000, 40000)
    ax.legend()

# Animate 3D trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
def init():
    return ax
ani = FuncAnimation(fig, animate_3d_traj_w_drag, frames=range(0, len(trajectory_w_drag), 1000), init_func=init, interval=1, repeat=False)

# Save the animation as a GIF
# filename = 'satellite_trajectory_gravity_drag.gif'
# ani.save(filename, writer='pillow')

plt.show()






