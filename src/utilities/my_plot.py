import numpy as np
import matplotlib.pyplot as plt

def draw_line(point1, point2, ax):
    """
    Draw a line between two points and save the figure.

    Parameters:
    point1 (tuple): The first point as (x1, y1)
    point2 (tuple): The second point as (x2, y2)
    filename (str): The filename to save the figure as (default: 'fig3.png')
    """
    x_values = [point1[0], point2[0]]
    y_values = [point1[1], point2[1]]
    

    ax.plot(x_values, y_values, marker='o')  # Draw the line with markers at the points
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Robot trajectory')
    ax.grid(True)

def plot_scp_traj(ax1, fig1, x_obs, r_safe, x_guess, x_opt, x0, t_query, dt, time_iter, steps, fig_name):
    """
    Plots the SCP trajectory and saves the figure.

    Parameters:
    - ax1: The axis object for plotting.
    - fig1: The figure object for saving the plot.
    - x_obs: Array of obstacle positions.
    - r_safe: Safety radius around the obstacles.
    - x_guess: Initial guess for the trajectory.
    - x_opt: Optimized trajectory.
    - x0: Initial position.
    - t_query: Time query parameter.
    - dt: Time step.
    - time_iter: Time iteration parameter.
    """
    l=1

    if time_iter <= -1:

        ax1.scatter(x_obs[0,0], x_obs[0,1], color='r', s=100, label="Obstacle")
        circle = plt.Circle(x_obs[0], r_safe, color='r', fill=False, label="Safety Radius")
        ax1.add_artist(circle)
        
        ax1.set_title("Optimal 2D Trajectory Avoiding Obstacle")
        ax1.set_xlabel("X position")
        ax1.set_ylabel("Y position")
        ax1.plot(x0[0], x0[1], 'g--', label="Initial guess")
        ax1.plot(x0[0], x0[1], 'b-', label="Path")
        ax1.plot(x0[0], x0[1], 'b--', label="Optimal Trajectory")
        ax1.legend()

        for obs in x_obs:
            ax1.scatter(obs[0], obs[1], color='r', s=100, label="Obstacle")
            circle = plt.Circle(obs, r_safe, color='r', fill=False, label="Safety Radius")
            ax1.add_artist(circle)
        
        ax1.scatter(x0[0], x0[1], color='b', s=100, label="Position")
        ax1.grid(True)
        ax1.axis('equal')  # This line ensures that the scale of the x and y axes is the same
    else:
        ax1.scatter(x0[0], x0[1], color='b', s=100, label="Position")
        ax1.plot(x_guess[:, 0], x_guess[:, 1], 'g--', label="Initial guess")
        ax1.plot(x_opt[:int(t_query / dt), 0], x_opt[:int(t_query / dt), 1], 'b-', label="Path")
        ax1.plot(x_opt[:, 0], x_opt[:, 1], 'b--', label="Optimal Trajectory")
    
    # if time_iter == 5:
    #     for i in range(steps+1):
    #         point1 = (x_opt[i, 0], x_opt[i, 1])
    #         point2 = (x_opt[i, 0]+np.cos(x_opt[i, 2])*l, x_opt[i, 1]+np.sin(x_opt[i, 2])*l)
    #         # point2 = point1
    #         draw_line(point1, point2, ax1)
    point1 = (x0[0], x0[1])
    point2 = (x0[0]+np.cos(x0[2])*l, x0[1]+np.sin(x0[2])*l)
    draw_line(point1, point2, ax1)

    fig1.savefig(fig_name)

def plot_SCP_dynamics(T, N, x_opt, u_opt, fig_name):
    """
    Plots the dynamics of the system.

    Parameters:
    - T: Total time
    - N: Number of time steps
    - x_opt: Optimized positions and velocities
    - u_opt: Optimized control inputs
    """
    t = np.linspace(0, T, N + 1)
    plt.figure(figsize=(12, 8))

    # Plot position x(t) and y(t)
    plt.subplot(2, 1, 1)
    plt.plot(t, x_opt[:, 0], label='Position x(t)', color='green')  # Plot x-coordinate
    plt.plot(t, x_opt[:, 1], label='Position y(t)', color='blue')  # Plot y-coordinate
    plt.plot(t, x_opt[:, 2], label='Position theta(t)', color='red')  # Plot theta-coordinate
    plt.xlabel('Time (s)')
    plt.ylabel('States')
    plt.title('States vs. Time')
    plt.legend()

    # # Plot velocity v_x(t) and v_y(t)
    # plt.subplot(3, 1, 2)
    # # plt.plot(t, x_opt[:, 2], label='Velocity v_x(t)')
    # plt.plot(t, x_opt[:, 1], label='Velocity v(t)')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Velocity (m/s)')
    # plt.title('Velocity vs. Time')
    # plt.legend()

    # Plot control inputs u1(t) and u2(t)
    plt.subplot(2, 1, 2)
    plt.plot(t[:-1], u_opt[:, 0], label='Control Input u1(t)')
    plt.plot(t[:-1], u_opt[:, 1], label='Control Input u2(t)')
    plt.xlabel('Time (s)')
    plt.ylabel('Control Input (m/s^2)')
    plt.title('Control Input vs. Time')
    plt.legend()

    plt.tight_layout()
    plt.savefig(fig_name)