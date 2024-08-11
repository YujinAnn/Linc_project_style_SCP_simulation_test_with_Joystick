#!/usr/bin/env python

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath('./utilities'))

from utilities.scp import SCPSubproblem
from utilities.my_plot import draw_line, plot_scp_traj, plot_SCP_dynamics
from utilities.SCP_utility_functions import get_x_at_t, SimpleGuess, compute_halfSpace, get_dyn

import rospy
from geometry_msgs.msg import TwistStamped

# Global variables to store the received Twist messages
linear_x = 0.0
toggle_ = 0.0
angular_z = 0.0

def callback(data):
    global linear_x, toggle_, angular_z
    # Extract linear.x and angular.z from the Twist message
    linear_x = data.twist.linear.x
    toggle_ = data.twist.linear.z
    angular_z = data.twist.angular.z

def get_opt_traj(x_opt, u_opt, dt, N, x_dim, x_obs, r_safe, x0, u_d):
    safe_mode = False
    x_opt, u_opt, cost = x_opt, u_opt, -1
    num_scp_iterations = 5
    

    try: 

        for ii_scp in range(num_scp_iterations):
            # solve subproblem
            Ad, Bd = get_dyn(x_opt, u_opt, dt, N, x_dim_=x_dim)
            Acs, bcs = compute_halfSpace(x_opt, x_obs, N, r_safe, x_dim=x_dim)
            x_opt, u_opt, cost = my_scpsubproblem.solve(x0=x0,
                                                        xT=x0,
                                                        ud=u_d,
                                                        xnom=np.array(x_opt[0:N+1]),
                                                        unom=np.array(u_opt[0:N]),
                                                        As=Ad, Bs=Bd, 
                                                        u_bound=input_lims, 
                                                        Acs=Acs, bcs=bcs)
            print(f"SCP Iteration {ii_scp}: cost = {cost}")
    except (cp.error.SolverError, Exception) as e:
        print(f"Failed due to: {str(e)}. Changing the initial_guess to a sequence of x0.")
        num_scp_iterations_ = 5
        
        # Reset x_opt to a constant initial guess
        x_opt = np.full(x_guess.shape, x_guess[0])
        
        for ii_scp in range(num_scp_iterations_):
            # solve subproblem
            Ad, Bd = get_dyn(x_opt, u_opt, dt, N, x_dim_=x_dim)
            Acs, bcs = compute_halfSpace(x_opt, x_obs, N, r_safe, x_dim=x_dim)
            try:
                x_opt, u_opt, cost = my_scpsubproblem.solve(x0=x0,
                                                            xT=x0,
                                                            ud=u_d,
                                                            xnom=np.array(x_opt[0:N+1]),
                                                            unom=np.array(u_opt[0:N]),
                                                            As=Ad, Bs=Bd, 
                                                            u_bound=input_lims, 
                                                            Acs=Acs, bcs=bcs)
                print(f"SCP Iteration {ii_scp}: cost = {cost}")
            except Exception as e:
                print(f"Failed again during SCP Iteration {ii_scp} due to: {str(e)}")
                safe_mode = True
                break  # Exit if it fails again

    return safe_mode, x_opt, u_opt, cost


# Parameters
N = int(400)  # Number of intervals
T = int(4)  # Total time
dt = T / N  # Time step
t_query = 4 #dt*5

x_obs1 = np.array([10, 1])  # Obstacle coordinates
x_obs2 = np.array([14, 1])  # Obstacle coordinates
x_obs = np.array([x_obs1, x_obs2])
x_dim=3

r_safe = 1.5  # Effective radius including safety distance
x_dim=3
x0 = np.array([7, 0.9, 0])
xf = np.array([15, 0.9, 0])

fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.scatter(x0[0], x0[1], color='b', s=100, label="Position")

R = np.diag(np.array([1, 1]))
Qf = np.diag(np.array([0, 0, 0]))
Q =  np.diag(np.array([0, 0, 0]))
input_lims = np.array([[-10.0, 10.0], [-10.0, 10.0]]) # default value

# initialize SCP subproblem solver
my_scpsubproblem = SCPSubproblem(steps=N, x_dim=x_dim, u_dim=2, R=R, Q=Q, Qf=Qf, use_u_bound=True, max_state_halfspaces=4, Trust_region = True)
myguesser = SimpleGuess(steps=N, x_dim=x_dim, R=R, Qf=Qf, Q=Q, input_lims=input_lims, dt=dt)
max_iter = 100
safe_mode = False

rospy.init_node('cmd_vel_listener', anonymous=True)
rospy.Subscriber("/cmd_vel/teleop/drive", TwistStamped, callback)


def update_desired_input(linear_value, angular_value):
    max_speed=1/2
    max_rotation=np.pi/6/2
    return np.array([max_speed*linear_value, max_rotation*angular_value])


#################################################################################################
############################## main #############################################################
#################################################################################################

plot_scp_traj(ax1, fig1, x_obs, r_safe, x_guess=None, x_opt=None, x0=x0, t_query=None, dt=dt, time_iter=-1, steps=N, fig_name="./figures/trajectory.png")

for time_iter in range(max_iter):
    print(time_iter)


    # Check if angular_z reaches 1 and update target using linear_x
    while(True):

        u_d = update_desired_input(linear_x, angular_z)

        print("Desired input:", u_d)
        if toggle_ >= 1:
            if u_d[0] <=0: # angular movement
                goLength = 1
            else: # linear movement
                goLength = 3 
            break


    print("Confirmed input:", u_d)

    x_guess, u_guess = myguesser.find_initialGuess(x0, u_d, goLength = goLength)
    print("here13")
    x_opt, u_opt = x_guess, u_guess

    safe_mode, x_opt, u_opt, cost = get_opt_traj(x_opt, u_opt, dt, N, x_dim, x_obs, r_safe, x0, u_d)
    print("here1")

    x0 = get_x_at_t(t_query, T, x_opt.T, N)

    # if safe_mode == True:
    #     safe_mode = False
    #     print("Safe_mode")
    #     # x0 = get_x_at_t(T, T, x_opt.T, N)


    plot_scp_traj(ax1, fig1, x_obs, r_safe, x_guess, x_opt, x0, t_query, dt, time_iter, steps=N, fig_name="./figures/trajectory.png")

# Plotting Results
plot_SCP_dynamics(T, N, x_opt, u_opt, fig_name="./figures/state_graph.png")