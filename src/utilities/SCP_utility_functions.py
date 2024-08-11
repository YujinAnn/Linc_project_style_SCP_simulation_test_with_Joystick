import numpy as np
from scp import SCPSubproblem

def get_x_at_t(t_query, T, x_values, N):
    dt = T / N
    time_index = int(t_query / dt)
    if time_index >= len(x_values[0,:]):
        raise ValueError("The queried time exceeds the time range of the solution.")
    return x_values[:,time_index]

def compute_halfSpace(x_opt, x_obss, steps, r_safe, x_dim):
    Acs, bcs = [], []
    l=1
    x_obs1 = x_obss[0]
    if len(x_obss) == 2:
        x_obs2 = x_obss[1]

    for i in range(steps + 1):
        Ac_t = np.zeros((len(x_obss)*2, x_dim))
        Bc_t = np.zeros((len(x_obss)*2,))

        # robot body constraint
        diff = x_obs1 - x_opt[i, :2]
        norm_diff = np.linalg.norm(diff, 2)
        Ac_t[0, :2] = (diff / norm_diff).reshape(1, 2)   
        Bc_t[0] = np.array([norm_diff - r_safe])

        # robot corner constraint
        R_theta = np.array([np.cos(x_opt[i, 2])*l, np.sin(x_opt[i, 2])*l])
        endpoint_pos_prev = np.hstack([x_opt[i, 0], x_opt[i, 1]]) + R_theta
        diff2 = x_obs1 - endpoint_pos_prev
        norm_diff2 = np.linalg.norm(diff2, 2)
        Ac_t[1, :2] = (diff2 / norm_diff2).reshape(1, 2)   
        Bc_t[1] = np.array([norm_diff2 - r_safe])

        if len(x_obss) == 2:
            # robot body constraint
            diff = x_obs2 - x_opt[i, :2]
            norm_diff = np.linalg.norm(diff, 2)
            Ac_t[2, :2] = (diff / norm_diff).reshape(1, 2)   
            Bc_t[2] = np.array([norm_diff - r_safe])

            # robot corner constraint
            R_theta = np.array([np.cos(x_opt[i, 2])*l, np.sin(x_opt[i, 2])*l])
            endpoint_pos_prev = np.hstack([x_opt[i, 0], x_opt[i, 1]]) + R_theta
            diff2 = x_obs2 - endpoint_pos_prev
            norm_diff2 = np.linalg.norm(diff2, 2)
            Ac_t[3, :2] = (diff2 / norm_diff2).reshape(1, 2)   
            Bc_t[3] = np.array([norm_diff2 - r_safe])


        Acs.append(Ac_t)
        bcs.append(Bc_t)
    return Acs, bcs

def get_dyn(xold_, uold_, dt, steps, x_dim_):

    Ad_, Bd_ = [], []
    for i in range(steps):
        xold = xold_[i]
        uold = uold_[i]
        th = xold[2]
        u1 = uold[0]

        A = np.zeros((x_dim_, x_dim_))
        A[0, 2] = -u1*np.sin(th)
        A[1, 2] = u1*np.cos(th)
  


        B = np.zeros((x_dim_, 2))
        B[0, 0] = np.cos(th)
        B[1, 0] = np.sin(th)
        B[2, 1] = 1



        Ad = A*dt+np.eye(x_dim_)
        Bd = B*dt


        Ad_.append(Ad)
        Bd_.append(Bd)

    return Ad_, Bd_

class SimpleGuess:
    def __init__(self, steps, x_dim, R, Qf, Q, input_lims, dt):
        self.x_dim = x_dim
        self.R = R
        self.Qf = Qf
        self.Q =  Q
        self.N = steps
        self.input_lims = input_lims
        self.SCP_wo_obs = SCPSubproblem(steps=self.N, x_dim=x_dim, u_dim=2, R=self.R, Q=self.Q, Qf=self.Qf, use_u_bound=False, max_state_halfspaces=None, Trust_region = False)
        self.dt = dt
        self.T = dt*steps
    def find_initialGuess(self, x0, ud, goLength):

        x0_new = x0.copy()
        # theta = x0_new[2]    
        # x_prev = np.linspace(x0_new, x0_new + [np.cos(theta)*goLength, np.sin(theta)*goLength, 0], self.N+1)  # Nominal positions
        # # u_prev = np.linspace([0, 0], [0, 0], self.N)
        # u_prev = np.linspace([goLength/self.T , 0], [goLength/self.T, 0], self.N)
        theta = x0_new[2]    
        x_prev = np.linspace(x0_new, x0_new, self.N+1)  # Nominal positions
        # u_prev = np.linspace([0, 0], [0, 0], self.N)
        u_prev = np.linspace([0 , 0], [0, 0], self.N)

        x_opt, u_opt = x_prev, u_prev

        for i in range(5):
            Ad, Bd = get_dyn(x_prev, u_prev, self.dt, self.N, x_dim_=self.x_dim)
            x_opt, u_opt, cost = self.SCP_wo_obs.solve(x0=x0_new,
                                                            xT=x0_new,
                                                            ud=ud,
                                                            xnom=np.array(x_prev),
                                                            unom=np.array(u_prev),
                                                            As=Ad, Bs=Bd, 
                                                            u_bound=self.input_lims, 
                                                            Acs=None, bcs=None)
            x_prev, u_prev = x_opt, u_opt
        return x_opt, u_opt


