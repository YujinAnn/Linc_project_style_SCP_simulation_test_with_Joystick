import warnings

import cvxpy as cp
import numpy as np


_SOLVER = "OSQP"


class SCPSubproblem:
    """Solver for Sequential Convex Programming subproblems.

    Uses CVXPY's 'parameters' feature so the reduction from modeling language
    to solver canonical form can be reused between invocations. This gives
    significant speedup.

    Currently only supports final-state cost (not constraint!) and quadratic
    input cost. More parameters can be surfaced to the user if needed.
    Optionally uses CVXPYgen to generate C code for the solver, giving major
    additional speedup.

    Based on a simplified version of Method 2 in the paper:
    Optimal Guidance and Control with Nonlinear Dynamics Using Sequential Convex Programming.
    Rebecca Foust, Soon-Jo Chung, Fred Y. Hadaegh.
    Journal of Guidance, Control, and Dynamics, Dec. 2019.

    Args:
        steps (int): Number of discrete-time dynamics steps.
        x_dim (int): State dimension.
        u_dim (int): Action dimension.
        R (array): Quadratic cost weight for actions. Must be (u_dim, u_dim) shape and PSD.
        Q (array, optional): Quadratic cost for trajectory tracking. Must be (x_dim, x_dim) shape.
        Qf (array, optional): Quadratic cost for end-of-horizon tracking cost. Must be (x_dim, x_dim) shape.
        u_bound (array, optional): Elementwise bound: |u[t, i]| <= u_bound[i] for all times t and dimensions i.
        u_diff_bound (array, optional): Elementwise bound: |u[t, i] - u[t+1, i]| <= u_bound[i] for all times t and dimensions i.
        use_u_norm1_bound (bool, optional): Whether to include the bound |v+s*w| <= c in the optimization problem.
        max_state_halfspaces: Number of halfspaces for state constraints.
        trust_x (float, optional): Infinity-norm trust region on state variables.
        trust_u (float, optional): Infinity-norm trust region on action variables.
    """
    def __init__(self, steps, x_dim, u_dim, R, Q=None, Qf=None, use_u_bound=False, max_state_halfspaces=None, trust_x=1.0, trust_u=1.0, Trust_region = True):
        self.max_state_halfspaces = max_state_halfspaces
        self.use_u_bound = use_u_bound
        if Q is None and Qf is None:
            warnings.warn("SCP: Q and Qf are both None.")

        n, m = x_dim, u_dim

        # Initial and final states.
        self.x0 = cp.Parameter((n), name="x0")
        self.xT = cp.Parameter((n), name="xT")
        self.ud = cp.Parameter((m), name="ud")


        # Nominal solution.
        self.xnom = cp.Parameter((steps + 1, n), name="xnom")
        self.unom = cp.Parameter((steps, m), name="unom")

        # Dynamics Jacobians linearized about nominal solution.
        self.A = [cp.Parameter((n, n), name=f"A_{i}") for i in range(steps)]
        self.B = [cp.Parameter((n, m), name=f"b_{i}") for i in range(steps)]

        # Decision variables: Changes in x and u from nominal.
        self.dx = cp.Variable((steps + 1, n))
        self.du = cp.Variable((steps, m))
        x = self.xnom + self.dx
        u = self.unom + self.du

        # Constraints...
        constraints = []

        # ...Initial state...
        constraints.append(x[0] == self.x0)
        
        # ...Dynamics...
        for t in range(steps):
            constraints.append(
                self.dx[t + 1]
                ==
                self.A[t] @ self.dx[t] + self.B[t] @ self.du[t]
            )

        # ...Input box bounds...
        if use_u_bound:
            self.u_bound = cp.Parameter((2,2), name="u_bound")
            constraints.append(u[:,0] >= self.u_bound[0,0])
            constraints.append(u[:,0] <= self.u_bound[0,1])
            constraints.append(u[:,1] >= self.u_bound[1,0])
            constraints.append(u[:,1] <= self.u_bound[1,1])
        else:
            self.u_bound = None

        # ...State constraints...
        if max_state_halfspaces is not None:
            r = max_state_halfspaces
            self.Ac = [cp.Parameter((r, n), name=f"Ac_{i}") for i in range(steps+1)]
            self.bc = [cp.Parameter(r, name=f"bc_{i}") for i in range(steps+1)]
            for t in range(steps + 1):
                constraints.append(self.Ac[t] @ self.dx[t] <= self.bc[t])

        # ...Trust region. TODO: Adaptive? Scheduled?
        if Trust_region == True:
            constraints.append(cp.abs(self.dx) <= trust_x)
            constraints.append(cp.abs(self.du) <= trust_u)

        # Cost function...
        cost = 0

        # ...Input costs...
        if R is not None:
            for t in range(steps):
                cost += cp.quad_form(u[t]- self.ud, R)

        # ....State costs:
        if Q is not None:
            for t in range(steps+1):
                # cost += cp.quad_form(x[t] - self.xnom[t], Q)
                cost += cp.quad_form(x[t] - self.xT, Q)

        # ...Final state penalty.
        if Qf is not None:
            cost += cp.quad_form(x[steps] - self.xT, Qf)

        # Compile and save for later.
        self.problem = cp.Problem(cp.Minimize(cost), constraints)
        assert self.problem.is_dcp(dpp=True)
        self.is_codegen = False

    def solve(self, x0, xT, ud, xnom, unom, As, Bs, uprev=None, u_bound=None, 
              tipover_rhs=None, tipover_omega_sign=None, u_norm1_bound=None, Acs=None, bcs=None, cvxpy_kwargs={}):
        """Solve a particular subproblem.

        Arguments `As` and `Bs` should come from the nonlinear dynamics
        linearized about the nominal solution (xnom, unom).

        Args:
            x0 (array): Initial state (x_dim).
            xT (array): Final target state (x_dim).
            xnom (array): Nominal state trajectory, (steps + 1, x_dim).
            unom (array): Nominal state trajectory, (steps, u_dim).
            As (array): Dynamics Jacobians w.r.t. state, (steps, x_dim, x_dim).
            Bs (array): Dynamics Jacobians w.r.t. action, (steps, x_dim, u_dim).
            xdes (array): Target state trajectory, (steps, xdim).
                          Only needed if Q != None in constructor.
            uprev (array): Previous u that was just executed, (udim).
                           Only needed if u_diff_bound != None in constructor.
            u_bound (array): Input limits, (udim, 2)
            tipover_rhs (float): Constant c for the bound |v + s*w| <= c.
                                   Only required if use_unorm1_bound = True in the constructor.
            tipover_omega_sign (float): Sign of the angular velocity w in the bound |v + s*w| <= c.
                                   Only required if use_unorm1_bound = True in the constructor.
                                   Should only ever take value 1 or -1.
            Acs (array): LHS matrices of per-step state constraints.
            bcs (array): RHS vectors of per-step state constraints.

        Returns:
            xs (array): New state trajectory, (steps + 1, x_dim).
            us (array): New action trajectory, (steps, u_dim).
            cost (float): Cost value.
        """

        if u_bound is None or self.use_u_bound is False:
            assert self.u_bound is None
        else:
            self.u_bound.save_value(u_bound)
        # assert np.allclose(x0, xnom[0])
        self.x0.save_value(x0)
        self.xT.save_value(xT)
        self.ud.save_value(ud)
        self.xnom.save_value(xnom)
        self.unom.save_value(unom)
        for Ap, A in zip(self.A, As):
            Ap.save_value(A)
        for Bp, B in zip(self.B, Bs):
            Bp.save_value(B)
        
        # State constraints.
        if self.max_state_halfspaces is not None:
            if Acs is not None:
                assert bcs is not None
                assert self.Ac is not None
                for Acp, bcp, Ac, bc, x in zip(self.Ac, self.bc, Acs, bcs, xnom):
                    Acp.save_value(Ac)
                    bcp.save_value(bc)

        
        if self.is_codegen:
            assert "method" not in cvxpy_kwargs
            cost = self.problem.solve(method="cpg", **cvxpy_kwargs)
        else:
            cost = self.problem.solve(solver=_SOLVER, **cvxpy_kwargs)


        
        # import matplotlib.pyplot as plt
        # t=99
        # a, b = Acs[t][0, :2]
        # c = bcs[t]

        
        # # Generate x-coordinates
        # x_points = np.linspace(-10, 10, 400)

        # # Solve for y-coordinates
        # y_points = (c - a * x_points) / b

        # # Plotting the line
        # plt.plot(x_points, y_points, label=f'Line: {a}*x + {b}*y = {c}')
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.title('Line Plot')
        # plt.legend()
        # plt.grid(True)

        # # Save the figure
        # plt.savefig('line_plot.png')  # Save as PNG file

        # print("Problem status: ", self.problem.status)

        # xnew = xnom + self.dx.value
        # plt.plot(xnew[:,0], xnew[:,1])
        # plt.savefig('line_plot.png')

        return xnom + self.dx.value, unom + self.du.value, cost


    def codegen(self):
        """Generate and compile solver code using CVXPYgen.

        Can be much faster, e.g. 5x. Implements caching on disk to avoid
        recompiling the same problem repeatedly.
        """
        import importlib
        import os
        import pickle
        from cvxpygen import cpg

        # Implement a 1-element cache on disk. CVXPYgen automatically pickles
        # our problem, so we build upon that.
        cachekey = pickle.dumps(self.problem)
        path = "codegen/problem.pickle"
        if (os.path.exists(path) and open(path, "rb").read() == cachekey):
            print("Found cached codegen, using.")
        else:
            cpg.generate_code(self.problem, code_dir="codegen", solver=_SOLVER)
            open(path, "wb").write(cachekey)

        solver = importlib.import_module("codegen.cpg_solver")
        self.problem.register_solve("cpg", solver.cpg_solve)
        self.is_codegen = True