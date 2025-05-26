import numpy as np
import time

import torch
from torch.utils.data import TensorDataset, random_split

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import casadi as ca
from qpth.qp import QPFunction

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_dtype(torch.float64)



###################################################################
# Base PROBLEM
###################################################################
class BaseProblem:
    def __init__(self, dataset, val_size, test_size, seed):
        self.input_L = torch.tensor(dataset['XL'] )
        self.input_U = torch.tensor(dataset['XU'] )
        self.L = torch.tensor(dataset['YL'] )
        self.U = torch.tensor(dataset['YU'] )
        self.X = torch.tensor(dataset['X'] )
        self.Y = torch.tensor(dataset['Y'] )
        self.num = dataset['X'].shape[0]
        self.device = DEVICE

        total_size = self.X.shape[0]
        train_size = int(total_size  - val_size - test_size)
        full_dataset = TensorDataset(self.X, self.Y)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(full_dataset, [train_size, val_size, test_size],
                                                                                generator=torch.Generator().manual_seed(seed))
        
    def eq_grad(self, X, Y):
        # Create a copy of Y that requires gradients for the whole batch
        Y_grad = Y.clone().detach().requires_grad_(True)
        
        # Compute equality residuals and their squares for the whole batch
        eq_resid = self.eq_resid(X, Y_grad) ** 2
        eq_penalty = torch.sum(eq_resid, dim=1, keepdim=True)
        
        # Compute gradients for all samples at once
        grad = torch.autograd.grad(
            outputs=eq_penalty,
            inputs=Y_grad,
            grad_outputs=torch.ones_like(eq_penalty),
            create_graph=False,
            retain_graph=False,
            only_inputs=True
        )[0]
        
        return grad

    def ineq_grad(self, X, Y):
        # Create a copy of Y that requires gradients
        Y_grad = Y.clone().detach().requires_grad_(True)
        
        # Compute inequality residuals and their squares for the whole batch
        ineq_resid = self.ineq_resid(X, Y_grad) ** 2
        ineq_penalty = torch.sum(ineq_resid, dim=1, keepdim=True)
        
        # Compute gradients for all samples at once
        grad = torch.autograd.grad(
            outputs=ineq_penalty,
            inputs=Y_grad,
            grad_outputs=torch.ones_like(ineq_penalty),
            create_graph=False,
            retain_graph=False,
            only_inputs=True
        )[0]
        
        return grad

    def scale_full(self, Y):
        lower_bound = self.L.view(1, -1)
        upper_bound = self.U.view(1, -1)
        # The last layer of NN is sigmoid, scale to Opt bound
        scale_Y = Y * (upper_bound - lower_bound) + lower_bound
        return scale_Y

    def scale_partial(self, Y):
        lower_bound = (self.L[self.partial_vars]).view(1, -1)
        upper_bound = (self.U[self.partial_vars]).view(1, -1)
        scale_Y = Y * (upper_bound - lower_bound) + lower_bound
        return scale_Y

    def scale(self, Y):
        if Y.shape[1] < self.ydim:
            Y_scale = self.scale_partial(Y)
        else:
            Y_scale = self.scale_full(Y)
        return Y_scale

    def cal_penalty(self, X, Y):
        penalty = torch.cat([self.ineq_resid(X, Y), self.eq_resid(X, Y)], dim=1)
        return torch.abs(penalty)

    def check_feasibility(self, X, Y):
        return self.cal_penalty(X, Y)


###################################################################
# QP PROBLEM
###################################################################
class QPProblem(BaseProblem):
    """
        minimize_y 1/2 * y^T Q y + p^Ty
        s.t.       Ay =  x
                   Gy <= h
                   L<= x <=U
    """
    def __init__(self, dataset, val_size, test_size, seed):
        super().__init__(dataset, val_size, test_size, seed)
        self.Q_np = dataset['Q']
        self.p_np = dataset['p']
        self.A_np = dataset['A']
        self.G_np = dataset['G']
        self.h_np = dataset['h']
        self.L_np = dataset['YL']
        self.U_np = dataset['YU']
        # self.X_np = dataset['X']
        self.Q = torch.tensor(dataset['Q'] )
        self.p = torch.tensor(dataset['p'] )
        self.A = torch.tensor(dataset['A'] )
        self.G = torch.tensor(dataset['G'] )
        self.h = torch.tensor(dataset['h'] )
        self.L = torch.tensor(dataset['YL'] )
        self.U = torch.tensor(dataset['YU'] )
        self.X = torch.tensor(dataset['X'] )
        self.Y = torch.tensor(dataset['Y'] )
        self.xdim = dataset['X'].shape[1]
        self.ydim = dataset['Q'].shape[0]
        self.neq = dataset['A'].shape[0]
        self.nineq = dataset['G'].shape[0]
        self.nknowns = 0

        best_partial = dataset['best_partial']
        self.partial_vars = best_partial
        self.partial_unknown_vars = best_partial
        self.other_vars = np.setdiff1d(np.arange(self.ydim), self.partial_vars)
        self.A_partial = self.A[:, self.partial_vars]
        self.A_other_inv = torch.inverse(self.A[:, self.other_vars])
        

    def __str__(self):
        return 'QPProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )

    def obj_fn(self, Y):
        return (0.5 * (Y @ self.Q) * Y + self.p * Y).sum(dim=1)
    
    def eq_resid(self, X, Y):
        return Y @ self.A.T - X
    
    def ineq_resid(self, X, Y):
        res = Y @ self.G.T - self.h.view(1, -1)
        l = self.L - Y
        u = Y - self.U
        resids = torch.cat([res, l, u], dim=1)
        return torch.clamp(resids, 0)

    def complete_partial(self, X, Y, backward=True):
        Y_full = torch.zeros(X.shape[0], self.ydim, device=X.device)
        Y_full[:, self.partial_vars] = Y
        Y_full[:, self.other_vars] = (X - Y @ self.A_partial.T) @ self.A_other_inv.T
        return Y_full   

   
    def opt_solve(self, X, accuracy='default', tol=5e-3):
        Q, p, A, G, h, L, U = \
            self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np, self.L_np, self.U_np
        X_np = X.detach().cpu().numpy()
        Y = []
        objs = []
        total_time = 0
        n = 0
        for Xi in X_np:
            y = cp.Variable(self.ydim)
            prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(y, Q) + p.T @ y),
                                [G @ y <= h, y <= U, y >= L,
                                A @ y == Xi])
            start_time = time.time()
            if accuracy == 'default':
                prob.solve()
            elif accuracy == 'reduced':
                prob.solve(
                    solver = 'OSQP',
                    eps_rel = tol,
                    eps_abs = tol,
                )
            else:
                raise NotImplementedError                
            end_time = time.time()
            print(n, np.max(y.value), np.min(y.value), y.value[0:5].T, end='\r')
            n += 1
            Y.append(y.value)
            objs.append(prob.value)
            total_time += (end_time - start_time)
        sols = np.array(Y)
        objs = np.array(objs)
        
        return sols, objs, total_time


    def qpth_projection(self, X, Y):
        batch_size = X.shape[0]
        n = self.ydim
        device = X.device
        dtype = X.dtype
        
        # Identity matrix for quadratic term (squared distance objective)
        Q = torch.eye(n, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, n, n)
        
        # Linear term: -y_pred
        p = -Y
                # Prepare inequality constraints: [G; I; -I] y <= [h; U; -L]
        G_top = self.G.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, self.nineq, n)
        G_middle = torch.eye(n, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, n, n)
        G_bottom = -torch.eye(n, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, n, n)
        G_combined = torch.cat([G_top, G_middle, G_bottom], dim=1)
        
        h_top = self.h.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, self.nineq)
        h_middle = self.U.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, n)
        h_bottom = -self.L.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, n)
        h_combined = torch.cat([h_top, h_middle, h_bottom], dim=1)
        
        # Use QPFunction to solve the projection problem
        Y_proj = QPFunction(verbose=-1)(Q, p, G_combined, h_combined, self.A, X)
        
        return Y_proj


    # def init_cvx_projection(self):
    #     ### for cvxpy projection
    #     y = cp.Variable(self.ydim)
    #     x = cp.Parameter(self.X.shape[1])
    #     y_pred = cp.Parameter(self.Y.shape[1])
    #     constraints = [self.G_np @ y <= self.h_np, y <= self.U_np, y >= self.L_np,
    #                                self.A_np @ y == x]
    #     prob = cp.Problem(cp.Minimize(cp.sum_squares(y - y_pred)), constraints)

    #     self.cvx_projection = CvxpyLayer(prob, parameters=[x, y_pred], variables=[y])
        ###

            
###################################################################
# QCQP Problem
###################################################################
class QCQPProblem(QPProblem):
    """
        minimize_y 1/2 * y^T Q y + p^Ty
        s.t.       Ay =  x
                   1/2 * y^T H y + G^T y <= h
                   L<= x <=U
    """
    def __init__(self, dataset, val_size, test_size, seed):
        super().__init__(dataset, val_size, test_size, seed)
        self.H_np = dataset['H']
        self.H = torch.tensor(dataset['H'] )

    def __str__(self):
        return 'QCQPProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )
    
    def ineq_resid(self, X, Y):
        res = []
        q = torch.matmul(self.H, Y.T).permute(2, 0, 1)
        q = (q * Y.view(Y.shape[0], 1, -1)).sum(-1)
        res = 0.5 * q + torch.matmul(Y, self.G.T) - self.h
        l = self.L - Y
        u = Y - self.U
        resids = torch.cat([res, l, u], dim=1)
        return torch.clamp(resids, 0)

    def opt_solve(self, X, accuracy='default', tol=1e-5):
        Q, p, A, G, H, h, L, U = \
            self.Q_np, self.p_np, self.A_np, self.G_np, self.H_np, self.h_np, self.L_np, self.U_np
        X_np = X.detach().cpu().numpy()
        Y = []
        objs = []
        total_time = 0
        n = 0
        for Xi in X_np:
            y = cp.Variable(self.ydim)
            constraints = [A @ y == Xi, y <= U, y >= L]
            for i in range(self.nineq):
                Ht = H[i]
                Gt = G[i]
                ht = h[i]
                constraints.append(0.5 * cp.quad_form(y, Ht) + Gt.T @ y <= ht)
            prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(y, Q) + p.T @ y),
                                constraints)
            start_time = time.time()
            if accuracy == 'default':
                prob.solve(solver = 'SCS')
            elif accuracy == 'reduced':
                prob.solve(
                    solver = 'SCS',
                    eps_rel = tol,
                    eps_abs = tol,
                )
            else:
                raise NotImplementedError
            end_time = time.time()
            print(n, np.max(y.value), np.min(y.value), y.value[0:5].T, end='\r')
            n += 1
            Y.append(y.value)
            objs.append(prob.value)
            total_time += (end_time - start_time)

        sols = np.array(Y)
        objs = np.array(objs)

        return sols, objs, total_time

    # def init_cvx_projection(self):
    #     ### for cvxpy projection
    #     y = cp.Variable(self.ydim)
    #     x = cp.Parameter(self.X.shape[1])
    #     y_pred = cp.Parameter(self.Y.shape[1])
    #     constraints = [self.A_np @ y == x, y <= self.U_np, y >= self.L_np]
    #     for i in range(self.nineq):
    #         constraints.append(0.5*cp.quad_form(y, self.H_np[i]) + self.G_np[i].T @ y <= self.h_np[i])

    #     prob = cp.Problem(cp.Minimize(cp.sum_squares(y - y_pred)), constraints)

    #     self.cvx_projection = CvxpyLayer(prob, parameters=[x, y_pred], variables=[y])
        

###################################################################
# SOCP Problem
###################################################################
class SOCPProblem(QPProblem):
    """
        minimize_y p^Ty
        s.t.       Ay =  x
                   ||G^T y + h||_2 <= c^Ty+d
                   L<= x <=U
    """

    def __init__(self, dataset, val_size, test_size, seed):
        super().__init__(dataset, val_size, test_size, seed)
        self.C_np = dataset['C']
        self.d_np = dataset['d']
        self.C = torch.tensor(dataset['C'] )
        self.d = torch.tensor(dataset['d'] )

    def __str__(self):
        return 'SOCPProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )
    
    def ineq_resid(self, X, Y):
        res = []
        q = torch.norm(torch.matmul(self.G, Y.T).permute(2, 0, 1) + self.h.unsqueeze(0), dim=-1, p=2)
        p = torch.matmul(Y, self.C.T) + self.d
        res = q - p
        l = self.L - Y
        u = Y - self.U
        resids = torch.cat([res, l, u], dim=1)
        return torch.clamp(resids, 0)

    def opt_solve(self, X, accuracy='default', tol=1e-5):
        Q, p, A, G, h, C, d, L, U = \
            self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np, self.C_np, self.d_np, self.L_np, self.U_np
        X_np = X.detach().cpu().numpy()
        Y = []
        objs = []
        total_time = 0
        n = 0
        for Xi in X_np:
            y = cp.Variable(self.ydim)
            soc_constraints = [cp.SOC(C[i].T @ y + d[i], G[i] @ y + h[i]) for i in range(self.nineq)]
            constraints = soc_constraints + [A @ y == Xi] + [y <= U] + [y >= L]
            prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(y, Q) + p.T @ y), constraints)
            start_time = time.time()
            if accuracy == 'default':
                prob.solve()
            elif accuracy == 'reduced':
                prob.solve(
                    solver = 'SCS',
                    eps_rel = tol,
                    eps_abs = tol,
                )
            else:
                raise NotImplementedError
            
            end_time = time.time()
            print(n, np.max(y.value), np.min(y.value), y.value[0:5].T, end='\r')
            n += 1
            Y.append(y.value)
            objs.append(prob.value)
            total_time += (end_time - start_time)

        sols = np.array(Y)
        objs = np.array(objs)
        
        return sols, objs, total_time
    
    # def init_cvx_projection(self):
    #     ## for cvxpy projection
    #     y = cp.Variable(self.ydim)
    #     x = cp.Parameter(self.X.shape[1])
    #     y_pred = cp.Parameter(self.Y.shape[1])
    #     soc_constraints = [cp.SOC(self.C_np[i].T @ y + self.d_np[i], self.G_np[i] @ y + self.h_np[i]) for i in range(self.nineq)]
    #     constraints = [self.A_np @ y == x, y <= self.U_np, y >= self.L_np] + soc_constraints
    #     prob = cp.Problem(cp.Minimize(cp.sum_squares(y - y_pred)), constraints)
    #     self.cvx_projection = CvxpyLayer(prob, parameters=[x, y_pred], variables=[y])





###################################################################
# NONCONVEX PROBLEM
###################################################################
class nonconvexQPProblem(QPProblem):
    def __str__(self):
        return 'QPProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )
    
    def obj_fn(self, Y):
        return (0.5 * (Y @ self.Q) * Y + self.p * torch.sin(Y)).sum(dim=1)
    
    def ineq_resid(self, X, Y):
        res = torch.sin(Y) @ self.G.T - self.h.view(1, -1)*(torch.cos(X))
        l = self.L - Y
        u = Y - self.U
        resids = torch.cat([res, l, u], dim=1)
        return torch.clamp(resids, 0)

    def opt_solve(self, X, accuracy='default', tol=1e-5):
        Q, p, A, G, h, L, U = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np, self.L_np, self.U_np
        X_np = X.detach().cpu().numpy()
        Y = []
        objs = []
        num_var = self.ydim
        num_eq = self.neq
        num_ineq = self.nineq
        start_time = time.time()
        print('running nonconvex qp', end='\r')
        for Xi in X_np:
            y = ca.MX.sym('y_var', num_var)
            obj_func = 0.5 * ca.mtimes(y.T, ca.mtimes(Q, y)) + ca.dot(p, ca.sin(y))
            eq_constraints = A @ y - Xi
            ineq_constraints = G @ ca.sin(y) - h*(ca.cos(Xi))
            # ineq_constraints = G @ y - h
            nlp = {'x': y, 'f': obj_func, 'g': ca.vertcat(eq_constraints, ineq_constraints)}
            opts = {'ipopt.print_level': 0, 'print_time': 0}
            solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
            # Define bounds for variables and constraints
            lbg = np.concatenate([np.zeros(num_eq), -np.inf * np.ones(num_ineq)])
            ubg = np.concatenate([np.zeros(num_eq), np.zeros(num_ineq)])
            lbx = L
            ubx = U
            # Solve the NLP
            res = solver(lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
            sol_x = res['x'].full().flatten().tolist()
            obj = res['f'].full()[0,0]
            Y.append(sol_x)
            objs.append(obj)
        end_time = time.time()
        total_time = end_time - start_time
        return np.array(Y), np.array(objs), total_time
        

class nonconvexQCQPProblem(QCQPProblem):
    def __str__(self):
        return 'QCQPProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )
    
    def obj_fn(self, Y):
        return (0.5 * (Y @ self.Q) * Y + self.p * torch.sin(Y)).sum(dim=1) 
    
    def ineq_resid(self, X, Y):
        res = []    
        q = torch.matmul(self.H, Y.T).permute(2, 0, 1)
        q = (q * Y.view(Y.shape[0], 1, -1)).sum(-1)
        res = 0.5 * q + torch.matmul(torch.cos(Y), self.G.T) - self.h
        l = self.L - Y
        u = Y - self.U
        resids = torch.cat([res, l, u], dim=1)
        return torch.clamp(resids, 0)
    
    def opt_solve(self, X, accuracy='default', tol=1e-5):
        Q, p, A, G, H, h, L, U = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.H_np, self.h_np, self.L_np, self.U_np
        X_np = X.detach().cpu().numpy()
        Y = []
        objs = []
        num_var = self.ydim
        num_eq = self.neq
        num_ineq = self.nineq
        start_time = time.time()
        print('running nonconvex qcqp', end='\r')
        for Xi in X_np:
            y = ca.MX.sym('y_var', num_var)
            obj_func = 0.5 * ca.mtimes(y.T, ca.mtimes(Q, y)) + ca.dot(p, ca.sin(y))

            eq_constraints = A @ y - Xi
            ineq_constraints = []
            for i in range(num_ineq):
                ineq_constraints.append(0.5 * ca.mtimes(y.T, ca.mtimes(H[i], y)) + ca.dot(G[i], ca.cos(y)) - h[i])
            ineq_constraints = ca.vertcat(*ineq_constraints)

            nlp = {'x': y, 'f': obj_func, 'g': ca.vertcat(eq_constraints, ineq_constraints)}
            opts = {'ipopt.print_level': 0, 'print_time': 0}
            solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
            # Define bounds for variables and constraints
            lbg = np.concatenate([np.zeros(num_eq), -np.inf * np.ones(num_ineq)])
            ubg = np.concatenate([np.zeros(num_eq), np.zeros(num_ineq)])
            lbx = L
            ubx = U
            # Solve the NLP
            res = solver(lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
            sol_x = res['x'].full().flatten().tolist()
            obj = res['f'].full()[0,0]
            Y.append(sol_x)
            objs.append(obj)
        end_time = time.time()
        total_time = end_time - start_time
        return np.array(Y), np.array(objs), total_time


class nonconvexSOCPProblem(SOCPProblem):

    def __str__(self):
        return 'SOCPProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )
    
    def obj_fn(self, Y):
        return (0.5 * (Y @ self.Q) * Y + self.p * torch.sin(Y)).sum(dim=1)
    
    def ineq_resid(self, X, Y):
        res = []
        q = torch.norm(torch.matmul(self.G, torch.cos(Y).T).permute(2, 0, 1) + self.h.unsqueeze(0), dim=-1, p=2)
        p = torch.matmul(Y, self.C.T) + self.d
        res = q - p
        l = self.L - Y
        u = Y - self.U
        resids = torch.cat([res, l, u], dim=1)
        return torch.clamp(resids, 0)
 
    def opt_solve(self, X, accuracy='default', tol=1e-5):
        Q, p, A, G, h, C, d, L, U = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np, self.C_np, self.d_np, self.L_np, self.U_np
        X_np = X.detach().cpu().numpy()
        Y = []
        objs = []
        num_var = self.ydim
        num_eq = self.neq
        num_ineq = self.nineq
        start_time = time.time()
        print('running nonconvex socp', end='\r')
        for Xi in X_np:
            y = ca.MX.sym('y_var', num_var)
            obj_func = 0.5 * ca.mtimes(y.T, ca.mtimes(Q, y)) + ca.dot(p, ca.sin(y))

            eq_constraints = A @ y - Xi

            ineq_constraints = []
            for i in range(num_ineq):
                ineq_constraints.append(ca.norm_2(G[i] @ ca.cos(y) + h[i]) - (ca.dot(C[i], y) + d[i]))
            ineq_constraints = ca.vertcat(*ineq_constraints)
            
            nlp = {'x': y, 'f': obj_func, 'g': ca.vertcat(eq_constraints, ineq_constraints)}
            opts = {'ipopt.print_level': 0, 'print_time': 0}
            solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
            # Define bounds for variables and constraints
            lbg = np.concatenate([np.zeros(num_eq), -np.inf * np.ones(num_ineq)])
            ubg = np.concatenate([np.zeros(num_eq), np.zeros(num_ineq)])
            lbx = L
            ubx = U
            # Solve the NLP
            res = solver(lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
            sol_x = res['x'].full().flatten().tolist()
            obj = res['f'].full()[0]
            Y.append(sol_x)
            objs.append(obj)
        end_time = time.time()
        total_time = end_time - start_time
        return np.array(Y), np.array(objs), total_time


###################################################################
# NONSMOOTH NONCONVEX
###################################################################
class nonsmooth_nonconvexQPProblem(QPProblem):
    def __str__(self):
        return 'QPProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )
    
    def obj_fn(self, Y):
        return (0.5 * (Y @ self.Q) * Y + self.p * torch.sin(Y)).sum(dim=1) + 0.1*torch.norm(Y, dim=1)
    
    def ineq_resid(self, X, Y):
        res = torch.sin(Y) @ self.G.T - self.h.view(1, -1)*(torch.cos(X))
        l = self.L - Y
        u = Y - self.U
        resids = torch.cat([res, l, u], dim=1)
        return torch.clamp(resids, 0)
    
    def opt_solve(self, X, accuracy='default', tol=1e-5):
        Q, p, A, G, h, L, U = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np, self.L_np, self.U_np
        X_np = X.detach().cpu().numpy()
        Y = []
        objs = []
        num_var = self.ydim
        num_eq = self.neq
        num_ineq = self.nineq
        start_time = time.time()
        print('running nonconvex qp', end='\r')
        for Xi in X_np:
            y = ca.MX.sym('y_var', num_var)
            t = ca.MX.sym('t_var')
            obj_func = 0.5 * ca.mtimes(y.T, ca.mtimes(Q, y)) + ca.dot(p, ca.sin(y)) + 0.1*t
            eq_constraints = A @ y - Xi
            ineq_constraints = G @ ca.sin(y) - h*(ca.cos(Xi))
            soc = ca.dot(y, y) - t**2
            # ineq_constraints = G @ y - h
            nlp = {'x': ca.vertcat(y, t), 'f': obj_func, 'g': ca.vertcat(eq_constraints, ineq_constraints, soc)}
            opts = {'ipopt.print_level': 0, 'print_time': 0}
            solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
            # Define bounds for variables and constraints
            lbg = np.concatenate([np.zeros(num_eq), -np.inf * np.ones(num_ineq+1)])
            ubg = np.concatenate([np.zeros(num_eq), np.zeros(num_ineq+1)])
            lbx = np.concatenate([L, [0]])
            ubx = np.concatenate([U, [np.inf]])
            # Solve the NLP
            res = solver(lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
            sol_x = res['x'].full().flatten().tolist()[:-1]
            obj = res['f'].full()[0,0]
            Y.append(sol_x)
            objs.append(obj)
        end_time = time.time()
        total_time = end_time - start_time
        return np.array(Y), np.array(objs), total_time
    

class nonsmooth_nonconvexQCQPProblem(QCQPProblem):
    def __str__(self):
        return 'QCQPProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )
    
    def obj_fn(self, Y):
        return (0.5 * (Y @ self.Q) * Y + self.p * torch.sin(Y)).sum(dim=1) + 0.1*torch.norm(Y, dim=1)
    
    def ineq_resid(self, X, Y):
        res = []    
        q = torch.matmul(self.H, Y.T).permute(2, 0, 1)
        q = (q * Y.view(Y.shape[0], 1, -1)).sum(-1)
        res = 0.5 * q + torch.matmul(torch.cos(Y), self.G.T) - self.h
        l = self.L - Y
        u = Y - self.U
        resids = torch.cat([res, l, u], dim=1)
        return torch.clamp(resids, 0)
    
    def opt_solve(self, X, accuracy='default', tol=1e-5):
        Q, p, A, G, H, h, L, U = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.H_np, self.h_np, self.L_np, self.U_np
        X_np = X.detach().cpu().numpy()
        Y = []
        objs = []
        num_var = self.ydim
        num_eq = self.neq
        num_ineq = self.nineq
        start_time = time.time()
        print('running nonconvex qcqp', end='\r')
        for Xi in X_np:
            y = ca.MX.sym('y_var', num_var)
            t = ca.MX.sym('t_var')
            obj_func = 0.5 * ca.mtimes(y.T, ca.mtimes(Q, y)) + ca.dot(p, ca.sin(y)) + 0.1*t

            eq_constraints = A @ y - Xi
            ineq_constraints = []
            soc = ca.dot(y, y) - t**2
            for i in range(num_ineq):
                ineq_constraints.append(0.5 * ca.mtimes(y.T, ca.mtimes(H[i], y)) + ca.dot(G[i], ca.cos(y)) - h[i])
            ineq_constraints.append(soc)
            ineq_constraints = ca.vertcat(*ineq_constraints)

            nlp = {'x': ca.vertcat(y, t), 'f': obj_func, 'g': ca.vertcat(eq_constraints, ineq_constraints)}
            opts = {'ipopt.print_level': 0, 'print_time': 0}
            solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
            # Define bounds for variables and constraints
            lbg = np.concatenate([np.zeros(num_eq), -np.inf * np.ones(num_ineq+1)])
            ubg = np.concatenate([np.zeros(num_eq), np.zeros(num_ineq+1)])
            lbx = np.concatenate([L, [0]])
            ubx = np.concatenate([U, [np.inf]])
            # Solve the NLP
            res = solver(lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
            sol_x = res['x'].full().flatten().tolist()[:-1]
            obj = res['f'].full()[0,0]
            Y.append(sol_x)
            objs.append(obj)
        end_time = time.time()
        total_time = end_time - start_time
        return np.array(Y), np.array(objs), total_time
 

class nonsmooth_nonconvexSOCPProblem(SOCPProblem):
    def __str__(self):
        return 'SOCPProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )
    def obj_fn(self, Y):
        return (0.5 * (Y @ self.Q) * Y + self.p * torch.sin(Y)).sum(dim=1) + 0.1*torch.norm(Y, dim=1)
    
    def ineq_resid(self, X, Y):
        res = []
        q = torch.norm(torch.matmul(self.G, torch.cos(Y).T).permute(2, 0, 1) + self.h.unsqueeze(0), dim=-1, p=2)
        p = torch.matmul(Y, self.C.T) + self.d
        res = q - p
        l = self.L - Y
        u = Y - self.U
        resids = torch.cat([res, l, u], dim=1)
        return torch.clamp(resids, 0)
    
    def opt_solve(self, X, accuracy='default', tol=1e-5):
        Q, p, A, G, h, C, d, L, U = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np, self.C_np, self.d_np, self.L_np, self.U_np
        X_np = X.detach().cpu().numpy()
        Y = []
        objs = []
        num_var = self.ydim
        num_eq = self.neq
        num_ineq = self.nineq
        start_time = time.time()
        print('running nonconvex socp', end='\r')
        for Xi in X_np:
            y = ca.MX.sym('y_var', num_var)
            t = ca.MX.sym('t_var')

            obj_func = 0.5 * ca.mtimes(y.T, ca.mtimes(Q, y)) + ca.dot(p, ca.sin(y)) + 0.1*t

            eq_constraints = A @ y - Xi
            soc = ca.dot(y, y) - t**2
            ineq_constraints = []
            for i in range(num_ineq):
                ineq_constraints.append(ca.norm_2(G[i] @ ca.cos(y) + h[i]) - (ca.dot(C[i], y) + d[i]))
            ineq_constraints.append(soc)
            ineq_constraints = ca.vertcat(*ineq_constraints)
            
            nlp = {'x': ca.vertcat(y, t), 'f': obj_func, 'g': ca.vertcat(eq_constraints, ineq_constraints)}
            opts = {'ipopt.print_level': 0, 'print_time': 0}
            solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
            # Define bounds for variables and constraints
            lbg = np.concatenate([np.zeros(num_eq), -np.inf * np.ones(num_ineq+1)])
            ubg = np.concatenate([np.zeros(num_eq), np.zeros(num_ineq+1)])
            lbx = np.concatenate([L, [0]])
            ubx = np.concatenate([U, [np.inf]])
            # Solve the NLP
            res = solver(lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
            sol_x = res['x'].full().flatten().tolist()[:-1]
            obj = res['f'].full()[0]
            Y.append(sol_x)
            objs.append(obj)
        end_time = time.time()
        total_time = end_time - start_time
        return np.array(Y), np.array(objs), total_time

# For DC3 correction
def ineq_partial_grad(data, X, Y):
    # Extract partial variables and create a copy that requires gradients
    Y_pred = Y[:, data.partial_vars].clone().detach().requires_grad_(True)
    # Complete to get full Y values for the entire batch at once
    y = data.complete_partial(X, Y_pred)
    # Compute inequality residuals squared (penalty) for the entire batch
    ineq_penalty = data.ineq_resid(X, y) ** 2
    ineq_penalty = torch.sum(ineq_penalty, dim=-1, keepdim=True)
    # Get gradients with respect to Y_pred for the entire batch at once
    grad_pred = torch.autograd.grad(ineq_penalty.sum(), Y_pred)[0]
    # Create the full gradient tensor for all samples
    grad = torch.zeros(Y.shape[0], data.ydim, device=X.device)
    grad[:, data.partial_vars] = grad_pred
    grad[:, data.other_vars] = - (grad_pred @ data.A_partial.T) @ data.A_other_inv.T
    return grad

# Correction for DC3 
def grad_steps(data, X, Y, config):
    lr = config['DC3']['corr_lr']
    max_corr_steps = config['DC3']['max_corr_steps']
    momentum = config['DC3']['corr_momentum']    
    Y_new = Y
    old_Y_step = 0
    for _ in range(max_corr_steps):
        Y_step = ineq_partial_grad(data, X, Y_new)    
        new_Y_step = lr * Y_step + momentum * old_Y_step
        Y_new = Y_new - new_Y_step
        
        old_Y_step = new_Y_step

    return Y_new