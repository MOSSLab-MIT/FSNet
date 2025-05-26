import numpy as np
import pickle
import casadi as ca

num_var = 200
num_ineq = 100
num_eq = 100
num_examples = 20000
seed = 2025

np.random.seed(seed)
Q = np.diag(np.random.rand(num_var)*0.5)
p = np.random.uniform(-1, 1, num_var)
A = np.random.uniform(-1, 1, size=(num_eq, num_var))
X = np.random.uniform(-1, 1, size=(num_examples, num_eq))
XL = X.min(axis=0)
XU = X.max(axis=0)
G = np.random.uniform(-1, 1, size=(num_ineq, num_var))
h = np.sum(np.abs(G@np.linalg.pinv(A)), axis=1)
H = np.random.uniform(0, 0.1,  size=(num_ineq, num_var))
H = [np.diag(H[i]) for i in range(num_ineq)]
H = np.array(H)
L = np.ones((num_var)) * -5
U = np.ones((num_var)) * 5
data = {'Q':Q,
        'p':p,
        'A':A,
        'X':X,
        'G':G,
        'H':H,
        'h':h,
        'YL':L,
        'YU':U,
        'XL':XL,
        'XU':XU,
        'Y':[]}
Y = []

for n in range(num_examples):
    Xi = X[n]
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
    # check if the solver converged
    if solver.stats()['success']:
        sol_x = res['x'].full().flatten()
        Y.append(sol_x)
    else:
        print("Solver failed to converge")
        break
    if n % 500 == 0:
        print("Example {}: Objective value: {}".format(n, res['f'].full().flatten()[0]))
data['Y'] = np.array(Y)

i = 0
det_min = 0
best_partial = 0
while i < 1000:
    np.random.seed(i)
    partial_vars = np.random.choice(num_var, num_var - num_eq, replace=False)
    other_vars = np.setdiff1d(np.arange(num_var), partial_vars)
    _, det = np.linalg.slogdet(A[:, other_vars])
    if det>det_min:
        det_min = det
        best_partial = partial_vars
    i += 1
print('best_det', det_min)
data['best_partial'] = best_partial



with open("datasets/nonconvex/qcqp/random{}_qcqp_dataset_var{}_ineq{}_eq{}_ex{}".format(seed, num_var, num_ineq, num_eq, num_examples), 'wb') as f:
    pickle.dump(data, f)