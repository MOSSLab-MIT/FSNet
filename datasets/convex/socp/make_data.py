import numpy as np
import pickle
import cvxpy as cp

num_var = 200
num_ineq = 100
num_eq = 100
num_examples = 20000
seed = 2025
print(num_ineq, num_eq)
np.random.seed(seed)
Q = np.diag(np.random.rand(num_var)*0.5)
p = np.random.uniform(-1, 1, num_var)
A = np.random.uniform(-1, 1, size=(num_eq, num_var))
X = np.random.uniform(-1, 1, size=(num_examples, num_eq))
XL = X.min(axis=0)
XU = X.max(axis=0)

L = np.ones((num_var))*-5
U = np.ones((num_var))*5
x0 = np.random.uniform(-1, 1, size=(num_var))
G = []
h = []
C = []
d = []
for i in range(num_ineq):
    G.append(np.random.uniform(-1, 1, size=(num_ineq, num_var)))
    h.append(np.random.uniform(-1, 1, size=(num_ineq)))
    C.append(np.random.uniform(-1, 1, size=(num_var)))
    d.append(np.linalg.norm(G[i] @ x0 + h[i], 2) - C[i].T @ x0)
    
data = {'Q':Q,
        'p':p,
        'A':A,
        'X':X,
        'G':np.array(G),
        'h':np.array(h),
        'C':np.array(C),
        'd':np.array(d),
        'YL':L,
        'YU':U,
        'XL':XL,
        'XU':XU,
        'Y':[]}
Y = []
for n in range(num_examples):
    Xi = X[n]
    y = cp.Variable(num_var)
    # We use cp.SOC(t, x) to create the SOC constraint ||x||_2 <= t.
    soc_constraints = [cp.SOC(C[i].T @ y + d[i], G[i] @ y + h[i]) for i in range(num_ineq)]
    prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(y, Q) + p.T @ y),
                  soc_constraints + [A @ y == Xi]+[y<=U] + [y>=L])
    prob.solve(solver='SCS')
    print(n, np.max(y.value), np.min(y.value), y.value[0:5].T, end='\r')
    Y.append(y.value)
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


with open("datasets/convex/socp/random{}_socp_dataset_var{}_ineq{}_eq{}_ex{}".format(seed, num_var, num_ineq, num_eq, num_examples), 'wb') as f:
    pickle.dump(data, f)