import torch


# @torch.compile(mode="default")
@torch.jit.script
def _search_direction(
    g: torch.Tensor,               # (B, n)
    S: torch.Tensor,               # (m, B, n)   stacked s‑vectors
    Y: torch.Tensor,               # (m, B, n)   stacked y‑vectors
    gamma: torch.Tensor            # (B, 1) or scalar
) -> torch.Tensor:                 # returns d  (B, n)
    """
    Compute d = −H_k^{-1} g_k   for L‑BFGS in batch mode.

    Parameters
    ----------
    g      : current gradient            shape (B, n)
    S, Y   : history of s_i, y_i         shape (m, B, n)
    gamma  : scalar or (B,1) scaling for the initial Hessian

    Returns
    -------
    d      : search direction            shape (B, n)
    """
    # if S.numel() == 0:           # first iteration → steepest descent
    #     return -g

    m = S.shape[0]               # history length
    rho = 1.0 / ((S * Y).sum(dim=2, keepdim=True) + 1e-10)      # (m,B,1)

    # ------------ first loop (reverse) -------------------------
    q = g
    alphas = []
    for i in range(m - 1, -1, -1):
        alpha_i = rho[i] * (S[i] * q).sum(dim=1, keepdim=True)  # (B,1)
        alphas.append(alpha_i)
        q = q - alpha_i * Y[i]

    # initial Hessian: gamma * I
    r = gamma * q

    # ------------ second loop (forward) ------------------------
    alphas = alphas[::-1]
    for i in range(m):
        beta = rho[i] * (Y[i] * r).sum(dim=1, keepdim=True)
        r = r + S[i] * (alphas[i] - beta)

    return -r                         # final search direction


# @torch.compile(mode="default")
@torch.jit.script
def compute_gamma(S: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return (S[-1] * Y[-1]).sum(dim=1, keepdim=True)/ ((Y[-1] * Y[-1]).sum(dim=1, keepdim=True) + 1e-10)

# ------------------------------------------------------------
# Differentiable L‑BFGS 
# -------------------------------------------------------------
def lbfgs_solve_vec(
    y_init: torch.Tensor,
    x: torch.Tensor,
    data,
    *,
    max_iter: int = 20,
    memory: int = 20,
    val_tol: float = 1e-6,
    grad_tol: float = 1e-6,
    scale: float = 1e3,
    c: float = 1e-4,
    rho_ls: float = 0.5,
    max_ls_iter: int = 10,
) -> torch.Tensor:
    
    """
    Differentiable L‑BFGS solver with a vectorized two‑loop recursion
    and light‑weight backtracking line search.
    """
    # Objective function closure
    def _obj(y: torch.Tensor) -> torch.Tensor:
        eq  = (data.eq_resid(x, y)** 2).sum(dim=1).mean(0)
        inq = (data.ineq_resid(x, y) ** 2).sum(dim=1).mean(0)
        return scale * (eq + inq)

    # Initialise
    y = y_init.clone() 
    B, n = y_init.shape
    # histories 
    S_hist = torch.zeros(memory, B, n, device=y_init.device, dtype=y_init.dtype)
    Y_hist = torch.zeros_like(S_hist)
    hist_len = 0
    hist_ptr = 0
    
    f_val = _obj(y)
    g     = torch.autograd.grad(f_val, y, create_graph=True)[0]
    for k in range(max_iter):

        # ----- convergence tests --------------------------------------------------
        converged = (f_val / scale < val_tol) | (g.norm(dim=1) < grad_tol)
        if converged.all():
            break
        # ----- search direction ---------------------------------------------------
        if hist_len > 0:                              
            # last hist_len entries in insertion order
            idx = (hist_ptr - hist_len + torch.arange(hist_len, device=S_hist.device)) % memory
            S = S_hist[idx]    # shape (hist_len, B, n)
            Y = Y_hist[idx]
            gamma = compute_gamma(S, Y)
            d = _search_direction(g, S, Y, gamma)
        else:
            d = -0.1*g

        # ----- backtracking line search (light autograd footprint) ---------------
        step = 1.0
        dir_deriv = (g * d).sum()
        with torch.no_grad():                       # explore α without grads
            for _ in range(max_ls_iter):
                y_trial = y + step * d
                f_trial = _obj(y_trial)
                ok = f_trial <= f_val + c*step*dir_deriv
                if ok.all(): 
                    break

                step *= rho_ls

        # Re‑evaluate objective/grad ON GRAPH at the accepted step
        y_next = y + step * d
        f_next = _obj(y_next)
        g_next = torch.autograd.grad(f_next, y_next, create_graph=True)[0]

        # ----- store in circular buffer -----
        S_hist[hist_ptr] = (y_next - y)
        Y_hist[hist_ptr] = (g_next - g)
        hist_ptr = (hist_ptr + 1) % memory
        hist_len = min(hist_len + 1, memory)

        y = y_next
        f_val = f_next.clone()
        g = g_next.clone()

        # if k % 9 == 0:                     # lightweight logging
        #     print(f"iter {k:3d}:  f = {f_next.item()/scale:.3e}  |g| = {g_next.norm():.3e} step = {step:.3e}")

    return y


def nondiff_lbfgs_solve_vec(
    y_init: torch.Tensor,
    x: torch.Tensor,
    data,
    *,
    max_iter: int = 20,
    memory: int = 20,
    val_tol: float = 1e-6,
    grad_tol: float = 1e-6,
    scale: float = 1e3,
    c: float = 1e-4,
    rho_ls: float = 0.5,
    max_ls_iter: int = 10,
    S_hist = None,
    Y_hist = None,
    hist_len = 0,
    hist_ptr = 0,
) -> torch.Tensor:
    
    """
    L‑BFGS identical to `lbfgs_solve_vec` but does not build a
    backward graph.  Suitable for inference or when you never need
    ∂ŷ/∂(anything).
    """
    # Objective function closure
    def _obj(y: torch.Tensor) -> torch.Tensor:
        eq  = (data.eq_resid(x, y)**2).sum(dim=1).mean(0)
        inq = (data.ineq_resid(x, y)**2).sum(dim=1).mean(0)
        return scale * (eq + inq)

    # --- init (no autograd attached) ---------------------------------
    y = y_init.detach().clone().requires_grad_(True)           # ensure leaf / no history
    
    # history buffers
    if S_hist is None:
        B, n = y_init.shape
        S_hist = torch.zeros(memory, B, n, device=y_init.device, dtype=y_init.dtype)
        Y_hist = torch.zeros_like(S_hist)
        hist_len = 0
        hist_ptr = 0
    
    f_val = _obj(y)
    g     = torch.autograd.grad(f_val, y, create_graph=False)[0]
    for k in range(max_iter):

        y.requires_grad_(False)          # stop tracing further ops
        g = g.detach()                   # gradient as plain tensor

        # -------- convergence checks --------------------------------
        converged = (f_val / scale < val_tol) | (g.norm(dim=1) < grad_tol)
        if converged.all():
            break
       
        # ----- search direction ---------------------------------------------------
        if hist_len > 0:                               
            # last hist_len entries in insertion order
            idx = (hist_ptr - hist_len + torch.arange(hist_len, device=S_hist.device)) % memory
            S = S_hist[idx]    # shape (hist_len, B, n)
            Y = Y_hist[idx]
            gamma = compute_gamma(S, Y)
            d = _search_direction(g, S, Y, gamma)
        else:
            d = -0.1*g

        # -------- backtracking line search (always no‑grad) ---------
        step = 1.0
        dir_deriv = (g * d).sum()
        for _ in range(max_ls_iter):
            y_try = y + step * d
            if _obj(y_try) <= f_val + c * step * dir_deriv:
                break
            step *= rho_ls
            
        y_next = y + step * d                # accepted move

        # -------- update history  (store *detached* tensors) --------
        # need g_next for y_hist
        y_next.requires_grad_(True)
        f_next = _obj(y_next)
        g_next, = torch.autograd.grad(f_next, y_next, create_graph=False)
        S_hist[hist_ptr] = (y_next - y).detach()
        Y_hist[hist_ptr] = (g_next - g).detach()
        hist_ptr = (hist_ptr + 1) % memory
        hist_len = min(hist_len + 1, memory)

        y = y_next.detach()                  # next iter starts clean

        f_val = f_next.clone()
        g = g_next.clone()
        # if k % 9 == 0:                     # lightweight logging
        #     print(f"iter {k:3d}:  f = {_obj(y_next).item()/scale:.3e}  |g| = {g_next.norm():.3e}")

    return y

def hybrid_lbfgs_solve_vec(
    y_init: torch.Tensor,
    x: torch.Tensor,
    data,
    *,
    max_iter: int = 50,
    max_diff_iter: int = 20,      #  how many steps to keep on the graph
    memory: int = 20,
    val_tol: float = 1e-6,
    grad_tol: float = 1e-6,
    scale: float = 1e3,
    c: float = 1e-4,
    rho_ls: float = 0.5,
    max_ls_iter: int = 10,
) -> torch.Tensor:
    
    """
    Hybrid L‑BFGS solver that starts with differentiable L‑BFGS
    and switches to nondifferentiable L‑BFGS after `max_diff_iter`
    iterations. This is useful for truncated backpropagation.
    """

    # Convenience closure for the scaled objective
    def _obj(y: torch.Tensor) -> torch.Tensor:
        # with amp.autocast(enabled=True): # Enable AMP context
        eq  = (data.eq_resid(x, y)   ** 2).sum(dim=1).mean(0)
        inq = (data.ineq_resid(x, y) ** 2).sum(dim=1).mean(0)
        return scale * (eq + inq)

    # Initialise
    y = y_init.clone()                   # keep graph
    
    B, n = y_init.shape
    S_hist = torch.zeros(memory, B, n, device=y_init.device, dtype=y_init.dtype)
    Y_hist = torch.zeros_like(S_hist)
    hist_len = 0
    hist_ptr = 0
    
    f_val = _obj(y)
    g = torch.autograd.grad(f_val, y, create_graph=True)[0]
    for k in range(max_diff_iter):
        # ----- convergence tests --------------------------------------------------
        converged = (f_val / scale < val_tol) | (g.norm(dim=1) < grad_tol)
        if converged.all():
            break
        
        # ----- search direction ---------------------------------------------------
        if hist_len > 0:                                   # stack history (m,B,n)
            # last hist_len entries in insertion order
            idx = (hist_ptr - hist_len + torch.arange(hist_len, device=S_hist.device)) % memory
            S = S_hist[idx]    # shape (hist_len, B, n)
            Y = Y_hist[idx]
            gamma = compute_gamma(S, Y)
            d = _search_direction(g, S, Y, gamma)
        else:
            d = -0.1*g

        # ----- backtracking line search (light autograd footprint) ---------------
        step = 1.0
        dir_deriv = (g * d).sum()
        with torch.no_grad():                       # explore α without grads
            for _ in range(max_ls_iter):
                y_trial = y + step * d
                f_trial = _obj(y_trial)
                if f_trial <= f_val + c * step * dir_deriv:
                    break
                step *= rho_ls

        # Re‑evaluate objective/grad ON GRAPH at the accepted step
        y_next     = y + step * d
        f_next     = _obj(y_next)
        g_next     = torch.autograd.grad(f_next, y_next, create_graph=True)[0]

        # ----- store in circular buffer -----
        S_hist[hist_ptr] = (y_next - y)
        Y_hist[hist_ptr] = (g_next - g)
        hist_ptr = (hist_ptr + 1) % memory
        hist_len = min(hist_len + 1, memory)

        y = y_next
        f_val = f_next.clone()
        g = g_next.clone()

        # if k%5 == 0:                     # lightweight logging
        #     print(f"iter {k:3d}:  f = {f_next.item()/scale:.3e}  |g| = {g_next.norm():.3e} step = {step:.3e}")

    # --- now switch to nondiff mode --------------------------------
    y_nondiff = nondiff_lbfgs_solve_vec(
        y, x, data,
        max_iter=max_iter - max_diff_iter,
        memory=memory,
        val_tol=val_tol,
        grad_tol=grad_tol,
        scale=scale,
        c=c,
        rho_ls=rho_ls,
        max_ls_iter=max_ls_iter,
        S_hist=S_hist,
        Y_hist=Y_hist,
        hist_len=hist_len,
        hist_ptr=hist_ptr,
    )

    return y + (y_nondiff - y).detach()
