import numpy as np

def em(model, Y, theta_init=None, n_iter=50, optimizer=None, verbose=True):

    if theta_init is None:
        theta = np.mean(Y, axis=0)
    else:
        theta = theta_init.copy()

    history = [theta.copy()]
    Q_history = []

    for k in range(n_iter):

        # ===== E-step =====
        weights = model.compute_weights(Y, theta)

        # ===== Monitoring =====
        Q_val = model.Q(Y, theta)
        Q_history.append(Q_val)

        # ===== M-step =====
        if optimizer is None:
            theta = model.solve_m_step(Y, weights)
        else:
            Q_fn = lambda th: model.Q(Y, th)
            grad_fn = lambda th: model.gradient_Q(Y, th)
            theta = optimizer.step(theta, Q_fn, grad_fn)

        history.append(theta.copy())

        if verbose and k % 10 == 0:
            print(f"Iteration {k} | Q = {Q_val:.4f}")

    return theta, history, Q_history