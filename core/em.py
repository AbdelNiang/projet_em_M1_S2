import numpy as np


def em_explicit(model, Y, theta_init=None, n_iter=50, verbose=True):
    """
    EM avec M-step explicite
    """

    if theta_init is None:
        theta = np.mean(Y, axis=0)
    else:
        theta = theta_init.copy()

    history = [theta.copy()]

    for k in range(n_iter):

        # E-step
        weights = model.compute_weights(Y, theta)

        # M-step
        theta = model.solve_m_step(Y, weights)

        history.append(theta.copy())

        if verbose and k % 10 == 0:
            print(f"Iteration {k}")

    return theta, history

def em_gradient(model, Y, theta_init=None, n_iter=50, lr=0.1):

    if theta_init is None:
        theta = np.mean(Y, axis=0)
    else:
        theta = theta_init.copy()

    history = [theta.copy()]

    for _ in range(n_iter):

        grad = model.gradient_Q(Y, theta)
        theta = theta + lr * grad

        history.append(theta.copy())

    return theta, history