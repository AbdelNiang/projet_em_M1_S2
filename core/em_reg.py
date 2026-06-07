import numpy as np


def em_regularized(
    model,
    Y,
    lambda_reg=1e-2,
    theta_init=None,
    n_iter=50,
    optimizer=None,
    verbose=True
):

    # =========================
    # Initialisation
    # =========================

    if theta_init is None:

        theta = np.mean(Y, axis=0)

    else:

        theta = theta_init.copy()

    history = [theta.copy()]

    Q_history = []

    # =========================
    # Boucle EM
    # =========================

    for k in range(n_iter):

        # =====================
        # E-step
        # =====================

        weights = model.compute_weights(
            Y,
            theta
        )

        # =====================
        # Monitoring
        # =====================

        Q_val = model.Q(
            Y,
            theta
        )

        Q_history.append(Q_val)

        # =====================
        # M-step
        # =====================

        # ----- EM explicite -----

        if optimizer is None:

            theta = model.solve_m_step(

                Y,
                weights,
                lambda_reg=lambda_reg

            )

        # ----- GEM régularisé -----

        else:

            Q_fn = lambda th: (
                model.Q(Y, th)
            )

            grad_fn = lambda th: (

                model.gradient_Q(Y, th)
                - lambda_reg * th

            )

            theta = optimizer.step(

                theta,
                Q_fn,
                grad_fn

            )

        history.append(theta.copy())

        # =====================
        # Affichage
        # =====================

        if verbose and k % 10 == 0:

            print(
                f"Iteration {k} | "
                f"Q = {Q_val:.4f}"
            )

    return theta, history, Q_history