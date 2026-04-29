import numpy as np

def generate_data(model, theta_true, n_samples, seed=0):
    """
    Génère Y_i = A_{Z_i} θ + bruit

    Returns:
        Y : (n, p) ou (n, p, p)
        Z : (n,)
    """

    np.random.seed(seed)

    Y_list = []
    Z_list = []

    for _ in range(n_samples):

        # tirage du latent
        z = np.random.choice(model.latent_grid)

        # observation sans bruit
        y = model.apply_operator(theta_true, z)

        # bruit
        noise = model.sigma * np.random.randn(*y.shape)

        Y_list.append(y + noise)
        Z_list.append(z)

    return np.array(Y_list), np.array(Z_list)