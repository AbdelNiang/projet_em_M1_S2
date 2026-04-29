from models.translation1d import Translation1D
from core.em import em_explicit
from utils.data import generate_data
from utils.metrics import translation_invariant_error
from utils.viz import plot_signals
import numpy as np

# paramètres
p = 50
theta_true = np.sin(np.linspace(0, 2*np.pi, p))

model = Translation1D(sigma=0.3, shifts=np.arange(-5, 6), p=p)

# données
Y, Z = generate_data(model, theta_true, n_samples=20)

# estimation
theta_init = np.random.randn(p)
theta_est, _ = em_explicit(model, Y, theta_init)

# erreur
err = translation_invariant_error(theta_est, theta_true)
print("Erreur (invariante):", err)

# plot
plot_signals(theta_true, theta_est)
import numpy as np

# ===== Models =====
from models.translation1d import Translation1D
from models.rotation2d import RotationProjection2D

# ===== EM =====
from core.em import em_explicit, em_gradient

# ===== Utils =====
from utils.data import generate_data
from utils.metrics import translation_invariant_error
from utils.viz import plot_signals


# ============================================================
# CAS 1 : Translation 1D
# ============================================================

def demo_translation():
    print("\n" + "="*50)
    print("CAS 1 : Translation 1D")
    print("="*50)

    # paramètres
    p = 50
    sigma = 0.3
    shifts = np.arange(-5, 6)

    # modèle
    model = Translation1D(sigma=sigma, shifts=shifts, p=p)

    # signal vrai
    x = np.linspace(0, 1, p)
    theta_true = np.exp(-50 * (x - 0.5)**2)

    # données
    Y, Z = generate_data(model, theta_true, n_samples=30)

    # initialisation
    theta_init = np.random.randn(p)

    # EM explicite (rapide ici)
    theta_est, history = em_explicit(model, Y, theta_init, n_iter=50)

    # erreur invariante
    err = translation_invariant_error(theta_est, theta_true)
    print(f"Erreur (invariante translation) : {err:.4f}")

    # visualisation
    plot_signals(theta_true, theta_est)


# ============================================================
# CAS 2 : Rotation + Projection 2D
# ============================================================

def demo_rotation():
    print("\n" + "="*50)
    print("CAS 2 : Rotation + Projection 2D")
    print("="*50)

    # paramètres
    p = 40
    sigma = 0.3
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)

    # grille 2D
    x = np.linspace(-1, 1, p)
    X, Y_grid = np.meshgrid(x, x)

    # modèle
    model = RotationProjection2D(
        sigma=sigma,
        angles=angles,
        X_grid=X,
        Y_grid=Y_grid,
        x_1d=x
    )

    # signal vrai (gaussienne 2D)
    theta_true = np.exp(-8 * (X**2 + Y_grid**2))
    theta_true = model.enforce_constraints(theta_true)

    # données
    Y, Z = generate_data(model, theta_true, n_samples=10)

    # initialisation
    theta_init = np.random.randn(p, p)

    # EM version gradient (important ici)
    theta_est, history = em_gradient(model, Y, theta_init, n_iter=30, lr=0.1)

    theta_est = model.enforce_constraints(theta_est)

    # erreur relative
    err = np.linalg.norm(theta_est - theta_true) / np.linalg.norm(theta_true)
    print(f"Erreur relative : {err:.4f}")

    # affichage simple
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(theta_true, cmap="viridis")
    plt.title("θ vrai")

    plt.subplot(1, 2, 2)
    plt.imshow(theta_est, cmap="viridis")
    plt.title("θ estimé")

    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    demo_translation()
    demo_rotation()