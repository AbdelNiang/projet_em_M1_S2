import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1. Comparaison signaux 1D
# ============================================================

def plot_signals(theta_true, theta_est):
    """
    Compare deux signaux 1D
    """

    plt.figure()

    plt.plot(theta_true, label="θ vrai")
    plt.plot(theta_est, label="θ estimé")

    plt.legend()
    plt.title("Comparaison des signaux")

    plt.xlabel("Index")
    plt.ylabel("Amplitude")

    plt.grid()
    plt.show()


# ============================================================
# 2. Visualisation des échantillons 1D
# ============================================================

def plot_samples_1d(Y, n_show=5):
    """
    Affiche quelques observations 1D
    """

    plt.figure(figsize=(8, 4))

    for i in range(min(n_show, len(Y))):
        plt.plot(Y[i], alpha=0.7)

    plt.title("Échantillons observés (1D)")
    plt.xlabel("Index")
    plt.ylabel("Amplitude")

    plt.grid()
    plt.show()


# ============================================================
# 3. Visualisation des échantillons 2D
# ============================================================

def plot_samples_2d(Y, n_show=5):
    """
    Affiche quelques observations 2D
    """

    n_show = min(n_show, len(Y))

    plt.figure(figsize=(2*n_show, 3))

    for i in range(n_show):
        plt.subplot(1, n_show, i+1)
        plt.imshow(Y[i], cmap="viridis")
        plt.title(f"Sample {i}")
        plt.axis("off")

    plt.suptitle("Échantillons observés (2D)")
    plt.show()


# ============================================================
# 4. Comparaison images 2D
# ============================================================

def plot_images(theta_true, theta_est):
    """
    Compare deux images 2D
    """

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(theta_true, cmap="viridis")
    plt.title("θ vrai")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(theta_est, cmap="viridis")
    plt.title("θ estimé")
    plt.axis("off")

    plt.show()


# ============================================================
# 5. Convergence EM (Q)
# ============================================================

def plot_convergence(Q_history, title="Convergence de Q"):
    """
    Affiche la convergence de Q
    """

    plt.figure()

    plt.plot(Q_history)

    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Q")

    plt.grid()
    plt.show()