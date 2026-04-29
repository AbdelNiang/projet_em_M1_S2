import matplotlib.pyplot as plt

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

    plt.show()
