# EM for Inverse Problems with Latent Variables
Niokhobaye Abdel

# Overview

Ce projet implémente un algorithme EM pour des problèmes inverses avec
variable latente.

Modèle considéré :

$$Y = A_Z \theta + \varepsilon$$

Fonctionnalités :

- Implémentation modulaire de EM
- Cas translation 1D
- Cas rotation + projection 2D
- Régularisation de type Ridge

------------------------------------------------------------------------

# 1. Introduction

Nous considérons le modèle :

$$Y_i = A_{Z_i} \theta + \sigma \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, I)$$

où :

- $\theta \in \mathbb{R}^p$ est le signal inconnu
- $Z_i$ est une variable latente
- $A_z$ est un opérateur linéaire

Objectif :

$$\hat{\theta} = \arg\max_\theta \sum_i \log p(Y_i \mid \theta)$$

------------------------------------------------------------------------

# 2. Algorithme EM

## E-step

On calcule :

$$q_i(z) = p(z \mid Y_i, \theta_k)$$

Dans le cas gaussien :

$$q_i(z_k) \propto \exp\left(-\frac{|Y_i - A_{z_k}\theta|^2}{2\sigma^2}\right)$$

------------------------------------------------------------------------

## M-step

On maximise :

$$Q(\theta) = -\frac{1}{2\sigma^2} \sum_i \sum_z q_i(z)\,\|Y_i - A_z \theta\|^2$$

Gradient :

$$\nabla Q(\theta) = \frac{1}{\sigma^2} \sum_i \sum_z q_i(z) A_z^*(Y_i - A_z \theta)$$

------------------------------------------------------------------------

# 3. Generalized EM (GEM)

On remplace la M-step exacte par :

$$\theta_{k+1} = \theta_k + \eta \nabla Q(\theta_k)$$

avec Armijo pour garantir :

$$Q(\theta_{k+1}) \ge Q(\theta_k)$$

------------------------------------------------------------------------

# 3. Implémentation

## Classe abstraite

``` python
import numpy as np
from abc import ABC, abstractmethod

class InverseProblemWithLatent(ABC):

    def __init__(self, sigma, latent_grid):
        self.sigma = sigma
        self.latent_grid = np.asarray(latent_grid)
        self.n_latent = len(latent_grid)

    @abstractmethod
    def apply_operator(self, theta, z):
        pass

    @abstractmethod
    def apply_adjoint(self, y, z):
        pass
```

------------------------------------------------------------------------

## E-step vectorisé

``` python
def compute_weights(self, Y, theta):

    AZ_theta = np.stack([
        self.apply_operator(theta, z)
        for z in self.latent_grid
    ])

    diff = Y[:, None, :] - AZ_theta[None, :, :]
    losses = np.sum(diff**2, axis=2)

    exponents = -losses / (2 * self.sigma**2)

    exponents -= np.max(exponents, axis=1, keepdims=True)

    weights = np.exp(exponents)
    weights /= np.sum(weights, axis=1, keepdims=True)

    return weights
```

Fonction Q:

``` python
def Q(self, Y, theta):
    weights = self.compute_weights(Y, theta)
    AZ = np.stack([self.apply_operator(theta, z) for z in self.latent_grid])
    diff = Y[:, None, :] - AZ[None, :, :]
    losses = np.sum(diff**2, axis=2)
    return -np.sum(weights * losses) / (2*self.sigma**2)
```

------------------------------------------------------------------------

# 4. Cas 1 : Translation 1D

$$(A_z \theta)(x) = \theta(x - z)$$

``` python
class Translation1D(InverseProblemWithLatent):

    def apply_operator(self, theta, z):
        return np.roll(theta, int(z))

    def apply_adjoint(self, y, z):
        return np.roll(y, -int(z))
```

Propriété :

$$A_z^* A_z = I $$

isométrique → problème bien posé

------------------------------------------------------------------------

# 5. Cas 2 : Rotation + Projection

$$A_z = P \circ R_z$$

avec :

$$P\theta(x) = \int \theta(x,y)\,dy$$

------------------------------------------------------------------------

## Implémentation

``` python
def apply_operator(self, theta, angle):
    rotated = self._rotate_array(theta, angle)
    return np.sum(rotated, axis=0)
```

L’opérateur de projection fait perdre de l’information → problème
inverse mal posé

Génération d’un signal de test : On condidére un signal 2D gaussien :

``` python
p = 40

x = np.linspace(-1, 1, p)
X, Y_grid = np.meshgrid(x, x)

theta_true = np.exp(-8 * (X**2 + Y_grid**2))
```

Expérimentation :

``` python
Y = generate_data(model, theta_true, n_samples=10)

theta_init = np.random.randn(p, p)

theta_est, history, Q_hist = em(model, Y, theta_init)

plt.plot(Q_hist)
plt.title("Convergence de Q")
plt.xlabel("Iteration")
plt.ylabel("Q")
plt.show()
```

On observe des signaux 1D projetés($Y_i = A_{z_i} \theta$) : prjection
d’une gaussienne 2D → gaussienne 1D :

``` python
plt.plot(Y[0])
plt.title("Projection observée")
plt.show()

---

# 6. Génération de données

```python
def generate_data(model, theta_true, n_samples):

    Y = []

    for _ in range(n_samples):
        z = np.random.choice(model.latent_grid)
        y = model.apply_operator(theta_true, z)
        noise = model.sigma * np.random.randn(*y.shape)
        Y.append(y + noise)

    return np.array(Y)
```

# Expérimentation

\# Translation

``` python
p = 50
theta_true = np.exp(-50*(np.linspace(0,1,p)-0.5)**2)

model = Translation1D(0.3, np.arange(-5,6))
Y = generate_data(model, theta_true, 30)

theta_init = np.random.randn(p)
theta_est, Q_hist = em(model, Y, theta_init)
```

``` python
import matplotlib.pyplot as plt

plt.plot(theta_true, label="true")
plt.plot(theta_est, label="estimated")
plt.legend()
plt.show()
```

\# Rotation + projection

``` python
p = 40
theta_true = np.exp(-8*(np.random.rand(p,p)**2))

# données projetées
Y = generate_data(model, theta_true, 10)
```

``` python
plt.plot(Y[0])
plt.title("Projection observée")
plt.show()
```

------------------------------------------------------------------------

# 7. Résultats expérimentaux

``` python
theta_est, history, Q_history = em(model, Y, theta_init)
```

Comparaison :

``` python
plot_signals(theta_true, theta_est)
```

------------------------------------------------------------------------

# 8. Analyse

Erreur :

$$\varepsilon_n = \inf_a |\tau_a \hat{\theta} - \theta|$$

Observations :

- Convergence lorsque $n \to \infty$
- Sensibilité au bruit $\sigma$
- Invariance de translation

------------------------------------------------------------------------

# 9. Discussion

- Cas translation : problème bien posé (isométrique)
- Cas rotation : problème inverse mal posé,perte d’information
- EM converge vers un point critique(un optimum local)

------------------------------------------------------------------------

# 10. Conclusion

Nous avons implémenté un algorithme EM pour des problèmes inverses avec
variable latente.

Perspectives :

- Analyse de convergence
- Étude asymptotique $\varepsilon_n(\sigma)$
- Régularisation avancée
- Approche continue

------------------------------------------------------------------------
