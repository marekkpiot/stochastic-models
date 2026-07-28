# Simulation d’équations différentielles stochastiques

Projet pédagogique en Python consacré à la simulation numérique de processus stochastiques utilisés en finance quantitative.

Le projet étudie principalement le **processus d’Ornstein–Uhlenbeck**, puis introduit le **modèle de Heston** comme extension. L’objectif est de relier les équations mathématiques, leur interprétation probabiliste et leur implémentation numérique.

## Objectifs du projet

- comprendre la structure d’une équation différentielle stochastique ;
- simuler un processus avec le schéma d’Euler–Maruyama ;
- étudier le retour à la moyenne du processus d’Ornstein–Uhlenbeck ;
- comparer les résultats simulés aux résultats théoriques ;
- comparer Euler–Maruyama à la transition exacte du processus OU ;
- estimer les paramètres d’un processus OU à partir de données synthétiques ;
- introduire une variance stochastique avec le modèle de Heston ;
- générer des chocs browniens corrélés ;
- utiliser Monte-Carlo pour calculer des prix d’options.

---

## 1. Processus d’Ornstein–Uhlenbeck

Le processus d’Ornstein–Uhlenbeck est défini par :

$$
dX_t = \theta(\mu-X_t)\,dt + \sigma\,dW_t
$$

avec :

- $X_t$ : valeur du processus à la date $t$ ;
- $\mu$ : niveau moyen de long terme ;
- $\theta$ : vitesse de retour vers la moyenne ;
- $\sigma$ : intensité du bruit aléatoire ;
- $W_t$ : mouvement brownien.

Le processus est attiré vers le niveau $\mu$, tout en étant continuellement perturbé par des chocs aléatoires.

### Schéma d’Euler–Maruyama

Pour simuler le processus, on découpe le temps en petits intervalles $\Delta t$ et on utilise :

$$
X_{k+1}
=
X_k
+
\theta(\mu-X_k)\Delta t
+
\sigma\sqrt{\Delta t}\,Z_k
$$

où :

$$
Z_k\sim\mathcal N(0,1).
$$

La mise à jour contient deux parties :

- un terme déterministe qui ramène le processus vers $\mu$ ;
- un terme aléatoire provenant du mouvement brownien.

---

## 2. Propriétés théoriques du processus OU

La moyenne théorique du processus est :

$$
\mathbb E[X_t]
=
\mu+(X_0-\mu)e^{-\theta t}.
$$

Lorsque $t$ devient grand :

$$
\mathbb E[X_t]\longrightarrow\mu.
$$

La variance théorique est :

$$
\operatorname{Var}(X_t)
=
\frac{\sigma^2}{2\theta}
\left(1-e^{-2\theta t}\right).
$$

À long terme :

$$
\operatorname{Var}(X_t)
\longrightarrow
\frac{\sigma^2}{2\theta}.
$$

Le processus possède donc la distribution stationnaire suivante :

$$
X_\infty
\sim
\mathcal N
\left(
\mu,
\frac{\sigma^2}{2\theta}
\right).
$$

Le projet vérifie numériquement ces résultats en comparant :

- la moyenne empirique des trajectoires à la moyenne théorique ;
- la variance empirique à la variance théorique ;
- l’histogramme des valeurs finales à la densité normale stationnaire.

---

## 3. Comparaison avec la transition exacte

Pour le processus d’Ornstein–Uhlenbeck, la transition exacte entre deux dates est connue :

$$
X_{t+\Delta t}
=
\mu
+
(X_t-\mu)e^{-\theta\Delta t}
+
\sigma
\sqrt{
\frac{1-e^{-2\theta\Delta t}}{2\theta}
}
Z,
$$

avec :

$$
Z\sim\mathcal N(0,1).
$$

Le projet compare cette simulation exacte au schéma d’Euler–Maruyama.

Cette comparaison permet d’observer que :

- Euler–Maruyama introduit une erreur de discrétisation ;
- cette erreur est plus visible lorsque le pas de temps est grand ;
- l’approximation s’améliore lorsque $\Delta t$ diminue.

---

## 4. Calibration naïve du processus OU

Le projet contient également une calibration simple du processus d’Ornstein–Uhlenbeck.

Des trajectoires synthétiques sont d’abord générées avec des paramètres connus. On cherche ensuite à retrouver ces paramètres en utilisant uniquement les valeurs observées.

La prédiction moyenne d’une transition est :

$$
\widehat X_{k+1}
=
X_k+\theta(\mu-X_k)\Delta t.
$$

Le résidu associé est :

$$
e_k
=
X_{k+1}-\widehat X_{k+1}.
$$

Les paramètres $\theta$ et $\mu$ sont estimés en minimisant l’erreur quadratique moyenne :

$$
\operatorname{MSE}
=
\frac{1}{n}
\sum_{k=1}^{n}e_k^2.
$$

L’optimisation est effectuée avec `scipy.optimize.minimize`.

Une fois $\theta$ et $\mu$ estimés, la volatilité est obtenue à partir de la dispersion des résidus :

$$
\widehat\sigma
=
\sqrt{
\frac{
\frac{1}{n}\sum_{k=1}^{n}e_k^2
}{
\Delta t
}
}.
$$

### Exemple de résultat

| Paramètre | Valeur réelle | Valeur estimée |
|---|---:|---:|
| Vitesse de retour $\theta$ | 1,5000 | 1,5022 |
| Niveau moyen $\mu$ | 1,0000 | 0,9898 |
| Volatilité $\sigma$ | 0,3000 | 0,3010 |

La calibration retrouve donc correctement les paramètres utilisés pour générer les données.

Cette méthode reste volontairement simple et sert principalement à comprendre le principe général de calibration d’un modèle.

---

## 5. Extension : modèle de Heston

Le modèle de Heston remplace la volatilité constante du mouvement brownien géométrique par une variance aléatoire.

Le prix de l’actif vérifie :

$$
dS_t
=
rS_t\,dt
+
\sqrt{v_t}S_t\,dW_t^S.
$$

La variance vérifie :

$$
dv_t
=
\kappa(\theta-v_t)\,dt
+
\xi\sqrt{v_t}\,dW_t^v.
$$

Les paramètres principaux sont :

- $v_t$ : variance instantanée ;
- $\theta$ : variance moyenne de long terme ;
- $\kappa$ : vitesse de retour vers la moyenne ;
- $\xi$ : volatilité de la variance ;
- $\rho$ : corrélation entre les chocs du prix et de la variance.

### Chocs corrélés

À partir de deux normales indépendantes :

$$
Z_1,Z_2\sim\mathcal N(0,1),
$$

on construit :

$$
Z_S=Z_1
$$

et :

$$
Z_v
=
\rho Z_1
+
\sqrt{1-\rho^2}\,Z_2.
$$

Les deux variables obtenues suivent chacune une loi normale centrée réduite et leur corrélation vaut $\rho$.

### Positivité de la variance

Le schéma numérique peut produire temporairement une variance négative.

Une troncature à zéro est donc utilisée :

$$
v_t^+=\max(v_t,0).
$$

Cette méthode est simple et adaptée à un projet pédagogique, même si des schémas plus précis existent.

### Mise à jour du prix

Pendant un petit pas de temps, la variance est supposée constante. Une mise à jour exponentielle inspirée du mouvement brownien géométrique est alors utilisée :

$$
S_{k+1}
=
S_k
\exp
\left[
\left(r-\frac{v_k^+}{2}\right)\Delta t
+
\sqrt{v_k^+\Delta t}\,Z_k^S
\right].
$$

Cette écriture garantit que le prix reste strictement positif.

---

## 6. Pricing Monte-Carlo sous Heston

Le prix d’un call européen est estimé par :

$$
C_0
=
e^{-rT}
\mathbb E
\left[
(S_T-K)^+
\right].
$$

En simulation Monte-Carlo :

$$
\widehat C_0
=
e^{-rT}
\frac{1}{N}
\sum_{i=1}^{N}
(S_T^{(i)}-K)^+.
$$

L’erreur standard de l’estimation est calculée par :

$$
\operatorname{SE}
=
\frac{s}{\sqrt N},
$$

où $s$ est l’écart-type empirique des payoffs actualisés.

Le projet contient également une introduction au calcul de volatilité implicite et à la visualisation d’un skew de volatilité.

Ces éléments constituent une extension du cœur principal du projet.

---

## Structure du projet

```text
sde-simulation/
├── figures/
│   ├── heston_implied_volatility_skew.png
│   ├── heston_stock_paths.png
│   ├── heston_variance_paths.png
│   ├── ou_euler_vs_exact.png
│   ├── ou_mean_comparison.png
│   ├── ou_paths.png
│   ├── ou_stationary_distribution.png
│   └── ou_variance_comparison.png
│
├── src/
│   ├── __init__.py
│   ├── black_scholes.py
│   ├── heston.py
│   ├── heston_pricer.py
│   ├── implied_volatility.py
│   ├── ornstein_uhlenbeck.py
│   └── ou_calibration.py
│
├── correlation_demo.py
├── heston_demo.py
├── heston_pricer_demo.py
├── heston_smile_demo.py
├── ou_calibration_demo.py
├── ou_demo.py
├── ou_discretization_demo.py
├── ou_statistics_demo.py
├── README.md
└── requirements.txt
```

---

## Installation

Créer un environnement virtuel :

```powershell
python -m venv .venv
```

L’activer sous Windows PowerShell :

```powershell
.\.venv\Scripts\Activate.ps1
```

Installer les dépendances :

```powershell
python -m pip install -r requirements.txt
```

---

## Exécution des programmes

### Processus d’Ornstein–Uhlenbeck

Simulation des trajectoires :

```powershell
python ou_demo.py
```

Comparaison des statistiques empiriques et théoriques :

```powershell
python ou_statistics_demo.py
```

Comparaison entre Euler–Maruyama et la transition exacte :

```powershell
python ou_discretization_demo.py
```

Calibration naïve des paramètres :

```powershell
python ou_calibration_demo.py
```

### Modèle de Heston

Vérification de la corrélation des chocs :

```powershell
python correlation_demo.py
```

Simulation des trajectoires de prix et de variance :

```powershell
python heston_demo.py
```

Pricing Monte-Carlo d’options européennes :

```powershell
python heston_pricer_demo.py
```

Calcul des volatilités implicites :

```powershell
python heston_smile_demo.py
```

Les graphiques sont enregistrés automatiquement dans le dossier `figures`.

---

## Visualisations

### Trajectoires d’Ornstein–Uhlenbeck

![Trajectoires OU](figures/ou_paths.png)

### Moyenne empirique et moyenne théorique

![Comparaison des moyennes](figures/ou_mean_comparison.png)

### Variance empirique et variance théorique

![Comparaison des variances](figures/ou_variance_comparison.png)

### Distribution stationnaire

![Distribution stationnaire](figures/ou_stationary_distribution.png)

### Euler–Maruyama et transition exacte

![Euler contre transition exacte](figures/ou_euler_vs_exact.png)

### Trajectoires de prix sous Heston

![Prix Heston](figures/heston_stock_paths.png)

### Trajectoires de variance sous Heston

![Variance Heston](figures/heston_variance_paths.png)

### Skew de volatilité implicite

![Skew de volatilité](figures/heston_implied_volatility_skew.png)

---

## Bibliothèques utilisées

- `NumPy` pour les calculs vectorisés et les simulations aléatoires ;
- `SciPy` pour les lois de probabilité, l’optimisation et la recherche de racines ;
- `Matplotlib` pour la création des graphiques.

---

## Limites du projet

Ce projet possède une finalité pédagogique et ne constitue pas une bibliothèque destinée à une utilisation professionnelle.

Ses principales limites sont :

- Euler–Maruyama introduit une erreur de discrétisation ;
- la calibration OU utilise des données synthétiques ;
- la calibration ne fournit pas d’intervalles de confiance ;
- la simulation Heston utilise une troncature simple de la variance ;
- les prix Monte-Carlo comportent une erreur statistique ;
- les paramètres ne sont pas calibrés sur des données financières réelles ;
- les dividendes et les taux d’intérêt variables ne sont pas pris en compte ;
- les performances du code ne sont pas optimisées pour des simulations de très grande taille.

---

## Compétences développées

Ce projet m’a permis de travailler les notions suivantes :

- mouvement brownien ;
- lois normales ;
- équations différentielles stochastiques ;
- schéma d’Euler–Maruyama ;
- retour à la moyenne ;
- moyenne, variance et distribution stationnaire ;
- simulation numérique ;
- calibration de paramètres ;
- optimisation numérique ;
- variables aléatoires corrélées ;
- modèle de Heston ;
- pricing Monte-Carlo ;
- erreur standard ;
- volatilité implicite ;
- organisation d’un projet Python.
