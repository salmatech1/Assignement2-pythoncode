import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Functions and their derivatives (derivatives are needed for Hermite)
# ------------------------------------------------------------------
def f(x):
    return np.log(x + 1) - 2 * x

def df(x):
    return 1 / (x + 1) - 2

def g(x):
    return np.exp(x**2) - 10 * x

def dg(x):
    return 2 * x * np.exp(x**2) - 10


# ------------------------------------------------------------------
# Lagrange polynomial
# ------------------------------------------------------------------
def lagrange(xn, yn, x):
    x = np.asarray(x, dtype=float)
    P = np.zeros_like(x)
    for i in range(len(xn)):
        L = np.ones_like(x)
        for j in range(len(xn)):
            if j != i:
                L *= (x - xn[j]) / (xn[i] - xn[j])
        P += yn[i] * L
    return P


# ------------------------------------------------------------------
# Newton polynomial (divided differences)
# ------------------------------------------------------------------
def newton_coeffs(xn, yn):
    n = len(xn)
    c = np.array(yn, dtype=float)
    for j in range(1, n):
        c[j:] = (c[j:] - c[j - 1:-1]) / (xn[j:] - xn[:n - j])
    return c

def newton_eval(xn, c, x):
    x = np.asarray(x, dtype=float)
    P = c[-1] * np.ones_like(x)
    for k in range(len(c) - 2, -1, -1):
        P = P * (x - xn[k]) + c[k]
    return P

def newton(xn, yn, x):
    return newton_eval(xn, newton_coeffs(xn, yn), x)


# ------------------------------------------------------------------
# Hermite polynomial (values + first derivatives at each node)
# ------------------------------------------------------------------
def hermite(xn, yn, dyn, x):
    n = len(xn)
    z = np.repeat(xn, 2)
    Q = np.zeros((2 * n, 2 * n))
    for i in range(n):
        Q[2 * i, 0] = Q[2 * i + 1, 0] = yn[i]
        Q[2 * i + 1, 1] = dyn[i]
        if i != 0:
            Q[2 * i, 1] = (Q[2 * i, 0] - Q[2 * i - 1, 0]) / (z[2 * i] - z[2 * i - 1])
    for j in range(2, 2 * n):
        for i in range(j, 2 * n):
            Q[i, j] = (Q[i, j - 1] - Q[i - 1, j - 1]) / (z[i] - z[i - j])
    return newton_eval(z, np.diag(Q).copy(), x)


# ------------------------------------------------------------------
# Main routine for one function
# ------------------------------------------------------------------
A, B = 0.0, 2.0                      # interval of study
xx = np.linspace(A, B, 500)          # fine grid for plotting

def compute(func, dfunc, n):
    xn = np.linspace(A, B, n)        # n interpolation nodes
    yn, dyn = func(xn), dfunc(xn)
    return (lagrange(xn, yn, xx),
            newton(xn, yn, xx),
            hermite(xn, yn, dyn, xx),
            xn, yn)

def plot_case(ax, func, dfunc, n, name):
    PL, PN, PH, xn, yn = compute(func, dfunc, n)
    ax.plot(xx, func(xx), "b-", lw=3, label=f"{name}(x)")
    ax.plot(xx, PL, "r-", lw=2, label="PL Lagrange")
    ax.plot(xx, PH, "g--", lw=2, label="PH Hermite")
    ax.plot(xx, PN, color="gold", ls=":", lw=3, label="PN Newton")
    ax.plot(xn, yn, "ko", ms=5)
    ax.set_title(f"{name} with {n} points")
    ax.grid(True)
    ax.legend(fontsize=8)
    # keep the picture readable if a polynomial explodes
    lo, hi = func(xx).min(), func(xx).max()
    m = 0.5 * (hi - lo)
    ax.set_ylim(lo - m, hi + m)
    err = [np.max(np.abs(P - func(xx))) for P in (PL, PN, PH)]
    print(f"{name}, n={n:3d} | max error  Lagrange={err[0]:.2e}  "
          f"Newton={err[1]:.2e}  Hermite={err[2]:.2e}")

def study(func, dfunc, name):
    # Part 1: the 3 points x = 0, 1, 2
    xn = np.array([0.0, 1.0, 2.0])
    print(f"\n=== {name} ===")
    for i in range(3):
        print(f"{name}({int(xn[i])}) = {func(xn[i]):.6f}")
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_case(ax, func, dfunc, 3, name)
    plt.tight_layout()

    # Part 2: increase the number of points (10, 20, 50)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, n in zip(axes, (10, 20, 50)):
        plot_case(ax, func, dfunc, n, name)
    plt.tight_layout()

study(f, df, "f")
study(g, dg, "g")
plt.show()
