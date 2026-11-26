"""Regenereaza DOAR convergence_variants_n100.png cu stiluri de linie distincte.
Ruleaza 10 rulari per varianta (graficul e media; forma e identica cu 30).
Salveaza istoricele in _hist_n100_mich_v4.npz, deci a doua oara desenezi instant.

Rulare:  python regen_convergenta.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PSO_Algorithm import run_many, BASE, variant_setup

OUT = "grafice_pso"
CACHE = "_hist_n100_mich_v4.npz"
FUNCS = ["rastrigin", "griewangk", "rosenbrock", "michalewicz"]
VARIANTS = ["baseline", "tuned", "tuned_hybrid", "tuned_hybrid2"]

TUNED = dict(vmax_frac=0.05, c1=2.0, c2=1.0)
TUNED2 = dict(vmax_frac=0.1, c1=2.0, c2=1.0, restart_patience=80, restart_frac=0.2,
              polish_every=500, polish_period_evals=2000, crossover=0.3, polish=True)
PARAMS = {
    "baseline": dict(BASE),
    "tuned": dict(BASE, **TUNED),
    "tuned_hybrid": dict(BASE, **TUNED, polish=True),
    "tuned_hybrid2": dict(BASE, **TUNED2),
}

RUNS = 10          # 10 e destul pentru forma curbei (media)
ITER = 6000

# 1) calculeaza istoricele (sau incarca din cache daca exista)
if os.path.exists(CACHE):
    print(f"Incarc istoricele din {CACHE} (sterge fisierul ca sa recalculezi).")
    data = np.load(CACHE)
    hist = {k: data[k] for k in data.files}
else:
    print(f"Calculez istoricele ({RUNS} rulari/varianta)... dureaza cateva minute.")
    hist = {}
    for lbl in VARIANTS:
        for f in FUNCS:
            params = PARAMS[lbl]
            iterations, params = variant_setup(lbl, f, 100, ITER, params)
            r = run_many(f, 100, RUNS, iterations=iterations, **params)
            hist[f"{lbl}|{f}"] = np.asarray(r["history"])
            print(f"  {lbl:14} {f:12} gata", flush=True)
    np.savez(CACHE, **hist)
    print(f"Salvat in {CACHE}.")

# 2) deseneaza cu stiluri de linie distincte
styles = ["-", "--", ":", "-."]
widths = [1.4, 1.8, 2.2, 1.6]
os.makedirs(OUT, exist_ok=True)
fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
for ax, f in zip(axes, FUNCS):
    for i, lbl in enumerate(VARIANTS):
        ax.plot(hist[f"{lbl}|{f}"], label=lbl, linestyle=styles[i], linewidth=widths[i])
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_title(f"{f.capitalize()}, n=100")
    ax.set_xlabel("iteration")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
axes[0].set_ylabel("best fitness (mean over runs)")
fig.tight_layout()
fig.savefig(f"{OUT}/convergence_variants_n100.png", dpi=150)
plt.close(fig)
print(f"Gata: {OUT}/convergence_variants_n100.png")
