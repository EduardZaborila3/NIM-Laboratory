import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))           # .../NIM/Hw2
PSO_CSV = os.path.join(BASE_DIR, "rezultate_pso.csv")
HC_CSV = os.path.join(BASE_DIR, "..", "Hw1", "rezultate_hc.csv")
GA_CSV = os.path.join(BASE_DIR, "..", "Hw1", "rezultate_ga.csv")
FUNCS = ["rastrigin", "griewangk", "rosenbrock", "michalewicz"]
DIMS = [2, 30, 100]


def read(path, config=None):
    out = {}
    if not os.path.exists(path):
        print(f"Lipseste {path}")
        return out
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if config and r.get("config") != config:
                continue
            try:
                out[(r["functie"].strip(), int(r["dimensiune"]))] = float(r["medie"])
            except (KeyError, ValueError):
                pass
    return out


methods = [("HC", read(HC_CSV)), ("GA", read(GA_CSV)),
           ("PSO", read(PSO_CSV, "baseline")),
           ("PSO hybrid", read(PSO_CSV, "tuned_hybrid"))]
methods = [(lbl, m) for lbl, m in methods if m]

print("\nRanduri LaTeX pentru Tabelul de comparatie (copy-paste):")
for n in DIMS:
    for f in FUNCS:
        vals = [f"{m[(f, n)]:.3f}" if (f, n) in m else "--" for _, m in methods]
        print(f"{f.capitalize():12} & {n:3} & " + " & ".join(vals) + r" \\")

os.makedirs("grafice_pso", exist_ok=True)
for n in DIMS:
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, f in zip(axes, FUNCS):
        lbls = [lbl for lbl, m in methods if (f, n) in m]
        vals = [m[(f, n)] for _, m in methods if (f, n) in m]
        ax.bar(range(len(vals)), vals, color=[f"C{i}" for i in range(len(vals))])
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(lbls, rotation=20, fontsize=8)
        ax.set_title(f"{f.capitalize()}, n={n}")
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("mean best fitness")
    fig.tight_layout()
    fig.savefig(f"grafice_pso/comparison_h1_n{n}.png", dpi=150)
    plt.close(fig)
print("\nGrafice salvate in grafice_pso/comparison_h1_n*.png")