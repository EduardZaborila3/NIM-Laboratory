import csv
import math
import os
import random
import time

import numpy as np

from HC_Algorithm import FUNCTIONS  # (func, a, b) pentru fiecare functie de test

# ----------------------------------------------------------------------
# Fisiere de iesire
# ----------------------------------------------------------------------
OUT_DIR = "grafice_pso"
MAIN_CSV = "rezultate_pso.csv"
PARAM_CSV = "rezultate_pso_parametri.csv"
RUNS_CSV = "rezultate_pso_rulari.csv"
TESTS_CSV = "rezultate_pso_teste.csv"
CMP_CSV = "rezultate_comparatie.csv"

# Rezultatele din H1 (pune numele corect al fisierului GA daca e altul).
# Se asteapta coloanele: functie, dimensiune, medie
HC_CSV = "rezultate_hc.csv"
GA_CSV = "rezultate_ga.csv"

# optimele cunoscute (Michalewicz 30/100: cele mai bune valori cunoscute, aproximative)
KNOWN_OPT = {
    ("rastrigin", 2): 0.0, ("rastrigin", 30): 0.0, ("rastrigin", 100): 0.0,
    ("griewangk", 2): 0.0, ("griewangk", 30): 0.0, ("griewangk", 100): 0.0,
    ("rosenbrock", 2): 0.0, ("rosenbrock", 30): 0.0, ("rosenbrock", 100): 0.0,
    ("michalewicz", 2): -1.8013, ("michalewicz", 30): -29.63, ("michalewicz", 100): -99.28,
}


# ----------------------------------------------------------------------
# Cautare locala (coordinate search) pentru hibridizare PSO + HC
# ----------------------------------------------------------------------
def local_search(func, x, a, b, step0, min_step=1e-7, max_evals=20000):
    """Imbunatateste x modificand cate o coordonata cu +/- step.
    Pasul se injumatateste cand nu mai exista imbunatatire."""
    x = np.array(x, dtype=float)
    fx = func(x.tolist())
    evals = 1
    step = step0
    while step > min_step and evals < max_evals:
        improved = False
        for i in range(len(x)):
            for d in (step, -step):
                old = x[i]
                x[i] = min(max(old + d, a), b)
                if x[i] == old:
                    continue
                f = func(x.tolist())
                evals += 1
                if f < fx:
                    fx = f
                    improved = True
                    break
                x[i] = old
            if evals >= max_evals:
                break
        if not improved:
            step /= 2
    return x, fx, evals


# ----------------------------------------------------------------------
# PSO (reprezentare reala, minimizare)
# ----------------------------------------------------------------------
def pso(func, a, b, n,
        swarm_size=50, iterations=1000,
        w=0.7298, w_min=None,          # w_min != None -> inertie liniar descrescatoare w -> w_min
        c1=1.49618, c2=1.49618,        # c1 = componenta cognitiva, c2 = componenta sociala
        vmax_frac=0.2,                 # vmax = vmax_frac * (b - a)
        topology="global",             # "global" (gbest) sau "ring" (lbest, 2 vecini)
        polish=False,                  # True -> cautare locala pe gbest la final (hibrid)
        polish_max_evals=20000):
    """Returneaza (best_x, best_score, history, evaluari)."""
    vmax = vmax_frac * (b - a)

    pos = np.random.uniform(a, b, (swarm_size, n))
    vel = np.random.uniform(-vmax, vmax, (swarm_size, n))

    def evaluate(P):
        return np.array([func(p.tolist()) for p in P])

    scores = evaluate(pos)
    evals = swarm_size
    pbest_pos = pos.copy()
    pbest_score = scores.copy()

    g = int(np.argmin(pbest_score))
    gbest_pos = pbest_pos[g].copy()
    gbest_score = pbest_score[g]

    # vecinii pentru topologia ring: i-1, i, i+1
    idx = np.arange(swarm_size)
    ring_nb = np.stack([(idx - 1) % swarm_size, idx, (idx + 1) % swarm_size], axis=1)

    history = []

    for it in range(iterations):
        # inertia (constanta sau liniar descrescatoare)
        w_now = w if w_min is None else w - (w - w_min) * it / max(1, iterations - 1)

        # liderul fiecarei particule
        if topology == "global":
            leader = np.tile(gbest_pos, (swarm_size, 1))
        else:
            best_nb = ring_nb[idx, np.argmin(pbest_score[ring_nb], axis=1)]
            leader = pbest_pos[best_nb]

        r1 = np.random.rand(swarm_size, n)
        r2 = np.random.rand(swarm_size, n)
        vel = (w_now * vel
               + c1 * r1 * (pbest_pos - pos)
               + c2 * r2 * (leader - pos))
        vel = np.clip(vel, -vmax, vmax)

        pos = pos + vel
        # in afara domeniului: tai la margine si opresc viteza pe acea dimensiune
        out = (pos < a) | (pos > b)
        pos = np.clip(pos, a, b)
        vel[out] = 0.0

        scores = evaluate(pos)
        evals += swarm_size

        improved = scores < pbest_score
        pbest_pos[improved] = pos[improved]
        pbest_score[improved] = scores[improved]

        g = int(np.argmin(pbest_score))
        if pbest_score[g] < gbest_score:
            gbest_score = pbest_score[g]
            gbest_pos = pbest_pos[g].copy()

        history.append(float(gbest_score))

    if polish:
        x, f, e = local_search(func, gbest_pos, a, b, 0.01 * (b - a),
                               max_evals=polish_max_evals)
        evals += e
        if f < gbest_score:
            gbest_pos, gbest_score = x, f
        history.append(float(gbest_score))  # ultimul punct = dupa cautarea locala

    return gbest_pos, float(gbest_score), history, evals


# ----------------------------------------------------------------------
# Rulare multipla + statistici
# ----------------------------------------------------------------------
def run_many(name, n, runs, seed0=1000, **params):
    func, a, b = FUNCTIONS[name]
    scores, histories, evals_list = [], [], []
    start = time.time()
    for r in range(runs):
        np.random.seed(seed0 + r)   # aceleasi seed-uri pentru toate variantele
        random.seed(seed0 + r)
        _, s, h, e = pso(func, a, b, n, **params)
        scores.append(s)
        histories.append(h)
        evals_list.append(e)
    duration = (time.time() - start) / runs
    mean = sum(scores) / runs
    std = math.sqrt(sum((s - mean) ** 2 for s in scores) / runs)
    return dict(scores=scores, min=min(scores), mean=mean, max=max(scores), std=std,
                time=duration, evals=sum(evals_list) / runs,
                history=np.mean(histories, axis=0))


def append_csv(path, header, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        wr = csv.writer(f)
        if not exists:
            wr.writerow(header)
        wr.writerow(row)


def run_experiment(name, n, runs, iterations, label="baseline", **params):
    """Experiment principal: scrie in rezultate_pso.csv si rezultate_pso_rulari.csv"""
    r = run_many(name, n, runs, iterations=iterations, **params)
    print(f"[{label:12}] {name:12} n={n:3}  min={r['min']:.5f}  medie={r['mean']:.5f}  "
          f"max={r['max']:.5f}  std={r['std']:.5f}  evaluari={r['evals']:.0f}  "
          f"timp/rulare={r['time']:.2f}s")

    append_csv(MAIN_CSV,
               ["functie", "dimensiune", "config", "iteratii", "evaluari",
                "min", "medie", "max", "std", "timp_s"],
               [name, n, label, iterations, f"{r['evals']:.0f}", f"{r['min']:.5f}",
                f"{r['mean']:.5f}", f"{r['max']:.5f}", f"{r['std']:.5f}", f"{r['time']:.2f}"])
    for i, s in enumerate(r["scores"]):
        append_csv(RUNS_CSV, ["functie", "dimensiune", "config", "rulare", "scor"],
                   [name, n, label, i, f"{s:.6f}"])
    return r


# ----------------------------------------------------------------------
# Studiul influentei parametrilor (variem UN parametru, restul = baseline)
# ----------------------------------------------------------------------
BASE = dict(swarm_size=50, w=0.7298, w_min=None, c1=1.49618, c2=1.49618,
            vmax_frac=0.2, topology="global")

STUDY = {
    "swarm_size": [10, 30, 50, 100],
    "w":          [0.2, 0.5, 0.7298, 0.9, 1.1],
    "w_decay":    [("fixed 0.7298", dict(w=0.7298, w_min=None)),
                   ("0.9 -> 0.4", dict(w=0.9, w_min=0.4))],
    "c1_c2":      [(0.5, 2.5), (1.5, 1.5), (2.0, 2.0), (2.5, 0.5), (0.0, 2.0), (2.0, 0.0)],
    "vmax_frac":  [0.05, 0.1, 0.2, 0.5, 1.0],
    "topology":   ["global", "ring"],
}


def parameter_study(n=30, runs=10, iterations=500, functions=None, plot=True):
    functions = functions or list(FUNCTIONS)
    curves = {}  # (functie, grup) -> {eticheta: history}

    for fname in functions:
        for group, values in STUDY.items():
            for v in values:
                params = dict(BASE)
                if group == "w_decay":
                    label, extra = v
                    params.update(extra)
                elif group == "c1_c2":
                    label = f"c1={v[0]}, c2={v[1]}"
                    params.update(c1=v[0], c2=v[1])
                else:
                    label = f"{group}={v}"
                    params[group] = v

                r = run_many(fname, n, runs, iterations=iterations, **params)
                print(f"{fname:12} n={n} [{group:10}] {label:18} "
                      f"medie={r['mean']:.5f} std={r['std']:.5f} t={r['time']:.1f}s")
                append_csv(PARAM_CSV,
                           ["functie", "dimensiune", "parametru", "valoare", "iteratii",
                            "min", "medie", "max", "std", "timp_s"],
                           [fname, n, group, label, iterations, f"{r['min']:.5f}",
                            f"{r['mean']:.5f}", f"{r['max']:.5f}", f"{r['std']:.5f}",
                            f"{r['time']:.2f}"])
                curves.setdefault((fname, group), {})[label] = r["history"]

    if plot:
        plot_param_curves(curves, n)


# ----------------------------------------------------------------------
# Teste statistice intre variante
# ----------------------------------------------------------------------
def statistical_tests(results, variants, functions, dims):
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        print("scipy nu este instalat -> testele statistice sunt sarite "
              "(pip install scipy)")
        return
    ref = variants[0]
    for name in functions:
        for n in dims:
            for v in variants[1:]:
                a = results[(ref, name, n)]["scores"]
                b = results[(v, name, n)]["scores"]
                if np.allclose(a, b):
                    p = 1.0
                else:
                    p = mannwhitneyu(a, b, alternative="two-sided").pvalue
                append_csv(TESTS_CSV,
                           ["functie", "dimensiune", "referinta", "varianta", "p_value",
                            "semnificativ_0.05"],
                           [name, n, ref, v, f"{p:.6f}", "da" if p < 0.05 else "nu"])
                print(f"Mann-Whitney {name:12} n={n:3} {ref} vs {v:12}: p={p:.4g}")


# ----------------------------------------------------------------------
# Grafice
# ----------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(OUT_DIR, exist_ok=True)
    return plt


def plot_param_curves(curves, n):
    plt = _plt()
    for (fname, group), d in curves.items():
        plt.figure(figsize=(7, 4.5))
        for label, hist in d.items():
            plt.plot(hist, label=label)
        plt.yscale("symlog", linthresh=1e-3)  # symlog: Michalewicz are valori negative
        plt.xlabel("iteration")
        plt.ylabel("best fitness (mean over runs)")
        plt.title(f"{fname.capitalize()}, n={n}: effect of {group}")
        plt.legend(fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/{fname}_n{n}_{group}.png", dpi=150)
        plt.close()


def plot_dimension_convergence(results, label, functions, dims):
    """Convergenta variantei 'label' pentru fiecare functie, pe toate dimensiunile."""
    plt = _plt()
    fig, axes = plt.subplots(1, len(functions), figsize=(4 * len(functions), 3.6))
    for ax, name in zip(np.atleast_1d(axes), functions):
        for n in dims:
            h = results[(label, name, n)]["history"]
            ax.plot(np.linspace(0, 1, len(h)), h, label=f"n={n}")
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_title(name.capitalize())
        ax.set_xlabel("fraction of iterations")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("best fitness (mean over runs)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/convergence_dimensions_{label}.png", dpi=150)
    plt.close(fig)


def plot_variants_convergence(results, variants, functions, n):
    plt = _plt()
    fig, axes = plt.subplots(1, len(functions), figsize=(4 * len(functions), 3.6))
    for ax, name in zip(np.atleast_1d(axes), functions):
        for v in variants:
            ax.plot(results[(v, name, n)]["history"], label=v)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_title(f"{name.capitalize()}, n={n}")
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("best fitness (mean over runs)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/convergence_variants_n{n}.png", dpi=150)
    plt.close(fig)


def plot_variants_boxplots(results, variants, functions, n):
    plt = _plt()
    fig, axes = plt.subplots(1, len(functions), figsize=(4 * len(functions), 3.8))
    for ax, name in zip(np.atleast_1d(axes), functions):
        data = [results[(v, name, n)]["scores"] for v in variants]
        ax.boxplot(data)
        ax.set_xticks(range(1, len(variants) + 1))
        ax.set_xticklabels(variants, rotation=20, fontsize=8)
        ax.set_title(f"{name.capitalize()}, n={n}")
        ax.grid(alpha=0.3, axis="y")
    np.atleast_1d(axes)[0].set_ylabel("final best fitness")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/boxplot_variants_n{n}.png", dpi=150)
    plt.close(fig)


def plot_error_vs_dimension(results, variants, functions, dims):
    """Eroarea medie fata de optim (log) in functie de dimensiune."""
    plt = _plt()
    fig, axes = plt.subplots(1, len(functions), figsize=(4 * len(functions), 3.6))
    for ax, name in zip(np.atleast_1d(axes), functions):
        for v in variants:
            err = [max(results[(v, name, n)]["mean"] - KNOWN_OPT[(name, n)], 1e-6)
                   for n in dims]
            ax.plot(dims, err, marker="o", label=v)
        ax.set_yscale("log")
        ax.set_xticks(dims)
        ax.set_title(name.capitalize())
        ax.set_xlabel("dimension n")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("mean error  f - f*")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/error_vs_dimension.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# Comparatie cu H1 (HC si GA)
# ----------------------------------------------------------------------
def read_reference(path):
    """Citeste media pe (functie, dimensiune). Daca sunt mai multe randuri, ultimul castiga."""
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                out[(row["functie"].strip(), int(row["dimensiune"]))] = float(row["medie"])
            except (KeyError, ValueError):
                continue
    return out


def compare_with_h1(results, variants, functions, dims):
    hc = read_reference(HC_CSV)
    ga = read_reference(GA_CSV)
    if not hc and not ga:
        print(f"\nNu am gasit {HC_CSV} / {GA_CSV} -> comparatia cu H1 este sarita.")
        return
    print("\n--- Comparatie cu H1 ---")

    methods = []
    if hc:
        methods.append(("HC", hc))
    if ga:
        methods.append(("GA", ga))
    for v in variants:
        methods.append((f"PSO {v}", {(name, n): results[(v, name, n)]["mean"]
                                     for name in functions for n in dims}))

    for name in functions:
        for n in dims:
            row = [name, n] + [f"{m[(name, n)]:.5f}" if (name, n) in m else "-"
                               for _, m in methods]
            append_csv(CMP_CSV, ["functie", "dimensiune"] + [lbl for lbl, _ in methods], row)
            print("  ".join(str(x) for x in row))

    plt = _plt()
    for n in dims:
        fig, axes = plt.subplots(1, len(functions), figsize=(4 * len(functions), 3.8))
        for ax, name in zip(np.atleast_1d(axes), functions):
            labels, vals = [], []
            for lbl, m in methods:
                if (name, n) in m:
                    labels.append(lbl)
                    vals.append(m[(name, n)])
            ax.bar(range(len(vals)), vals, color=[f"C{i}" for i in range(len(vals))])
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(labels, rotation=25, fontsize=8)
            ax.set_title(f"{name.capitalize()}, n={n}")
            ax.grid(alpha=0.3, axis="y")
        np.atleast_1d(axes)[0].set_ylabel("mean best fitness")
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/comparison_h1_n{n}.png", dpi=150)
        plt.close(fig)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    RUN_MAIN = True       # experimentul principal (toate variantele)
    RUN_STUDY = True      # studiul influentei parametrilor

    RUNS = 30
    DIMENSIONS = [2, 30, 100]
    ITERATIONS = {2: 1000, 30: 3000, 100: 6000}   # buget care creste cu dimensiunea
    FUNCTIONS_TO_RUN = ["rastrigin", "griewangk", "rosenbrock", "michalewicz"]

    TUNED = dict(vmax_frac=0.05, c1=2.0, c2=1.0)   # ales pe baza studiului de parametri
    VARIANTS = {
        "baseline":     dict(BASE),
        "tuned":        dict(BASE, **TUNED),
        "tuned_hybrid": dict(BASE, **TUNED, polish=True),
    }

    PARAM_N = 30
    PARAM_RUNS = 10
    PARAM_ITERATIONS = 500

    # sterg rezultatele vechi ca sa nu se dubleze randurile
    for path in (MAIN_CSV, PARAM_CSV, RUNS_CSV, TESTS_CSV, CMP_CSV):
        if os.path.exists(path):
            os.remove(path)

    if RUN_MAIN:
        print("--- Experiment principal PSO ---")
        results = {}
        for label, params in VARIANTS.items():
            for n in DIMENSIONS:
                for name in FUNCTIONS_TO_RUN:
                    results[(label, name, n)] = run_experiment(
                        name, n, RUNS, ITERATIONS[n], label=label, **params)

        variant_names = list(VARIANTS)
        print("\n--- Teste statistice ---")
        statistical_tests(results, variant_names, FUNCTIONS_TO_RUN, DIMENSIONS)

        plot_dimension_convergence(results, "baseline", FUNCTIONS_TO_RUN, DIMENSIONS)
        plot_error_vs_dimension(results, variant_names, FUNCTIONS_TO_RUN, DIMENSIONS)
        for n in DIMENSIONS:
            plot_variants_convergence(results, variant_names, FUNCTIONS_TO_RUN, n)
            plot_variants_boxplots(results, variant_names, FUNCTIONS_TO_RUN, n)

        compare_with_h1(results, variant_names, FUNCTIONS_TO_RUN, DIMENSIONS)

    if RUN_STUDY:
        print("\n--- Influenta parametrilor ---")
        parameter_study(n=PARAM_N, runs=PARAM_RUNS, iterations=PARAM_ITERATIONS,
                        functions=FUNCTIONS_TO_RUN)

    print(f"\nGata. CSV-uri: {MAIN_CSV}, {RUNS_CSV}, {PARAM_CSV}, {TESTS_CSV}"
          f"{', ' + CMP_CSV if os.path.exists(CMP_CSV) else ''}. Grafice in {OUT_DIR}/")