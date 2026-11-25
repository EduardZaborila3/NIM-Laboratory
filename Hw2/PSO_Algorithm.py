import csv
import math
import os
import random
import time

import numpy as np

from HC_Algorithm import FUNCTIONS


OUT_DIR = "grafice_pso"
MAIN_CSV = "rezultate_pso.csv"
PARAM_CSV = "rezultate_pso_parametri.csv"
RUNS_CSV = "rezultate_pso_rulari.csv"
TESTS_CSV = "rezultate_pso_teste.csv"
CMP_CSV = "rezultate_comparatie.csv"


_H1_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Hw1")
HC_CSV = os.path.join(_H1_DIR, "rezultate_hc.csv")
GA_CSV = os.path.join(_H1_DIR, "rezultate_ga.csv")


KNOWN_OPT = {
    ("rastrigin", 2): 0.0, ("rastrigin", 30): 0.0, ("rastrigin", 100): 0.0,
    ("griewangk", 2): 0.0, ("griewangk", 30): 0.0, ("griewangk", 100): 0.0,
    ("rosenbrock", 2): 0.0, ("rosenbrock", 30): 0.0, ("rosenbrock", 100): 0.0,
    ("michalewicz", 2): -1.8013, ("michalewicz", 30): -29.63, ("michalewicz", 100): -99.28,
}


def local_search(func, x, a, b, step0, min_step=1e-7, max_evals=20000):
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


def pso(func, a, b, n,
        swarm_size=50, iterations=1000,
        w=0.7298, w_min=None,                                                                     
        c1=1.49618, c2=1.49618,                                                            
        vmax_frac=0.2,                                             
        topology="global",                                                            
        polish=False,                                                                     
        polish_max_evals=20000,

        restart_patience=0,                                                                           
        restart_frac=0.2,                                                             
        polish_every=0,                                                                        
        polish_period_evals=2000,                                                       
        crossover=0.0):                                                                                     
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


    idx = np.arange(swarm_size)
    ring_nb = np.stack([(idx - 1) % swarm_size, idx, (idx + 1) % swarm_size], axis=1)

    history = []
    stagnation = 0                                                          

    for it in range(iterations):

        w_now = w if w_min is None else w - (w - w_min) * it / max(1, iterations - 1)


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

        out = (pos < a) | (pos > b)
        pos = np.clip(pos, a, b)
        vel[out] = 0.0

        scores = evaluate(pos)
        evals += swarm_size

        improved = scores < pbest_score
        pbest_pos[improved] = pos[improved]
        pbest_score[improved] = scores[improved]

        g = int(np.argmin(pbest_score))
        if pbest_score[g] < gbest_score - 1e-12:
            gbest_score = pbest_score[g]
            gbest_pos = pbest_pos[g].copy()
            stagnation = 0
        else:
            stagnation += 1


                                                                              

                                                                       
        if restart_patience and stagnation >= restart_patience:
            k = max(1, int(restart_frac * swarm_size))
            worst = np.argsort(pbest_score)[-k:]
            k_glob = k // 2
            glob, loc = worst[:k_glob], worst[k_glob:]
            pos[glob] = np.random.uniform(a, b, (len(glob), n))
            span = 0.1 * (b - a)
            pos[loc] = np.clip(gbest_pos + np.random.uniform(-span, span, (len(loc), n)), a, b)
            vel[worst] = np.random.uniform(-vmax, vmax, (k, n))
            s = evaluate(pos[worst])
            evals += k
            pbest_pos[worst] = pos[worst]
            pbest_score[worst] = s
            stagnation = 0


        if polish_every and it and it % polish_every == 0:
            x, f, e = local_search(func, gbest_pos, a, b, 0.1 * (b - a),
                                   max_evals=polish_period_evals)
            evals += e
            if f < gbest_score:
                gbest_pos, gbest_score = x, f
                pbest_pos[0] = x
                pbest_score[0] = f
                pos[0] = x


                                                                             

                                                                      
        if crossover and np.random.rand() < crossover:
            order = np.argsort(pbest_score)
            p1, p2 = order[0], order[np.random.randint(1, max(2, swarm_size // 2))]
            alpha = np.random.rand(n)
            child = alpha * pbest_pos[p1] + (1 - alpha) * pbest_pos[p2]
            child = np.clip(child, a, b)
            fc = func(child.tolist())
            evals += 1
            worst_i = int(np.argmax(pbest_score))
            if fc < pbest_score[worst_i]:
                pos[worst_i] = child
                pbest_pos[worst_i] = child
                pbest_score[worst_i] = fc
                if fc < gbest_score:
                    gbest_score = fc
                    gbest_pos = child.copy()
                    stagnation = 0

        history.append(float(gbest_score))

    if polish:
        x, f, e = local_search(func, gbest_pos, a, b, 0.01 * (b - a),
                               max_evals=polish_max_evals)
        evals += e
        if f < gbest_score:
            gbest_pos, gbest_score = x, f
        history.append(float(gbest_score))                                        

    return gbest_pos, float(gbest_score), history, evals


                              

def run_many(name, n, runs, seed0=1000, **params):
    func, a, b = FUNCTIONS[name]
    scores, histories, evals_list = [], [], []
    start = time.time()
    for r in range(runs):
        np.random.seed(seed0 + r)                                              
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
    curves = {}                                          

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
        plt.yscale("symlog", linthresh=1e-3)                                           
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
    styles = ["-", "--", ":", "-."]
    widths = [1.4, 1.8, 2.2, 1.6]
    fig, axes = plt.subplots(1, len(functions), figsize=(4 * len(functions), 3.6))
    for ax, name in zip(np.atleast_1d(axes), functions):
        for i, v in enumerate(variants):
            ax.plot(results[(v, name, n)]["history"], label=v,
                    linestyle=styles[i % len(styles)],
                    linewidth=widths[i % len(widths)])
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


if __name__ == "__main__":
    RUN_MAIN = True                                                  
    RUN_STUDY = True                                       

    RUNS = 30
    DIMENSIONS = [2, 30, 100]
    ITERATIONS = {2: 1000, 30: 3000, 100: 6000}                                     
    FUNCTIONS_TO_RUN = ["rastrigin", "griewangk", "rosenbrock", "michalewicz"]

    TUNED = dict(vmax_frac=0.05, c1=2.0, c2=1.0)                                        

    TUNED2 = dict(vmax_frac=0.1, c1=2.0, c2=1.0,
                  restart_patience=120, restart_frac=0.2,
                  polish_every=500, polish_period_evals=4000,
                  polish_max_evals=60000,
                  crossover=0.3, polish=True)
    VARIANTS = {
        "baseline":     dict(BASE),
        "tuned":        dict(BASE, **TUNED),
        "tuned_hybrid": dict(BASE, **TUNED, polish=True),
        "tuned_hybrid2": dict(BASE, **TUNED2),
    }

    PARAM_N = 30
    PARAM_RUNS = 10
    PARAM_ITERATIONS = 500


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