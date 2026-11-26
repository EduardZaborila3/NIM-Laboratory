"""Ruleaza doar experimentele de 100D necesare figurilor din raport."""
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from PSO_Algorithm import (
    BASE, compare_with_h1, plot_variants_boxplots,
    plot_variants_convergence, run_many, variant_setup,
)

FUNCTIONS = ["rastrigin", "griewangk", "rosenbrock", "michalewicz"]
TUNED = dict(vmax_frac=0.05, c1=2.0, c2=1.0)
VARIANTS = {
    "baseline": BASE,
    "tuned": dict(BASE, **TUNED),
    "tuned_hybrid": dict(BASE, **TUNED, polish=True),
    "tuned_hybrid2": dict(BASE, swarm_size=80, vmax_frac=0.07, c1=2.0,
                          c2=1.0, restart_patience=120, restart_frac=0.2,
                          polish_every=500, polish_period_evals=4000,
                          polish_max_evals=100000, crossover=0.3,
                          polish=True, polish_restarts=4),
}

# Medii 30D din rularea completa furnizata pentru raport.
MEANS_30 = {
    ('baseline', 'rastrigin'): 59.10041,
    ('baseline', 'griewangk'): 0.02040,
    ('baseline', 'rosenbrock'): 15.61683,
    ('baseline', 'michalewicz'): -25.21621,
    ('tuned', 'rastrigin'): 30.87687,
    ('tuned', 'griewangk'): 0.01547,
    ('tuned', 'rosenbrock'): 20.54768,
    ('tuned', 'michalewicz'): -26.89510,
    ('tuned_hybrid', 'rastrigin'): 30.87687,
    ('tuned_hybrid', 'griewangk'): 0.01547,
    ('tuned_hybrid', 'rosenbrock'): 12.91007,
    ('tuned_hybrid', 'michalewicz'): -27.31093,
    ('tuned_hybrid2', 'rastrigin'): 0.00000,
    ('tuned_hybrid2', 'griewangk'): 0.00910,
    ('tuned_hybrid2', 'rosenbrock'): 0.76174,
    ('tuned_hybrid2', 'michalewicz'): -29.63069,
}


def main():
    os.chdir(Path(__file__).resolve().parent)
    Path("grafice_pso").mkdir(exist_ok=True)
    results = {(v, f, 30): {"mean": mean} for (v, f), mean in MEANS_30.items()}
    for variant, config in VARIANTS.items():
        for function in FUNCTIONS:
            iterations, params = variant_setup(
                variant, function, 100, 6000, config
            )
            print(f"{variant} / {function} 100D: 30 rulari", flush=True)
            results[variant, function, 100] = run_many(
                function, 100, 30, iterations=iterations, **params
            )
    plot_variants_boxplots(results, list(VARIANTS), FUNCTIONS, 100)
    plot_variants_convergence(results, list(VARIANTS), FUNCTIONS, 100)
    # Refoloseste functia din proiect fara a scrie in CSV-ul comparativ.
    from unittest.mock import patch
    with patch("PSO_Algorithm.append_csv"):
        compare_with_h1(results, list(VARIANTS), FUNCTIONS, [30, 100])


if __name__ == "__main__":
    main()
