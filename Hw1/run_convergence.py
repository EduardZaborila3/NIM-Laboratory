import numpy as np

import HC_Algorithm
from GA_Algorithm import genetic_algorithm
from HC_Algorithm import FUNCTIONS, calculate_bits_number


def collect(name, n, runs=30, gray=True, hc_every=200, generations=1000):
    HC_Algorithm.USE_GRAY = gray
    func, a, b = FUNCTIONS[name]
    bits_number = calculate_bits_number(a, b, 5)
    histories = []
    diversities = []
    for r in range(runs):
        _, _, h, d = genetic_algorithm(func, a, b, n, bits_number, generations=generations,
                                       hc_every=hc_every, log=True)
        histories.append(h)
        diversities.append(d)
        print(name, n, "run", r + 1, "final", h[-1])
    tag = f"{name}_{n}_{'gray' if gray else 'bin'}_hc{hc_every}"
    np.save(f"hist_{tag}.npy", np.array(histories))
    np.save(f"div_{tag}.npy", np.array(diversities))
    HC_Algorithm.USE_GRAY = True


if __name__ == "__main__":
    collect("rosenbrock", 30, gray=True)
    collect("rosenbrock", 30, gray=False)
    for fname in ["rastrigin", "griewangk", "michalewicz"]:
        collect(fname, 30, gray=True)
    collect("rastrigin", 100, gray=True)