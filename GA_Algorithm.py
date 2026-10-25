import math
import random
import time
import os

import HC_Algorithm
from HC_Algorithm import (FUNCTIONS, gen_bits_array, calculate_bits_number,decode, hill_climbing_binar)

# Optimizare: folosim codul Gray
HC_Algorithm.USE_GRAY = True


# PASUL 1: populatia si evaluarea
def init_population(pop_size, bits_arr_len):
    return [gen_bits_array(bits_arr_len) for _ in range(pop_size)]


def evaluate_population(population, func, a, b, n, bits_number):
    return [func(decode(individual, a, b, n, bits_number)) for individual in population]

# PASUL 2: selectia prin turneu
def tournament_selection(population, scores, k=3):
    # alegem k indivizi la intamplare, castiga cel cu scorul cel mai mic
    candidates = random.sample(range(len(population)), k)
    winner = min(candidates, key=lambda i: scores[i])
    return population[winner].copy()


# PASUL 3: incrucisarea cu un punct de taiere
def crossover(parent1, parent2, crossover_prob):
    if random.random() < crossover_prob:
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    return parent1.copy(), parent2.copy()


# PASUL 4: mutatia
def mutation(individual, mutation_prob):
    for i in range(len(individual)):
        if random.random() < mutation_prob:
            individual[i] = 1 - individual[i]
    return individual


# PASII 5 + 6: algoritmul genetic, elitism, hibridizare cu HC
def genetic_algorithm(func, a, b, n, bits_number,
                      pop_size=100, generations=1000,
                      crossover_prob=0.8, mutation_prob=None,
                      elite_size=2, hc_every=200):
    bits_arr_len = n * bits_number
    if mutation_prob is None:
        mutation_prob = 5 / bits_arr_len      # in medie 5 biti mutati per individ

    population = init_population(pop_size, bits_arr_len)
    scores = evaluate_population(population, func, a, b, n, bits_number)

    for generation in range(generations):
        # elitism: cei mai buni elite_size indivizi trec direct mai departe
        order = sorted(range(pop_size), key=lambda i: scores[i])
        new_population = [population[i].copy() for i in order[:elite_size]]

        # restul populatiei: selectie -> incrucisare -> mutatie
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, scores)
            parent2 = tournament_selection(population, scores)
            child1, child2 = crossover(parent1, parent2, crossover_prob)
            new_population.append(mutation(child1, mutation_prob))
            if len(new_population) < pop_size:
                new_population.append(mutation(child2, mutation_prob))

        population = new_population
        scores = evaluate_population(population, func, a, b, n, bits_number)

        # hibridizare: din cand in cand, HC pe cel mai bun individ
        if hc_every and (generation + 1) % hc_every == 0:
            best = min(range(pop_size), key=lambda i: scores[i])
            population[best], scores[best] = hill_climbing_binar(
                func, a, b, n, bits_number, start=population[best])

    # la final: HC pe cel mai bun individ
    best = min(range(pop_size), key=lambda i: scores[i])
    best_solution, best_score = hill_climbing_binar(
        func, a, b, n, bits_number, start=population[best])
    return best_solution, best_score


# ---------------------------------------------------------------
# PASUL 8: experimentele
# ---------------------------------------------------------------
def run_experiment(name, n, runs, generations):
    func, a, b = FUNCTIONS[name]
    bits_number = calculate_bits_number(a, b, 5)

    scores = []
    start = time.time()
    for _ in range(runs):
        _, score = genetic_algorithm(func, a, b, n, bits_number, generations=generations)
        scores.append(score)
    duration = (time.time() - start) / runs

    mean = sum(scores) / runs
    std = math.sqrt(sum((s - mean) ** 2 for s in scores) / runs)

    print(f"{name:12} n={n:3}  min={min(scores):.5f}  medie={mean:.5f}  "
          f"max={max(scores):.5f}  std={std:.5f}  timp/rulare={duration:.2f}s")

    file_exists = os.path.exists("rezultate_ga.csv")
    with open("rezultate_ga.csv", "a") as f:
        if not file_exists:
            f.write("functie,dimensiune,generatii,min,medie,max,std,timp_s\n")
        f.write(f"{name},{n},{generations},{min(scores):.5f},{mean:.5f},"
                f"{max(scores):.5f},{std:.5f},{duration:.2f}\n")


if __name__ == "__main__":
    RUNS = 30
    GENERATIONS = 1000
    DIMENSIONS = [2, 30]

    FUNCTIONS_TO_RUN = ["rastrigin", "griewangk", "rosenbrock", "michalewicz"]

    for n in DIMENSIONS:
        for name in FUNCTIONS_TO_RUN:
            run_experiment(name, n, RUNS, GENERATIONS)