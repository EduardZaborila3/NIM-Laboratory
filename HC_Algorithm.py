import math
import random
from math import prod
import time
import os

def rastrigin(array):
    return 10 * len(array) + sum(x ** 2 - 10 * math.cos(2 * math.pi * x) for x in array)

def griewangk(array):
    return sum(x ** 2 / 4000 for x in array) - prod(math.cos(array[i] / math.sqrt(i + 1)) for i in range(len(array))) + 1

def rosenbrock(array):
    return sum(100 * (array[i + 1] - array[i] ** 2) ** 2 + (1 - array[i]) ** 2 for i in range(len(array) - 1))

def michalewicz(array):
    return -sum(math.sin(array[i]) * (math.sin((i + 1) * array[i] ** 2 / math.pi)) ** 20 for i in range(len(array)))

#genereaza coordonatele de start aleator
def gen_bits_array(n):
    bits_array = [random.randint(0, 1) for _ in range(n)]
    return bits_array

def calculate_bits_number(a, b, precision):
    return math.ceil(math.log2((b - a) * 10 ** precision))

def decode_variable(chunk, a, b, bits_number):
    integer = int("".join(map(str, chunk)), 2)
    return a + integer * (b - a) / (2 ** bits_number - 1)

def decode(bits_array, a, b, n, bits_number):
    values = []
    for i in range(n):
        chunk = bits_array[i * bits_number:(i + 1) * bits_number]
        values.append(decode_variable(chunk, a, b, bits_number))
    return values

FUNCTIONS = {
    "rastrigin":(rastrigin, -5.12, 5.12),
    "griewangk":(griewangk, -600,600),
    "rosenbrock": (rosenbrock,-2.048, 2.048),
    "michalewicz":(michalewicz, 0,math.pi),
}

def hill_climbing_binar(func, a, b, n, bits_number):
    bits_arr_len = n * bits_number
    current_solution = gen_bits_array(bits_arr_len)
    current_x = decode(current_solution, a, b, n, bits_number)
    current_score = func(current_x)

    while True:
        best_i = None
        best_value = None
        best_neighbour_score = float('inf')

        for i in range(bits_arr_len):
            var = i // bits_number
            start = var * bits_number

            current_solution[i] = 1 - current_solution[i]
            new_value = decode_variable(current_solution[start:start + bits_number], a, b, bits_number)
            current_solution[i] = 1 - current_solution[i]

            old_value = current_x[var]
            current_x[var] = new_value
            neighbour_score = func(current_x)
            current_x[var] = old_value

            if neighbour_score < best_neighbour_score:
                best_i = i
                best_value = new_value
                best_neighbour_score = neighbour_score

        if best_neighbour_score < current_score:
            current_solution[best_i] = 1 - current_solution[best_i]
            current_x[best_i // bits_number] = best_value
            current_score = best_neighbour_score
        else:
            break

    return current_solution, current_score

#ruleaza HC de mai multe ori ptr a nu se bloca in minime locale
def hill_climbing_restarts(func, a, b, n, bits_number, restarts):
    best_solution = None
    best_score = float('inf')
    for _ in range(restarts):
        solution, score = hill_climbing_binar(func, a, b, n, bits_number)
        if score < best_score:
            best_solution = solution
            best_score = score
    return best_solution, best_score
            
def run_experiment(name, n, runs, restarts):
    func, a, b = FUNCTIONS[name]
    bits_number = calculate_bits_number(a, b, 5)

    scores = []
    start = time.time()
    for _ in range(runs):
        _, score = hill_climbing_restarts(func, a, b, n, bits_number, restarts)
        scores.append(score)
    duration = (time.time() - start) / runs

    mean = sum(scores) / runs
    std = math.sqrt(sum((s - mean) ** 2 for s in scores) / runs)

    print(f"{name:12} n={n:3}  min={min(scores):.5f}  medie={mean:.5f}  "
          f"max={max(scores):.5f}  std={std:.5f}  timp/rulare={duration:.2f}s")

    file_exists = os.path.exists("rezultate_hc.csv")
    with open("rezultate_hc.csv", "a") as f:
        if not file_exists:
            f.write("functie,dimensiune,restarts,min,medie,max,std,timp_s\n")
        f.write(f"{name},{n},{restarts},{min(scores):.5f},{mean:.5f},"
                f"{max(scores):.5f},{std:.5f},{duration:.2f}\n")

if __name__ == "__main__":
    RUNS = 30
    RESTARTS = {2: 50, 30: 10, 100: 1}
    DIMENSIONS = [2, 30]

    for n in DIMENSIONS:
        for name in FUNCTIONS:
            run_experiment(name, n, RUNS, RESTARTS[n])