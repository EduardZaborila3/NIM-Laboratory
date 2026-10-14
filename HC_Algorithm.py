import math
import random
from math import prod

def rastrigin(array):
    return 10 * len(array) + sum(x ** 2 - 10 * math.cos(2 * math.pi * x) for x in array)

def griewangk(array):
    return sum(x ** 2 / 4000 for x in array) - prod(math.cos(array[i]) / math.sqrt[i] for i in range(len(array))) + 1

def rosenbrock(array):
    return sum(100 * (array[i + 1] - array[i] ** 2) ** 2 + (1 - array[i]) ** 2 for i in range(len(array) - 1))

def michalewicz(array):
    return sum(math.sin(array[i]) * (math.sin(i * array[i] ** 2 / math.pi)) ** 20 for i in range(len(array) - 1))

def gen_bits_array(n):
    bits_array = [random.randint(0, 1) for _ in range(n)]
    return bits_array

def calculate_bits_number(a, b, precision):
    return math.ceil(math.log2((b - a) * 10 ** precision))

def hill_climbing_binar(bits_arr_len, evaluate):
    current_solution = gen_bits_array(bits_arr_len)
    current_score = evaluate(current_solution)

    while True:
        best_neighbour = None
        best_neighbour_score = float('inf')

        for i in range(bits_arr_len):
            neighbour = current_solution.copy()
            neighbour[i] = 1 - neighbour[i]
            neighbour_score = evaluate(neighbour)

            if neighbour_score < best_neighbour_score:
                best_neighbour = neighbour
                best_neighbour_score = neighbour_score

        if best_neighbour_score < current_score:
            current_solution = best_neighbour
            current_score = best_neighbour_score
        else:
            break

    return current_solution, current_score
            

