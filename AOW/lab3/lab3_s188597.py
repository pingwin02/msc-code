##########################################################################################
## Tutorial based on the one presented at: https://pymoo.org/getting_started/part_1.html
## Multi-Objective Optimization Algorithms, lecture #4
## J. Szlapczynska, GUT, 2025
##########################################################################################

import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
import matplotlib.pyplot as plt
from pymoo.config import Config
from topsispy import topsis

Config.warnings['not_compiled'] = False

ALPHA = 0.01

def d(theta):
    return 100 + 0.5 * theta

class MyProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,
                         n_ieq_constr=1,
                         xl=np.array([5, 0]),
                         xu=np.array([25, 90]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = d(x[1]) / x[0]
        f2 = ALPHA * d(x[1]) * (1 + x[0] ** 2 / 100)
        g = d(x[1]) - 140
        out["F"] = [f1, f2]
        out["G"] = [g]


problem = MyProblem()
algorithm = NSGA2(
    pop_size=40,
    n_offsprings=10,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)
termination = get_termination("n_gen", 40)

boss_weights = [0.35, 0.65]
client_weights = [0.60, 0.40]
criteria = [-1, -1]

best_boss_variables = []
best_boss_values = []
best_client_variables = []
best_client_values = []

for _ in range(10):
    res = minimize(problem,
                   algorithm,
                   termination,
                   save_history=False,
                   verbose=False)
    X = res.X  # decision variables
    F = res.F  # Pareto Front

    best_boss, t_boss = topsis(F, boss_weights, criteria)
    best_client, t_client = topsis(F, client_weights, criteria)

    print(f"Iteration {_ + 1}")
    print(f"BOSS PREFERENCE\ni, X[i] [speed, degree], F[i] [time, energy]: {best_boss}, {X[best_boss]}, {F[best_boss]}")
    print(f"CLIENT PREFERENCE\ni, X[i] [speed, degree], F[i] [time, energy]: {best_client}, {X[best_client]}, {F[best_client]}\n\n")

    best_boss_values.append(F[best_boss])
    best_client_values.append(F[best_client])
    best_boss_variables.append(X[best_boss])
    best_client_variables.append(X[best_client])

print(f"MEAN BOSS DECISION VARIABLES [speed, degree]\nF: {np.mean(best_boss_variables, axis=0)}")
print(f"MEAN BOSS PREFERENCE [time, energy]\nF: {np.mean(best_boss_values, axis=0)}")
print(f"MEAN CLIENT DECISION VARIABLES [speed, degree]\nF: {np.mean(best_client_variables, axis=0)}")
print(f"MEAN CLIENT PREFERENCE [time, energy]\nF: {np.mean(best_client_values, axis=0)}")

plt.figure(figsize=(7, 5))
plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.scatter(F[best_boss, 0], F[best_boss, 1], marker="x", color="red", s=200, label="Boss")
plt.scatter(F[best_client, 0], F[best_client, 1], marker="x", color="green", s=200, label="Client")
plt.legend()
plt.title("Objective Space")
plt.xlabel("$f_1$")
plt.ylabel("$f_2$")
plt.show()
