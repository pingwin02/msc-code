# 2025, joanna.szlapczynska@pg.edu.pl
# sample code showing how own problem class in pymoo can be defined
# this example is working with matplotlib 3.9.0 or newer (solved problems with colormap)

from pymoo.problems import get_problem
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.termination import get_termination
from pymoo.visualization.scatter import Scatter
from pymoo.visualization.fitness_landscape import FitnessLandscape
import numpy as np
import matplotlib.pyplot as plt


class MyBoothProblem(Problem):

    def __init__(self):
        super().__init__(n_var=2,  # number of variables
                         n_obj=1,  # number of objectives
                         n_constr=0,  # number of constraints (here there are no constraints!)
                         xl=np.array([-10, -10]),  # variables' lower bounds
                         xu=np.array([10, 10]))  # variables' upper bounds

    def _evaluate(self, x, out, *args, **kwargs):
        f = (x[:, 0] + 2 * x[:, 1] - 7) ** 2 + (2 * x[:, 0] + x[:, 1] - 5) ** 2
        out["F"] = f.reshape(-1, 1)  # pymoo expects out["F"] to be a 2D array with shape (n_samples, n_obj)
        # if we had any constraints a similar assigment for "G" key should be given for the constraints like out["G"] = ......


# Set up the problem's object
problem = MyBoothProblem()

# # The very same results would be achieved if we loaded
# # the built-in Ackley problem with 2 decision variables :D :D :D
# problem = get_problem("ackley", n_var=2)

# Get colormap  - to avoid misconfiguration problem between pymoo and matplotlib (if you do not have such error just
# remove the line below and references to cmap parameters later on
cmap = plt.get_cmap("viridis")
FitnessLandscape(problem, title="Booth problem, fitness landscape - surface", angle=(45, 45), _type="surface",
                 cmap=cmap).show()  # shows the landscape of the objective function (as surface)
FitnessLandscape(problem, title="Booth problem, fitness landscape - contour", _type="contour", colorbar=True,
                 cmap=cmap).show()  # additionally shows the landscape of the objective function (as contour)

# Set up the algorithm's object
algorithm = GA(pop_size=100)  # create Genetic Algorithm with population size = 100

for n_gen in (100, 1000, 5000):
    # set an object for termination
    termination = get_termination("n_gen", n_gen)
    solutions = []
    print(f"Number of generations: {n_gen}")
    for _ in range(20):
        # Run the pymoo optimization (by running the minimize() function)
        res = minimize(
            problem=problem,  # set the problem object
            algorithm=algorithm,  # set the algorithm object
            termination=termination,  # set the termination condition
            # seed=42, # use positive values for seed only if you DO NOT want any randomness and want reproducible results !!!
            verbose=False  # the output should be printed during the optimization runs
        )

        solutions.append(res.F[0])
        print(f"\r{_ + 1}/20", end="")
        # Output the best solution found
        # print("Best solution found:")
        # print("Decision variables: x1=", res.X[0], " x2=", res.X[1])  # decision variables' optimal values
        # print("Objective function F(X)=", res.F[0])  # objective functions' optimal values (here just one objective!)

        # Plot the decision and objective spaces
        # Scatter(title="Decision Space", labels=["X1", "X2"], cmap=cmap).add(res.X).show()
        # Scatter(title="Objective Space", labels=["f1"], cmap=cmap).add(res.F).show()
    print(f"\nMean solution for {n_gen} generations: {np.mean(solutions)}")
    print(f"Standard deviation for {n_gen} generations: {np.std(solutions)}")
