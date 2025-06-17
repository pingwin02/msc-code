import os
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.core.callback import Callback
from pymoo.decomposition.tchebicheff import Tchebicheff
from pymoo.indicators.gd import GD
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions


class MetricLogger(Callback):
    def __init__(self, problem):
        super().__init__()
        self.problem = problem
        self.pf = problem.pareto_front()
        self.data = []

    def notify(self, algorithm):
        F = algorithm.opt.get("F")
        metrics = {
            "generation": algorithm.n_gen,
            "GD": GD(self.pf)(F),
            "IGD": IGD(self.pf)(F),
            "HV": HV(ref_point=np.array([2.0] * self.problem.n_obj))(F)
        }
        self.data.append(metrics)

    def to_csv(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df = pd.DataFrame(self.data)
        df.to_csv(filename, index=False)


def run_wfg_experiment(problem_name):
    print(f"\nRunning: {problem_name}")
    problem = get_problem(problem_name, n_var=10, n_obj=3)
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
    algorithm = MOEAD(ref_dirs=ref_dirs, decomposition=Tchebicheff())
    termination = get_termination("n_gen", 500)
    callback = MetricLogger(problem)

    minimize(
        problem=problem,
        algorithm=algorithm,
        termination=termination,
        callback=callback,
        verbose=False
    )

    output_path = f"Data/MOEAD/metrics_{problem_name}.csv"
    callback.to_csv(output_path)
    print(f"Saved metrics to {output_path}")


run_wfg_experiment("wfg4")
run_wfg_experiment("wfg5")
run_wfg_experiment("wfg6")
