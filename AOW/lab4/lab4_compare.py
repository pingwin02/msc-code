import os
import numpy as np
import pandas as pd
from scipy.io import loadmat


def load_pymoo_csv(filepath):
    df = pd.read_csv(filepath)
    df = df[df['generation'] % 10 == 1]
    df = df.reset_index(drop=True)
    return df


def load_platemo_mat(filepath):
    mat = loadmat(filepath, squeeze_me=True)

    metrics = {
        'GD': np.array(mat['metric']['GD'].item()).flatten(),
        'IGD': np.array(mat['metric']['IGD'].item()).flatten(),
        'HV': np.array(mat['metric']['HV'].item()).flatten()
    }

    return metrics


def compare_metrics(pymoo_df, platemo_metrics, metric_name):
    comparison = pd.DataFrame()
    # comparison['generation'] = pymoo_df['generation']
    comparison['pymoo'] = pymoo_df[metric_name]
    comparison['platemo'] = platemo_metrics[metric_name][:len(pymoo_df)]
    comparison['diff'] = (comparison['pymoo'] - comparison['platemo']).abs()
    for col in ['pymoo', 'platemo', 'diff']:
        comparison[col] = comparison[col].apply(lambda x: f"{x:.5f}")
    return comparison


def run_comparisons():
    problems = ["wfg4", "wfg5", "wfg6"]
    metrics = ['GD', 'IGD', 'HV']

    for problem in problems:
        path = f"Data/MOEAD"
        pymoo_file = f"{path}/metrics_{problem}.csv"
        platemo_file = f"{path}/MOEAD_{problem.upper()}_M3_D10_1.mat"

        pymoo_df = load_pymoo_csv(pymoo_file)
        platemo_metrics = load_platemo_mat(platemo_file)

        for metric in metrics:
            comp_df = compare_metrics(pymoo_df, platemo_metrics, metric)
            output_file = f"Results/compare_{problem}_{metric}.csv"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            comp_df.to_csv(output_file, index=False)
            print(f"Saved: {output_file}")


run_comparisons()