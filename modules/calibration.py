import numpy as np
import pandas as pd

from modules.sofa_sim_launcher import run_forward_simulation


def read_dataset(dataset, from_real):
    """
    Read dataset and return pairs of (motor commands, end-effector positions).

    Args:
        dataset: Path to the CSV file
        from_real: Whether to use real positions or simulated effector positions

    Returns:
        List of tuples (motor_commands, end_effector_position)
    """
    print(f"Loading dataset from {dataset}")

    # Load the dataset
    df = pd.read_csv(dataset, delimiter=";", skiprows=8)

    # Helper function to parse list strings
    def clean_and_eval_list_string(list_string):
        import ast
        import re

        # Add commas between numbers in the string
        cleaned_string = re.sub(r"(?<=\d)\s+(?=[-\d])", ",", list_string)
        return ast.literal_eval(cleaned_string)

    # Extract motor angles (input) and effector positions (output)
    motor_commands = np.array(
        [clean_and_eval_list_string(angle) for angle in df["Motor angle"].tolist()]
    )

    if from_real and "Real Position" in df.columns:
        # Use real measured positions if available
        end_effector_positions = np.array(
            [clean_and_eval_list_string(pos) for pos in df["Real Position"].tolist()]
        )
    else:
        # Use simulated effector positions
        end_effector_positions = np.array(
            [
                clean_and_eval_list_string(pos)
                for pos in df["Effector position"].tolist()
            ]
        )

    # Return as list of (motor_command, end_effector_position) pairs
    dataset_pairs = list(zip(motor_commands, end_effector_positions))

    return dataset_pairs


def calibrate_young(dataset, from_real=False):

    # read dataset of motor angle - end effector position pairs
    dataset_pairs = read_dataset(dataset, from_real)
    print("Done reading dataset")

    delta = 1e4  # finite-diff parameter
    alpha = 1e1  # stepsize

    E = 2800.0  # starting value of E
    converged = False
    msg = "reached maximum number of iterations"

    tol = 1e-8
    max_iter = 10
    for i in range(max_iter):

        # select random minibatch
        pair_indices = np.random.choice(len(dataset_pairs), size=3)

        for j in pair_indices:
            m, p = dataset_pairs[i]
            # run forward simulator with E, m
            p_sim = run_forward_simulation(E, m)
            f_sim = np.linalg.norm(p_sim - p)

            p_sim_delta = run_forward_simulation(E + delta, m)
            f_sim_delta = np.linalg.norm(p_sim_delta - p)

            # calculate gradient using forward difference
            gradient = (f_sim_delta - f_sim) / delta

            # update Young modulus
            E -= alpha * gradient

            # check convergence
            print(
                f"iteration {i}\t datapoint {j}\t E {E:.0f}\t error {f_sim:.4f}, gradient {gradient:.4e}"
            )
            if np.abs(gradient) <= tol:
                msg = "converged in gradient norm"

    results = {"msg": msg, "success": converged}
    return E, results
