import numpy as np

def r2_score_numpy(y_true, y_pred) -> float:
    import numpy as np

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

def clean_and_eval_list_string(list_string) -> list:
    """Function to clean and evaluate the string representation of lists"""
    import ast
    import re

    # Add commas between numbers in the string
    cleaned_string = re.sub(r"(?<=\d)\s+(?=[-\d])", ",", list_string)
    return ast.literal_eval(cleaned_string)


def load_dataset(file_path, get_real=False) -> np.ndarray:
    import pandas as pd
    from sklearn.model_selection import train_test_split

    print(f"Loading dataset from {file_path}")

    # Loading  du dataset
    df_data_raw = pd.read_csv(file_path, delimiter=";", skiprows=8)

    # Shuffle the dataframe
    df_shuffled = df_data_raw.sample(
        frac=1.0, random_state=42
    )  # Added random_state for reproducibility

    # Split the dataframe into training and test sets
    train_size = 0.8
    df_data_training, df_data_test = train_test_split(
        df_shuffled, train_size=train_size, random_state=42
    )  # Added random_state for reproducibility

    # Separate features (X) and target (y) for both training and test sets
    X_train = np.array(
        [
            clean_and_eval_list_string(pos)
            for pos in df_data_training["Effector position"].tolist()
        ]
    )
    y_train = np.array(
        [
            clean_and_eval_list_string(angle)
            for angle in df_data_training["Motor angle"].tolist()
        ]
    )
    if get_real:
        X_train = (
            np.array(
                [
                    clean_and_eval_list_string(pos)
                    for pos in df_data_training["Real Position"].tolist()
                ]
            )
            if "Real Position" in df_data_training.columns
            else None
        )

    X_test = np.array(
        [
            clean_and_eval_list_string(pos)
            for pos in df_data_test["Effector position"].tolist()
        ]
    )
    y_test = np.array(
        [
            clean_and_eval_list_string(angle)
            for angle in df_data_test["Motor angle"].tolist()
        ]
    )
    if get_real:
        X_test = (
            np.array(
                [
                    clean_and_eval_list_string(pos)
                    for pos in df_data_test["Real Position"].tolist()
                ]
            )
            if "Real Position" in df_data_test.columns
            else None
        )

    return X_train, y_train, X_test, y_test
