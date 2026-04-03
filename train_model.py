import os
import sys
from pathlib import Path

if os.name == "posix":
    os.environ["SOFA_ROOT"] = "/opt/emio-labs/resources/sofa"
    sys.path.insert(0, "~/emio-labs/v25.12.01/assets")
    sys.path.insert(
        0,
        "/opt/emio-labs/resources/sofa/plugins/SofaPython3/lib/python3/site-packages/",
    )
else:
    home = Path.home()
    appdata = os.getenv("LOCALAPPDATA")
    os.environ["SOFA_ROOT"] = os.path.join(
        appdata, "Programs\\emio-labs\\resources\\sofa"
    )
    sys.path.append(home.joinpath("/emio-labs/v25.12.01/assets"))
    sys.path.append(
        os.path.join(
            os.environ["SOFA_ROOT"], "plugins\\SofaPython3\\lib\\python3\\site-packages"
        )
    )

from modules.calibration import calibrate_young
from modules.lab_utils import load_dataset
from modules.pytorch_mlp import PytorchMLPReg

DEFAULT = "pytorch"


def train_pytorch_model(dataset_path, from_real=False):
    x_train, y_train, x_test, y_test = load_dataset(dataset_path, from_real)

    mlp = PytorchMLPReg()

    mlp.train(x_train, y_train, x_test, y_test, n_epochs=2_000)

    dataset_fname = dataset_path.parts[-1].strip(".csv")
    fname = f"data/results/{dataset_fname}.pth"
    mlp.save(fname)
    print(f"Trained model saved at {fname}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train model using dataset")

    parser.add_argument(
        "--model-type",
        type=str,
        choices=["pytorch", "calibrated"],
        default=DEFAULT,
        help="Model type: pytorch or calibrated",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path(
            "/home/frederike/emio-labs/v25.12.01/assets/labs/Practical1/data/results/blueleg_beam_sphere.csv"
        ),
        help="Path to dataset CSV",
    )
    parser.add_argument(
        "-r",
        "--from-real",
        action="store_true",
        help="Use real-world dataset instead of synthetic",
    )
    args = parser.parse_args()

    dataset_path = args.dataset_path
    learn_from_real = args.from_real
    model_type = args.model_type
    if not os.path.exists(dataset_path):
        print(f"Dataset file not found: {dataset_path}")
        sys.exit(1)

    if model_type == "calibrated":
        calibrate_young(dataset_path, learn_from_real)
    elif model_type == "pytorch":
        train_pytorch_model(dataset_path, learn_from_real)
    else:
        print(f"Unknown model type: {model_type}")
        sys.exit(1)
