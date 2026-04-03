import os
import sys
from pathlib import Path

import numpy as np

DEBUG = True


def _bootstrap_sofa_python():
    project_dir = Path(__file__).resolve().parents[1]
    assets_dir = Path(__file__).resolve().parents[3]

    if os.name == "posix":
        sofa_root = os.environ.setdefault("SOFA_ROOT", "/opt/emio-labs/resources/sofa")
        sofa_python = (
            Path(sofa_root)
            / "plugins"
            / "SofaPython3"
            / "lib"
            / "python3"
            / "site-packages"
        )
    else:
        appdata = os.getenv("LOCALAPPDATA", "")
        sofa_root = os.environ.setdefault(
            "SOFA_ROOT", os.path.join(appdata, "Programs", "emio-labs", "resources", "sofa")
        )
        sofa_python = (
            Path(sofa_root)
            / "plugins"
            / "SofaPython3"
            / "lib"
            / "python3"
            / "site-packages"
        )

    for path in (project_dir, assets_dir, sofa_python):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_bootstrap_sofa_python()


def run_forward_simulation(young_modulus, motor_angles):
    """
    Simulate forward kinematics using SOFA simulation.

    Args:
        young_modulus: Young's modulus value for the material
        motor_angles: Array of motor angles

    Returns:
        End-effector position as numpy array [x, y, z]
    """
    import parameters
    import Sofa
    import SofaRuntime

    # Create the Sofa simulation scene
    SofaRuntime.importPlugin("Sofa.Component")
    SofaRuntime.importPlugin("Sofa.GUI.Component")
    SofaRuntime.importPlugin("Sofa.GL.Component")
    root = Sofa.Core.Node("root")

    # Set the Young's modulus parameter
    parameters.youngModulus = young_modulus

    # important to call this after parameters has been set.
    from parts.emio import createScene

    createScene(root)

    Sofa.Simulation.init(root)

    # Set motor angles
    emio = root.Simulation.Emio
    for i in range(4):
        emio.getChild(f"Motor{i}").JointActuator.value = motor_angles[i]

    # Run the simulation for a fixed number of steps
    dt = 0.01
    for _ in range(50):
        Sofa.Simulation.animate(root, dt)
        if DEBUG:
            position = emio.CenterPart.Effector.getMechanicalState().position[0]
            print(position[:3].round(2))

    # Retrieve the position of the effector
    position = emio.CenterPart.Effector.getMechanicalState().position[0]

    # Return as numpy array (first 3 components: x, y, z)
    return np.array(position[:3])


def main():
    """
    Main function for testing the forward simulation.
    """
    myYoungModulus = 2800
    myMotorAngles = [0.0, 0.0, 0.0, 0.0]  # Default motor angles

    position = run_forward_simulation(myYoungModulus, myMotorAngles)
    print(f"End-effector position: {position}")


if __name__ == "__main__":
    main()
