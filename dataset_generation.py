import os
import sys
from pathlib import Path


def _bootstrap_sofa_python():
    lab_dir = Path(__file__).resolve().parent
    assets_dir = lab_dir.parent.parent

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

    for path in (assets_dir, sofa_python):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_bootstrap_sofa_python()

import csv
import numpy as np
import Sofa

from modules.targets import Targets

resultsDirectory = os.path.dirname(os.path.realpath(__file__))+"/data/results/"
STEP=10  # Number of steps to wait before changing target

class TargetController(Sofa.Core.Controller):
    """
        A Controller to change the target of Emio, and save the collected data in a CSV file.

        emio: Sofa node of Emio
        target: Sofa node containing a MechanicalObject with the targets position
        effector: PositionEffector component
        assembly: Controller component for the assembly of Emio (set up animation of the legs and center part)
        steps: number of simulation steps to wait before going to the next target  
    """

    def __init__(self, emio, target, effector, assembly, shape, steps=20):
        Sofa.Core.Controller.__init__(self)
        self.name="TargetController"

        self.emio = emio
        self.targetsPosition = target.getMechanicalState().position.value
        self.targetIndex = len(self.targetsPosition) - 1

        self.effector = effector
        self.assembly = assembly
        self.targetReached = False

        self.shape = shape

        self.animationSteps = steps 
        self.animationStep = self.animationSteps
        self.createCSVFile()

    def onAnimateBeginEvent(self, _):
        """
            Change the target when it's time
        """
        delta = np.array(self.emio.effector.getMechanicalState().position.value[0][0:3]) - np.array(self.targetsPosition[self.targetIndex])
        if np.linalg.norm(delta) < 0.5:
            self.targetReached = True

        if self.assembly.done:
            self.animationStep -= 1
            if self.targetIndex >= 0 and (self.animationStep <= 0 or self.targetReached):
                self.writeToCSVFile()
                self.targetIndex -= 1
                self.animationStep = self.animationSteps
                self.effector.effectorGoal = [list(self.targetsPosition[self.targetIndex]) + [0, 0, 0, 1]]
                self.targetReached = False

    def getFilename(self):
        legname = self.emio.legsName[0]
        legmodel = self.emio.legsModel[0]
        count_positions = len(self.emio.getRoot().Modelling.SphereTargets.getMechanicalState().position.value)
        return resultsDirectory + legname + "_"+ legmodel + '_'+self.shape+str(count_positions)+'.csv'

    def createCSVFile(self):
        """
            Clear or create the csv file in which we'll save the data
        """
        with open(self.getFilename(), 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=';')
            csvwriter.writerow(["# extended ", self.emio.extended.value])
            csvwriter.writerow(["# legs ", self.emio.legsName.value])
            csvwriter.writerow(["# legs model ", self.emio.legsModel.value])
            csvwriter.writerow(["# legs young modulus ", self.emio.legsYoungModulus.value])
            csvwriter.writerow(["# legs poisson ratio ", self.emio.legsPoissonRatio.value])
            csvwriter.writerow(["# legs position on motor ", self.emio.legsPositionOnMotor.value])
            csvwriter.writerow(["# connector ", self.emio.centerPartName.value])
            csvwriter.writerow(["# connector type ", self.emio.centerPartType.value])
            csvwriter.writerow(["Effector position", "Motor angle"])

    def writeToCSVFile(self):
        """
            Save the data in a csv file
        """
        with open(self.getFilename(), 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=';')
            csvwriter.writerow([self.emio.effector.getMechanicalState().position.value[0][0:3],
                                [self.emio.Motor0.JointActuator.angle.value,
                                self.emio.Motor1.JointActuator.angle.value,
                                self.emio.Motor2.JointActuator.angle.value,
                                self.emio.Motor3.JointActuator.angle.value]
                                ]) 


def createScene(rootnode):
    """
        Emio simulation
    """
    from utils.header import addHeader, addSolvers
    from parts.gripper import Gripper
    from parts.controllers.assemblycontroller import AssemblyController
    from parts.emio import Emio
    import argparse
    import sys

    ## Parse args
    parser = argparse.ArgumentParser(prog=sys.argv[0],
                                     description='Simulate a leg.')
    parser.add_argument(metavar='shape', type=str, nargs='?', help="the shape of the trajectory to follow",
                        choices=["cube", "sphere"], default='sphere', dest="shape")
    parser.add_argument(metavar='ratio', type=float, nargs='?', help="the division ratio of the target object's size",
                        default=0.08, dest="ratio")

    try:
        args = parser.parse_args()
    except SystemExit:
        Sofa.msg_error(sys.argv[0], "Invalid arguments, get defaults instead.")
        args = parser.parse_args([])

    
    Sofa.msg_info(os.path.basename(__file__), f"Using shape: {args.shape}, ratio: {args.ratio}")

    settings, modelling, simulation = addHeader(rootnode, inverse=True)

    rootnode.dt = 0.03
    rootnode.gravity = [0., -9810., 0.]
    addSolvers(simulation)

    # Add Emio to the scene
    emio = Emio(name="Emio",
                legsName=["blueleg"],
                legsModel=["beam"],
                legsPositionOnMotor=["counterclockwisedown", "clockwisedown", "counterclockwisedown", "clockwisedown"],
                centerPartName="whitepart",  # choose the gripper as the center part
                centerPartType="deformable",  # the gripper is deformable
                centerPartModel="beam",
                centerPartClass=Gripper,  # specify that the center part is a Gripper
                platformLevel=2,
                extended=True)
    if not emio.isValid():
        return

    simulation.addChild(emio)
    emio.attachCenterPartToLegs()
    assembly = AssemblyController(emio)
    emio.addObject(assembly)

    # Generation of the targets
    spherePositions = Targets(ratio=args.ratio, center=[0, -130, 0], size=80).__getattribute__(args.shape)()
    sphere = modelling.addChild("SphereTargets")
    sphere.addObject("MechanicalObject", position=spherePositions, showObject=True, showObjectScale=10, drawMode=0)

    # Effector
    emio.effector.addObject("MechanicalObject", template="Rigid3", position=[0, 0, 0, 0, 0, 0, 1])
    emio.effector.addObject("RigidMapping", index=0)

    # Inverse components and GUI
    emio.addInverseComponentAndGUI(spherePositions[-1] + [0, 0, 0, 1], withGUI=False)
    emio.effector.EffectorCoord.maxSpeed.value = 100 # Limit the speed of the effector's motion

    # Components for the connection to the real robot 
    emio.addConnectionComponents()

    # We add a controller to go through the targets
    rootnode.addObject(TargetController(emio=emio,
                                        target=sphere, 
                                        effector=emio.effector.EffectorCoord, 
                                        assembly=assembly,
                                        shape=args.shape,
                                        steps=STEP))

    return rootnode
