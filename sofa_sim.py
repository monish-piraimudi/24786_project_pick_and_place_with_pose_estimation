import csv
import os
from math import pi
import struct
import asyncio

import numpy as np

import Sofa
import Sofa.ImGui as MyGui

from modules.targets import Targets
from modules.lab_utils import *

INPUT_FMT = "!3f"     # network byte order
OUTPUT_FMT = "!4f"
INPUT_SIZE = struct.calcsize(INPUT_FMT)
OUTPUT_SIZE = struct.calcsize(OUTPUT_FMT)

resultsDirectory = os.path.dirname(os.path.realpath(__file__)) + "/data/results/"
STEP = 25

import asyncio
import threading

class AsyncWorker:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self._run_loop, daemon=True
        )
        self.thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coro):
        """Run async coroutine synchronously"""
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()
    
async def createAsyncObject():
    reader, writer = await asyncio.open_connection(
        "127.0.0.1", 5000
    )
    return reader, writer

async def writeAdnRead(input_vec, writer, reader):
    """
    input_vec: iterable of 3 floats (x, y, z)
    returns: tuple of 4 floats (a0, a1, a2, a3)
    """

    # Serialize input
    data_in = struct.pack(
        INPUT_FMT,
        float(input_vec[0]),
        float(input_vec[1]),
        float(input_vec[2]),
    )

    # Send
    writer.write(data_in)
    await writer.drain()
    # Receive exactly 4 floats
    data_out = await reader.readexactly(OUTPUT_SIZE)
    a0, a1, a2, a3 = struct.unpack(OUTPUT_FMT, data_out)
    return np.array([[a0, a1, a2, a3]])


class MLPController(Sofa.Core.Controller):
    """
    A Controller that loads a trained MLP model to predict the motor angles for Emio
    """


    def __init__(self, emio, model_file):
        Sofa.Core.Controller.__init__(self)
        self.name = "MLPController"
        self.emio = emio
        self.model_file = model_file

        #### ASYNC COMMUNICATION ####
        if os.name == 'posix':
            self.async_worker = AsyncWorker()
            self.reader, self.writer = self.async_worker.run(
                createAsyncObject()
            )
        else:
            from modules.pytorch_mlp import PytorchMLPReg
            self.regr = PytorchMLPReg(model_file=self.model_file, batch_size=1)

        #### GUI ####
        self.emio.addData(name="target_X", type="float", value=0.0)
        self.emio.addData(name="target_Y", type="float", value=-100.0)
        self.emio.addData(name="target_Z", type="float", value=0.0)
        group = "MLP Controller"
        MyGui.MyRobotWindow.addSettingInGroup(
            "TCP X", self.emio.target_X, -150.0, 150.0, group
        )
        MyGui.MyRobotWindow.addSettingInGroup(
            "TCP Y", self.emio.target_Y, -200.0, -50.0, group
        )
        MyGui.MyRobotWindow.addSettingInGroup(
            "TCP Z", self.emio.target_Z, -150.0, 150.0, group
        )

    def onAnimateBeginEvent(self, _):
         if self.emio.AssemblyController.done:
            input = np.array([
                float(self.emio.target_X.value),
                float(self.emio.target_Y.value),
                float(self.emio.target_Z.value),
            ])

            if os.name == 'posix':
                output = self.async_worker.run(
                    writeAdnRead(input, self.writer, self.reader)
                )
                motors_angles = output
            else:
                import torch
                target = torch.tensor(
                    [
                        input
                    ],
                    dtype=torch.float32,
                    device="cpu",
                )
                with torch.inference_mode():
                    motors_angles = self.regr.predict(target)
            for i in range(4):
                self.emio.getChild(f"Motor{i}").JointActuator.value = motors_angles[0][i]


class TargetController(Sofa.Core.Controller):
    """
    A Controller to change the target of Emio, and save the collected data in a CSV file.

    emio: Sofa node of Emio
    target: Sofa node containing a MechanicalObject with the targets position
    effector: PositionEffector component
    assembly: Controller component for the assembly of Emio (set up animation of the legs and center part)
    steps: number of simulation steps to wait before going to the next target
    """

    def __init__(self, emio, target, assembly, steps=20):
        Sofa.Core.Controller.__init__(self)
        self.name = "TargetController"

        self.emio = emio
        self.targetsPosition = target.getMechanicalState().position.value
        self.targetIndex = len(self.targetsPosition) - 1

        self.assembly = assembly
        self.firstTargetReached = False

        self.animationSteps = steps
        self.animationStep = self.animationSteps
        self.index = 0

        #### Plotting the error ####
        self.addData(name="error", type="float", value=0)
        self.addData(name="errorX", type="float", value=0)
        self.addData(name="errorY", type="float", value=0)
        self.addData(name="errorZ", type="float", value=0)
        self.addData(name="r2", type="float", value=0)
        MyGui.PlottingWindow.addData("error", self.error)
        MyGui.PlottingWindow.addData("errorX", self.errorX)
        MyGui.PlottingWindow.addData("errorY", self.errorY)
        MyGui.PlottingWindow.addData("errorZ", self.errorZ)
        MyGui.PlottingWindow.addData("r2", self.r2)

    def onAnimateBeginEvent(self, _):
        """
        Change the target when it's time
        """
        if self.assembly.done:
            self.animationStep -= 1
            if self.targetIndex >= 0 and self.animationStep == 0:

                # Store effector position in Trajectory MechanicalObject
                position = list(
                    np.copy(
                        self.emio.getRoot()
                        .Modelling.Trajectory.getMechanicalState()
                        .position.value
                    )
                )
                position[self.index] = (
                    self.emio.effector.getMechanicalState().position.value[0][0:3]
                )
                self.index += 1
                self.emio.getRoot().Modelling.Trajectory.getMechanicalState().position.value = (
                    position
                )

                # calculate the error
                delta = np.array(
                    self.emio.effector.getMechanicalState().position.value[0][0:3]
                ) - np.array(self.targetsPosition[self.targetIndex])
                self.error.value = np.linalg.norm(delta)
                self.errorX.value = delta[0]
                self.errorY.value = delta[1]
                self.errorZ.value = delta[2]

                # calculate the r2 score using AI_models.r2_score_numpy
                targets = np.array(self.targetsPosition[self.targetIndex :])
                self.r2 = r2_score_numpy(
                    targets, position[: len(self.targetsPosition) - self.targetIndex]
                )

                # Change target and update the motors angles
                self.targetIndex -= 1
                self.animationStep = self.animationSteps
                self.emio.target_X.value = self.targetsPosition[self.targetIndex][0]
                self.emio.target_Y.value = self.targetsPosition[self.targetIndex][1]
                self.emio.target_Z.value = self.targetsPosition[self.targetIndex][2]

    def getFilename(self):
        legname = self.emio.legsName[0]
        legmodel = self.emio.legsModel[0]
        return (
            resultsDirectory
            + legname
            + "_"
            + STEP
            + "STEP"
            + "_"
            + legmodel
            + "_sphere.csv"
        )

    def createCSVFile(self):
        """
        Clear or create the csv file in which we'll save the data
        """
        with open(self.getFilename(), "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=";")
            csvwriter.writerow(["# extended ", self.emio.extended.value])
            csvwriter.writerow(["# legs ", self.emio.legsName.value])
            csvwriter.writerow(["# legs model ", self.emio.legsModel.value])
            csvwriter.writerow(
                ["# legs young modulus ", self.emio.legsYoungModulus.value]
            )
            csvwriter.writerow(
                ["# legs poisson ratio ", self.emio.legsPoissonRatio.value]
            )
            csvwriter.writerow(
                ["# legs position on motor ", self.emio.legsPositionOnMotor.value]
            )
            csvwriter.writerow(["# connector ", self.emio.centerPartName.value])
            csvwriter.writerow(["# connector type ", self.emio.centerPartType.value])
            csvwriter.writerow(["Effector position", "Motor angle"])

    def writeToCSVFile(self):
        """
        Save the data in a csv file
        """
        with open(self.getFilename(), "a", newline="") as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=";")
            csvwriter.writerow(
                [
                    self.emio.effector.getMechanicalState().position.value[0][0:3],
                    [
                        self.emio.Motor0.JointActuator.angle.value,
                        self.emio.Motor1.JointActuator.angle.value,
                        self.emio.Motor2.JointActuator.angle.value,
                        self.emio.Motor3.JointActuator.angle.value,
                    ],
                ]
            )


def createScene(rootnode):
    """
    Emio simulation
    """
    import argparse
    import sys

    from parts.controllers.assemblycontroller import AssemblyController
    from parts.emio import Emio
    from utils.header import addHeader, addSolvers

    ## Parse args
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="Simulate a leg.")
    parser.add_argument(
        metavar="model_file",
        type=str,
        nargs="?",
        help="the path to the file containing the model",
        dest="model_file",
    )
    parser.add_argument(
        metavar="shape",
        type=str,
        nargs="?",
        help="the shape of the trajectory to follow",
        choices=["cube", "sphere"],
        default="sphere",
        dest="shape",
    )
    parser.add_argument(
        metavar="ratio",
        type=float,
        nargs="?",
        help="the division ratio of the target object's size",
        default=0.1,
        dest="ratio",
    )

    try:
        args = parser.parse_args()
    except SystemExit:
        Sofa.msg_error(sys.argv[0], "Invalid arguments, get defaults instead.")
        args = parser.parse_args([])

    Sofa.msg_info(
        os.path.basename(__file__),
        f"Using model file: {args.model_file}, shape: {args.shape}, ratio: {args.ratio}",
    )

    settings, modelling, simulation = addHeader(rootnode, inverse=False)

    rootnode.dt = 0.03
    rootnode.gravity = [0.0, -9810.0, 0.0]
    addSolvers(simulation)

    # Add Emio to the scene
    emio = Emio(
        name="Emio",
        legsName=["blueleg"],
        legsModel=["beam"],
        legsPositionOnMotor=[
            "counterclockwisedown",
            "clockwisedown",
            "counterclockwisedown",
            "clockwisedown",
        ],
        centerPartName="bluepart",
        centerPartType="rigid",
        extended=True,
    )
    if not emio.isValid():
        return

    simulation.addChild(emio)
    emio.attachCenterPartToLegs()
    assembly = AssemblyController(emio)
    emio.addObject(assembly)

    # Generation of the targets
    targetsPositions = (
        Targets(ratio=args.ratio, center=[0, -130, 0], size=80).sphere()
        if args.shape == "sphere"
        else Targets(ratio=args.ratio, center=[0, -130, 0], size=80).cube()
    )
    targets = modelling.addChild("SphereTargets")
    targets.addObject(
        "MechanicalObject",
        position=targetsPositions,
        showObject=True,
        showObjectScale=10,
        drawMode=0,
    )

    # Trajectory storage
    trajectory = modelling.addChild("Trajectory")
    trajectory.addObject(
        "MechanicalObject",
        position=[[0, 0, 0] for i in range(len(targetsPositions))],
        showObject=True,
        showObjectScale=10,
        drawMode=0,
        showColor=[1, 0, 0, 1],
    )

    # Effector
    emio.effector.addObject(
        "MechanicalObject", template="Rigid3", position=[0, 0, 0, 0, 0, 0, 1]
    )
    emio.effector.addObject("RigidMapping", index=0)

    for motor in emio.motors:
        motor.addObject(
            "JointConstraint",
            name="JointActuator",
            minDisplacement=-pi,
            maxDisplacement=pi,
            index=0,
            value=0,
            valueType="displacement",
        )

    # Adds components to connect to the robot
    emio.addConnectionComponents()

    # MLP Controller
    rootnode.addObject(MLPController(emio=emio, model_file=args.model_file))

    # We add a controller to go through the targets
    rootnode.addObject(
        TargetController(emio=emio, target=targets, assembly=assembly, steps=STEP)
    )
    return rootnode
