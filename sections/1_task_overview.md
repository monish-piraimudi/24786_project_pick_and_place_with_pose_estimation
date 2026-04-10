:::: collapse Task Overview
## Task Overview

The task is a single-cube pick-and-place problem:
- the cube starts on the tray
- the target location is marked in green
- Emio approaches the cube, closes the gripper, lifts, moves to the target, releases, and retreats

The current pipeline is split across a few small modules:
- `modules/pick_place_il.py`
  Defines the SOFA scene, the evaluator, the tuned task parameters, and the scripted `AutoPickAndPlaceDemo`.
- `modules/pick_place_il_runtime.py`
  Builds that exact scene headlessly, samples tray-wide object poses, records episodes, and runs learned-policy rollouts.
- `modules/emio_camera_observation.py`
  Opens `EmioCamera`, updates frames and trackers, and provides real RGB observations plus tracker positions.
- `modules/sim_emio_camera_observation.py`
  Provides the default simulation-only RGB observation path by rendering an Emio-like perspective camera view when real RGB is disabled.
- `modules/imitation_data.py`
  Saves episodes, loads hybrid image/state datasets, splits by episode, and computes rollout metrics.
- `modules/imitation_policy.py`
  Defines the hybrid implicit BC energy model, action search, and checkpoint save/load helpers.

The scene is the source of truth. The runtime scripts do not define a separate task; they only drive the scene for:
- collection
- evaluation
- observation capture
- logging

::: exercise
**Exercise:**

Open `modules/pick_place_il.py` and find:
1. `_default_task_tuning()`
2. `PickAndPlaceEvaluator`
3. `AutoPickAndPlaceDemo`
4. `createScene(rootnode)`

How does the scene decide whether the block has been attached, lifted, and placed?

:::

::::
