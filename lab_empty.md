# Emio Imitation Learning
::: highlight
##### Overview

This lab uses the Emio pick-and-place scene as the source of truth for the full imitation-learning pipeline.

The workflow is:
- watch the scripted expert in SOFA
- collect randomized expert episodes from that same scene
- train a behavior-cloning CNN on saved RGB observations
- evaluate the learned policy in closed loop

The task is simulation-only. The saved policy input is a compact rendered RGB observation, and the expert target is a 4D action:
- `dx`
- `dy`
- `dz`
- gripper command

The scene uses a compact scripted pick-and-place structure:
- kinematic block
- scripted attach/release grasp logic
- timed pick, lift, place, and retreat demo

:::

:::: collapse Install Dependencies
## Install Dependencies

This lab uses `numpy`, `scipy`, `svg.path`, and `torch`.

Install the Python packages listed in `requirements.txt`:

#python-button("-m pip install -r 'assets/labs/24786_project_pick_and_place_with_pose_estimation/requirements.txt'")

Manual command:

```bash
python -m pip install -r assets/labs/24786_project_pick_and_place_with_pose_estimation/requirements.txt
```

SOFA-based collection and evaluation scripts must run with SOFA's Python 3.10 interpreter. In this lab that is:

```bash
/opt/emio-labs/resources/sofa/bin/python/bin/python3.10
```

::: exercise
**Exercise:**

Open these files and identify their roles:
- `modules/pick_place_il.py`
- `modules/pick_place_il_runtime.py`
- `modules/camera_observation.py`
- `modules/imitation_data.py`
- `modules/imitation_policy.py`

:::

::::

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
  Builds that exact scene headlessly, applies randomized initial conditions, records episodes, and runs learned-policy rollouts.
- `modules/camera_observation.py`
  Renders the synthetic RGB observation used by training and inference.
- `modules/imitation_data.py`
  Saves episodes, loads datasets, splits by episode, and computes rollout metrics.
- `modules/imitation_policy.py`
  Defines the CNN policy and checkpoint save/load helpers.

The scene is the source of truth. The runtime scripts do not define a separate task; they only drive the scene for:
- collection
- evaluation
- image rendering
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

:::: collapse Step 1: Watch The Expert In SOFA
## Step 1: Watch The Expert In SOFA

The SOFA scene launched from `lab_empty.py` runs the scripted expert demo directly. This is the behavior that later generates the imitation-learning episodes.

Launch the scene:

#runsofa-button("assets/labs/24786_project_pick_and_place_with_pose_estimation/lab_empty.py")

Equivalent command-line version:

```bash
/opt/emio-labs/resources/sofa/bin/runSofa -a assets/labs/24786_project_pick_and_place_with_pose_estimation/lab_empty.py
```

The demo phases are:
- `approach_pick`
- `descend_pick`
- `close_gripper`
- `lift`
- `approach_place`
- `descend_place`
- `open_gripper`
- `retreat`

::: exercise
**Exercise:**

Watch one full expert rollout in SOFA and describe what happens in each of the eight phases.

:::

::::

:::: collapse Step 2: Collect Expert Demonstrations
## Step 2: Collect Expert Demonstrations

`collect_il_dataset.py` runs the pick-and-place scene headlessly and records successful expert episodes.

The saved episodes come from the scene's own timed demo. For each seed:
- the block spawn is jittered in X/Z
- the place target stays fixed by default
- the pick waypoint is derived from the jittered block pose
- the script records RGB observations and expert 4D actions at each step

By default:
- `object_jitter_mm = 15`
- `place_jitter_mm = 0`

Set both to `0` if you want the untuned base layout with no randomization. If you want more pickup diversity while keeping placement fixed, increase only `object_jitter_mm`.

Collect a starter dataset:

#python-button("assets/labs/24786_project_pick_and_place_with_pose_estimation/collect_il_dataset.py --episodes 100 --max-attempts 40 --object-jitter-mm 15 --place-jitter-mm 0 --save-failed-episodes")

Recommended manual command:

```bash
/opt/emio-labs/resources/sofa/bin/python/bin/python3.10 \
  assets/labs/24786_project_pick_and_place_with_pose_estimation/collect_il_dataset.py \
  --episodes 100 \
  --max-attempts 40 \
  --object-jitter-mm 15 \
  --place-jitter-mm 0 \
  --save-failed-episodes
```

Saved outputs:
- `data/results/il_pick_place/episodes/episode_*.npz`
- `data/results/il_pick_place/episodes/manifest.json`

::: exercise
**Exercise:**

Collect a small dataset and inspect the printed summaries. How many attempts were needed to gather the requested number of successful episodes?

:::

::::

:::: collapse Step 3: Inspect The Episode Format
## Step 3: Inspect The Episode Format

Each rollout is saved as one `.npz` trajectory file. This preserves temporal order and makes it easy to inspect whole episodes before flattening them for supervised learning.

Important saved keys include:
- `observation`
- `action`
- `executed_action`
- `phase_index`
- `episode_step`
- `cube_pose`
- `object_position`
- `pick_position`
- `target_position`
- `effector_pose`
- `gripper_opening`
- `task_score`
- `task_lifted`
- `task_placed`
- `pick_success`
- `place_success`
- `total_success`

For learning:
- policy input is `observation`
- training target is `action`

In `modules/imitation_data.py`:
- `EpisodeRecorder` stores one rollout
- `load_episode_paths(...)` finds saved episodes
- `split_episode_paths(...)` splits by episode
- `flatten_episode_dataset(...)` concatenates step data after splitting
- `aggregate_rollout_metrics(...)` summarizes closed-loop performance

Why split by episode instead of by timestep?
- consecutive frames from one rollout are highly correlated
- mixing them across train and validation would leak near-duplicate samples

::: exercise
**Exercise:**

Inspect one saved `.npz` file and answer:
1. what is the shape of `observation`?
2. what is the shape of `action`?
3. why is splitting by episode better than splitting by row?

:::

::::

:::: collapse Step 4: Train The Image Policy
## Step 4: Train The Image Policy

`train_il_policy.py` trains a convolutional behavior-cloning model from the saved episodes.

With the default collection settings, the learned policy sees image observations from:
- varying pickup locations
- one fixed placement target
- the same rendered RGB observation format at every timestep

Training expects the newly saved image episodes from this lab folder. If the dataset directory contains old legacy vector episodes mixed with image episodes, training now stops with a clear error instead of silently skipping them.

The training script:
- loads episode files
- splits them into train, validation, and test episodes
- flattens each split into stepwise arrays
- normalizes RGB observations channel-wise
- trains the CNN with mean-squared error
- saves the checkpoint with image normalization statistics

The current policy in `modules/imitation_policy.py` is image-only:
- input shape `[H, W, 3]`
- output dimension `4`

Train the policy:

#python-button("assets/labs/24786_project_pick_and_place_with_pose_estimation/train_il_policy.py --dataset-dir data/results/il_pick_place/episodes --output-path data/results/il_pick_place/bc_policy.pth")

Manual command:

```bash
python assets/labs/24786_project_pick_and_place_with_pose_estimation/train_il_policy.py \
  --dataset-dir /home/dan/emio-labs/v25.12.01/assets/labs/24786_project_pick_and_place_with_pose_estimation/data/results/il_pick_place/episodes \
  --output-path /home/dan/emio-labs/v25.12.01/assets/labs/24786_project_pick_and_place_with_pose_estimation/data/results/il_pick_place/bc_policy.pth
```

Output:
- `data/results/il_pick_place/bc_policy.pth`

::: exercise
**Exercise:**

Run training and inspect the printed losses. Why must the exact same image normalization be reused at inference time?

:::

::::

:::: collapse Step 5: Evaluate The Learned Policy
## Step 5: Evaluate The Learned Policy

`evaluate_il_policy.py` rolls out either:
- the expert controller
- the learned policy

Evaluation is closed-loop and reports rollout-level metrics rather than just supervised loss.

Useful metrics include:
- `pick_success`
- `place_success`
- `total_success`
- `final_place_error_mm`
- `dropped_object_rate`
- `failure_phase_counts`

Evaluate the learned policy:

#python-button("assets/labs/24786_project_pick_and_place_with_pose_estimation/evaluate_il_policy.py --mode policy --policy-path data/results/il_pick_place/bc_policy.pth")

Recommended manual command:

```bash
/opt/emio-labs/resources/sofa/bin/python/bin/python3.10 \
  assets/labs/24786_project_pick_and_place_with_pose_estimation/evaluate_il_policy.py \
  --mode policy \
  --policy-path data/results/il_pick_place/bc_policy.pth
```

Saved outputs:
- `data/results/il_pick_place/eval/rollout_manifest.json`
- `data/results/il_pick_place/eval/metrics.json`

::: exercise
**Exercise:**

Compare policy evaluation metrics to the expert baseline. If the learned policy has lower success, which phase tends to fail first?

:::

::::

:::: collapse Step 6: Compare Expert And Policy
## Step 6: Compare Expert And Policy

Step 6 now supports an interactive GUI policy-inspection mode inside the same SOFA scene.

The recommended comparison flow is:
1. watch the scripted expert in the default GUI scene
2. launch the scene in `policy_inspect` mode with a trained checkpoint
3. choose cube `x/z` positions with the GUI controls
4. watch one learned-policy rollout start automatically
5. compare that visual behavior to the scripted evaluation metrics

Use the `runSofa` button below to open the interactive learned-policy scene in Emio:

#runsofa-button("assets/labs/24786_project_pick_and_place_with_pose_estimation/policy_inspect_scene.py")

```bash
/opt/emio-labs/resources/sofa/bin/runSofa -a \
  assets/labs/24786_project_pick_and_place_with_pose_estimation/policy_inspect_scene.py
```

Inside the GUI:
- use `Cube start X (mm)` and `Cube start Z (mm)` to choose the initial cube pose
- keep the place target fixed
- the learned-policy rollout starts automatically when the scene opens
- after the rollout finishes, adjust the cube position to trigger another trial automatically

Scripted evaluation is still useful for aggregate metrics. Example expert evaluation:

```bash
/opt/emio-labs/resources/sofa/bin/python/bin/python3.10 \
  assets/labs/24786_project_pick_and_place_with_pose_estimation/evaluate_il_policy.py \
  --mode expert \
  --episodes 20 \
  --start-seed 1000
```

Example policy evaluation on the same seeds:

```bash
/opt/emio-labs/resources/sofa/bin/python/bin/python3.10 \
  assets/labs/24786_project_pick_and_place_with_pose_estimation/evaluate_il_policy.py \
  --mode policy \
  --policy-path data/results/il_pick_place/bc_policy.pth \
  --episodes 20 \
  --start-seed 1000
```

::: exercise
**Exercise:**

Open the GUI in `policy_inspect` mode and test at least three different cube `x/z` positions. Which pickup offsets does the learned policy handle well, and where does it diverge from the scripted expert?

:::

::::

:::: collapse Summary
## Summary

In this lab you:
- inspected the pick-and-place scene in SOFA
- used that exact scene to generate imitation-learning episodes
- trained an image-based behavior-cloning policy
- evaluated the learned controller in closed loop
- compared expert and learned rollout performance

The key idea is that one scene drives the whole pipeline:
- scene definition in `modules/pick_place_il.py`
- runtime collection and evaluation in `modules/pick_place_il_runtime.py`
- image rendering in `modules/camera_observation.py`
- offline training in `train_il_policy.py`

From here, you can experiment with:
- more demonstrations
- more jitter
- different CNN architectures
- richer synthetic camera observations

::::
