# Emio Imitation Learning
::: highlight
##### Overview

This lab uses the Emio pick-and-place scene as the source of truth for the full imitation-learning pipeline.

The workflow is:
- watch the scripted expert in SOFA
- collect successful expert episodes from that same scene
- train an implicit behavior-cloning policy on saved RGB observations plus a small low-dimensional state vector
- evaluate the learned policy in closed loop

The deployed policy is hybrid. It consumes:
- an RGB observation
- a 5D `state_observation`

The state vector is:
- normalized phase index
- normalized gripper opening
- held flag
- normalized TCP height
- normalized cube height

The policy predicts a 4D action by minimizing an energy model over observation-action pairs:
- `dx`
- `dy`
- `dz`
- gripper command

The scene uses a compact scripted pick-and-place structure:
- kinematic block
- scripted attach/release grasp logic
- state-machine pick, lift, place, and retreat progression

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
- `modules/emio_camera_observation.py`
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

The saved episodes now come from the same state-machine phase progression used at learned-policy inference time. For each seed:
- the block spawn is sampled from a continuous tray workspace in X/Z
- the pick location uses the same X/Z as the block
- the place target stays fixed by default
- the script records RGB observations, a 5D `state_observation`, expert 4D actions, executed actions, and tracker-assisted cube metadata at each step

By default the workspace bounds are:
- `x in [-35, 10]`
- `z in [-30, 20]`

The default collection path is camera-free so it can run without attached hardware. When a camera is connected, enable live RGB observations and marker-assisted tracking explicitly with:
- `--real-rgb-observation`
- `--camera-tracking`

Collect a starter dataset:

#python-button("assets/labs/24786_project_pick_and_place_with_pose_estimation/collect_il_dataset.py --episodes 100 --max-attempts 140 --workspace-bounds-mm -35 10 -30 20 --save-failed-episodes")

Recommended manual command:

```bash
/opt/emio-labs/resources/sofa/bin/python/bin/python3.10 \
  assets/labs/24786_project_pick_and_place_with_pose_estimation/collect_il_dataset.py \
  --episodes 100 \
  --max-attempts 140 \
  --workspace-bounds-mm -35 10 -30 20 \
  --save-failed-episodes
```

To collect with the physical camera attached, add:

```bash
  --real-rgb-observation \
  --camera-tracking
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
- `state_observation`
- `action`
- `executed_action`
- `tracked_cube_position`
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
- policy input also includes `state_observation`
- training target is `action`

In this version of the lab:
- `observation` is an RGB image with shape `[N, H, W, 3]`
- `state_observation` has shape `[N, 5]`

In `modules/imitation_data.py`:
- `EpisodeRecorder` stores one rollout
- `load_episode_paths(...)` finds saved episodes
- `split_episode_paths(...)` splits by episode
- `flatten_episode_dataset(...)` concatenates step data after splitting
- `flatten_hybrid_episode_dataset(...)` concatenates image/state/action step data after splitting
- `aggregate_rollout_metrics(...)` summarizes closed-loop performance

Why split by episode instead of by timestep?
- consecutive frames from one rollout are highly correlated
- mixing them across train and validation would leak near-duplicate samples

::: exercise
**Exercise:**

Inspect one saved `.npz` file and answer:
1. what is the shape of `observation`?
2. what is the shape of `state_observation`?
3. what is the shape of `action`?
4. why is splitting by episode better than splitting by row?

:::

::::

:::: collapse Step 4: Train The Hybrid Policy
## Step 4: Train The Hybrid Policy

`train_il_policy.py` trains an implicit behavior-cloning policy from the saved episodes.

With the default collection settings, the learned policy sees:
- varying pickup locations
- one fixed placement target
- the same RGB observation format at every timestep
- a fixed-order 5D low-dimensional state vector at every timestep

Training expects newly saved hybrid episodes from this lab folder. If the dataset directory contains old legacy image-only episodes mixed with hybrid episodes, training stops with a clear error instead of silently skipping them.

The training script:
- loads episode files
- splits them into train, validation, and test episodes
- flattens each split into stepwise arrays
- normalizes RGB observations channel-wise
- normalizes the 5D state vector feature-wise
- trains an energy model on positive expert actions plus sampled negative actions
- saves the checkpoint with image normalization and state normalization statistics

The current policy in `modules/imitation_policy.py` is hybrid and implicit:
- input shape `[H, W, 3]`
- state shape `[5]`
- action dimension `4`
- inference selects the action with minimum predicted energy over a bounded search distribution

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

Run training and inspect the printed losses. Why must the exact same image normalization, state normalization, and action bounds be reused at inference time?

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

#python-button("assets/labs/24786_project_pick_and_place_with_pose_estimation/evaluate_il_policy.py --mode policy --policy-path data/results/il_pick_place/bc_policy.pth --workspace-bounds-mm -35 10 -30 20")

Recommended manual command:

```bash
/opt/emio-labs/resources/sofa/bin/python/bin/python3.10 \
  assets/labs/24786_project_pick_and_place_with_pose_estimation/evaluate_il_policy.py \
  --mode policy \
  --policy-path data/results/il_pick_place/bc_policy.pth \
  --workspace-bounds-mm -35 10 -30 20
```

Saved outputs:
- `data/results/il_pick_place/eval/rollout_manifest.json`
- `data/results/il_pick_place/eval/metrics.json`

By default evaluation uses:
- simulated Emio-view RGB observations and no tracker, so it can run without attached hardware
- the same 5D state vector used during hybrid training
- the same fixed place target used during collection

To evaluate with the camera attached, add:
- `--real-rgb-observation`
- `--camera-tracking`

::: exercise
**Exercise:**

Compare policy evaluation metrics to the expert baseline. If the learned policy has lower success, which phase tends to fail first?

:::

::::

:::: collapse Step 6: Compare Expert And Policy
## Step 6: Compare Expert And Policy

Step 6 now supports two interactive GUI policy-inspection modes inside the same SOFA scene:
- a default inspection scene that does not require camera hardware
- an optional live-camera inspection scene for real RGB policy input

The recommended comparison flow is:
1. watch the scripted expert in the default GUI scene
2. launch the default policy-inspection scene with a trained checkpoint
3. inspect the learned rollout without requiring a camera
4. if a camera is attached, launch the live-camera inspection scene
5. move the cube within the tray workspace while the camera sees it
6. compare that visual behavior to the scripted evaluation metrics

Use the default `runSofa` button below to open the camera-free learned-policy scene in Emio:

#runsofa-button("assets/labs/24786_project_pick_and_place_with_pose_estimation/policy_inspect_scene.py")

```bash
/opt/emio-labs/resources/sofa/bin/runSofa -a \
  assets/labs/24786_project_pick_and_place_with_pose_estimation/policy_inspect_scene.py
```

This default scene:
- does not require an attached Emio camera
- uses the simulated Emio-view observation path
- computes the same 5D `state_observation` used during training
- keeps the place target fixed
- starts the learned-policy rollout automatically when the scene opens
- lets you move the cube in the scene and rerun after each rollout

If you have a camera attached and want live RGB policy input, use this second scene:

#runsofa-button("assets/labs/24786_project_pick_and_place_with_pose_estimation/policy_inspect_camera_scene.py")

```bash
/opt/emio-labs/resources/sofa/bin/runSofa -a \
  assets/labs/24786_project_pick_and_place_with_pose_estimation/policy_inspect_camera_scene.py
```

Inside the live-camera GUI:
- keep the place target fixed
- the learned-policy rollout starts automatically when the scene opens
- the policy consumes live RGB frames from the Emio camera
- the policy also consumes the same 5D state vector built from the live scene state
- tracker updates are used to align the cube pose and diagnostics with the observed object
- after the rollout finishes, move the cube to a new tray position to trigger another trial automatically

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

Open the default GUI policy-inspection scene and test at least three different cube `x/z` positions. If a camera is attached, repeat the same test in the live-camera scene. Which pickup offsets does the learned policy handle well, and where does it diverge from the scripted expert?

:::

::::

:::: collapse Summary
## Summary

In this lab you:
- inspected the pick-and-place scene in SOFA
- used that exact scene to generate imitation-learning episodes
- trained a hybrid image-plus-state implicit behavior-cloning policy
- evaluated the learned controller in closed loop
- compared expert and learned rollout performance

The key idea is that one scene drives the whole pipeline:
- scene definition in `modules/pick_place_il.py`
- runtime collection and evaluation in `modules/pick_place_il_runtime.py`
- real RGB observation capture in `modules/emio_camera_observation.py`
- offline training in `train_il_policy.py`

From here, you can experiment with:
- more demonstrations
- wider workspace bounds
- richer low-dimensional state features
- different implicit-policy search hyperparameters
- stronger image encoders

::::
