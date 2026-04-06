# Emio Imitation Learning
::: highlight
##### Overview

The goal of this lab is to teach a simulated Emio robot to pick up a cube and place it in a target zone by imitation learning.

Instead of hand-writing a controller for every situation, you will:
- inspect a scripted expert that already solves the task
- collect expert demonstrations in simulation
- train a behavior-cloning policy with PyTorch
- evaluate the learned policy offline
- compare the expert and the trained policy directly in SOFA

This lab uses privileged simulator state rather than camera images. The policy observes the robot state, cube pose, target pose, and task phase, then predicts the next end-effector motion and gripper command.

:::

:::: collapse Install Dependencies
## Install Dependencies

This lab uses `numpy`, `scipy`, `svg.path`, and `torch`.

Install the Python packages listed in this lab's `requirements.txt`:

#python-button("-m pip install -r 'assets/labs/24786_project_pick_and_place_with_pose_estimation/requirements.txt'")

You can also run the same command manually:

```bash
python -m pip install -r assets/labs/24786_project_pick_and_place_with_pose_estimation/requirements.txt
```

::: exercise
**Exercise:**

Install the dependencies, then open the following files and skim their top-level docstrings:
- `modules/pick_place_il.py`
- `modules/imitation_data.py`
- `modules/imitation_policy.py`

Identify which file defines the task and controller, which file handles dataset utilities, and which file defines the learned policy.

:::

::::

:::: collapse Task Overview
## Task Overview

The task is a single-cube pick-and-place problem:
- a cube starts at a randomized position on the table
- a green target zone is placed at another randomized position
- Emio must move down, grasp the cube, lift it, move to the target, release it, and retreat

Three code areas drive the pipeline:
- `modules/pick_place_il.py`
  Builds the SOFA scene, defines the 29-dimensional observation, the 4-dimensional action, the scripted expert finite-state machine, and the rollout controller.
- `modules/imitation_data.py`
  Records episodes, splits datasets by episode, flattens trajectories for training, and aggregates rollout metrics.
- `modules/imitation_policy.py`
  Defines the behavior-cloning neural network, saves the checkpoint, and reloads the checkpoint with normalization statistics for inference.

The observation seen by the policy contains:
- end-effector position
- cube position
- target position
- relative vectors between these objects
- four motor values
- gripper opening
- held/contact state
- one-hot task phase

The action predicted by the expert or learned policy contains:
- `dx`, `dy`, `dz` end-effector motion
- one scalar gripper command

::: exercise
**Exercise:**

Open `modules/pick_place_il.py` and find where the observation vector and action vector are assembled. Match each observation block to a physical quantity in the scene.

:::

::::

:::: collapse Step 1: Inspect The Task And Expert
## Step 1: Inspect The Task And Expert

Before training a model, it helps to watch the behavior we want the policy to imitate.

The scripted expert in `modules/pick_place_il.py` uses a finite-state machine with these phases:
- `approach_pick`
- `descend_pick`
- `close_gripper`
- `lift`
- `approach_place`
- `descend_place`
- `open_gripper`
- `retreat`

This expert is the "before training" baseline. It shows the desired task structure and produces the demonstrations used for behavior cloning.

#input("il_rollout_seed", "Rollout seed", "1000")

Run the expert baseline in SOFA:

#runsofa-button("assets/labs/24786_project_pick_and_place_with_pose_estimation/pick_place_il_scene.py", "--mode", "expert", "--seed", "il_rollout_seed", "--max-steps", "300")

You can also launch the same scene manually:

```bash
/opt/emio-labs/resources/sofa/bin/runSofa -a assets/labs/24786_project_pick_and_place_with_pose_estimation/pick_place_il_scene.py --argv "--mode expert --seed 1000 --max-steps 300"
```

::: exercise
**Exercise:**

Run the expert scene and describe the eight phases of the task in your own words. What information do you think the policy will need in order to imitate this behavior?

:::

::::

:::: collapse Step 2: Collect Demonstrations
## Step 2: Collect Demonstrations

The script `collect_il_dataset.py` rolls out the expert controller and saves each successful trajectory as an `.npz` episode file in `data/results/il_pick_place/episodes`.

Each saved rollout also contributes an entry to `manifest.json`, which summarizes:
- episode id
- seed
- success flags
- final place error
- failure phase
- saved file path

Collect a starter dataset:

#python-button("assets/labs/24786_project_pick_and_place_with_pose_estimation/collect_il_dataset.py --episodes 20 --max-attempts 40")

Equivalent command-line version:

```bash
python assets/labs/24786_project_pick_and_place_with_pose_estimation/collect_il_dataset.py --episodes 20 --max-attempts 40
```

By default, this stores trajectories in:
- `data/results/il_pick_place/episodes/episode_*.npz`
- `data/results/il_pick_place/episodes/manifest.json`

::: exercise
**Exercise:**

Collect a small dataset, then inspect `data/results/il_pick_place/episodes`. How many attempts were needed to save the requested number of successful expert episodes?

:::

::::

:::: collapse Step 3: Understand The Dataset Format
## Step 3: Understand The Dataset Format

Imitation-learning data is first stored by episode, not as one giant table. This preserves rollout order and makes it easier to inspect success and failure.

In `modules/imitation_data.py`:
- `EpisodeRecorder` logs one rollout step by step
- `split_episode_paths(...)` creates train, validation, and test splits by episode
- `flatten_episode_dataset(...)` turns many episodes into stacked training arrays
- `aggregate_rollout_metrics(...)` computes rollout-level metrics after evaluation

Each saved episode contains arrays such as:
- `observation`
- `action`
- `executed_action`
- `phase_index`
- `episode_step`
- `cube_pose`
- `target_position`
- `effector_pose`
- `gripper_opening`
- success flags

Why split by episode instead of by individual row? Consecutive timesteps from the same rollout are highly correlated. If they leak across train and validation sets, the model can appear better than it really is.

::: exercise
**Exercise:**

Open `modules/imitation_data.py` and explain:
1. why the split is done by episode
2. why the training dataset is flattened only after splitting
3. why rollout metrics are different from training loss

Then inspect one saved `.npz` episode and identify which keys correspond to the policy input and the expert target action.

:::

::::

:::: collapse Step 4: Train The Policy
## Step 4: Train The Policy

The learned policy is defined in `modules/imitation_policy.py` as a small multilayer perceptron with:
- input dimension `29`
- two hidden layers of size `128`
- ReLU activations
- output dimension `4`

Training is performed by `train_il_policy.py`. The script:
- loads saved episode files
- splits them into train, validation, and test sets
- flattens the trajectories
- normalizes the observations
- trains the behavior-cloning model with mean-squared error
- saves the checkpoint together with `obs_mean` and `obs_std`

Those normalization statistics are saved because inference must preprocess observations the same way training did.

Train the policy:

#python-button("assets/labs/24786_project_pick_and_place_with_pose_estimation/train_il_policy.py --dataset-dir data/results/il_pick_place/episodes --output-path data/results/il_pick_place/bc_policy.pth")

Equivalent command-line version:

```bash
python assets/labs/24786_project_pick_and_place_with_pose_estimation/train_il_policy.py --dataset-dir data/results/il_pick_place/episodes --output-path data/results/il_pick_place/bc_policy.pth
```

The trained checkpoint is saved to:
- `data/results/il_pick_place/bc_policy.pth`

::: exercise
**Exercise:**

Train the model and inspect the printed training and validation losses. Why is observation normalization especially important here, given that positions, motor values, gripper opening, and phase indicators all appear in the same input vector?

:::

::::

:::: collapse Step 5: Evaluate The Trained Policy
## Step 5: Evaluate The Trained Policy

The script `evaluate_il_policy.py` runs the learned controller in closed loop and reports rollout-level metrics.

These metrics measure task performance, not just prediction error:
- `pick_success`
- `place_success`
- `total_success`
- `final_place_error_mm`
- `dropped_object_rate`
- `failure_phase_counts`

Evaluate the trained policy:

#python-button("assets/labs/24786_project_pick_and_place_with_pose_estimation/evaluate_il_policy.py --mode policy --policy-path data/results/il_pick_place/bc_policy.pth")

Equivalent command-line version:

```bash
python assets/labs/24786_project_pick_and_place_with_pose_estimation/evaluate_il_policy.py --mode policy --policy-path data/results/il_pick_place/bc_policy.pth
```

Evaluation writes results to:
- `data/results/il_pick_place/eval/rollout_manifest.json`
- `data/results/il_pick_place/eval/metrics.json`

::: exercise
**Exercise:**

Compare the evaluation metrics to the expert behavior you observed earlier. If the total success rate is lower than the expert baseline, what kinds of mistakes do you think behavior cloning is making?

:::

::::

:::: collapse Step 6: Watch The Trained Policy In SOFA
## Step 6: Watch The Trained Policy In SOFA

Now compare the scripted expert and the trained policy on the same randomized task.

#input("il_policy_path", "Path to trained policy", "assets/labs/24786_project_pick_and_place_with_pose_estimation/data/results/il_pick_place/bc_policy.pth")

Use the same `il_rollout_seed` input from Step 1 so both controllers face the same cube and target placement.

Run the trained policy in SOFA:

#runsofa-button("assets/labs/24786_project_pick_and_place_with_pose_estimation/pick_place_il_scene.py", "--mode", "policy", "--policy-path", "il_policy_path", "--seed", "il_rollout_seed", "--max-steps", "300")

Equivalent command-line version:

```bash
/opt/emio-labs/resources/sofa/bin/runSofa -a assets/labs/24786_project_pick_and_place_with_pose_estimation/pick_place_il_scene.py --argv "--mode policy --policy-path assets/labs/24786_project_pick_and_place_with_pose_estimation/data/results/il_pick_place/bc_policy.pth --seed 1000 --max-steps 300"
```

To compare fairly:
1. run the expert baseline with one seed
2. keep the seed the same
3. run the trained policy
4. compare where the learned controller succeeds, hesitates, or fails

::: exercise
**Exercise:**

Run the expert and trained policy on the same seed. Describe at least one behavior that the learned policy copies well and one behavior where it differs from the expert.

:::

::::

:::: collapse Optional Extension: DAgger
## Optional Extension: DAgger

So far, the policy only learned from states visited by the expert. A common next step is DAgger: let the policy act, but query the expert for the correct action at the visited states and add those corrections to the dataset.

This lab supports a simple version of that idea:
- `collect_il_dataset.py --mode dagger`
- `pick_place_il_scene.py --mode dagger`

In `dagger` mode:
- the policy action is executed in the scene
- the expert action is still logged as the learning target

Example collection command:

```bash
python assets/labs/24786_project_pick_and_place_with_pose_estimation/collect_il_dataset.py --mode dagger --policy-path data/results/il_pick_place/bc_policy.pth --episodes 20 --max-attempts 40
```

::: exercise
**Exercise:**

Why might DAgger improve a behavior-cloning policy that performs well near expert trajectories but fails after making a small mistake of its own?

:::

::::

:::: collapse Summary
## Summary

In this lab you:
- observed the scripted expert that defines the target behavior
- collected expert demonstrations from the simulator
- inspected how episodes are stored and prepared for training
- trained a behavior-cloning policy
- evaluated the learned controller with rollout metrics
- compared the expert and the trained policy directly in SOFA

This is the standard imitation-learning workflow in its simplest form:
- expert policy
- demonstration dataset
- supervised learning
- closed-loop rollout evaluation

Once this baseline works, you can explore better policies, more data, DAgger, or richer observations.

::::
