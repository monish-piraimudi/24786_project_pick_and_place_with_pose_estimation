# Emio Imitation Learning
::: highlight
##### Overview

This lab uses the Emio pick-and-place scene as the source of truth for the full imitation-learning pipeline.

The workflow is:
- watch the scripted expert in SOFA
- collect successful expert episodes from that same scene
- train an implicit behavior-cloning policy on saved compact state observations
- evaluate the learned policy in closed loop

The deployed policy is state-only. It consumes:
- a 17D `state_observation`

The state vector is:
- normalized TCP `x/y/z`
- normalized cube `x/y/z`
- normalized goal `x/y/z`
- normalized TCP-to-cube delta in `x/y/z`
- normalized cube-to-goal delta in `x/y/z`
- normalized gripper opening
- held flag

Saved rollout files may still include RGB `observation` frames for debugging and inspection, but the learned policy ignores them.

The policy predicts a 4D action by minimizing an energy model over state-action pairs:
- `motor0_angle`
- `motor1_angle`
- `motor2_angle`
- `motor3_angle`

At runtime the controller logs:
- `action` as the raw energy-minimizing proposal
- `executed_action` as the smoothed and clipped command that is actually applied

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
#include(assets/labs/24786_project_pick_and_place_with_pose_estimation/sections/1_task_overview.md)
#include(assets/labs/24786_project_pick_and_place_with_pose_estimation/sections/2_watch_expert.md)
#include(assets/labs/24786_project_pick_and_place_with_pose_estimation/sections/3_collect_expert_demonstrations.md)
#include(assets/labs/24786_project_pick_and_place_with_pose_estimation/sections/4_inspect_episode_format.md)
#include(assets/labs/24786_project_pick_and_place_with_pose_estimation/sections/5_train_motor_policy.md)
#include(assets/labs/24786_project_pick_and_place_with_pose_estimation/sections/6_evaluate_learned_policy.md)
#include(assets/labs/24786_project_pick_and_place_with_pose_estimation/sections/7_compare_expert_and_policy.md)
#include(assets/labs/24786_project_pick_and_place_with_pose_estimation/sections/8_summary.md)
