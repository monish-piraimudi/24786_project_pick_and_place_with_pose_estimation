# Emio Imitation Learning
::: highlight
##### Overview

This lab uses the Emio pick-and-place scene as the source of truth for the full imitation-learning pipeline.

In this lab, you will:
- watch the scripted expert in SOFA
- collect successful expert episodes from that same scene
- train an implicit behavior-cloning policy on saved compact state observations
- evaluate the learned policy in closed loop

By the end of the lab, you should understand how one scene supports the full workflow from expert demonstration to imitation-learning evaluation, and why this lab uses an implicit behavior-cloning policy instead of direct action regression.

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
#include(assets/labs/24786_project_pick_and_place_with_pose_estimation/sections/1b_implicit_behavioral_cloning_primer.md)
#include(assets/labs/24786_project_pick_and_place_with_pose_estimation/sections/2_watch_expert.md)
#include(assets/labs/24786_project_pick_and_place_with_pose_estimation/sections/3_collect_expert_demonstrations.md)
#include(assets/labs/24786_project_pick_and_place_with_pose_estimation/sections/4_inspect_episode_format.md)
#include(assets/labs/24786_project_pick_and_place_with_pose_estimation/sections/5_train_motor_policy.md)
#include(assets/labs/24786_project_pick_and_place_with_pose_estimation/sections/6_evaluate_learned_policy.md)
#include(assets/labs/24786_project_pick_and_place_with_pose_estimation/sections/7_compare_expert_and_policy.md)
#include(assets/labs/24786_project_pick_and_place_with_pose_estimation/sections/8_summary.md)
#include(assets/labs/24786_project_pick_and_place_with_pose_estimation/sections/9_bonus_design_exercise.md)
