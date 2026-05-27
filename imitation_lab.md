# Emio Imitation Learning
::: highlight
##### Overview

In this lab, you will learn about imitation learning and implement your own pipeline to teach Emio how to perform pick-and-place tasks. 

In particular, you will:
- watch the scripted expert controller in SOFA
- collect successful expert episodes from a SOFA simulation scene
- train an implicit behavior-cloning policy on compact state observations
- evaluate the learned policy in closed loop

By the end of the lab, you should understand the full workflow from expert demonstration collection to policy learning and evaluation. 
You will also learn the principles of implicit behavior cloning and its advantages over direct action regression.

:::

:::: collapse Install Dependencies
## Install Dependencies

This lab uses `numpy`, `scipy`, `svg.path`, and `torch`.

Install the Python packages listed in `requirements.txt`:
#python-button("assets/labs/lab_imitation/button_install_dependencies.py")

```bash
/opt/emio-labs/resources/sofa/bin/python/bin/python3.10 -m pip install -r assets/labs/lab_imitation/requirements.txt
```

Windows 11 PowerShell:

```powershell
& "$env:LOCALAPPDATA\Programs\emio-labs\resources\sofa\bin\python\python.exe" -m pip install -r assets/labs/lab_imitation/requirements.txt
```

SOFA-based collection and evaluation scripts must run with SOFA's Python 3.10 interpreter. In this lab that is typically:

- Linux: `/opt/emio-labs/resources/sofa/bin/python/bin/python3.10`
- Windows 11 PowerShell: `$env:LOCALAPPDATA\Programs\emio-labs\resources\sofa\bin\python\python.exe`

::: exercise
**Exercise:**

Open these files and identify their roles:
- `modules/pick_place_il.py`
- `modules/pick_place_il_runtime.py`
- `modules/emio_camera_observation.py`
- `modules/imitation_data.py`
- `modules/imitation_policy.py`
:::

:::: collapse Answer
## Answer

- `modules/pick_place_il.py` — defines the pick-and-place task, phases, and scripted expert controller
- `modules/pick_place_il_runtime.py` — handles the runtime loop, state machine transitions, and closed-loop execution
- `modules/emio_camera_observation.py` — manages camera initialization and RGB frame capture
- `modules/imitation_data.py` — episode recording, loading, splitting, and flattening utilities
- `modules/imitation_policy.py` — the implicit BC policy, CEM inference, and EMA smoothing
::::

#include(assets/labs/lab_imitation/sections/1_implicit_behavioral_cloning_primer.md)
#include(assets/labs/lab_imitation/sections/2_watch_expert.md)
#include(assets/labs/lab_imitation/sections/3_collect_expert_demonstrations.md)
#include(assets/labs/lab_imitation/sections/4_inspect_episode_format.md)
#include(assets/labs/lab_imitation/sections/5_train_motor_policy.md)
#include(assets/labs/lab_imitation/sections/6_evaluate_learned_policy.md)
#include(assets/labs/lab_imitation/sections/7_compare_expert_and_policy.md)
#include(assets/labs/lab_imitation/sections/8_summary.md)
#include(assets/labs/lab_imitation/sections/9_bonus_design_exercise.md)
