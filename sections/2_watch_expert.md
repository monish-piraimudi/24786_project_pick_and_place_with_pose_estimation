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
