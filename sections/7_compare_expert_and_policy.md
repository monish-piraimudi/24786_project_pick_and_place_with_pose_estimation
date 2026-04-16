:::: collapse Step 6: Compare Expert And Policy
## Step 6: Compare Expert And Policy

Step 6 uses one interactive GUI inspection scene inside SOFA.

Use the `runSofa` button below to open the camera-free learned-policy scene in Emio:

#runsofa-button("assets/labs/24786_project_pick_and_place_with_pose_estimation/policy_inspect_scene.py")

```bash
/opt/emio-labs/resources/sofa/bin/runSofa -a \
  assets/labs/24786_project_pick_and_place_with_pose_estimation/policy_inspect_scene.py
```

This default scene:
- does not require an attached Emio camera
- computes the same 17D `state_observation` used during training
- keeps the place target fixed
- starts the learned-policy rollout automatically when the scene opens
- lets you move the cube in the scene and rerun after each rollout
- is intended to be used in simulation first

This matches the usual Emio GUI workflow used in the other labs:
- launch one SOFA scene
- press *Play* to start the simulation
- stay in simulation mode while testing the learned policy
- use the GUI's *Simulation / Robot* switch only if you later want to connect to the physical robot outside this lab step

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

Open the GUI policy-inspection scene and test at least three different cube `x/z` positions in simulation mode. Which pickup offsets does the learned policy handle well, and where does it diverge from the scripted expert?

:::

::::
