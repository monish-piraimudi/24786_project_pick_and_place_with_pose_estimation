:::: collapse Step 6: Compare Expert And Policy
## Step 6: Compare Expert And Policy

Step 6 now supports two interactive GUI policy-inspection modes inside the same SOFA scene:
- a default inspection scene that does not require camera hardware
- an optional live-camera inspection scene for tracker-assisted cube updates

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
- computes the same 17D `state_observation` used during training
- keeps the place target fixed
- starts the learned-policy rollout automatically when the scene opens
- lets you move the cube in the scene and rerun after each rollout

If you have a camera attached and want tracker-assisted inspection, use this second scene:

#runsofa-button("assets/labs/24786_project_pick_and_place_with_pose_estimation/policy_inspect_camera_scene.py")

```bash
/opt/emio-labs/resources/sofa/bin/runSofa -a \
  assets/labs/24786_project_pick_and_place_with_pose_estimation/policy_inspect_camera_scene.py
```

Inside the live-camera GUI:
- keep the place target fixed
- the learned-policy rollout starts automatically when the scene opens
- the state-only policy consumes the same 17D geometric state vector built from the live scene state
- tracker updates are used to align the cube pose and diagnostics with the observed object
- live RGB frames remain useful for visual debugging
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
