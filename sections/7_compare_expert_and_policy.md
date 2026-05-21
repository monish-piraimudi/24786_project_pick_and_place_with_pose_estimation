:::: collapse Step 6: Compare Expert And Policy
## Step 6: Compare Expert And Policy

In imitation learning, the policy does not invent a strategy from scratch. Instead, it learns to reproduce patterns from the expert demonstrations you collected earlier in the lab.

Step 6 is where you inspect what that learned behavior looks like in practice. By launching the policy-inspection scene at different cube start positions, you can see where the learned controller behaves like the expert, where it generalizes well, and where it starts to diverge from the demonstrated behavior.

Before you launch the scene, you can choose the initial cube start position on the tray:

#input("inspect_cube_x_mm", "Cube start X (mm)", "-3")
#input("inspect_cube_z_mm", "Cube start Z (mm)", "12")

Use the `runSofa` button below to open the camera-free learned-policy scene in Emio with that selected start pose. As you watch the rollout, focus on what the policy appears to have learned from the demonstrations, not just on whether it succeeds or fails:

#runsofa-button("assets/labs/24786_project_pick_and_place_with_pose_estimation/policy_inspect_scene.py", "--cube-x-mm", "inspect_cube_x_mm", "--cube-z-mm", "inspect_cube_z_mm")

```bash
/opt/emio-labs/resources/sofa/bin/runSofa -a \
  assets/labs/24786_project_pick_and_place_with_pose_estimation/policy_inspect_scene.py \
  --cube-x-mm -3 \
  --cube-z-mm 12
```

Windows 11 PowerShell:

```powershell
& "$env:LOCALAPPDATA\Programs\emio-labs\resources\sofa\bin\runSofa.exe" -a `
  "assets/labs/24786_project_pick_and_place_with_pose_estimation/policy_inspect_scene.py" `
  --cube-x-mm -3 `
  --cube-z-mm 12
```

::: exercise
**Exercise:**

Open the GUI policy-inspection scene and test at least three different cube `x/z` positions in simulation mode.

For each position, ask yourself:
- Where does the learned policy behave like the scripted expert, and where does it diverge?
- What does that suggest about what the policy learned from the demonstrations?
- If the policy fails, is the problem more likely to come from missing demonstrations, limited state information, or the action-search and control process?

Summarize which pickup offsets the policy handles well, which ones are harder, and what those patterns tell you about the strengths and limits of imitation learning in this lab.

:::

::::
