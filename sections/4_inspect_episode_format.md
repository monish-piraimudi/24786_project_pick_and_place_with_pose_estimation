:::: collapse Step 3: Inspect The Episode Format
## Step 3: Inspect The Episode Format

Each rollout is saved as one `.npz` trajectory file. This preserves temporal order and makes it easy to inspect whole episodes before flattening them for supervised learning.

The most important keys for the learning pipeline are:
- `state_observation` — the 17D geometric state vector used as policy input
- `action` — the raw expert target action used as the supervised learning target
- `executed_action` — the smoothed and clipped command actually sent to the controller
- `phase_index` — which phase of the pick-and-place sequence the robot is currently in
- `pick_success`, `place_success`, `total_success` — per-step success flags used to filter and evaluate episodes

Additional diagnostic keys such as `cube_pose`, `effector_pose`, `gripper_opening`, and `task_score` are also saved and can be useful for debugging, but are not required for training.

For learning:
- policy input is `state_observation`
- policy target is `action`

The `observation` field may be present as an RGB image with shape `[N, H, W, 3]` when the camera is enabled, but training ignores it by default.

`state_observation` has shape `[N, 17]` and includes:
- TCP position in `x/y/z`
- cube position in `x/y/z`
- goal position in `x/y/z`
- TCP-to-cube delta in `x/y/z`
- cube-to-goal delta in `x/y/z`
- normalized gripper opening
- held flag (1.0 when the gripper is currently holding the cube, 0.0 otherwise)

Action logging separates:
- `action`: the raw policy proposal or expert target action
- `executed_action`: the smoothed and clipped command actually sent to the controller

**Why split by episode rather than by timestep?**

When building train and validation sets, it is important to split at the episode boundary rather than at the individual timestep level. Because consecutive frames within a single rollout share the same cube start position and follow a continuous trajectory, they are highly correlated with each other. If you split by row instead, near-identical frames from the same rollout can appear in both the training and validation sets, which makes validation loss an unreliable measure of how well the policy generalises to new situations. Splitting by episode ensures that every step in a validation episode is genuinely unseen during training.

::: exercise
**Exercise:**

Inspect one saved `.npz` file and answer:
1. What is the shape of `state_observation`?
2. What is the shape of `action`?
3. During which phase does the `held` flag first become 1.0?
4. Open two consecutive frames from the same episode and compare their `state_observation` vectors. What does this tell you about why validation must be split at the episode level?

:::

::::
