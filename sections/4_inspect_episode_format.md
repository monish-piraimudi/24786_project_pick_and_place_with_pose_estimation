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
- policy input is `state_observation`
- training target is `action`

In this version of the lab:
- `observation` may be present as an RGB image with shape `[N, H, W, 3]`
- `state_observation` has shape `[N, 17]`

The `state_observation` includes:
- TCP position in `x/y/z`
- cube position in `x/y/z`
- goal position in `x/y/z`
- TCP-to-cube delta in `x/y/z`
- cube-to-goal delta in `x/y/z`
- normalized gripper opening
- held flag

Action logging separates:
- `action`: the raw policy proposal or expert target action
- `executed_action`: the smoothed and clipped command actually sent to the controller

In `modules/imitation_data.py`:
- `EpisodeRecorder` stores one rollout
- `load_episode_paths(...)` finds saved episodes
- `split_episode_paths(...)` splits by episode
- `flatten_episode_dataset(...)` concatenates step data after splitting
- `flatten_state_episode_dataset(...)` concatenates state/action step data after splitting
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
