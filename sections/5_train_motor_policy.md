:::: collapse Step 4: Train The State-Only Policy
## Step 4: Train The State-Only Policy

`train_il_policy.py` trains an implicit behavior-cloning policy from the saved episodes.

With the default collection settings, the learned policy sees:
- varying pickup locations
- one fixed placement target
- a fixed-order 17D geometric state vector at every timestep

Training expects episodes with `state_observation` and `action`. Saved RGB observations may still be present for logging, but training ignores them.

The training script:
- loads episode files
- splits them into train, validation, and test episodes
- flattens each split into stepwise state/action arrays
- normalizes the 17D state vector feature-wise
- trains an energy model on positive expert actions plus sampled negative actions
- saves the checkpoint with state normalization statistics plus CEM and smoothing metadata

The current policy in `modules/imitation_policy.py` is state-only and implicit:
- state shape `[17]`
- action dimension `4` of absolute motor angles
- inference selects the action with minimum predicted energy over a bounded CEM search distribution
- CEM warm-starts from the previous executed action
- the applied command is an EMA-smoothed version of the raw policy proposal

The only supported checkpoint type is `implicit_bc_motor_state_v1`.

Train the policy:

#python-button("assets/labs/24786_project_pick_and_place_with_pose_estimation/train_il_policy.py --dataset-dir data/results/il_pick_place/episodes --output-path data/results/il_pick_place/bc_policy.pth")

Manual command:

```bash
python assets/labs/24786_project_pick_and_place_with_pose_estimation/train_il_policy.py \
  --dataset-dir /home/dan/emio-labs/v25.12.01/assets/labs/24786_project_pick_and_place_with_pose_estimation/data/results/il_pick_place/episodes \
  --output-path /home/dan/emio-labs/v25.12.01/assets/labs/24786_project_pick_and_place_with_pose_estimation/data/results/il_pick_place/bc_policy.pth
```

Output:
- `data/results/il_pick_place/bc_policy.pth`

::: exercise
**Exercise:**

Run training and inspect the printed losses. Why must the exact same state normalization and action bounds be reused at inference time?

:::

::::
