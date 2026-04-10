:::: collapse Step 4: Train The Hybrid Policy
## Step 4: Train The Hybrid Policy

`train_il_policy.py` trains an implicit behavior-cloning policy from the saved episodes.

With the default collection settings, the learned policy sees:
- varying pickup locations
- one fixed placement target
- the same RGB observation format at every timestep
- a fixed-order 5D low-dimensional state vector at every timestep

Training expects newly saved hybrid episodes from this lab folder. If the dataset directory contains old legacy image-only episodes mixed with hybrid episodes, training stops with a clear error instead of silently skipping them.

The training script:
- loads episode files
- splits them into train, validation, and test episodes
- flattens each split into stepwise arrays
- normalizes RGB observations channel-wise
- normalizes the 5D state vector feature-wise
- trains an energy model on positive expert actions plus sampled negative actions
- saves the checkpoint with image normalization and state normalization statistics

The current policy in `modules/imitation_policy.py` is hybrid and implicit:
- input shape `[H, W, 3]`
- state shape `[5]`
- action dimension `4`
- inference selects the action with minimum predicted energy over a bounded search distribution

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

Run training and inspect the printed losses. Why must the exact same image normalization, state normalization, and action bounds be reused at inference time?

:::

::::
