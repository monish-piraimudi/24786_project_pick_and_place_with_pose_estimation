:::: collapse Step 4: Train The State-Only Policy
## Step 4: Train The State-Only Policy

Now that you have a dataset of expert demonstrations, you can train the behavior-cloning policy. `train_il_policy.py` takes the saved episodes and learns to reproduce the expert's actions from the 17D geometric state vector alone — no camera image is required.

With the default collection settings, the learned policy sees:
- varying pickup locations
- one fixed placement target
- a fixed-order 17D geometric state vector at every timestep

The training script prepares the data by loading the episode files, splitting them into train, validation, and test episodes at the episode boundary (not the timestep level), and flattening each split into stepwise state/action arrays. The 17D state vector is then normalized feature-wise so that all input dimensions are on a comparable scale. The normalization statistics computed here are saved as part of the checkpoint and must be reused exactly at inference time — any mismatch will silently corrupt the inputs the policy receives.

The model itself is an energy-based implicit policy: given a state and a candidate action, it predicts a scalar energy score. At inference time, a Cross-Entropy Method (CEM) search finds the action with the lowest predicted energy within the allowed motor-angle bounds. The search warm-starts from the previously executed action to keep motion smooth, and the selected action is further stabilized by an exponential moving average before being sent to the controller.

The only supported checkpoint type is `implicit_bc_motor_state_v1`.

Train the policy:

#python-button("assets/labs/24786_project_pick_and_place_with_pose_estimation/button_train_il_policy.py")

Manual command:

```bash
python assets/labs/24786_project_pick_and_place_with_pose_estimation/train_il_policy.py \
  --dataset-dir ~/emio-labs/v25.12.01/assets/labs/24786_project_pick_and_place_with_pose_estimation/data/results/il_pick_place/episodes \
  --output-path ~/emio-labs/v25.12.01/assets/labs/24786_project_pick_and_place_with_pose_estimation/data/results/il_pick_place/bc_policy.pth
```

Output:
- `data/results/il_pick_place/bc_policy.pth`

::: exercise
**Exercise:**

Run training and watch how the training and validation losses evolve across epochs. Do they decrease together, or does one plateau or diverge earlier? What does the gap between them suggest about how well the policy is generalising to unseen episodes? If you collected more demonstrations, what would you expect to happen to that gap?

:::

::::
