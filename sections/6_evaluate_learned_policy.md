:::: collapse Step 5: Evaluate The Learned Policy
## Step 5: Evaluate The Learned Policy

`evaluate_il_policy.py` rolls out either:
- the expert controller
- the learned policy

Evaluation is closed-loop and reports rollout-level metrics rather than just supervised loss.

Useful metrics include:
- `pick_success`
- `place_success`
- `total_success`
- `final_place_error_mm`
- `dropped_object_rate`
- `failure_phase_counts`

Evaluate the learned policy:

#python-button("assets/labs/24786_project_pick_and_place_with_pose_estimation/evaluate_il_policy.py --mode policy --policy-path data/results/il_pick_place/bc_policy.pth --workspace-bounds-mm -35 10 -30 20")

Recommended manual command:

```bash
/opt/emio-labs/resources/sofa/bin/python/bin/python3.10 \
  assets/labs/24786_project_pick_and_place_with_pose_estimation/evaluate_il_policy.py \
  --mode policy \
  --policy-path data/results/il_pick_place/bc_policy.pth \
  --workspace-bounds-mm -35 10 -30 20
```

Saved outputs:
- `data/results/il_pick_place/eval/rollout_manifest.json`
- `data/results/il_pick_place/eval/metrics.json`

By default evaluation uses:
- the same 17D geometric state vector used during state-only training
- the same fixed place target used during collection
- no camera input is required unless you want live tracker updates or rollout image logging

To evaluate with the camera attached, add:
- `--real-rgb-observation`
- `--camera-tracking`

::: exercise
**Exercise:**

Compare policy evaluation metrics to the expert baseline. If the learned policy has lower success, which phase tends to fail first?

:::

::::
