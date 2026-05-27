:::: collapse Step 2: Collect Expert Demonstrations
## Step 2: Collect Expert Demonstrations

`collect_il_dataset.py` runs the pick-and-place scene headlessly and records successful expert episodes.

The saved episodes now come from the same state-machine phase progression used at inference time. For each seed:
- a new block position is sampled from a continuous tray workspace in X/Z
- the pick location uses the same X/Z as the block
- the place target stays fixed by default
- the script records a 17D geometric `state_observation`, expert 4D motor-angle actions, executed actions, and tracker-assisted cube metadata at each step
- RGB `observation` frames may also be saved for debugging and comparison, but the state-only training path ignores them

By default the workspace bounds are:
- `x in [-35, 10]`
- `z in [-30, 20]`

The default collection path is camera-free so it can run without attached hardware. When a camera is connected, enable live RGB observations and marker-assisted tracking explicitly with:
- `--real-rgb-observation`
- `--camera-tracking`

Collect a starter dataset:

#python-button("assets/labs/lab_imitation/button_collect_il_dataset.py")

Recommended manual command:

```bash
/opt/emio-labs/resources/sofa/bin/python/bin/python3.10 \
  assets/labs/lab_imitation/collect_il_dataset.py \
  --episodes 100 \
  --max-attempts 140 \
  --workspace-bounds-mm -35 10 -30 20 \
  --save-failed-episodes
```

Windows 11 PowerShell:

```powershell
& "$env:LOCALAPPDATA\Programs\emio-labs\resources\sofa\bin\python\python.exe" `
  "assets/labs/lab_imitation/collect_il_dataset.py" `
  --episodes 100 `
  --max-attempts 140 `
  --workspace-bounds-mm -35 10 -30 20 `
  --save-failed-episodes
```

To collect with the physical camera attached, add:

```bash
  --real-rgb-observation \
  --camera-tracking
```

On Windows PowerShell, append:

```powershell
  --real-rgb-observation `
  --camera-tracking
```

Saved outputs:
- `data/results/il_pick_place/episodes/episode_*.npz`
- `data/results/il_pick_place/episodes/manifest.json`

::: exercise
**Exercise:**

Collect a small dataset and inspect the printed summaries. How many attempts were needed to gather the requested number of successful episodes?

:::

::::
