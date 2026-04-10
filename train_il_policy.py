import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from modules.imitation_data import flatten_episode_dataset, load_episode_paths, split_episode_paths
from modules.imitation_policy import DEFAULT_ACTION_BOUNDS, ImplicitBCPolicy, save_policy_checkpoint


PROJECT_DIR = Path(__file__).resolve().parent
NEGATIVE_SAMPLES = 63


def _resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_DIR / path)


def _describe_episode_signature(path: Path) -> tuple[tuple[int, ...], tuple[int, ...]]:
    with np.load(path, allow_pickle=False) as episode:
        observation_shape = tuple(int(dim) for dim in episode["observation"].shape)
        action_shape = tuple(int(dim) for dim in episode["action"].shape)
    return observation_shape, action_shape


def _validate_image_episode_paths(episode_paths: list[Path]) -> tuple[tuple[int, int, int], tuple[int]]:
    if not episode_paths:
        raise SystemExit("Need at least 3 saved episodes to train/validate/test the policy")

    incompatible = []
    image_shapes = set()
    action_shapes = set()

    for episode_path in episode_paths:
        observation_shape, action_shape = _describe_episode_signature(episode_path)
        is_image_episode = (
            len(observation_shape) == 4
            and observation_shape[-1] == 3
            and len(action_shape) == 2
            and action_shape[-1] == 4
        )
        if not is_image_episode:
            incompatible.append((episode_path.name, observation_shape, action_shape))
            continue
        image_shapes.add(observation_shape[1:])
        action_shapes.add(action_shape[1:])

    if incompatible:
        message_lines = [
            "Dataset directory contains incompatible episodes. Implicit BC training requires an image-only dataset.",
            "Found the following incompatible episode shapes:",
        ]
        for episode_name, observation_shape, action_shape in incompatible[:10]:
            message_lines.append(
                f"  {episode_name}: observation_shape={observation_shape} action_shape={action_shape}"
            )
        if len(incompatible) > 10:
            message_lines.append(f"  ... and {len(incompatible) - 10} more")
        message_lines.append(
            "Move legacy vector/synthetic episodes out of this folder or point --dataset-dir to a clean real-RGB episode directory."
        )
        raise SystemExit("\n".join(message_lines))

    if len(image_shapes) != 1 or len(action_shapes) != 1:
        raise SystemExit(
            "Dataset directory contains inconsistent image or action shapes across episodes. "
            f"image_shapes={sorted(image_shapes)} action_shapes={sorted(action_shapes)}"
        )

    image_shape = next(iter(image_shapes))
    action_shape = next(iter(action_shapes))
    return image_shape, action_shape


def _make_loader(observations, actions, batch_size, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(observations).float(),
        torch.from_numpy(actions).float(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _prepare_images(train_obs, val_obs, test_obs):
    image_shape = tuple(int(dim) for dim in train_obs.shape[1:])
    train_obs = train_obs.astype(np.float32)
    val_obs = val_obs.astype(np.float32)
    test_obs = test_obs.astype(np.float32)

    if float(np.max(train_obs)) > 1.5:
        train_obs /= 255.0
        val_obs /= 255.0
        test_obs /= 255.0

    image_mean = train_obs.mean(axis=(0, 1, 2)).astype(np.float32)
    image_std = train_obs.std(axis=(0, 1, 2)).astype(np.float32)
    image_std = np.where(image_std < 1e-6, 1.0, image_std).astype(np.float32)

    def _norm(obs):
        if obs.size == 0:
            return np.zeros((0, *image_shape), dtype=np.float32)
        obs = ((obs - image_mean.reshape(1, 1, 1, -1)) / image_std.reshape(1, 1, 1, -1)).astype(np.float32)
        return np.transpose(obs, (0, 3, 1, 2)).astype(np.float32)

    return _norm(train_obs), _norm(val_obs), _norm(test_obs), image_mean, image_std, image_shape


def _sample_negative_actions(actions: torch.Tensor, action_bounds: np.ndarray, num_negatives: int) -> torch.Tensor:
    batch_size, action_dim = actions.shape
    lower = torch.as_tensor(action_bounds[:, 0], dtype=actions.dtype, device=actions.device)
    upper = torch.as_tensor(action_bounds[:, 1], dtype=actions.dtype, device=actions.device)
    global_samples = lower + (upper - lower) * torch.rand(
        batch_size,
        num_negatives,
        action_dim,
        device=actions.device,
        dtype=actions.dtype,
    )
    local_noise = 0.15 * (upper - lower)
    local_samples = actions.unsqueeze(1) + torch.randn_like(global_samples) * local_noise.view(1, 1, -1)
    local_samples = torch.clamp(local_samples, lower.view(1, 1, -1), upper.view(1, 1, -1))
    selector = torch.rand(batch_size, num_negatives, 1, device=actions.device, dtype=actions.dtype)
    mixed = torch.where(selector < 0.5, global_samples, local_samples)
    return mixed


def _implicit_bc_loss(model, batch_x, batch_y, action_bounds: np.ndarray, num_negatives: int) -> torch.Tensor:
    obs_embed = model.encode_observation(batch_x)
    pos_energy = model.forward_with_embedding(obs_embed, batch_y).unsqueeze(1)
    neg_actions = _sample_negative_actions(batch_y, action_bounds, num_negatives)
    obs_embed_neg = obs_embed.unsqueeze(1).repeat(1, num_negatives, 1).reshape(-1, obs_embed.shape[-1])
    neg_energy = model.forward_with_embedding(
        obs_embed_neg,
        neg_actions.reshape(-1, batch_y.shape[-1]),
    ).reshape(batch_x.shape[0], num_negatives)
    logits = -torch.cat([pos_energy, neg_energy], dim=1)
    targets = torch.zeros(batch_x.shape[0], dtype=torch.long, device=batch_x.device)
    return F.cross_entropy(logits, targets)


def _evaluate(model, loader, device, action_bounds: np.ndarray):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.inference_mode():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            loss = _implicit_bc_loss(model, batch_x, batch_y, action_bounds, NEGATIVE_SAMPLES)
            total_loss += float(loss.item())
            num_batches += 1
    return total_loss / max(1, num_batches)


def main():
    parser = argparse.ArgumentParser(description="Train the implicit BC policy for Emio pick and place")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_DIR / "data/results/il_pick_place/episodes",
        help="Directory containing episode_*.npz files. Relative paths are resolved from this lab folder.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROJECT_DIR / "data/results/il_pick_place/bc_policy.pth",
        help="Path where the trained policy checkpoint will be saved. Relative paths are resolved from this lab folder.",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--workspace-bounds-mm",
        type=float,
        nargs=4,
        metavar=("X_MIN", "X_MAX", "Z_MIN", "Z_MAX"),
        default=(-35.0, 10.0, -30.0, 20.0),
        help="Continuous object workspace bounds recorded in checkpoint metadata.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_dir = _resolve_project_path(Path(args.dataset_dir))
    output_path = _resolve_project_path(Path(args.output_path))

    episode_paths = load_episode_paths(dataset_dir)
    if len(episode_paths) < 3:
        raise SystemExit("Need at least 3 saved episodes to train/validate/test the policy")
    image_shape_signature, action_shape_signature = _validate_image_episode_paths(episode_paths)

    train_paths, val_paths, test_paths = split_episode_paths(episode_paths, seed=args.seed)
    train_obs, train_actions = flatten_episode_dataset(train_paths, observation_key="observation")
    val_obs, val_actions = flatten_episode_dataset(val_paths, observation_key="observation")
    test_obs, test_actions = flatten_episode_dataset(test_paths, observation_key="observation")

    if train_obs.ndim != 4 or train_obs.shape[-1] != 3:
        raise SystemExit(
            f"Expected RGB image observations with shape [N, H, W, 3], got {train_obs.shape}"
        )

    train_obs, val_obs, test_obs, image_mean, image_std, image_shape = _prepare_images(
        train_obs,
        val_obs,
        test_obs,
    )

    print("Dataset summary:")
    print(f"  dataset_dir={dataset_dir.resolve()}")
    print(f"  num_episodes={len(episode_paths)}")
    print(
        f"  observation_signature=[N, {image_shape_signature[0]}, {image_shape_signature[1]}, {image_shape_signature[2]}]"
    )
    print(f"  action_signature=[N, {action_shape_signature[0]}]")
    print(f"  split_episodes train={len(train_paths)} val={len(val_paths)} test={len(test_paths)}")
    print(f"  split_steps train={int(train_obs.shape[0])} val={int(val_obs.shape[0])} test={int(test_obs.shape[0])}")

    train_loader = _make_loader(train_obs, train_actions, args.batch_size, shuffle=True)
    val_loader = _make_loader(val_obs, val_actions, args.batch_size, shuffle=False)
    test_loader = _make_loader(test_obs, test_actions, args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImplicitBCPolicy(input_channels=int(image_shape[2])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    action_bounds = DEFAULT_ACTION_BOUNDS.copy()
    best_val_loss = float("inf")
    best_state_dict = None
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        num_batches = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = _implicit_bc_loss(model, batch_x, batch_y, action_bounds, NEGATIVE_SAMPLES)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            num_batches += 1

        train_loss /= max(1, num_batches)
        val_loss = _evaluate(model, val_loader, device, action_bounds)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_loss = _evaluate(model, test_loader, device, action_bounds)
    metadata = {
        "dataset_dir": str(dataset_dir.resolve()),
        "train_episodes": len(train_paths),
        "val_episodes": len(val_paths),
        "test_episodes": len(test_paths),
        "train_steps": int(train_obs.shape[0]),
        "val_steps": int(val_obs.shape[0]),
        "test_steps": int(test_obs.shape[0]),
        "best_val_loss": float(best_val_loss),
        "test_loss": float(test_loss),
        "model_type": "implicit_bc",
        "action_bounds": action_bounds.tolist(),
        "search_config": {
            "num_samples": 192,
            "num_elites": 24,
            "num_iters": 4,
            "min_std": 0.05,
        },
        "workspace_bounds_mm": [float(value) for value in args.workspace_bounds_mm],
    }
    save_policy_checkpoint(
        output_path,
        model.cpu(),
        image_mean=image_mean,
        image_std=image_std,
        image_shape=image_shape,
        metadata=metadata,
    )

    print(f"Saved policy checkpoint to {output_path.resolve()}")
    print(metadata)


if __name__ == "__main__":
    main()
