import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from modules.imitation_data import flatten_episode_dataset, load_episode_paths, split_episode_paths
from modules.imitation_policy import BehaviorCloningCNN, save_policy_checkpoint


PROJECT_DIR = Path(__file__).resolve().parent


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
            "Dataset directory contains incompatible episodes. Training now requires an image-only dataset.",
            "Found the following incompatible episode shapes:",
        ]
        for episode_name, observation_shape, action_shape in incompatible[:10]:
            message_lines.append(
                f"  {episode_name}: observation_shape={observation_shape} action_shape={action_shape}"
            )
        if len(incompatible) > 10:
            message_lines.append(f"  ... and {len(incompatible) - 10} more")
        message_lines.append(
            "Move legacy vector episodes out of this folder or point --dataset-dir to a clean image-episode directory."
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


def _evaluate(model, loader, device):
    criterion = torch.nn.MSELoss()
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.inference_mode():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            loss = criterion(model(batch_x), batch_y)
            total_loss += float(loss.item())
            num_batches += 1
    return total_loss / max(1, num_batches)


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

    train_obs = ((train_obs - image_mean.reshape(1, 1, 1, -1)) / image_std.reshape(1, 1, 1, -1)).astype(np.float32)
    if val_obs.size != 0:
        val_obs = ((val_obs - image_mean.reshape(1, 1, 1, -1)) / image_std.reshape(1, 1, 1, -1)).astype(np.float32)
    else:
        val_obs = np.zeros((0, *image_shape), dtype=np.float32)
    if test_obs.size != 0:
        test_obs = ((test_obs - image_mean.reshape(1, 1, 1, -1)) / image_std.reshape(1, 1, 1, -1)).astype(np.float32)
    else:
        test_obs = np.zeros((0, *image_shape), dtype=np.float32)

    train_obs = np.transpose(train_obs, (0, 3, 1, 2)).astype(np.float32)
    val_obs = np.transpose(val_obs, (0, 3, 1, 2)).astype(np.float32)
    test_obs = np.transpose(test_obs, (0, 3, 1, 2)).astype(np.float32)
    return train_obs, val_obs, test_obs, image_mean, image_std, image_shape


def main():
    parser = argparse.ArgumentParser(description="Train the image behavior-cloning policy for Emio pick and place")
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
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
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
            f"Expected image observations with shape [N, H, W, 3], got {train_obs.shape}"
        )

    train_obs, val_obs, test_obs, image_mean, image_std, image_shape = _prepare_images(
        train_obs,
        val_obs,
        test_obs,
    )

    print("Dataset summary:")
    print(f"  dataset_dir={dataset_dir.resolve()}")
    print(f"  num_episodes={len(episode_paths)}")
    print(f"  observation_signature=[N, {image_shape_signature[0]}, {image_shape_signature[1]}, {image_shape_signature[2]}]")
    print(f"  action_signature=[N, {action_shape_signature[0]}]")
    print(
        f"  split_episodes train={len(train_paths)} val={len(val_paths)} test={len(test_paths)}"
    )
    print(
        f"  split_steps train={int(train_obs.shape[0])} val={int(val_obs.shape[0])} test={int(test_obs.shape[0])}"
    )

    train_loader = _make_loader(train_obs, train_actions, args.batch_size, shuffle=True)
    val_loader = _make_loader(val_obs, val_actions, args.batch_size, shuffle=False)
    test_loader = _make_loader(test_obs, test_actions, args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BehaviorCloningCNN(input_channels=int(image_shape[2])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss()

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
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            num_batches += 1

        train_loss /= max(1, num_batches)
        val_loss = _evaluate(model, val_loader, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_loss = _evaluate(model, test_loader, device)
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
