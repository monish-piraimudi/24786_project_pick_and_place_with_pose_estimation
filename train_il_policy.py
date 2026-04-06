import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from modules.imitation_data import flatten_episode_dataset, load_episode_paths, split_episode_paths
from modules.imitation_policy import BehaviorCloningMLP, save_policy_checkpoint


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


def main():
    parser = argparse.ArgumentParser(description="Train the behavior-cloning policy for Emio pick and place")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/results/il_pick_place/episodes"),
        help="Directory containing episode_*.npz files",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/results/il_pick_place/bc_policy.pth"),
        help="Path where the trained policy checkpoint will be saved",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    episode_paths = load_episode_paths(args.dataset_dir)
    if len(episode_paths) < 3:
        raise SystemExit("Need at least 3 saved episodes to train/validate/test the policy")

    train_paths, val_paths, test_paths = split_episode_paths(episode_paths, seed=args.seed)
    train_obs, train_actions = flatten_episode_dataset(train_paths)
    val_obs, val_actions = flatten_episode_dataset(val_paths)
    test_obs, test_actions = flatten_episode_dataset(test_paths)

    obs_mean = train_obs.mean(axis=0).astype(np.float32)
    obs_std = train_obs.std(axis=0).astype(np.float32)
    obs_std = np.where(obs_std < 1e-6, 1.0, obs_std).astype(np.float32)

    train_obs = ((train_obs - obs_mean) / obs_std).astype(np.float32)
    val_obs = ((val_obs - obs_mean) / obs_std).astype(np.float32)
    test_obs = ((test_obs - obs_mean) / obs_std).astype(np.float32)

    train_loader = _make_loader(train_obs, train_actions, args.batch_size, shuffle=True)
    val_loader = _make_loader(val_obs, val_actions, args.batch_size, shuffle=False)
    test_loader = _make_loader(test_obs, test_actions, args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BehaviorCloningMLP().to(device)
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
        "train_episodes": len(train_paths),
        "val_episodes": len(val_paths),
        "test_episodes": len(test_paths),
        "train_steps": int(train_obs.shape[0]),
        "val_steps": int(val_obs.shape[0]),
        "test_steps": int(test_obs.shape[0]),
        "best_val_loss": float(best_val_loss),
        "test_loss": float(test_loss),
    }
    save_policy_checkpoint(args.output_path, model.cpu(), obs_mean, obs_std, metadata=metadata)

    print(f"Saved policy checkpoint to {args.output_path}")
    print(metadata)


if __name__ == "__main__":
    main()
