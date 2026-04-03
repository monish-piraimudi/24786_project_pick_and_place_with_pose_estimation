import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class PytorchMLPReg:
    def __init__(self, batch_size=200, model_file=None):
        self.batch_size = batch_size
        self.model_file = model_file

        # ============ Exercise 1: Fill in below ===========
        # Implement the neural network as described in emio-labs "Create Model"

        self.model = nn.Sequential(
            nn.Linear(3, 128),
            nn.Sigmoid(),
            nn.Linear(128, 128),
            nn.Sigmoid(),
            nn.Linear(128, 4),
        )

        # =========================================

        if model_file:
            self.load(model_file)
            print(f"Loaded model from {model_file}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def save(self, file_path):
        torch.save(self.model.state_dict(), file_path)

    def load(self, file_path):
        self.model.load_state_dict(
            torch.load(file_path, weights_only=True, map_location=torch.device("cpu"))
        )
        self.model.eval()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X).float().to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
        return predictions

    def get_score(self, test_loader):
        loss_function = nn.MSELoss()
        with torch.no_grad():
            test_loss = 0.0
            all_y_true = []
            all_y_pred = []
            for batch_x, batch_y in test_loader:
                outputs = self.model(batch_x)
                test_loss += loss_function(outputs, batch_y).item()
                all_y_true.append(batch_y)
                all_y_pred.append(outputs)
            test_loss /= len(test_loader)
            return test_loss

    def score(self, X_test, y_test) -> float:
        self.model.eval()

        if isinstance(X_test, np.ndarray):
            X_test = torch.from_numpy(X_test).float().to(self.device)
            y_test = torch.from_numpy(y_test).float().to(self.device)

        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
        return self.get_score(test_loader)

    def train(self, X_train, y_train, X_test=None, y_test=None, n_epochs=10_000):
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir="data/logs/pytorch_mlp")

        torch.manual_seed(1)

        # Convert numpy arrays to PyTorch tensors
        X_train = torch.from_numpy(X_train).float().to(self.device)
        y_train = torch.from_numpy(y_train).float().to(self.device)
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if X_test is not None and y_test is not None:
            X_test = torch.from_numpy(X_test).float().to(self.device)
            y_test = torch.from_numpy(y_test).float().to(self.device)
            test_dataset = TensorDataset(X_test, y_test)
            test_loader = DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=False
            )

        # ============ Exercise 2: Fill in below ===========
        # Implement the training loop, by replacing "None" below by the
        # appropriate function call.

        # solution:
        optimizer = optim.Adam(self.model.parameters())
        loss_function = nn.MSELoss()

        print("Starting training...")
        for epoch in range(n_epochs):
            for batch_x, batch_y in loader:

                # important: have to reset gradients
                optimizer.zero_grad()

                # Forward pass: Create predictions
                outputs = self.model(batch_x)

                # Compare to labels using the cost function
                loss = loss_function(outputs, batch_y)

                # Backward pass: calculate gradients
                loss.backward()

                # Take a step using the optimizer.
                optimizer.step()

                # =======================
                writer.add_scalar("train_loss", loss.item(), epoch)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}", end=", ")
                if X_test is not None and y_test is not None:
                    self.model.eval()

                    test_loss = self.get_score(test_loader)
                    print(f"Test Loss: {test_loss:.4f}")
                    writer.add_scalar("test_loss", test_loss, epoch)
                print("")
        writer.close()
        print("Training completed. To view loss curves, run from terminal:")
        print("tensorboard --logdir=data/logs/pytorch_mlp/")
