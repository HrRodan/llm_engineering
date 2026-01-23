import numpy as np
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.feature_extraction.text import HashingVectorizer


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, dropout_prob):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual  # Skip connection
        return self.relu(out)


class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_size, num_layers=10, hidden_size=4096, dropout_prob=0.2):
        super(DeepNeuralNetwork, self).__init__()

        # First layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(num_layers - 2):
            self.residual_blocks.append(ResidualBlock(hidden_size, dropout_prob))

        # Output layer
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.input_layer(x)

        for block in self.residual_blocks:
            x = block(x)

        return self.output_layer(x)


class DeepNeuralNetworkRunner:
    def __init__(self, train, val):
        self.train_data = train
        self.val_data = val
        self.vectorizer = None
        self.model = None
        self.device = None
        self.loss_function = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataset = None
        self.train_loader = None
        self.y_mean = None
        self.y_std = None

        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        # L4 OPTIMIZATION: Enable TF32 for faster matrix multiplications on Ampere/Ada GPUs
        torch.set_float32_matmul_precision("high")

    def setup(self):
        self.vectorizer = HashingVectorizer(
            n_features=5000, stop_words="english", binary=True
        )

        train_documents = [item.summary for item in self.train_data]
        X_train_np = self.vectorizer.fit_transform(train_documents)
        self.X_train = torch.FloatTensor(X_train_np.toarray())
        y_train_np = np.array([float(item.price) for item in self.train_data])
        self.y_train = torch.FloatTensor(y_train_np).unsqueeze(1)

        val_documents = [item.summary for item in self.val_data]
        X_val_np = self.vectorizer.transform(val_documents)
        self.X_val = torch.FloatTensor(X_val_np.toarray())
        y_val_np = np.array([float(item.price) for item in self.val_data])
        self.y_val = torch.FloatTensor(y_val_np).unsqueeze(1)

        y_train_log = torch.log(self.y_train + 1)
        y_val_log = torch.log(self.y_val + 1)
        self.y_mean = y_train_log.mean()
        self.y_std = y_train_log.std()
        self.y_train_norm = (y_train_log - self.y_mean) / self.y_std
        self.y_val_norm = (y_val_log - self.y_mean) / self.y_std

        self.model = DeepNeuralNetwork(self.X_train.shape[1])
        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Deep Neural Network created with {total_params:,} parameters")

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"Using {self.device}")

        # L4 OPTIMIZATION: Move data to GPU immediately to avoid host-to-device transfer overhead
        self.X_train = self.X_train.to(self.device)
        self.y_train_norm = self.y_train_norm.to(self.device)

        # Validation data to GPU as well
        self.X_val = self.X_val.to(self.device)
        self.y_val_norm = self.y_val_norm.to(self.device)
        self.y_val = self.y_val.to(self.device)

        self.model.to(self.device)
        # L4 OPTIMIZATION: Compile the model to fuse kernels and optimize graph execution
        # mode="reduce-overhead" is crucial for small batches
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
            print("Model compiled with torch.compile(mode='reduce-overhead')")
        except Exception as e:
            print(f"Could not compile model: {e}")

        self.loss_function = nn.L1Loss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=0.001, weight_decay=0.01
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10, eta_min=0)

        # L4 OPTIMIZATION: Removed DataLoader. We will manually batch on GPU.
        # This eliminates the iterator overhead which is significant for small batches.
        self.batch_size = 512

    def train(self, epochs=5):
        # L4 OPTIMIZATION: Added tqdm to outer loop for overall progress
        # L4 OPTIMIZATION: Manual batching on GPU
        num_samples = self.X_train.shape[0]

        for epoch in tqdm(range(1, epochs + 1), desc="Epochs"):
            self.model.train()
            train_losses = []

            # Shuffle indices on GPU
            indices = torch.randperm(num_samples, device=self.device)

            # Helper to generate batches
            num_batches = (num_samples + self.batch_size - 1) // self.batch_size

            # Iterate through batches
            # Using range acts as the iterator, we slice the tensors directly
            for i in tqdm(
                range(num_batches), desc=f"Training epoch {epoch}", leave=False
            ):
                start_idx = i * self.batch_size
                end_idx = min(start_idx + self.batch_size, num_samples)

                batch_indices = indices[start_idx:end_idx]

                # Data is already on device, just slice it
                batch_X = self.X_train[batch_indices]
                batch_y = self.y_train_norm[batch_indices]

                self.optimizer.zero_grad()

                # L4 OPTIMIZATION: Mixed Precision training with BFloat16 (native on L4)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = self.model(batch_X)
                    loss = self.loss_function(outputs, batch_y)

                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                train_losses.append(loss.item())

            # Validation
            self.model.eval()
            with torch.no_grad():
                # Data already on device
                val_outputs = self.model(self.X_val)
                val_loss = self.loss_function(val_outputs, self.y_val_norm)

                # Convert back to original scale for meaningful metrics
                val_outputs_orig = torch.exp(val_outputs * self.y_std + self.y_mean) - 1
                mae = torch.abs(val_outputs_orig - self.y_val).mean()

            avg_train_loss = np.mean(train_losses)
            print(f"Epoch [{epoch}/{epochs}]")
            print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}")
            print(f"Val mean absolute error: ${mae.item():.2f}")
            print(f"Learning rate: {self.scheduler.get_last_lr()[0]:.6f}")

            self.scheduler.step()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)

    def inference(self, item):
        self.model.eval()
        with torch.no_grad():
            vector = self.vectorizer.transform([item.summary])
            vector = torch.FloatTensor(vector.toarray()).to(self.device)
            pred = self.model(vector)[0]
            result = torch.exp(pred * self.y_std + self.y_mean) - 1
            result = result.item()
        return max(0, result)
