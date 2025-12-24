import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from lmbm.models.lstm import NextIntLSTM
from lmbm.models.mlp import NextIntMLP
from lmbm.models.transformer import NextIntTransformer


class SequenceTrainer:
    def __init__(self, device: torch.device):
        self.device = device
        self.run_history = []

    def train(self, model: nn.Module, train_loader: DataLoader,
              val_loader: DataLoader, config: Dict[str, Any],
              model_dir: str = "checkpoints") -> Dict[str, Any]:
        """
        Train model with early stopping, save best model, return history.
        """
        model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=config['lr_patience'], factor=0.5)

        best_val_loss = float('inf')
        patience_counter = 0
        train_losses, val_losses = [], []

        Path(model_dir).mkdir(exist_ok=True)
        run_id = len(self.run_history)
        best_model_path = f"{model_dir}/best_model_run_{run_id}.pt"

        for epoch in range(config['max_epochs']):
            # Train
            model.train()
            train_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validate
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    logits = model(x)
                    loss = criterion(logits, y)
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'epoch': epoch,
                    'val_loss': val_loss
                }, best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= config['patience']:
                    break

        # Record run
        run_result = {
            'run_id': run_id,
            'best_val_loss': best_val_loss,
            'epochs_trained': len(train_losses),
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            **config
        }
        self.run_history.append(run_result)

        return run_result


def create_model(model_type: str, config: dict) -> nn.Module:
    vocab_size = config['vocab_size']
    seq_len = config['seq_len']

    if model_type == 'lstm':
        return NextIntLSTM(
            vocab_size=vocab_size,
            embed_dim=config['embed_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
        )

    elif model_type == 'mlp':
        return NextIntMLP(
            vocab_size=vocab_size,
            embed_dim=config['embed_dim'],
            seq_len=seq_len,
            hidden_dim=config['hidden_dim'],
            num_hidden_layers=config['num_layers'],
        )

    elif model_type == 'transformer':
        return NextIntTransformer(
            vocab_size=vocab_size,
            seq_len=seq_len,
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config.get('dropout', 0.1),
        )

    raise ValueError(f"Unknown model_type: {model_type}")


def run_sweep(hyperparams: Dict[str, list], train_dataset, val_dataset,
              batch_size: int = 32, csv_path: str = "runs.csv"):
    """
    Run hyperparameter sweep, log everything to CSV.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = SequenceTrainer(device)

    # Split datasets if needed
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    all_results = []

    # Cartesian product of hyperparams
    import itertools
    keys, values = zip(*hyperparams.items())
    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        config.update({
            'batch_size': batch_size,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset)
        })

        print(f"Training {config['model_type']} with config: {config}")

        model = create_model(config['model_type'], config)
        result = trainer.train(model, train_loader, val_loader, config)
        all_results.append(result)

    # Save to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(all_results)} runs to {csv_path}")

    # Best run
    best_run = df.loc[df['best_val_loss'].idxmin()]
    print(f"Best run: {best_run['best_val_loss']:.4f} val loss")

    return df
