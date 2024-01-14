import json
import optuna
import torch
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss
from pytorch_lightning.callbacks import EarlyStopping

# Load existing configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Extract trainer parameters
trainer_params = config["trainer"]

def objective(trial, training, train_dataloader, val_dataloader):
    # Define hyperparameter search space
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", [8, 16, 32])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    attention_head_size = trial.suggest_categorical("attention_head_size", [1, 2, 4])
    hidden_continuous_size = trial.suggest_categorical("hidden_continuous_size", [8, 16, 32])

    # Model
    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        loss=QuantileLoss(),
        log_interval=10
    )

    # Trainer setup with loaded parameters
    trainer = pl.Trainer(
        **trainer_params,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[EarlyStopping(monitor="val_loss", patience=3, mode="min")]
    )

    # Train the model
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

    # Objective: the best validation loss
    return trainer.callback_metrics["val_loss"].item()

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)  # Adjust the number of trials as needed

# Extract the best parameters for TFT
best_tft_params = study.best_trial.params

# Update the config dictionary
config["tft"] = best_tft_params

# Save updated configuration to JSON file
with open('train_config_optuna.json', 'w') as config_file:
    json.dump(config, config_file, indent=4)

# Print the best trial results
print("Best trial:")
print(f"Value: {study.best_trial.value}")
print("Params: ")
for key, value in best_tft_params.items():
    print(f"  {key}: {value}")
