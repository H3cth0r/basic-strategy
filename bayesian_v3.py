

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Imports""")
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import yfinance as yf
    import ta
    import joblib # For saving scalers
    import json   # For saving metadata

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from blitz.modules import BayesianLinear
    from blitz.utils import variational_estimator

    import matplotlib.pyplot as plt
    import warnings
    import os 
    return (
        BayesianLinear,
        MinMaxScaler,
        joblib,
        json,
        mean_squared_error,
        nn,
        np,
        optim,
        os,
        pd,
        plt,
        ta,
        torch,
        train_test_split,
        variational_estimator,
        warnings,
        yf,
    )


@app.cell
def _(warnings):
    warnings.filterwarnings('ignore')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Configuration Constants""")
    return


@app.cell
def _():
    TICKER = 'BTC-USD'
    DATA_PERIOD = '7d'
    INTERVAL = '1m'
    N_LAGS = 15
    N_STEPS_AHEAD = 5
    TEST_SIZE = 0.2
    SEED = 42
    return (
        DATA_PERIOD,
        INTERVAL,
        N_LAGS,
        N_STEPS_AHEAD,
        SEED,
        TEST_SIZE,
        TICKER,
    )


@app.cell
def _():
    # Bayesian NN parameters
    HIDDEN_UNITS_1 = 64
    HIDDEN_UNITS_2 = 32
    LEARNING_RATE = 0.006
    EPOCHS = 200 # Reduced for quicker testing, you can increase it back
    # BATCH_SIZE = 128
    BATCH_SIZE = 256
    SAMPLE_NBR_ELBO = 5
    COMPLEXITY_WEIGHT_BASE = 1
    N_PREDICTION_SAMPLES = 100
    return (
        BATCH_SIZE,
        COMPLEXITY_WEIGHT_BASE,
        EPOCHS,
        HIDDEN_UNITS_1,
        HIDDEN_UNITS_2,
        LEARNING_RATE,
        N_PREDICTION_SAMPLES,
        SAMPLE_NBR_ELBO,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Helper Functions""")
    return


@app.cell
def _(np, torch):
    def set_seed(seed_value: int):
        """Sets random seeds for reproducibility."""
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value) 
        print(f"Seed set to {seed_value}")

    def get_device() -> torch.device:
        """Determines and returns the available device (GPU or CPU)."""
        if torch.cuda.is_available():
            print("CUDA (GPU) is available. Using GPU.")
            return torch.device("cuda")
        else:
            print("CUDA (GPU) not available. Using CPU.")
            return torch.device("cpu")
    return get_device, set_seed


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Data Processing Functions""")
    return


@app.cell
def _(MinMaxScaler, np, pd, ta, torch, train_test_split, yf):
    def load_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
        """Loads historical market data using yfinance."""
        print(f"Loading data for {ticker} | Period: {period}, Interval: {interval}")
        data = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
        data = data.drop(columns=['Dividends', 'Stock Splits'], errors='ignore') # Ignore errors if columns don't exist
        print(f"  Loaded data shape: {data.shape}")
        return data

    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Adds technical analysis features to the DataFrame."""
        print("Adding technical indicators...")
        df_ta = ta.add_all_ta_features(df.copy(), open='Open', high='High', low='Low', close='Close', volume='Volume', fillna=True)
        print(f"  Data shape after adding TAs: {df_ta.shape}")
        return df_ta

    def select_features(df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
        """Selects a predefined list of features from the DataFrame."""
        print("Selecting features...")
        available_features = [f for f in feature_list if f in df.columns]
        print(f"  Selected {len(available_features)} features: {available_features}")
        return df[available_features]

    def create_lagged_features(df: pd.DataFrame, feature_cols: list, n_lags: int) -> tuple[pd.DataFrame, list]:
        """Creates lagged versions of specified features."""
        print(f"Creating {n_lags} lagged features...")
        df_lagged = df.copy()
        lagged_feature_cols = []
        for lag in range(1, n_lags + 1):
            for col in feature_cols:
                new_col_name = f'{col}_lag_{lag}'
                df_lagged[new_col_name] = df_lagged[col].shift(lag)
                lagged_feature_cols.append(new_col_name)
        print(f"  Added {len(lagged_feature_cols)} lagged feature columns.")
        return df_lagged, lagged_feature_cols

    def create_target_variables(df: pd.DataFrame, n_steps_ahead: int, target_base_col: str = 'Close') -> tuple[pd.DataFrame, list]:
        """Creates target variables by shifting the target base column."""
        print(f"Creating {n_steps_ahead} target steps ahead based on '{target_base_col}'...")
        df_targets = df.copy()
        target_cols = []
        for i in range(1, n_steps_ahead + 1):
            target_col = f'target_{i}'
            df_targets[target_col] = df_targets[target_base_col].shift(-i)
            target_cols.append(target_col)
        print(f"  Created target columns: {target_cols}")
        return df_targets, target_cols

    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Removes rows with NaN values resulting from lagging/shifting."""
        print("Cleaning data (dropping NaNs)...")
        initial_rows = df.shape[0]
        df_cleaned = df.dropna()
        final_rows = df_cleaned.shape[0]
        print(f"  Dropped {initial_rows - final_rows} rows containing NaNs.")
        print(f"  Final data shape for modeling: {df_cleaned.shape}")
        return df_cleaned

    def prepare_final_data(df: pd.DataFrame, base_feature_cols: list, lagged_feature_cols: list, target_cols: list) -> tuple[pd.DataFrame, pd.DataFrame, list]:
        """Separates features (X) and targets (y) into final matrices."""
        print("Preparing final feature (X) and target (y) matrices...")
        feature_cols_final = base_feature_cols + lagged_feature_cols
        X = df[feature_cols_final]
        y = df[target_cols]
        print(f"  Feature Matrix X shape: {X.shape}")
        print(f"  Target matrix y shape: {y.shape}")
        return X, y, feature_cols_final

    def scale_data(X: pd.DataFrame, y: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
        """Scales features and targets using MinMaxScaler."""
        print("Scaling data...")
        feature_scaler = MinMaxScaler()
        X_scaled = feature_scaler.fit_transform(X)

        target_scaler = MinMaxScaler()
        y_scaled = target_scaler.fit_transform(y)
        print("  Data scaled using MinMaxScaler.")
        return X_scaled, y_scaled, feature_scaler, target_scaler

    def split_data(X_scaled: np.ndarray, y_scaled: np.ndarray, test_size: float, seed: int, shuffle: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Splits data into training and testing sets."""
        print(f"Splitting data into train/test sets (Test size: {test_size}, Shuffle: {shuffle})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=test_size, shuffle=shuffle, random_state=seed
        )
        print(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"  X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        return X_train, X_test, y_train, y_test

    def create_dataloaders(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, batch_size: int, device: torch.device) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.Tensor, torch.Tensor]:
        """Creates PyTorch DataLoaders for training and testing."""
        print(f"Creating PyTorch DataLoaders (Batch size: {batch_size})...")
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Shuffle batches each epoch

        test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print("  DataLoaders created.")
        # Return test tensors as well for later evaluation if needed outside loader
        return train_loader, test_loader, X_test_tensor, y_test_tensor
    return (
        add_technical_indicators,
        clean_data,
        create_dataloaders,
        create_lagged_features,
        create_target_variables,
        load_data,
        prepare_final_data,
        scale_data,
        select_features,
        split_data,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Model Definition""")
    return


@app.cell
def _(BayesianLinear, nn, variational_estimator):
    @variational_estimator
    class BayesianStockPredictor(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_units_1, hidden_units_2):
            super().__init__()
            self.blinear1 = BayesianLinear(input_dim, hidden_units_1)
            self.relu1 = nn.ReLU()
            self.blinear2 = BayesianLinear(hidden_units_1, hidden_units_2)
            self.relu2 = nn.ReLU()
            self.blinear_out = BayesianLinear(hidden_units_2, output_dim)

        def forward(self, x):
            x = self.relu1(self.blinear1(x))
            x = self.relu2(self.blinear2(x))
            x = self.blinear_out(x)
            return x
    return (BayesianStockPredictor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Model Training and Evaluation Functions""")
    return


@app.cell
def _(BayesianStockPredictor, nn, optim, torch):
    def build_model(input_dim: int, output_dim: int, hidden_units_1: int, hidden_units_2: int, device: torch.device) -> BayesianStockPredictor:
        """Initializes the Bayesian Neural Network model."""
        print("Building Bayesian NN model...")
        model = BayesianStockPredictor(input_dim, output_dim, hidden_units_1, hidden_units_2).to(device)
        print(model)
        return model

    def train_model(model: nn.Module, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader,
                    learning_rate: float, epochs: int, complexity_weight_base: float, sample_nbr_elbo: int,
                    device: torch.device) -> tuple[nn.Module, list, list]:
        """Trains the Bayesian model."""
        print("Starting model training...")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss() # Used for evaluation part of ELBO and test loss

        # Calculate complexity weight based on training set size
        complexity_weight = complexity_weight_base / len(train_loader.dataset)
        print(f"  Using complexity weight: {complexity_weight:.2e}")

        train_losses = []
        test_losses = []
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                # Ensure batch tensors are on the correct device (redundant if DataLoader does it, but safe)
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                loss = model.sample_elbo(inputs=batch_x,
                                         labels=batch_y,
                                         criterion=criterion,
                                         sample_nbr=sample_nbr_elbo,
                                         complexity_cost_weight=complexity_weight)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_epoch_loss)

            # Evaluate on test set periodically
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for batch_x_test, batch_y_test in test_loader:
                    batch_x_test, batch_y_test = batch_x_test.to(device), batch_y_test.to(device)
                    outputs = model(batch_x_test) # Get mean prediction for eval
                    loss = criterion(outputs, batch_y_test)
                    test_loss += loss.item()
            avg_test_loss = test_loss / len(test_loader)
            test_losses.append(avg_test_loss)

            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                print(f'  Epoch [{epoch + 1}/{epochs}], Train ELBO Loss: {avg_epoch_loss:.6f}, Test MSE Loss: {avg_test_loss:.6f}')

        print('\nTraining finished.')
        return model, train_losses, test_losses
    return build_model, train_model


@app.cell
def _(MinMaxScaler, mean_squared_error, nn, np, plt, torch):
    def plot_losses(train_losses: list, test_losses: list, epochs: int):
        """Plots the training and validation loss curves."""
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epochs + 1), train_losses, label='Training ELBO Loss')
        plt.plot(range(1, epochs + 1), test_losses, label='Test MSE Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()

    def generate_predictions(model: nn.Module, test_loader: torch.utils.data.DataLoader, n_samples: int, device: torch.device) -> np.ndarray:
        """Generates multiple prediction samples from the Bayesian model."""
        print(f"Generating {n_samples} prediction samples from the test set...")
        model.eval()
        predictions_list = []
        with torch.no_grad():
            for _ in range(n_samples):
                sample_preds = []
                for batch_x, _ in test_loader:
                    batch_x = batch_x.to(device)
                    batch_preds = model(batch_x)
                    sample_preds.append(batch_preds.cpu().numpy())
                # Concatenate predictions from all batches for this sample
                predictions_list.append(np.concatenate(sample_preds, axis=0))

        # Stack samples along a new axis (samples, time_steps, features)
        y_pred_samples_scaled = np.stack(predictions_list)
        print(f"  Generated prediction samples shape: {y_pred_samples_scaled.shape}")
        return y_pred_samples_scaled

    def inverse_transform_predictions(y_pred_samples_scaled: np.ndarray, y_test_scaled: np.ndarray, target_scaler: MinMaxScaler) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Inverse transforms scaled predictions and actuals back to original scale."""
        print("Inverse transforming predictions and actuals...")
        # Calculate mean and std dev across samples (axis 0)
        y_pred_mean_scaled = y_pred_samples_scaled.mean(axis=0)
        y_pred_std_scaled = y_pred_samples_scaled.std(axis=0)

        # Inverse transform mean predictions and original test data
        y_pred_mean = target_scaler.inverse_transform(y_pred_mean_scaled)
        y_test_orig = target_scaler.inverse_transform(y_test_scaled) # Assuming y_test_scaled is numpy

        # Calculate uncertainty bounds (e.g., 95% CI) in scaled space and inverse transform
        y_pred_upper_scaled = y_pred_mean_scaled + 1.96 * y_pred_std_scaled
        y_pred_lower_scaled = y_pred_mean_scaled - 1.96 * y_pred_std_scaled
        y_pred_upper = target_scaler.inverse_transform(y_pred_upper_scaled)
        y_pred_lower = target_scaler.inverse_transform(y_pred_lower_scaled)

        print("  Inverse transformation complete.")
        return y_pred_mean, y_pred_lower, y_pred_upper, y_test_orig, y_pred_std_scaled # Also return std dev if needed

    def evaluate_performance(y_test_orig: np.ndarray, y_pred_mean: np.ndarray, n_steps_ahead: int):
        """Calculates and prints RMSE for each prediction step."""
        print("\nEvaluating performance (RMSE)...")
        rmses = []
        for i in range(n_steps_ahead):
            rmse_step_i = np.sqrt(mean_squared_error(y_test_orig[:, i], y_pred_mean[:, i]))
            print(f'  Test RMSE for Step {i + 1} Ahead: {rmse_step_i:.4f}')
            rmses.append(rmse_step_i)
        return rmses

    return (
        evaluate_performance,
        generate_predictions,
        inverse_transform_predictions,
        plot_losses,
    )


@app.cell
def _(np, plt):
    def plot_predictions(y_test_orig: np.ndarray, y_pred_mean: np.ndarray, y_pred_lower: np.ndarray, y_pred_upper: np.ndarray, step_index: int, ticker: str, n_steps_ahead: int):
        """Plots actual vs predicted values for a specific step ahead."""
        step_label = f"Step {step_index + 1}"
        title = f'{ticker} - Prediction vs Actual ({step_label} Ahead / {n_steps_ahead} Total Steps)'

        print(f"\nPlotting predictions vs actuals for {step_label}...")
        plt.figure(figsize=(15, 7))
        plt.plot(y_test_orig[:, step_index], label=f'Actual Price ({step_label})', color='blue', alpha=0.7)
        plt.plot(y_pred_mean[:, step_index], label=f'Predicted Mean Price ({step_label})', color='orange')

        # Plot uncertainty bounds
        plt.fill_between(range(len(y_pred_mean)),
                         y_pred_lower[:, step_index],
                         y_pred_upper[:, step_index],
                         color='orangered', alpha=0.3, label='95% Confidence Interval')

        plt.title(title)
        plt.xlabel('Time Steps (Test Set)')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()
    return (plot_predictions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Export Function""")
    return


@app.cell
def _(MinMaxScaler, joblib, json, nn, os, torch):
    def export_artifacts(model: nn.Module, feature_scaler: MinMaxScaler, target_scaler: MinMaxScaler,
                         feature_cols_final: list, target_cols: list, config: dict, filename_prefix: str = "btc_model"):
        """Exports the trained model state, scalers, and necessary metadata."""
        print(f"Exporting artifacts with prefix '{filename_prefix}'...")
        export_dir = "exported_model"
        os.makedirs(export_dir, exist_ok=True)

        # 1. Save Model State Dictionary
        model_path = os.path.join(export_dir, f"{filename_prefix}_state_dict.pth")
        torch.save(model.state_dict(), model_path)
        print(f"  Model state dict saved to: {model_path}")

        # 2. Save Scalers and Metadata (Column names, N_Steps_Ahead etc.)
        scaler_metadata_path = os.path.join(export_dir, f"{filename_prefix}_scalers_metadata.joblib")
        artifacts = {
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler,
            'feature_cols_final': feature_cols_final, # Needed to structure input data for prediction
            'target_cols': target_cols,               # Needed for understanding output structure
            'config': config                          # Save relevant config like N_LAGS, N_STEPS_AHEAD
        }
        joblib.dump(artifacts, scaler_metadata_path)
        print(f"  Scalers and metadata saved to: {scaler_metadata_path}")

        # (Optional) Save config separately as JSON for easy reading
        config_path = os.path.join(export_dir, f"{filename_prefix}_config.json")
        # Filter config to only include relevant items for prediction if desired
        prediction_config = {k: v for k, v in config.items() if k in ['N_LAGS', 'N_STEPS_AHEAD', 'HIDDEN_UNITS_1', 'HIDDEN_UNITS_2']}
        with open(config_path, 'w') as f:
            json.dump(prediction_config, f, indent=4)
        print(f"  Configuration saved to: {config_path}")
    return (export_artifacts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Main Pipeline Function""")
    return


@app.cell
def _(
    add_technical_indicators,
    build_model,
    clean_data,
    create_dataloaders,
    create_lagged_features,
    create_target_variables,
    evaluate_performance,
    export_artifacts,
    generate_predictions,
    get_device,
    inverse_transform_predictions,
    load_data,
    plot_losses,
    plot_predictions,
    prepare_final_data,
    scale_data,
    select_features,
    set_seed,
    split_data,
    train_model,
):
    def train_and_test(
        ticker: str, period: str, interval: str,
        base_features: list, n_lags: int, n_steps_ahead: int,
        test_size: float, seed: int, shuffle_split: bool,
        hidden_units_1: int, hidden_units_2: int,
        learning_rate: float, epochs: int, batch_size: int,
        sample_nbr_elbo: int, complexity_weight_base: float,
        n_prediction_samples: int,
        use_random_seed: bool = True,
        export_prefix: str = "model_export"

    ) -> dict:
        """
        Runs the complete training and testing pipeline.

        Returns:
            dict: A dictionary containing results like the trained model, scalers,
                  predictions, actual values, and metrics.
        """
        # --- Setup ---
        if use_random_seed:
            set_seed(seed)
        device = get_device()

        # Capture config for export
        config = {
            'TICKER': ticker, 'DATA_PERIOD': period, 'INTERVAL': interval,
            'N_LAGS': n_lags, 'N_STEPS_AHEAD': n_steps_ahead, 'TEST_SIZE': test_size, 'SEED': seed,
            'HIDDEN_UNITS_1': hidden_units_1, 'HIDDEN_UNITS_2': hidden_units_2,
            'LEARNING_RATE': learning_rate, 'EPOCHS': epochs, 'BATCH_SIZE': batch_size,
            'SAMPLE_NBR_ELBO': sample_nbr_elbo, 'COMPLEXITY_WEIGHT_BASE': complexity_weight_base,
            'N_PREDICTION_SAMPLES': n_prediction_samples
        }

        # --- Data Loading and Preprocessing ---
        df_raw = load_data(ticker, period, interval)
        df_ta = add_technical_indicators(df_raw)
        df_selected = select_features(df_ta, base_features)
        base_feature_cols = df_selected.columns.tolist() # Get actual base features after selection
        df_lagged, lagged_feature_cols = create_lagged_features(df_selected, base_feature_cols, n_lags)
        df_targets, target_cols = create_target_variables(df_lagged, n_steps_ahead)
        df_cleaned = clean_data(df_targets)
        X, y, feature_cols_final = prepare_final_data(df_cleaned, base_feature_cols, lagged_feature_cols, target_cols)

        # --- Scaling and Splitting ---
        # Note: Scaling before splitting (as in the original code) can lead to data leakage.
        # Ideally, fit scaler only on training data and transform both train and test.
        # Replicating original logic here:
        X_scaled, y_scaled, feature_scaler, target_scaler = scale_data(X, y)
        X_train, X_test, y_train, y_test = split_data(X_scaled, y_scaled, test_size, seed, shuffle_split)

        # --- DataLoaders ---
        train_loader, test_loader, X_test_tensor, y_test_tensor = create_dataloaders(
            X_train, y_train, X_test, y_test, batch_size, device
        )

        # --- Model Building and Training ---
        input_dim = X_train.shape[1]
        output_dim = n_steps_ahead
        model = build_model(input_dim, output_dim, hidden_units_1, hidden_units_2, device)
        trained_model, train_losses, test_losses = train_model(
            model, train_loader, test_loader, learning_rate, epochs,
            complexity_weight_base, sample_nbr_elbo, device
        )
        plot_losses(train_losses, test_losses, epochs)

        # --- Prediction and Evaluation ---
        y_pred_samples_scaled = generate_predictions(trained_model, test_loader, n_prediction_samples, device)
        # Ensure y_test is numpy for inverse transform function
        y_test_numpy = y_test_tensor.cpu().numpy()
        y_pred_mean, y_pred_lower, y_pred_upper, y_test_orig, _ = inverse_transform_predictions(
            y_pred_samples_scaled, y_test_numpy, target_scaler
        )
        rmses = evaluate_performance(y_test_orig, y_pred_mean, n_steps_ahead)

        # --- Plotting Results ---
        plot_predictions(y_test_orig, y_pred_mean, y_pred_lower, y_pred_upper, 0, ticker, n_steps_ahead) # First step
        plot_predictions(y_test_orig, y_pred_mean, y_pred_lower, y_pred_upper, n_steps_ahead - 1, ticker, n_steps_ahead) # Last step

        # --- Exporting Artifacts ---
        export_artifacts(trained_model, feature_scaler, target_scaler, feature_cols_final, target_cols, config, export_prefix)

        # --- Return Results ---
        results = {
            "model": trained_model,
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
            "feature_cols_final": feature_cols_final,
            "target_cols": target_cols,
            "X_test_orig_index": X.index[-len(X_test):], # Keep original index for test set if needed
            "y_test_orig": y_test_orig,
            "y_pred_mean": y_pred_mean,
            "y_pred_lower": y_pred_lower,
            "y_pred_upper": y_pred_upper,
            "rmse_per_step": rmses,
            "train_losses": train_losses,
            "test_losses": test_losses,
            "config": config
        }
        print("\nPipeline finished.")
        return results
    return (train_and_test,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Model Execution""")
    return


@app.cell
def _(
    BATCH_SIZE,
    COMPLEXITY_WEIGHT_BASE,
    DATA_PERIOD,
    EPOCHS,
    HIDDEN_UNITS_1,
    HIDDEN_UNITS_2,
    INTERVAL,
    LEARNING_RATE,
    N_LAGS,
    N_PREDICTION_SAMPLES,
    N_STEPS_AHEAD,
    SAMPLE_NBR_ELBO,
    SEED,
    TEST_SIZE,
    TICKER,
    train_and_test,
):
    selected_features_list = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'trend_sma_fast',
        'trend_ema_fast', 'momentum_rsi', 'trend_macd_diff',
        'volatility_bbh', 'volatility_bbl', 'volatility_atr', 'volume_obv'
    ]

    # Run the pipeline
    results = train_and_test(
        ticker=TICKER,
        period=DATA_PERIOD,
        interval=INTERVAL,
        base_features=selected_features_list,
        n_lags=N_LAGS,
        n_steps_ahead=N_STEPS_AHEAD,
        test_size=TEST_SIZE,
        seed=SEED,
        shuffle_split=False,
        hidden_units_1=HIDDEN_UNITS_1,
        hidden_units_2=HIDDEN_UNITS_2,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        sample_nbr_elbo=SAMPLE_NBR_ELBO,
        complexity_weight_base=COMPLEXITY_WEIGHT_BASE,
        n_prediction_samples=N_PREDICTION_SAMPLES,
        use_random_seed=True,
        export_prefix=f"{TICKER}_model"
    )
    return (selected_features_list,)


@app.cell
def _():
    """
    results_eth = train_and_test(
        ticker="ETH-USD",
        period=DATA_PERIOD,
        interval=INTERVAL,
        base_features=selected_features_list,
        n_lags=N_LAGS,
        n_steps_ahead=N_STEPS_AHEAD,
        test_size=TEST_SIZE,
        seed=SEED,
        shuffle_split=False,
        hidden_units_1=HIDDEN_UNITS_1,
        hidden_units_2=HIDDEN_UNITS_2,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        sample_nbr_elbo=SAMPLE_NBR_ELBO,
        complexity_weight_base=COMPLEXITY_WEIGHT_BASE,
        n_prediction_samples=N_PREDICTION_SAMPLES,
        use_random_seed=True,
        export_prefix=f"{TICKER}_model"
    )
    """
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## On Real Time""")
    return


@app.cell
def _():
    import time
    import datetime
    return datetime, time


@app.cell
def _(BayesianStockPredictor, get_device, joblib, os, torch):
    # --- Helper function to load artifacts ---
    def load_prediction_artifacts(prefix: str, export_dir: str = "exported_model"):
        """Loads model, scalers, and metadata needed for prediction."""
        print(f"Loading artifacts with prefix '{prefix}' from '{export_dir}'...")

        # Load scalers and metadata
        scaler_metadata_path = os.path.join(export_dir, f"{prefix}_scalers_metadata.joblib")
        if not os.path.exists(scaler_metadata_path):
            raise FileNotFoundError(f"Scaler/metadata file not found: {scaler_metadata_path}")
        artifacts = joblib.load(scaler_metadata_path)
        feature_scaler = artifacts['feature_scaler']
        target_scaler = artifacts['target_scaler']
        feature_cols_final = artifacts['feature_cols_final']
        # target_cols = artifacts['target_cols'] # Less critical for prediction input
        config = artifacts['config']
        n_lags = config['N_LAGS']
        n_steps_ahead = config['N_STEPS_AHEAD']
        hidden_units_1 = config['HIDDEN_UNITS_1']
        hidden_units_2 = config['HIDDEN_UNITS_2']
        print("  Scalers and metadata loaded.")

        # Load model
        input_dim = len(feature_cols_final)
        output_dim = n_steps_ahead
        device = get_device() # Re-determine device
        model = BayesianStockPredictor(input_dim, output_dim, hidden_units_1, hidden_units_2)

        model_path = os.path.join(export_dir, f"{prefix}_state_dict.pth")
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model state file not found: {model_path}")
        # Load state dict, ensuring it's mapped to the correct device
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval() # Set model to evaluation mode
        print(f"  Model loaded and set to '{device}' device in evaluation mode.")

        return model, feature_scaler, target_scaler, feature_cols_final, n_lags, n_steps_ahead, device, config
    return (load_prediction_artifacts,)


@app.cell
def _(
    add_technical_indicators,
    create_lagged_features,
    datetime,
    load_prediction_artifacts,
    np,
    pd,
    plt,
    select_features,
    selected_features_list,
    time,
    torch,
    yf,
):
    def validate_real_time(
        model_prefix: str,
        ticker: str,
        validation_duration_minutes: int = 25,
        update_interval_seconds: int = 60,
        n_prediction_samples: int = 100, # Use the same as training or adjust
        history_fetch_period: str = '5d' # How much history to fetch each time for TA/lags
        ):
        """
        Loads a trained model and performs validation using real-time data.

        Args:
            model_prefix (str): Prefix used when saving model artifacts (e.g., "BTC-USD_model").
            ticker (str): The ticker symbol to fetch real-time data for (e.g., "BTC-USD").
            validation_duration_minutes (int): How long to run the validation loop.
            update_interval_seconds (int): How often to fetch data and predict (should be >= 60 for 1m interval).
            n_prediction_samples (int): Number of samples for Bayesian prediction uncertainty.
            history_fetch_period (str): yfinance period string to fetch sufficient history
                                         for calculating TAs and lags (e.g., '5d', '3d'). Needs to be
                                         long enough to cover max lookback of TAs + n_lags.
        """
        try:
            # --- 1. Load Artifacts ---
            model, feature_scaler, target_scaler, feature_cols_final, n_lags, n_steps_ahead, device, config = \
                load_prediction_artifacts(model_prefix)
            base_features_used = config.get('base_features', selected_features_list) # Get base features from config if saved, else use global


            # --- 2. Initialize Storage for Plotting ---
            timestamps = []
            actual_prices = []
            predicted_means_step1 = []
            predicted_lowers_step1 = []
            predicted_uppers_step1 = []

            # --- 3. Real-time Validation Loop ---
            end_time = time.time() + validation_duration_minutes * 60
            print(f"\n--- Starting Real-time Validation for {ticker} ---")
            print(f"Duration: {validation_duration_minutes} minutes")
            print(f"Fetching data every {update_interval_seconds} seconds")
            print(f"Predicting {n_steps_ahead} steps ahead.")
            print(f"Using history period: {history_fetch_period} for TA/Lags calculation.")
            print(f"Press Ctrl+C to stop early.")

            while time.time() < end_time:
                loop_start_time = time.time()
                current_dt = datetime.datetime.now()
                print(f"\n{current_dt.strftime('%Y-%m-%d %H:%M:%S')} - Fetching latest data...")

                try:
                    # --- 3a. Fetch Data ---
                    # Fetch sufficient historical data ending now, with 1m interval
                    data_hist = yf.Ticker(ticker).history(period=history_fetch_period, interval='1m', auto_adjust=True)
                    data_hist = data_hist.drop(columns=['Dividends', 'Stock Splits'], errors='ignore')

                    if data_hist.empty:
                        print("  Warning: No data received. Skipping this interval.")
                        time.sleep(max(0, update_interval_seconds - (time.time() - loop_start_time)))
                        continue

                    # Ensure timezone is consistent or removed if necessary (yfinance can be tricky)
                    if data_hist.index.tz is not None:
                         data_hist.index = data_hist.index.tz_convert(None) # Or convert to UTC: data_hist.index.tz_convert('UTC')

                    latest_timestamp = data_hist.index[-1]
                    latest_actual_close = data_hist['Close'].iloc[-1]
                    print(f"  Latest data point timestamp: {latest_timestamp}, Close: {latest_actual_close:.2f}")

                    # --- 3b. Preprocess Data ---
                    # Important: Apply TAs and lags to the whole fetched dataframe
                    # to ensure calculations are correct for the last point.
                    df_ta = add_technical_indicators(data_hist)
                    base_feature_cols_available = [f for f in base_features_used if f in df_ta.columns]
                    df_selected = select_features(df_ta, base_feature_cols_available) # Select only base features first
                    df_lagged, lagged_feature_cols = create_lagged_features(df_selected, base_feature_cols_available, n_lags)

                    # Combine base + lagged features required by the model
                    all_needed_features = base_feature_cols_available + lagged_feature_cols
                    df_final_features = df_lagged[all_needed_features]

                    # Clean NaNs resulting from TAs/lags
                    df_final_features_clean = df_final_features.dropna()

                    if df_final_features_clean.empty:
                        print("  Warning: Not enough data to calculate all features/lags after cleaning. Skipping.")
                        time.sleep(max(0, update_interval_seconds - (time.time() - loop_start_time)))
                        continue

                    # Select the *very last* row for prediction
                    X_latest = df_final_features_clean.iloc[-1:] # Keep as DataFrame row

                    # Ensure columns are in the exact same order as during training
                    X_latest = X_latest[feature_cols_final]

                    # Scale features
                    X_latest_scaled = feature_scaler.transform(X_latest)

                    # Convert to Tensor
                    X_latest_tensor = torch.tensor(X_latest_scaled, dtype=torch.float32).to(device)

                    # --- 3c. Predict ---
                    with torch.no_grad():
                        pred_samples_scaled = [model(X_latest_tensor).cpu().numpy() for _ in range(n_prediction_samples)]
                    # Shape: (n_samples, 1, n_steps_ahead) -> (n_samples, n_steps_ahead)
                    pred_samples_scaled = np.squeeze(np.array(pred_samples_scaled), axis=1)

                    # Calculate mean and std dev across samples
                    pred_mean_scaled = pred_samples_scaled.mean(axis=0)
                    pred_std_scaled = pred_samples_scaled.std(axis=0)

                    # Inverse transform predictions
                    # Reshape mean/std to (1, n_steps_ahead) for the scaler
                    pred_mean = target_scaler.inverse_transform(pred_mean_scaled.reshape(1, -1)).flatten()
                    # For bounds, calculate in scaled space and inverse transform
                    lower_bound_scaled = pred_mean_scaled - 1.96 * pred_std_scaled
                    upper_bound_scaled = pred_mean_scaled + 1.96 * pred_std_scaled
                    pred_lower = target_scaler.inverse_transform(lower_bound_scaled.reshape(1, -1)).flatten()
                    pred_upper = target_scaler.inverse_transform(upper_bound_scaled.reshape(1, -1)).flatten()

                    # We are interested in the prediction for the *next* step (t+1)
                    predicted_mean_next_step = pred_mean[0]
                    predicted_lower_next_step = pred_lower[0]
                    predicted_upper_next_step = pred_upper[0]

                    print(f"  Predicted Close for next step ({latest_timestamp + pd.Timedelta(minutes=1)}):")
                    print(f"    Mean: {predicted_mean_next_step:.2f}")
                    print(f"    95% CI: [{predicted_lower_next_step:.2f}, {predicted_upper_next_step:.2f}]")

                    # --- 3d. Store Results ---
                    timestamps.append(latest_timestamp) # Store timestamp of the data *used* for prediction
                    actual_prices.append(latest_actual_close)
                    predicted_means_step1.append(predicted_mean_next_step)
                    predicted_lowers_step1.append(predicted_lower_next_step)
                    predicted_uppers_step1.append(predicted_upper_next_step)

                except ConnectionError as e:
                    print(f"  Error fetching data: {e}. Retrying next cycle.")
                except Exception as e:
                    print(f"  An error occurred during processing: {e}")
                    import traceback
                    traceback.print_exc() # Print detailed traceback for debugging
                    print("  Skipping this interval.")

                # --- 3e. Wait for next interval ---
                elapsed_time = time.time() - loop_start_time
                sleep_time = update_interval_seconds - elapsed_time
                if sleep_time > 0:
                    # print(f"  Sleeping for {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)

            print("\n--- Validation Loop Finished ---")

            # --- 4. Plot Results ---
            if not timestamps:
                print("No data collected during validation run. Cannot plot.")
                return None

            print("Plotting results...")
            plt.figure(figsize=(15, 8))
            # Plot actual prices observed during the run
            plt.plot(timestamps, actual_prices, label='Actual Close Price', color='blue', marker='.', linestyle='-')

            # Plot the 1-step ahead predicted mean prices
            # Shift predicted times by 1 minute to align with what they are predicting
            predicted_timestamps = [ts + pd.Timedelta(minutes=1) for ts in timestamps]
            plt.plot(predicted_timestamps, predicted_means_step1, label='Predicted Mean (1-Step Ahead)', color='orange', marker='x', linestyle='--')

            # Plot the 1-step ahead uncertainty bounds
            plt.fill_between(predicted_timestamps,
                             predicted_lowers_step1,
                             predicted_uppers_step1,
                             color='orangered', alpha=0.3, label='95% CI (1-Step Ahead)')

            plt.title(f'{ticker} Real-time Validation ({validation_duration_minutes} min) - Actual vs 1-Step Prediction')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            # --- 5. Return Collected Data (Optional) ---
            results_df = pd.DataFrame({
                'Timestamp': timestamps,
                'ActualClose': actual_prices,
                'PredictedMean_Step1': predicted_means_step1,
                'PredictedLower_Step1': predicted_lowers_step1,
                'PredictedUpper_Step1': predicted_uppers_step1
            }).set_index('Timestamp')

            return results_df

        except FileNotFoundError as e:
            print(f"Error: {e}. Make sure the model artifacts exist in the 'exported_model' directory.")
            print("Did you run the training script successfully first?")
            return None
        except KeyboardInterrupt:
            print("\nValidation interrupted by user.")
            # Still try to plot if some data was collected
            if timestamps:
                 print("Plotting collected data...")
                 plt.figure(figsize=(15, 8))
                 plt.plot(timestamps, actual_prices, label='Actual Close Price', color='blue', marker='.', linestyle='-')
                 predicted_timestamps = [ts + pd.Timedelta(minutes=1) for ts in timestamps]
                 plt.plot(predicted_timestamps, predicted_means_step1, label='Predicted Mean (1-Step Ahead)', color='orange', marker='x', linestyle='--')
                 plt.fill_between(predicted_timestamps, predicted_lowers_step1, predicted_uppers_step1, color='orangered', alpha=0.3, label='95% CI (1-Step Ahead)')
                 plt.title(f'{ticker} Real-time Validation (Interrupted) - Actual vs 1-Step Prediction')
                 plt.xlabel('Time')
                 plt.ylabel('Price')
                 plt.legend()
                 plt.grid(True)
                 plt.xticks(rotation=45)
                 plt.tight_layout()
                 plt.show()
                 # Return partial data
                 results_df = pd.DataFrame({
                     'Timestamp': timestamps,
                     'ActualClose': actual_prices,
                     'PredictedMean_Step1': predicted_means_step1,
                     'PredictedLower_Step1': predicted_lowers_step1,
                     'PredictedUpper_Step1': predicted_uppers_step1
                 }).set_index('Timestamp')
                 return results_df
            else:
                 print("No data collected before interruption.")
                 return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
            return None
    return


@app.cell
def _():
    """
    MODEL_EXPORT_PREFIX = f"{TICKER}_model" # e.g., "BTC-USD_model"

    # Run the real-time validation
    validation_results = validate_real_time(
        model_prefix=MODEL_EXPORT_PREFIX,
        ticker=TICKER, # e.g., "BTC-USD"
        validation_duration_minutes=25,
        update_interval_seconds=60,
        n_prediction_samples=100, # Should match or be appropriate for prediction
        history_fetch_period='5d' # Adjust if needed based on TAs/lags
    )

    if validation_results is not None:
        print("\nValidation Results DataFrame:")
        print(validation_results.head())
        # You can save this dataframe if needed:
        # validation_results.to_csv("realtime_validation_results.csv")
    """
    return


if __name__ == "__main__":
    app.run()
