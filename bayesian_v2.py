

import marimo

__generated_with = "0.13.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Imports""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import yfinance as yf
    import ta

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from blitz.modules import BayesianLinear
    from blitz.utils import variational_estimator

    import matplotlib.pyplot as plt

    import warnings
    return (
        BayesianLinear,
        MinMaxScaler,
        nn,
        np,
        optim,
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
    mo.md(r"""## Configuration""")
    return


@app.cell
def _():
    # General Configuration
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
    LEARNING_RATE = 0.005
    EPOCHS = 200
    BATCH_SIZE = 128
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


@app.cell
def _(SEED, np, torch):
    # RANDOM SEED ?
    USE_RANDOM_SEED = True
    if USE_RANDOM_SEED:
      np.random.seed(SEED)
      torch.manual_seed(SEED)

    if torch.cuda.is_available():
      torch.cuda.manual_seed(SEED)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Fetch Data""")
    return


@app.cell
def _(DATA_PERIOD, INTERVAL, TICKER, yf):
    data = yf.Ticker(TICKER).history(period=DATA_PERIOD, interval=INTERVAL, auto_adjust=True)
    data = data.drop(columns=['Dividends', 'Stock Splits'])
    data.head()
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Feature Engineering""")
    return


@app.cell
def _(data, ta):
    data_1 = ta.add_all_ta_features(data, open='Open', high='High', low='Low', close='Close', volume='Volume', fillna=True)
    print(data_1.shape)
    data_1.head()
    return (data_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Feature Selection""")
    return


@app.cell
def _(data_1):
    selected_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'trend_sma_fast', 'trend_ema_fast', 'momentum_rsi', 'trend_macd_diff', 'volatility_bbh', 'volatility_bbl', 'volatility_atr', 'volume_obv']
    selected_features = [f for f in selected_features if f in data_1.columns]
    data_2 = data_1[selected_features]
    data_2.head()
    return (data_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Create Lagged Features""")
    return


@app.cell
def _(N_LAGS, data_2):
    feature_cols = data_2.columns.tolist()
    lagged_feature_cols = []
    for lag in range(1, N_LAGS + 1):
        for col in feature_cols:
            new_col_name = f'{col}_lag_{lag}'
            data_2[new_col_name] = data_2[col].shift(lag)
            lagged_feature_cols.append(new_col_name)
    print(lagged_feature_cols)
    return feature_cols, lagged_feature_cols


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Create Target Variables (future steps)""")
    return


@app.cell
def _(N_STEPS_AHEAD, data_2):
    target_cols = []
    for i in range(1, N_STEPS_AHEAD + 1):
        target_col = f'target_{i}'
        data_2[target_col] = f'target_{i}'
        data_2[target_col] = data_2['Close'].shift(-i)
        target_cols.append(target_col)
    print(target_cols)
    return (target_cols,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Drop rows with NaN values""")
    return


@app.cell
def _(data_2):
    initial_rows = data_2.shape[0]
    data_2.dropna(inplace=True)
    final_rows = data_2.shape[0]
    print('Dropped {initial_rows - final_rows} rows.')
    print(f'  Final data shape for modeling: {data_2.shape}')
    data_2.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Separate Features and Targets""")
    return


@app.cell
def _(data_2, feature_cols, lagged_feature_cols, target_cols):
    feature_cols_final = feature_cols + lagged_feature_cols
    X = data_2[feature_cols_final]
    y = data_2[target_cols]
    print(f'Feature Matrix X shape: {X.shape}')
    print(f'Target matrix y shape: {y.shape}')
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Data Scaling""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Scaling Features""")
    return


@app.cell
def _(MinMaxScaler, X):
    feature_scaler = MinMaxScaler()
    X_scaled = feature_scaler.fit_transform(X)
    return (X_scaled,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Scaling Targets""")
    return


@app.cell
def _(MinMaxScaler, y):
    target_scaler = MinMaxScaler()
    y_scaled = target_scaler.fit_transform(y)
    return target_scaler, y_scaled


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Splitting data into training and testing sets""")
    return


@app.cell
def _(SEED, TEST_SIZE, X_scaled, train_test_split, y_scaled):
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=TEST_SIZE, shuffle=False, random_state=SEED
    )

    print(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"  X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Convert to PyTorch Tensors""")
    return


@app.cell
def _(X_test, X_train, device, torch, y_test, y_train):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    return X_test_tensor, X_train_tensor, y_test_tensor, y_train_tensor


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Create DataLoader for batching""")
    return


@app.cell
def _(
    BATCH_SIZE,
    X_test_tensor,
    X_train_tensor,
    torch,
    y_test_tensor,
    y_train_tensor,
):
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) # Shuffle batches each epoch

    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return test_loader, train_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Define Bayesian Neural Network""")
    return


@app.cell
def _(
    BayesianLinear,
    HIDDEN_UNITS_1,
    HIDDEN_UNITS_2,
    nn,
    variational_estimator,
):
    @variational_estimator
    class BayesianStockPredictor(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            # Define Bayesian Linear layers
            self.blinear1 = BayesianLinear(input_dim, HIDDEN_UNITS_1)
            self.relu1 = nn.ReLU()
            self.blinear2 = BayesianLinear(HIDDEN_UNITS_1, HIDDEN_UNITS_2)
            self.relu2 = nn.ReLU()
            self.blinear_out = BayesianLinear(HIDDEN_UNITS_2, output_dim)

        def forward(self, x):
            x = self.relu1(self.blinear1(x))
            x = self.relu2(self.blinear2(x))
            x = self.blinear_out(x)
            return x
    return (BayesianStockPredictor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Initiate the model""")
    return


@app.cell
def _(BayesianStockPredictor, N_STEPS_AHEAD, X_train, device):
    input_dim = X_train.shape[1]
    output_dim = N_STEPS_AHEAD
    model = BayesianStockPredictor(input_dim, output_dim).to(device)

    print(model)
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Start Training""")
    return


@app.cell
def _(LEARNING_RATE, model, nn, optim):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    return criterion, optimizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Calculate complexity weight (adjusts KL divergence term based on dataset size)

        A common heuristic is 1 / dataset_size or 1 / num_batches
        """
    )
    return


@app.cell
def _(COMPLEXITY_WEIGHT_BASE, train_loader):
    complexity_weight = COMPLEXITY_WEIGHT_BASE / len(train_loader.dataset)
    print(f"  Using complexity weight: {complexity_weight:.2e}")
    return (complexity_weight,)


@app.cell
def _(
    EPOCHS,
    SAMPLE_NBR_ELBO,
    complexity_weight,
    criterion,
    model,
    optimizer,
    test_loader,
    torch,
    train_loader,
):
    train_losses = []
    test_losses = []
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for i_1, (batch_x, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = model.sample_elbo(inputs=batch_x, labels=batch_y, criterion=criterion, sample_nbr=SAMPLE_NBR_ELBO, complexity_cost_weight=complexity_weight)
            loss.backward()
            optimizer.step()
            epoch_loss = epoch_loss + loss.item()
        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                test_loss = test_loss + loss.item()
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            print(f'Epoch [{epoch + 1}/{EPOCHS}], Train ELBO Loss: {avg_epoch_loss:.6f}, Test MSE Loss: {avg_test_loss:.6f}')
    print('\nTraining finished.')
    return test_losses, train_losses


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Plot training and validation loss""")
    return


@app.cell
def _(EPOCHS, plt, test_losses, train_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS + 1), train_losses, label='Training ELBO Loss')
    plt.plot(range(1, EPOCHS + 1), test_losses, label='Test MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Eval Model""")
    return


@app.cell
def _(N_PREDICTION_SAMPLES, model, np, test_loader, torch):
    model.eval()
    predictions_list = []
    with torch.no_grad():
        for _ in range(N_PREDICTION_SAMPLES):
            sample_preds = []
            for batch_x_1, _ in test_loader:
                batch_preds = model(batch_x_1)
                sample_preds.append(batch_preds.cpu().numpy())
            predictions_list.append(np.concatenate(sample_preds, axis=0))
    return (predictions_list,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Stack predictions accross""")
    return


@app.cell
def _(np, predictions_list):
    y_pred_samples = np.stack(predictions_list)
    return (y_pred_samples,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Calculate meand and standard deviation across samples""")
    return


@app.cell
def _(y_pred_samples):
    y_pred_mean_scaled = y_pred_samples.mean(axis=0)
    y_pred_std_scaled = y_pred_samples.std(axis=0)
    return y_pred_mean_scaled, y_pred_std_scaled


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Inverse Transform to Original Scale""")
    return


@app.cell
def _(target_scaler, y_pred_mean_scaled, y_test_tensor):
    y_pred_mean = target_scaler.inverse_transform(y_pred_mean_scaled)
    y_test_orig = target_scaler.inverse_transform(y_test_tensor.cpu().numpy())
    return y_pred_mean, y_test_orig


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Note: Inverse transforming std dev directly isn't strictly correct,
        but we can use it to get an idea of the uncertainty bounds in the original scale.
        A more proper way involves transforming the upper/lower bounds.
        """
    )
    return


@app.cell
def _(target_scaler, y_pred_mean_scaled, y_pred_std_scaled):
    y_pred_upper_scaled = y_pred_mean_scaled + 1.96 * y_pred_std_scaled # 95% CI approx
    y_pred_lower_scaled = y_pred_mean_scaled - 1.96 * y_pred_std_scaled
    y_pred_upper = target_scaler.inverse_transform(y_pred_upper_scaled)
    y_pred_lower = target_scaler.inverse_transform(y_pred_lower_scaled)
    return y_pred_lower, y_pred_upper


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Calculate Metrics""")
    return


@app.cell
def _(N_STEPS_AHEAD, np, y_pred_mean, y_test_orig):
    from sklearn.metrics import mean_squared_error
    rmse_step1 = np.sqrt(mean_squared_error(y_test_orig[:, 0], y_pred_mean[:, 0]))
    print(f'\nTest RMSE for Step 1 Ahead: {rmse_step1:.4f}')
    for i_2 in range(N_STEPS_AHEAD):
        rmse_step_i = np.sqrt(mean_squared_error(y_test_orig[:, i_2], y_pred_mean[:, i_2]))
        print(f'  Test RMSE for Step {i_2 + 1} Ahead: {rmse_step_i:.4f}')
    return


@app.cell
def _(TICKER, plt, y_pred_lower, y_pred_mean, y_pred_upper, y_test_orig):
    print("\nPlotting predictions vs actuals for the first step ahead...")
    plt.figure(figsize=(15, 7))
    plt.plot(y_test_orig[:, 0], label='Actual Price (Step 1)', color='blue', alpha=0.7)
    plt.plot(y_pred_mean[:, 0], label='Predicted Mean Price (Step 1)', color='orange')

    # Plot uncertainty bounds
    plt.fill_between(range(len(y_pred_mean)),
                     y_pred_lower[:, 0],
                     y_pred_upper[:, 0],
                     color='orangered', alpha=0.3, label='95% Confidence Interval')

    plt.title(f'{TICKER} - Prediction vs Actual (First Step Ahead)')
    plt.xlabel('Time Steps (Test Set)')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
    return


@app.cell
def _(
    N_STEPS_AHEAD,
    TICKER,
    plt,
    y_pred_lower,
    y_pred_mean,
    y_pred_upper,
    y_test_orig,
):
    print(f"\nPlotting predictions vs actuals for the last step ahead (Step {N_STEPS_AHEAD})...")
    plt.figure(figsize=(15, 7))
    plt.plot(y_test_orig[:, N_STEPS_AHEAD-1], label=f'Actual Price (Step {N_STEPS_AHEAD})', color='blue', alpha=0.7)
    plt.plot(y_pred_mean[:, N_STEPS_AHEAD-1], label=f'Predicted Mean Price (Step {N_STEPS_AHEAD})', color='green')

    # Plot uncertainty bounds
    plt.fill_between(range(len(y_pred_mean)),
                     y_pred_lower[:, N_STEPS_AHEAD-1],
                     y_pred_upper[:, N_STEPS_AHEAD-1],
                     color='limegreen', alpha=0.3, label='95% Confidence Interval')

    plt.title(f'{TICKER} - Prediction vs Actual (Step {N_STEPS_AHEAD} Ahead)')
    plt.xlabel('Time Steps (Test Set)')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
