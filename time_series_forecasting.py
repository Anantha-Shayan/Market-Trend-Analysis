# import pandas as pd
# import glob
# import os

# # 1. Get a list of all CSV file paths
# path = '/content/drive/MyDrive/Agmarket/Price'
# all_files = glob.glob(os.path.join(path, "*.csv"))

# # 2. Use a list comprehension to read all files
# # We add a column 'Commodity' based on the filename for tracking
# df_list = []
# for filename in all_files:
#     df = pd.read_csv(filename)
#     # Extract filename without extension as the label
#     df['commodity_name'] = os.path.basename(filename).replace('.csv', '')
#     df_list.append(df)

# # 3. Combine everything into one DataFrame
# combined_df = pd.concat(df_list, axis=0, ignore_index=True)

# # 4. Export to a high-performance format
# combined_df.to_csv('combined_agriculture_data.csv', index=False)
def prepare_data():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    data = pd.read_csv('./combined_agriculture_data.csv')

    df = pd.DataFrame(data)
    df

    df = df.drop(['commodity_name','market_id','state_id','district_id'], axis=1)

    df.rename(columns={'t': 'Date','cmdty':'Commodity','market_name':'Market','state_name':'State','district_name':'District','variety':'Variety'}, inplace=True)

    df = df.sort_values(by=['Date', 'Commodity', 'State'],ignore_index=True)

    """# **EDA and pre-processing**"""

    # df.info()
    # df.set_index('Date', inplace=True)

    print(df['Commodity'].nunique())
    print(df['Market'].nunique())
    print(df['State'].nunique())

    # outliers
    invalid_prices = df[
        (df['p_min'] > df['p_modal']) | (df['p_modal'] > df['p_max'])
    ]

    len(invalid_prices)

    # Check if price is -ve (outliers again)
    (df[['p_min', 'p_max', 'p_modal']] < 0).sum()

    df['p_modal'].describe()

    # Check if data is balanced
    df['Commodity'].value_counts().head(10)

    df

    df['Commodity'].unique()

    # check unique for state, district, market, variety
    df['State'].unique()

    df['District'].nunique()

    df.isna().sum()

    #df = df.reset_index()

    df

    def missing_dates(group):
        full_range = pd.date_range(group['Date'].min(), group['Date'].max())
        return len(full_range) - group['Date'].nunique()

    missing_by_group = (
        df.groupby(['Commodity', 'Market'])
        .apply(missing_dates, include_groups=False)
        .sort_values(ascending=False)
        .head(10)
    )

    missing_by_group

    df['Date'] = pd.to_datetime(df['Date'])

    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['week'] = df['Date'].dt.isocalendar().week
    df['dayofweek'] = df['Date'].dt.dayofweek

    potato = df[(df['Commodity'] == 'Tomato') & (df['year'] == 2025)]

    monthly_avg = (
        potato.groupby('month')['p_modal']
        .mean()
    )

    monthly_avg

    potato_market_stats = (
        potato.groupby('State')['p_modal']
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    potato_market_stats

    df = df.sort_values(['Commodity', 'Market', 'Date'], ignore_index = True)

    df['price_change'] = (
        df.groupby(['Commodity', 'Market'])['p_modal']
        .diff()
    )

    df['pct_change'] = (
        df.groupby(['Commodity', 'Market'])['p_modal']
        .pct_change(fill_method=None)
    )

    df

    print(df['price_change'].isna().sum())
    print(df['pct_change'].isna().sum())

    df['price_spike'] = df['pct_change'] > 0.15  # 15% daily spike
    df['price_spike'].mean()

    final_df_for_analysis = df.copy('')
    final_df_for_analysis

    df['State_District_Market'] = df['State'] + '_' + df['District'] + '_' + df['Market']
    df

    """# **Data Preparation**"""

    deepar_df = df[['Date', 'Commodity', 'State_District_Market', 'p_modal']].copy()
    deepar_df['Union'] = df['Commodity'] + "_" + df['State_District_Market']
    deepar_df = deepar_df.sort_values(by=['Date', 'Union'], ignore_index=True)
    deepar_df = deepar_df.drop('State_District_Market', axis = 1)
    deepar_df

    """DeepAR requires an integer time index, not datetime.
    Because internally
    DeepAR uses RNNs and
    RNNs operate on ordered sequences, not calendar dates
    """

    deepar_df['time_idx'] = (
            deepar_df.groupby('Union')['Date']
            .rank(method='dense')
            .astype(int)
        )

    deepar_df.drop(['Commodity'], axis=1, inplace=True)

    deepar_df = deepar_df.dropna(subset=["p_modal"])

    deepar_test = deepar_df[deepar_df["Date"] >= "2023-01-01"].copy()
    deepar_train = deepar_df[deepar_df["Date"] < "2023-01-01"].copy() #same data for train and val

    deepar_test.drop(['Date'], axis=1, inplace=True)
    deepar_train.drop(['Date'], axis=1, inplace=True)

    deepar_train

    return deepar_train, deepar_test


def build_datasets(deepar_train, max_encoder_length, max_prediction_length):
    from pytorch_forecasting.data import TimeSeriesDataSet, GroupNormalizer

    training_cut_off = deepar_train['time_idx'].max() - max_prediction_length

    training = TimeSeriesDataSet(
        deepar_train[deepar_train.time_idx <= training_cut_off],
        time_idx="time_idx",
        target="p_modal",
        group_ids=["Union"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=["p_modal"],
        target_normalizer=GroupNormalizer(groups=["Union"]), #normalize y
        allow_missing_timesteps=True
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,  #inherit all imp parameters like 'groupby', 'max_enco','max_pred', etc., from training set
        deepar_train,
        predict=True, #this says that the set is for prediction (therefore validation set)
        stop_randomization=True
        #stop randomization of enc and deco len within specified min,max_enc_len and min,max_pred_len ranges cuz you want a consistent and reproducible benchmark.
        # this is false in training cuz it exposes the model to a wider variety of i/p sequence len and pred horizons during training, making it more robust and less prone to overfitting to specific sequence len.
    )

    return training, validation


def build_model(training):
    from pytorch_forecasting import DeepAR
    from pytorch_forecasting.metrics import NormalDistributionLoss

    model = DeepAR.from_dataset(
        training,
        learning_rate=1e-3, #DeepAR uses Adam optimizer. Adam is stable in the range: 1e-4 to 1e-3
        hidden_size=64,
        rnn_layers=2, # layer1 - short term, layer2 - mideium term
        # 3 is overkill, 1 leads to overfit
        dropout=0.1,
        loss=NormalDistributionLoss(),
        # DeepAR models a probability distribution, not quantiles directly — therefore it must be trained with a distribution-based loss.
    )

    return model


def build_trainer():
    import lightning as pl

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="cuda",
        devices=1,
        gradient_clip_val=0.1, # to prevent exploding gradients during training
        enable_model_summary=True
    )

    return trainer


def train(checkpoint_path):
    import os

    deepar_train, _ = prepare_data()

    max_prediction_length = 7
    max_encoder_length = 14

    training, validation = build_datasets(
        deepar_train,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length
    )

    batch_size = 64
    train_loader = training.to_dataloader(
        train=True, # Randomly samples forecast windows and shuffles them (shuffles windows not time order) which improves generalization
        batch_size=batch_size,
        num_workers=2
    )

    val_loader = validation.to_dataloader(
        train=False, # No randomization. No gradient updates
        batch_size=batch_size,
        num_workers=2
    )

    model = build_model(training)
    trainer = build_trainer()

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    trainer.save_checkpoint(checkpoint_path)


def evaluate_validation(checkpoint_path):
    import torch

    deepar_train, _ = prepare_data()

    max_prediction_length = 7
    max_encoder_length = 14

    training, validation = build_datasets(
        deepar_train,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length
    )

    batch_size = 64
    val_loader = validation.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=2
    )

    from pytorch_forecasting import DeepAR
    model = DeepAR.load_from_checkpoint(checkpoint_path)
    model.eval()

    val_preds, val_x = model.predict(
        val_loader,
        mode="quantiles",
        quantiles=[0.1, 0.5, 0.9],
        return_x=True
    )

    val_actuals = torch.cat([y[0] for y in val_x["decoder_target"]])
    val_median = val_preds[..., 1]
    valid_mask = val_actuals != 0

    val_mape = torch.mean(
        torch.abs((val_actuals[valid_mask] - val_median[valid_mask]) / val_actuals[valid_mask])
    )
    val_mae = torch.mean(torch.abs(val_actuals - val_median))

    print("VAL MAPE:", val_mape.item())
    print("VAL MAE :", val_mae.item())


def evaluate_test(checkpoint_path):
    import torch
    from pytorch_forecasting.data import TimeSeriesDataSet

    deepar_train, deepar_test = prepare_data()

    max_prediction_length = 7
    max_encoder_length = 14

    training, _ = build_datasets(
        deepar_train,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length
    )

    test_dataset = TimeSeriesDataSet.from_dataset(
        training,
        deepar_test,
        predict=True,
        stop_randomization=True
    )

    batch_size = 64
    test_loader = test_dataset.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=2
    )

    from pytorch_forecasting import DeepAR
    model = DeepAR.load_from_checkpoint(checkpoint_path)
    model.eval()

    test_preds, test_x = model.predict(
        test_loader,
        mode="quantiles",
        quantiles=[0.1, 0.5, 0.9],
        return_x=True
    )

    test_actuals = torch.cat([y[0] for y in test_x["decoder_target"]])
    test_median = test_preds[..., 1]
    valid_mask = test_actuals != 0

    test_mape = torch.mean(
        torch.abs((test_actuals[valid_mask] - test_median[valid_mask]) / test_actuals[valid_mask])
    )
    test_mae = torch.mean(torch.abs(test_actuals - test_median))

    print("TEST MAPE:", test_mape.item())
    print("TEST MAE :", test_mae.item())


def predict_and_signal(checkpoint_path):
    import torch
    from pytorch_forecasting.data import TimeSeriesDataSet

    deepar_train, deepar_test = prepare_data()

    max_prediction_length = 7
    max_encoder_length = 14

    training, _ = build_datasets(
        deepar_train,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length
    )

    test_dataset = TimeSeriesDataSet.from_dataset(
        training,
        deepar_test,
        predict=True,
        stop_randomization=True
    )

    batch_size = 64
    test_loader = test_dataset.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=2
    )

    from pytorch_forecasting import DeepAR
    model = DeepAR.load_from_checkpoint(checkpoint_path)
    model.eval()

    test_preds, test_x = model.predict(
        test_loader,
        mode="quantiles",
        quantiles=[0.1, 0.5, 0.9],
        return_x=True
    )

    i = 0
    actual = test_x["decoder_target"][i][0].cpu()
    q10 = test_preds[i, :, 0].cpu()
    q50 = test_preds[i, :, 1].cpu()
    q90 = test_preds[i, :, 2].cpu()

    last_price = test_x["encoder_target"][i][-1].item()
    mean_future = torch.mean(q50).item()
    pct_change = (mean_future - last_price) / last_price if last_price != 0 else 0.0

    if pct_change > 0.08:
        signal = "SELL_LATER"
    elif pct_change < -0.08:
        signal = "SELL_NOW"
    else:
        signal = "HOLD"

    print("Signal:", signal)
    print("Expected % change:", pct_change * 100)


def main(stage):
    import os
    import pytorch_forecasting
    import inspect
    from pytorch_forecasting import DeepAR
    import lightning.pytorch as pl

    checkpoint_path = "deepar_agmarket.ckpt"

    if stage == "train":
        train(checkpoint_path)
    elif stage == "val":
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError("Checkpoint not found. Run: python time_series_forecasting.py train")
        evaluate_validation(checkpoint_path)
    elif stage == "test":
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError("Checkpoint not found. Run: python time_series_forecasting.py train")
        evaluate_test(checkpoint_path)
    elif stage == "predict":
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError("Checkpoint not found. Run: python time_series_forecasting.py train")
        predict_and_signal(checkpoint_path)
    else:
        raise ValueError("Stage must be one of: train | val | test | predict")

    print(pytorch_forecasting.__version__)
    print(inspect.getmodule(DeepAR))
    print(pl.LightningModule)

    # !pip install --no-cache-dir -v lightning pytorch-forecasting

    # !pip uninstall pytorch-lightning

    # !pip show lightning
    # !pip show pytorch-forecasting

if __name__ == "__main__":
    import sys
    import torch.multiprocessing as mp

    mp.freeze_support()

    if len(sys.argv) < 2:
        raise ValueError("Usage: python time_series_forecasting.py [train|val|test|predict]")

    main(sys.argv[1])
