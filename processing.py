"""
Preprocessing functions for data manipulation.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler

def load_df(csv, additional_features=False, population=None):
    """
    Preprocesses the data from the csv file.
    :param csv: The csv filepath to preprocess.
    :param return_target: Whether to return the target or not.
    :param combine_cases: Whether to combine the vaccinated and unvaccinated cases
    :param cumulative: whether to make cases cumulative; only returns active cases if false.
    :return: pd.DataFrame The preprocessed data.
    """
    df = pd.read_csv(csv)

    return process_df(df, additional_features=additional_features)


def process_df(df, additional_features=False):
    """
    Preprocesses the data from the csv file.
    return: pd.DataFrame
    """
    df['total_cases'] = df['infected_unvaccinated'] + df['infected_vaccinated']
    df['total_cases_nextday'] = df['total_cases'].shift(1)
    df.drop(df.head(1).index,inplace=True)
    if additional_features:
        # cumulative, increasing days consecuetive, eligible to be sick
        prev = 0.0
        cum = 0
        rolling_cum = []
        increasing_days = []
        days = 0
        for i in df['total_cases'].values:
            if i>prev:
                cum += i
                rolling_cum.append(cum)
                days += 1
                increasing_days.append(days)
            else:
                days=0
                increasing_days.append(days)
                rolling_cum.append(cum)

        df['days_increasing'] = increasing_days
        df['cumulative_cases'] = rolling_cum
        df = df[[i for i in df.columns if i != 'total_cases'] + ['total_cases']]

    return df


def series_to_supervised_old_(data, window=20):
    """
    Old/deprecated version of series_to_supervised. Used for testing.
    """
    data = np.array(data)
    x = []
    y = []
    for i in range(0, len(data)-window-1):
        x.append(data[i:i+window])
        y.append(data[i+window+1])

    x = np.concatenate(x).reshape(-1, window)
    y = np.concatenate(y)
    return x, y


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True, shift=0):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
        Returns np.ndarray if np.ndarray is inputted.
    """
    df = pd.DataFrame(data)
    cols = []
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i-shift))
    agg = pd.concat(cols, axis=1)
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values


def scale_data(df):
    """
    Scale data using scikit StandardScaler
    """
    scaler = StandardScaler()
    scaler.fit(df)
    scaled_data = scaler.transform(df)
    return scaled_data, scaler


def oof_idx(data, te_idx, window=20):
    """
    Convert out of fold index to accomidate for a pre-window
    """
    return list(range(min(te_idx)-window, max(te_idx)+1))


def infer_only_meta(df, tf_meta_models, future_pred=100, num_cols=6):
    """Infer on metadata only models
    Args:
        df (pd.DataFrame): Dataframe to infer on (read directly from csv)
        tf_meta_models (list): List of tensorflow models with metadata columns
        future_pred (int, optional): Number of days in the future to predict. Defaults to 100.
        num_cols (int, optional): Number of metadata columns. Defaults to 6.
    Returns:
        np.ndarray: Saves to `predictions.csv` and returns len(future_pred) long array of predictions
    """
    meta_df = process_df(df, additional_features=True)
    del meta_df['total_cases_nextday']
    meta_data, meta_scaler = scale_data(meta_df.values)
    meta_windowed = series_to_supervised(meta_data, n_in=50)
    meta_windowed = np.asarray(meta_windowed).reshape(-1, 6, 50+1)

    final_output = []
    for i in range(future_pred):
        meta_window = meta_windowed[-1, :, :]
        meta_pred = []
        for model in tf_meta_models:
            meta_pred.append(model.predict(np.asarray([meta_windowed[i, :, :-1]]))[0])
        meta_pred = np.array(meta_pred)
        meta_pred = np.mean(meta_pred, axis=0)
        final_output.append(meta_scaler.inverse_transform(meta_pred)[-1])

        meta_window = np.append(meta_window, np.expand_dims(np.array(meta_pred), axis=1), axis=1)[1:]

    pd.DataFrame({'Predicted_infections': np.array(final_output)}).to_csv('predictions.csv', index=False)
    data_frame = pd.DataFrame({'Predicted_infections': np.array(final_output)})
    logging.info('Predictions.csv Saved')
    return final_output, data_frame