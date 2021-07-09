import glob
import os
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

TIME_TO_SLEEP = 15


def get_end_time(csv_path):
    sync_t = os.path.splitext(os.path.basename(csv_path))[0]
    end_sec = int(sync_t[len('sync_'):])
    return pd.to_datetime(end_sec, unit='s')


def load_data(data_path):
    filepaths = glob.glob(os.path.join(data_path, 'sync_*.csv'))
    dfs = []
    for csv_path in tqdm(filepaths, desc='Loading csv data'):
        try:
            df = pd.read_csv(csv_path, skipinitialspace=True)
        except pd.errors.EmptyDataError:
            continue
        time = pd.date_range(end=get_end_time(csv_path), freq=f'{TIME_TO_SLEEP}S', periods=len(df))
        df = df.assign(time=time)
        df = df.set_index(time)
        dfs.append(df)

    return pd.concat(dfs).sort_index()


def main():
    os.makedirs('plots', exist_ok=True)
    df = load_data('data/')

    measures = [
        'co2', 'tvoc',
        'hum_ext', 'hum_room', 'hum_wall', 'hum_ceiling',
        'temp_ext', 'temp_room', 'temp_wall', 'temp_ceiling'
    ]

    # filter outliers and nans
    df.dropna(inplace=True)
    df = df[(np.abs(stats.zscore(df[measures])) < 3).all(axis=1)]

    start_date = df['time'][0].date()
    end_date = df['time'][-1].date()
    n_days = (end_date - start_date).days
    one_day = pd.DateOffset(days=1)

    with tqdm(range(n_days)) as progress_bar:
        for delta_day in progress_bar:
            ii_date = start_date + one_day * delta_day
            progress_bar.set_description(f'Plotting {ii_date.date()}')

            fig, axs = plt.subplots(2, 1, sharex=True)
            plot_df = df[((ii_date <= df['time']) & (df['time'] < ii_date + one_day))]
            plot_df[['co2', 'tvoc']].plot(ax=axs[0])
            plot_df[['hum_ext', 'hum_room', 'hum_wall', 'hum_ceiling',
                     'temp_ext', 'temp_room', 'temp_wall', 'temp_ceiling']].plot(ax=axs[1])
            filename = f'{ii_date.strftime("%Y-%m-%d")}.png'
            fig.savefig(os.path.join('plots', filename))
            plt.close(fig)


if __name__ == "__main__":
    main()
