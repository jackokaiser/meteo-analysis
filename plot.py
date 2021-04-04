import glob
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

TIME_TO_SLEEP = 15


def get_end_time(csv_path):
    end_sec = int(os.path.splitext(os.path.basename(csv_path))[0])
    return pd.to_datetime(end_sec, unit='s')


if __name__ == "__main__":
    filepaths = glob.glob('data/*.csv')

    dfs = []
    for csv_path in filepaths:
        df = pd.read_csv(csv_path, skipinitialspace=True)
        time = pd.date_range(end=get_end_time(csv_path), freq=f'{TIME_TO_SLEEP}S', periods=len(df))
        df = df.assign(time=time)
        df = df.set_index(time)
        dfs.append(df)

    df = pd.concat(dfs).sort_index()
    df = df[df['time'] < '2000-01-01']  # until NTP problem is solved
    measures = [
        'co2', 'tvoc',
        'hum_ext', 'hum_room', 'hum_wall', 'hum_ceiling',
        'temp_ext', 'temp_room', 'temp_wall', 'temp_ceiling'
    ]
    df = df[(np.abs(stats.zscore(df[measures])) < 3).all(axis=1)]

    ii_date = df['time'][0]
    date_offset = pd.DateOffset(days=1)
    while ii_date < df['time'][-1]:
        plot_df = df[((ii_date <= df['time']) & (df['time'] < ii_date + date_offset))]
        plt.figure()
        plot_df[['co2', 'tvoc']].plot()
        filename = f'{ii_date.strftime("%Y-%m-%d")}_air.png'
        plt.savefig(filename)

        plot_df[['hum_ext', 'hum_room', 'hum_wall', 'hum_ceiling',
                 'temp_ext', 'temp_room', 'temp_wall', 'temp_ceiling']].plot()
        filename = f'{ii_date.strftime("%Y-%m-%d")}_hum_temp.png'
        plt.savefig(filename)
        ii_date += date_offset
