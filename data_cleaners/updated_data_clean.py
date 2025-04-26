from data_cleanup import data_review
import pandas as pd
import glob

df_master = pd.read_csv('train_data.csv')
df_master = data_review(df_master, train=True)

for path in glob.glob('stats_*.csv'):
    df_i = pd.read_csv(path)
    year = df_i['year'].unique()[0]
    season = df_i['season'].unique()[0]
    suffix = f"_{year}_{season}"
    df_i.columns = [f"{col}{suffix}" if col not in ['year', 'season'] else col for col in df_i.columns]
    df_i = df_i.drop(columns=['year', 'season'])
    df_master = pd.concat([df_master, df_i], axis=1)