import glob
import pandas as pd

for path in glob.glob('stats_*.csv'):
    df = pd.read_csv(path)
    df.drop(columns=['id.1'], inplace=True)
    df.to_csv(path, index=False)
    print(f"Dropping id.1 from {path}")