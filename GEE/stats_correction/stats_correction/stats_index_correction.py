import glob
import pandas as pd

# Load the original DataFrame to get its index
df_orig = pd.read_csv('train_data_cleaned.csv')

for path in glob.glob('stats_*.csv'):
    # Read the stats CSV
    df_stats = pd.read_csv(path)
    
    # Assign the original index to the stats DataFrame
    df_stats.index = df_orig.index
    
    # Write out a new file including the index as an 'id' column
    out_path = path.replace('.csv', '_with_index.csv')
    df_stats.to_csv(out_path, index=True, index_label='id')
    
    print(f'✔ Wrote {out_path}')
