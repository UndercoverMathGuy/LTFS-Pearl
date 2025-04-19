import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import your local function
from soilgrids import soilgrids_df

def evi_locs():
    df = pd.read_csv('train_data_cleaned.csv')
    return df[['VILLAGE', 'DISTRICT', 'State', 'Zipcode']]

def return_coordinates_from_file(coord_csv='loc_coords_checkpoint.csv') -> pd.DataFrame:
    coords = pd.read_csv(coord_csv, index_col=0)
    df = evi_locs()
    coords = coords.reindex(df.index)
    return coords[['lat', 'lon']]

def soilgrids_multithreaded(
    lat_lon: pd.DataFrame,
    checkpoint_file='soilgrids_checkpoint.csv',
    output_file='soilgrids.csv',
    max_workers=6,
    chunk_size=1000,
    checkpoint_every=5
) -> pd.DataFrame:
    # 1) Load or init checkpoint
    if os.path.exists(checkpoint_file):
        out = pd.read_csv(checkpoint_file, index_col=0)
        out = out.reindex(lat_lon.index)
        done = out.dropna(how='all').shape[0]
        print(f"Resuming: {done}/{len(lat_lon)} points already fetched.")
    else:
        # sample via iloc[[0]]
        sample = soilgrids_df(lat_lon.iloc[[0]])
        out = pd.DataFrame(index=lat_lon.index, columns=sample.columns)
        print("Starting fresh: no checkpoint file found.")

    # 2) Chunk indices
    indices = [
        lat_lon.index[i:i+chunk_size]
        for i in range(0, len(lat_lon), chunk_size)
    ]
    total_chunks = len(indices)
    print(f"Total points: {len(lat_lon)}, chunk_size: {chunk_size}, total_chunks: {total_chunks}")

    completed_chunks = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(soilgrids_df, lat_lon.loc[idx_chunk]): idx_i
            for idx_i, idx_chunk in enumerate(indices)
        }
        for fut in as_completed(futures):
            idx_i = futures[fut]
            try:
                res = fut.result()
                out.loc[res.index] = res
                print(f"[Chunk {idx_i+1}/{total_chunks}] Completed")
            except Exception as e:
                print(f"[Chunk {idx_i+1}/{total_chunks}] FAILED: {e}")

            completed_chunks += 1
            if completed_chunks % checkpoint_every == 0 or completed_chunks == total_chunks:
                out.to_csv(checkpoint_file)
                print(f"Checkpoint saved after {completed_chunks}/{total_chunks} chunks")

    # 3) Final save
    out.to_csv(output_file)
    print(f"All done! SoilGrids output written to '{output_file}'")
    return out

if __name__ == '__main__':
    coords = return_coordinates_from_file()
    soilgrids_multithreaded(
        coords,
        checkpoint_file='soilgrids_checkpoint.csv',
        output_file='soilgrids.csv',
        max_workers=100,
        chunk_size=1,
        checkpoint_every=5
    )