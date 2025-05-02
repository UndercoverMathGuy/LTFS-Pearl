import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from constants import google_maps_api_key
import time

def request_maps(village: str, district: str, state: str, zipcode: str):
    """
    Return (lat, lon, status_code).
    Status codes:
      200 → OK
      204 → No results
      429 → Rate limit
      other → HTTP error
    """
    address = f"{village}, {district}, {state}, {zipcode}"
    url = (
        "https://maps.googleapis.com/maps/api/geocode/json"
        f"?address={requests.utils.quote(address)}"
        f"&key={google_maps_api_key}"
    )
    resp = requests.get(url)
    code = resp.status_code

    if code == 429:
        return None, None, 429
    if code != 200:
        return None, None, code

    data = resp.json()
    results = data.get("results", [])
    if not results:
        return None, None, 204

    loc = results[0]["geometry"]["location"]
    time.sleep(1/25)
    return loc["lat"], loc["lng"], 200

def fetch_one(args):
    """
    Worker function that calls request_maps and returns:
      (index, lat, lon, status_code)
    """
    idx, village, district, state, zipcode = args
    lat, lon, status = request_maps(village, district, state, zipcode)

    if status == 200:
        return idx, lat, lon, status

    # Log non-200 statuses
    if status == 429:
        print(f"[{idx}] RATE LIMIT (429)")
    elif status == 204:
        print(f"[{idx}] NO RESULTS")
    else:
        print(f"[{idx}] HTTP ERROR {status}")
    return idx, None, None, status

def loc_df_multithreaded(
    csv_path='train_data_cleaned.csv',
    max_workers=16,
    checkpoint_every=100,
    checkpoint_file='loc_coords_checkpoint.csv'
):
    # 1) Load input CSV
    df = pd.read_csv(csv_path)[['VILLAGE', 'DISTRICT', 'State', 'Zipcode']]

    # 2) Load or initialize checkpointed output
    try:
        out = pd.read_csv(checkpoint_file, index_col=0)
        out = out.reindex(df.index)
        done = out['lat'].notnull().sum()
        print(f"Resuming: {done}/{len(df)} rows already completed.")
    except FileNotFoundError:
        out = pd.DataFrame(columns=['lat','lon'], index=df.index)
        print("No checkpoint found; starting from row 0.")

    # 3) Build task list, skipping completed rows
    tasks = [
        (i, village, district, state, zipcode)
        for i, village, district, state, zipcode
        in df.itertuples(index=True, name=None)
        if pd.isna(out.loc[i, 'lat'])
    ]

    # 4) Execute in threads with checkpointing
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, task): task[0] for task in tasks}
        for fut in as_completed(futures):
            idx, lat, lon, status = fut.result()
            if status == 200:
                out.loc[idx] = [lat, lon]

            completed += 1
            if completed % checkpoint_every == 0:
                out.to_csv(checkpoint_file)
                print(f"  → checkpointed {completed} new rows…")

    # 5) Final save
    out.to_csv(checkpoint_file)
    print(f"Done! All results saved to '{checkpoint_file}'")

def requery_no_results(
    csv_path='train_data_cleaned.csv',
    checkpoint_file='loc_coords_checkpoint.csv',
    max_workers=16,
    checkpoint_every=20
):
    # 1) Load inputs and checkpoint
    df = pd.read_csv(csv_path)[['VILLAGE','DISTRICT','State','Zipcode']]
    out = pd.read_csv(checkpoint_file, index_col=0)
    out = out.reindex(df.index)

    # 2) Do we have a status column?
    has_status = 'status' in out.columns

    # 3) Identify rows to retry:
    if has_status:
        # If you ever store numeric codes, change '204' to match that.
        to_retry = out.index[out['status'] == 'ZERO_RESULTS'].tolist()
    else:
        # No status stored → retry any row that still has NaN lat
        to_retry = out.index[out['lat'].isna()].tolist()

    if not to_retry:
        print("Nothing to re-query (no NaN or ZERO_RESULTS).")
        return out

    print(f"Re-querying {len(to_retry)} rows...")

    # 4) Build tasks
    tasks = []
    for i in to_retry:
        row = df.loc[i]
        tasks.append((i,
                      row['VILLAGE'],
                      row['DISTRICT'],
                      row['State'],
                      row['Zipcode']))

    # 5) Fire up threads
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(fetch_one, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            idx, lat, lon, status = fut.result()

            # update lat/lon
            if lat is not None and lon is not None:
                out.loc[idx, ['lat','lon']] = [lat, lon]
            # store status if you have the column (or create it)
            if has_status:
                out.loc[idx, 'status'] = status

            completed += 1
            if completed % checkpoint_every == 0:
                out.to_csv(checkpoint_file)
                print(f"  → checkpointed {completed}/{len(to_retry)} retried rows…")

    # 6) Final save & report
    out.to_csv(checkpoint_file)
    filled = out['lat'].notna().sum()
    total  = len(out)
    print(f"Retry done. {filled}/{total} rows now have coordinates.")
    return out

# Usage after your main run:
if __name__ == '__main__':
    df_coords = requery_no_results()
