import os
import sys
import requests as r
import pandas as pd
import json
from datetime import datetime
import scipy.stats as st
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import math
import time
import threading
import logging
import pickle
import random
from pathlib import Path
from multiprocessing import Manager, freeze_support

MAX_WORKERS = 1024

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

years = [2020, 2021, 2022]
dates = {
    'kharif': ['-06-01', '-10-31'],
    'rabi': ['-11-01', '-04-30'],
}

class RateLimiter:
    def __init__(self, max_calls, time_period):
        self.max_calls = max_calls  # Maximum calls per time period
        self.time_period = time_period  # Time period in seconds
        self.calls = []  # Timestamps of calls
        self.lock = threading.Lock()  # Thread-safe lock

    def wait_if_needed(self):
        with self.lock:
            now = time.time()

            # Remove timestamps older than our time period
            self.calls = [t for t in self.calls if now - t < self.time_period]

            # If at capacity, wait until we can make another call
            if len(self.calls) >= self.max_calls:
                sleep_time = self.calls[0] + self.time_period - now
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    # After sleeping, we need to update our list again
                    now = time.time()
                    self.calls = [t for t in self.calls if now - t < self.time_period]

            # Add the current call timestamp
            self.calls.append(now)

# Global rate limiter: 100 calls per 60 seconds
api_limiter = RateLimiter(max_calls=100, time_period=60)

def request_url(latitude, longitude, startdate, enddate, request_counter=None, checkpoint_lock=None, failed_requests=None):
    # Apply rate limiting before making the API call
    api_limiter.wait_if_needed()

    start = datetime.strptime(startdate, "%Y-%m-%d").strftime("%Y%m%d")
    end = datetime.strptime(enddate, "%Y-%m-%d").strftime("%Y%m%d")
    url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=CDD10,PRECTOTCORR&community=AG&longitude={longitude}&latitude={latitude}&start={start}&end={end}&format=JSON"

    # Implement retry logic for 429 errors
    max_retries = 5
    retry_delay = 10  # Initial delay in seconds

    for attempt in range(max_retries):
        try:
            response = r.get(url, verify=True)

            # Increment the global request counter and log progress
            if request_counter and checkpoint_lock:
                with checkpoint_lock:
                    request_counter.value += 1
                    current_count = request_counter.value
                    logger.info(f"API Request #{current_count} - Location: ({latitude}, {longitude})")

                    # Save progress every 100 requests
                    if current_count % 100 == 0:
                        logger.info(f"Checkpoint: Completed {current_count} API requests")

            if response.status_code == 429:
                wait_time = retry_delay * (2 ** attempt) + random.uniform(1, 5)
                logger.warning(f"Rate limit exceeded (429). Waiting for {wait_time:.2f} seconds before retry.")
                time.sleep(wait_time)
                continue
            
            if response.status_code != 200:
                # For non-429 errors, add to failed requests list for later retry
                failed_req = {
                    'latitude': latitude, 'longitude': longitude,
                    'startdate': startdate, 'enddate': enddate,
                    'error': f"HTTP {response.status_code}: {response.text[:100]}",
                    'attempt': attempt + 1
                }
                if failed_requests is not None:
                    failed_requests.append(failed_req)
                logger.error(f"Request failed with status {response.status_code}: {response.text[:100]}. Will retry later.")
                raise r.exceptions.HTTPError(f"HTTP {response.status_code}: {response.text[:100]}")

            response.raise_for_status()
            content = json.loads(response.content.decode('utf-8'))
            return content

        except r.exceptions.HTTPError as e:
            if response.status_code == 429:
                wait_time = retry_delay * (2 ** attempt) + random.uniform(1, 5)
                logger.warning(f"Rate limit exceeded (429). Waiting for {wait_time:.2f} seconds before retry.")
                time.sleep(wait_time)
            else:
                logger.error(f"HTTP Error: {e}")
                # Non-429 HTTP errors get added to failed requests
                if attempt == max_retries - 1:
                    failed_req = {
                        'latitude': latitude, 'longitude': longitude,
                        'startdate': startdate, 'enddate': enddate,
                        'error': str(e),
                        'attempt': attempt + 1
                    }
                    if failed_requests is not None:
                        failed_requests.append(failed_req)
                    logger.error(f"Request failed after {attempt+1} attempts. Will retry at the end.")
                raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            if attempt == max_retries - 1:
                failed_req = {
                    'latitude': latitude, 'longitude': longitude,
                    'startdate': startdate, 'enddate': enddate,
                    'error': f"JSON decode error: {str(e)}",
                    'attempt': attempt + 1
                }
                if failed_requests is not None:
                    failed_requests.append(failed_req)
                logger.error(f"JSON decoding failed after {attempt+1} attempts. Will retry at the end.")
            raise
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            if attempt == max_retries - 1:
                failed_req = {
                    'latitude': latitude, 'longitude': longitude,
                    'startdate': startdate, 'enddate': enddate,
                    'error': str(e),
                    'attempt': attempt + 1
                }
                if failed_requests is not None:
                    failed_requests.append(failed_req)
                logger.error(f"Request failed after {attempt+1} attempts with error {str(e)}. Will retry at the end.")
            raise

    raise Exception(f"Failed to get data after {max_retries} retries")

def get_data(latitude, longitude, startdate, enddate, request_counter=None, checkpoint_lock=None, failed_requests=None):
    data = request_url(latitude, longitude, startdate, enddate, 
                      request_counter, checkpoint_lock, failed_requests)
    prectotcorr = data['properties']['parameter']['PRECTOTCORR']
    cdd10 = data['properties']['parameter']['CDD10']
    dates_p = data['properties']['parameter']['PRECTOTCORR'].keys()
    dates_c = data['properties']['parameter']['CDD10'].keys()
    if dates_p != dates_c:
        raise ValueError("Dates for PRECTOTCORR and CDD10 do not match.")
    dates = sorted(dates_p)
    df = pd.DataFrame(index=dates, columns=['PRECTOTCORR', 'CDD10'])
    for date in dates:
        df.at[date, 'PRECTOTCORR'] = prectotcorr[date]
        df.at[date, 'CDD10'] = cdd10[date]
    df.index = pd.to_datetime(df.index, format='%Y%m%d')
    return df

def return_final(latitude, longitude, startdate, enddate, request_counter=None, checkpoint_lock=None, failed_requests=None):
    df_orig = get_data(latitude, longitude, startdate, enddate, 
                      request_counter, checkpoint_lock, failed_requests)
    degdays = df_orig['CDD10'].sum()
    rain_series = df_orig['PRECTOTCORR']
    rain_cv = rain_series.std(ddof=0) / rain_series.mean()

    roll_sum = rain_series.rolling(window=90, min_periods=90).sum().dropna()
    alpha, loc, beta = st.gamma.fit(roll_sum, floc=0)
    cdf_vals = st.gamma.cdf(roll_sum, alpha, loc=loc, scale=beta)
    spi3 = pd.Series(st.norm.ppf(cdf_vals), index=roll_sum.index)
    spi3_mean = spi3.mean()

    data = {
        'gdd': degdays,
        'rain_cv': rain_cv,
        'spi3_mean': spi3_mean
    }

    return data

def save_checkpoint(year, season, completed_locations, results):
    """Save progress to a checkpoint file"""
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_file = checkpoint_dir / f"checkpoint_{year}_{season}.pkl"

    with open(checkpoint_file, 'wb') as f:
        pickle.dump({
            'year': year,
            'season': season,
            'completed_locations': completed_locations,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }, f)

    logger.info(f"Saved checkpoint for {season}_{year} with {len(completed_locations)} completed locations")
    # Print to terminal for clear visibility
    print(f"\n===== CHECKPOINT SAVED =====")
    print(f"Year: {year}, Season: {season}")
    print(f"Completed locations: {len(completed_locations)}")
    print(f"Checkpoint file: {checkpoint_file.absolute()}")
    print(f"============================\n")

# Add a new function to list all checkpoints
def list_checkpoints():
    """List all available checkpoints in the checkpoints directory"""
    checkpoint_dir = Path('checkpoints')
    if not checkpoint_dir.exists():
        print("No checkpoints directory found.")
        return []
        
    checkpoints = list(checkpoint_dir.glob("checkpoint_*.pkl"))
    if not checkpoints:
        print("No checkpoint files found in the checkpoints directory.")
        return []
        
    print("\n===== AVAILABLE CHECKPOINTS =====")
    checkpoint_info = []
    for cp_file in checkpoints:
        try:
            with open(cp_file, 'rb') as f:
                cp_data = pickle.load(f)
                info = {
                    'year': cp_data['year'],
                    'season': cp_data['season'],
                    'completed': len(cp_data['completed_locations']),
                    'timestamp': cp_data['timestamp'],
                    'file': cp_file
                }
                checkpoint_info.append(info)
                print(f"Year: {info['year']}, Season: {info['season']}, Completed: {info['completed']}")
                print(f"Timestamp: {info['timestamp']}")
                print(f"File: {cp_file.absolute()}")
                print("--------------------------")
        except Exception as e:
            print(f"Error reading checkpoint file {cp_file}: {e}")
    
    print("===============================\n")
    return checkpoint_info

def load_checkpoint(year, season):
    """Load progress from a checkpoint file if it exists"""
    checkpoint_file = Path(f"checkpoints/checkpoint_{year}_{season}.pkl")

    if checkpoint_file.exists():
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
            logger.info(f"Loaded checkpoint for {season}_{year} with {len(checkpoint['completed_locations'])} completed locations")
            return checkpoint['completed_locations'], checkpoint['results']

    return set(), {}

def process_location_batch(loc_batch, year, season, startdate, enddate, request_counter=None, checkpoint_lock=None, failed_requests=None):
    """Process a batch of locations using ThreadPoolExecutor"""
    results = {}
    completed_locations = set()
    batch_id = hash(f"{random.random()}")
    logger.info(f"Starting batch {batch_id} with {len(loc_batch)} locations")

    # Load any existing checkpoint for this year/season
    existing_completed, existing_results = load_checkpoint(year, season)

    # Use 3 threads per process for IO-bound API calls
    with ThreadPoolExecutor(max_workers=3) as thread_executor:
        futures = {}
        for idx, (lat, lon) in loc_batch.iterrows():
            # Skip already completed locations
            if idx in existing_completed:
                results[idx] = existing_results.get(idx)
                logger.info(f"Skipping already completed location {idx}")
                continue

            future = thread_executor.submit(return_final, lat, lon, startdate, enddate, 
                                          request_counter, checkpoint_lock, failed_requests)
            futures[idx] = future

        for idx, future in futures.items():
            try:
                data = future.result()
                results[idx] = data
                completed_locations.add(idx)

                # Check if we need to save a checkpoint (based on global counter)
                if checkpoint_lock and request_counter:
                    with checkpoint_lock:
                        if request_counter.value % 100 == 0:
                            all_completed = existing_completed.union(completed_locations)
                            all_results = {**existing_results, **results}
                            save_checkpoint(year, season, all_completed, all_results)

            except Exception as e:
                logger.error(f"Error processing location {idx}: {e}")

    logger.info(f"Completed batch {batch_id}, processed {len(completed_locations)} locations")
    # Add existing results to our return value
    results.update(existing_results)
    return results, completed_locations.union(existing_completed)

def process_year_season(loc_df, year, season):
    """Process a specific year-season combination"""
    if season == 'rabi':
        # Rabi of Y runs Nov Y → Apr (Y+1)
        enddate = f"{year+1}{dates['rabi'][1]}"
    else:
        enddate = f"{year}{dates[season][1]}"
    startdate = str(year) + dates[season][0]

    df = pd.DataFrame(columns=['gdd', 'rain_cv', 'spi3_mean'], index=loc_df.index)

    return (df, year, season, startdate, enddate, loc_df)

def retry_failed_requests(failed_requests):
    """Retry all failed requests at a slower rate (1 per 10 seconds)"""
    if not failed_requests:
        logger.info("No failed requests to retry")
        return
    
    logger.info(f"Starting to retry {len(failed_requests)} failed requests at 1 per 10 seconds")
    retry_results = {}
    
    # Save list of all failed requests for reference
    failed_log_path = Path('failed_requests_log.json')
    with open(failed_log_path, 'w') as f:
        json.dump([dict(req) for req in failed_requests], f, indent=2)
    
    logger.info(f"Saved log of all failed requests to {failed_log_path}")
    
    # Try to process each failed request
    for i, req in enumerate(failed_requests):
        logger.info(f"Retrying request {i+1}/{len(failed_requests)}: ({req['latitude']}, {req['longitude']})")
        try:
            # Wait 10 seconds between requests
            if i > 0:
                logger.info("Waiting 10 seconds before next retry...")
                time.sleep(10)
                
            # Attempt to process this request again
            data = return_final(req['latitude'], req['longitude'], req['startdate'], req['enddate'])
            retry_results[(req['latitude'], req['longitude'])] = data
            logger.info(f"Successfully retrieved data for ({req['latitude']}, {req['longitude']})")
        except Exception as e:
            logger.error(f"Final retry failed for ({req['latitude']}, {req['longitude']}): {e}")
    
    # Save successful retries
    if retry_results:
        retry_results_path = Path('retry_results.pkl')
        with open(retry_results_path, 'wb') as f:
            pickle.dump(retry_results, f)
        logger.info(f"Successfully retried {len(retry_results)}/{len(failed_requests)} requests")
        logger.info(f"Saved retry results to {retry_results_path}")
    else:
        logger.warning("No requests were successfully retried")
    
    return retry_results

def return_csv():
    # Create Manager instance inside this function
    manager = Manager()
    request_counter = manager.Value('i', 0)
    checkpoint_lock = manager.Lock()
    failed_requests = manager.list()
    
    loc_df = pd.read_csv('loc_coords_checkpoint.csv', index_col=0)

    # Create a list of all year-season combinations to process
    tasks = []
    for year in years:
        for season in dates.keys():
            tasks.append((loc_df, year, season))

    # Create output directory if it doesn't exist
    power_data_dir = 'power_data'
    if not os.path.exists(power_data_dir):
        os.makedirs(power_data_dir)

    # First process all year-season combinations
    year_season_results = []
    for loc_df, year, season in tasks:
        result = process_year_season(loc_df, year, season)
        year_season_results.append(result)

    # Then process locations in parallel using 6 processes (matching core count)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as process_executor:
        for df, year, season, startdate, enddate, loc_df in year_season_results:
            # Split locations into batches for parallel processing
            num_batches = 6  # Match the number of cores
            batch_size = math.ceil(len(loc_df) / num_batches)
            loc_batches = [loc_df.iloc[i:i+batch_size] for i in range(0, len(loc_df), batch_size)]

            # Submit batch processing tasks to the process pool with shared objects
            batch_futures = [process_executor.submit(process_location_batch,
                                                     batch, year, season, startdate, enddate,
                                                     request_counter, checkpoint_lock, failed_requests)
                             for batch in loc_batches]

            # Collect results
            for future in batch_futures:
                results, completed_locations = future.result()
                for idx, data in results.items():
                    df.at[idx, 'gdd'] = data['gdd']
                    df.at[idx, 'rain_cv'] = data['rain_cv']
                    df.at[idx, 'spi3_mean'] = data['spi3_mean']

            # Save the results
            output_path = f'{power_data_dir}/{season}_{year}_data.csv'
            df.to_csv(output_path)
            print(f"Wrote {output_path}")

    # Retry any failed requests at the end
    logger.info("Main processing complete. Starting retry of failed requests...")
    retry_results = retry_failed_requests(failed_requests)
    
    print(f"All data written to {power_data_dir}/")
    
    if retry_results:
        print(f"Additionally recovered {len(retry_results)} previously failed requests")
    
    if failed_requests:
        print(f"Warning: {len(failed_requests) - len(retry_results)} requests ultimately failed. See logs for details.")

if __name__ == "__main__":
    # Add freeze_support for Windows
    freeze_support()
    
    # Show all existing checkpoints before starting
    list_checkpoints()
    
    try:
        return_csv()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user. Checkpoint information:")
        list_checkpoints()
        print("\nTo resume, simply run the script again. It will automatically use the latest checkpoints.")
        sys.exit(1)