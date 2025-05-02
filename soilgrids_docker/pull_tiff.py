import os
import requests
import rasterio
import time
import urllib.parse
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor  # Changed to ThreadPoolExecutor for better reliability
import logging
from rasterio.warp import transform_bounds
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('download_log.txt')
    ]
)
logger = logging.getLogger('tiff_downloader')

# ─── CONFIG ────────────────────────────────────────────────────────────────
BASE_URL = "https://files.isric.org/soilgrids/latest/data/"
INDIA_BBOX = (68.0, 8.0, 97.0, 37.0)
MAX_RETRIES = 5  # Increased retries
TIMEOUT = 60  # Add timeout to requests

# Thread configuration - more reliable than processes for this I/O bound task
MAX_THREADS = 32  # More conservative thread count
MAX_CHECK_THREADS = 16  # More conservative check thread count
DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1MB chunks for faster downloads

# Thread-safe stats tracking
stats = {
    "total_tiles_checked": 0,
    "tiles_covering_india": 0,
    "total_tifs_checked": 0,
    "tifs_covering_india": 0,
    "files_downloaded": 0,
    "errors": 0
}
stats_lock = threading.Lock()
print_lock = threading.Lock()

# Known tile IDs that cover India (fallback if automatic detection fails)
INDIA_TILE_IDS = [
    "tileSG-017-053", "tileSG-018-052", "tileSG-018-053", "tileSG-018-054", 
    "tileSG-019-052", "tileSG-019-053", "tileSG-019-054", "tileSG-019-055",
    "tileSG-020-052", "tileSG-020-053", "tileSG-020-054", "tileSG-020-055"
]

def safe_print(*args, **kwargs):
    """Thread-safe print function"""
    with print_lock:
        print(*args, **kwargs)
        logger.info(" ".join(str(arg) for arg in args))

# Create a session with appropriate retry strategy
session = requests.Session()
session.mount('https://', requests.adapters.HTTPAdapter(
    max_retries=MAX_RETRIES,
    pool_connections=MAX_THREADS,
    pool_maxsize=MAX_THREADS
))

def list_directory_contents(url, retries=MAX_RETRIES):
    """List all files and directories at the given URL."""
    for attempt in range(retries):
        try:
            safe_print(f"Accessing URL: {url}")
            r = session.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            
            # Try to parse HTML index page
            soup = BeautifulSoup(r.text, "html.parser")
            
            # Debug: Check if we have proper HTML
            if not soup.find_all("a"):
                safe_print(f"WARNING: No links found in HTML at {url}")
                safe_print(f"First 300 chars of response: {r.text[:300]}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
            
            # Get all links
            links = []
            dirs = []
            files = []
            
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("../") or href == "/":
                    continue
                    
                full_url = urljoin(url, href)
                links.append(full_url)
                
                # Classify as directory or file
                if href.endswith("/"):
                    dirs.append(full_url)
                else:
                    files.append(full_url)
            
            safe_print(f"Found {len(dirs)} directories and {len(files)} files at {url}")
            return {"dirs": dirs, "files": files, "all": links}
            
        except requests.RequestException as e:
            if attempt < retries - 1:
                wait_time = 2 ** attempt
                safe_print(f"Error accessing {url}: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                safe_print(f"Failed to access {url} after {retries} attempts: {e}")
                with stats_lock:
                    stats["errors"] += 1
                return {"dirs": [], "files": [], "all": []}
        except Exception as e:
            if attempt < retries - 1:
                wait_time = 2 ** attempt
                safe_print(f"Unexpected error accessing {url}: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                safe_print(f"Critical error accessing {url}: {e}")
                with stats_lock:
                    stats["errors"] += 1
                return {"dirs": [], "files": [], "all": []}

def list_tile_directories(url, retries=MAX_RETRIES):
    """List all tile directories under the given URL."""
    contents = list_directory_contents(url)
    # Filter for tile directories (tileSG-*)
    tile_dirs = [d for d in contents["dirs"] if "tileSG-" in d]
    
    if not tile_dirs:
        safe_print(f"WARNING: No tile directories found at {url}")
        # If no tiles found, try direct URL construction for known India tiles
        base_path = url.rstrip("/")
        constructed_tiles = [
            f"{base_path}/{tile_id}/" for tile_id in INDIA_TILE_IDS
        ]
        safe_print(f"Trying direct URL construction for known India tiles: {constructed_tiles[:2]}...")
        return constructed_tiles
        
    return tile_dirs

def check_tif_coverage(tif_url, retries=MAX_RETRIES):
    """Check if a specific TIF file covers India by examining its coordinates."""
    filename = os.path.basename(tif_url)
    thread_id = threading.get_ident() % 10000
    
    # If the TIF URL is part of a known India tile, return True immediately
    if any(india_tile in tif_url for india_tile in INDIA_TILE_IDS):
        safe_print(f"[Thread {thread_id}] TIF from known India tile: {filename}")
        return True
        
    # Check coordinates
    for attempt in range(retries):
        try:
            safe_print(f"[Thread {thread_id}] Checking coverage of {filename}")
            with rasterio.open(f"/vsicurl/{tif_url}") as src:
                w, s, e, n = src.bounds
                lon0, lat0, lon1, lat1 = transform_bounds(src.crs, "EPSG:4326", w, s, e, n)
            
            west, south, east, north = INDIA_BBOX
            covers = not (lon1 < west or lon0 > east or lat1 < south or lat0 > north)
            
            if covers:
                safe_print(f"[Thread {thread_id}] TIF bounds: ({lon0:.2f},{lat0:.2f}) to ({lon1:.2f},{lat1:.2f}) - COVERS India")
            else:
                safe_print(f"[Thread {thread_id}] TIF bounds: ({lon0:.2f},{lat0:.2f}) to ({lon1:.2f},{lat1:.2f}) - OUTSIDE India")
                
            return covers
            
        except Exception as e:
            if attempt < retries - 1:
                wait_time = 2 ** attempt
                safe_print(f"[Thread {thread_id}] Error checking TIF coverage: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                safe_print(f"[Thread {thread_id}] Failed to check TIF coverage: {e}")
                for tile_id in INDIA_TILE_IDS:
                    if tile_id in tif_url:
                        safe_print(f"[Thread {thread_id}] TIF from known India tile ID {tile_id}, accepting")
                        return True
                return False

def check_and_download_tif(tif_url, local_file):
    """Check if a TIF covers India and download it if it does."""
    filename = os.path.basename(tif_url)
    thread_id = threading.get_ident() % 10000
    
    # Skip if file already exists and has content
    if os.path.exists(local_file) and os.path.getsize(local_file) > 0:
        safe_print(f"[Thread {thread_id}] File already exists: {filename}")
        return True
        
    # First check if the TIF covers India
    with stats_lock:
        stats["total_tifs_checked"] += 1
        
    if not check_tif_coverage(tif_url):
        safe_print(f"[Thread {thread_id}] [SKIP] TIF does not cover India: {filename}")
        return False
        
    with stats_lock:
        stats["tifs_covering_india"] += 1
    safe_print(f"[Thread {thread_id}] [KEEP] TIF covers India: {filename}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(local_file), exist_ok=True)
    
    # Download with retries
    for attempt in range(MAX_RETRIES):
        try:
            safe_print(f"[Thread {thread_id}] Downloading {filename}...")
            with session.get(tif_url, stream=True, timeout=TIMEOUT) as resp:
                resp.raise_for_status()
                total_size = int(resp.headers.get('content-length', 0))
                
                # Ensure temp file is in the same directory for atomic move
                temp_file = f"{local_file}.part"
                with open(temp_file, "wb") as f:
                    downloaded = 0
                    for chunk in resp.iter_content(DOWNLOAD_CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            # Periodically report progress for large files
                            if total_size > 10*1024*1024 and downloaded % (5*1024*1024) < DOWNLOAD_CHUNK_SIZE:
                                safe_print(f"[Thread {thread_id}] Downloaded {downloaded/1024/1024:.1f}MB of {total_size/1024/1024:.1f}MB")
                
                # Atomic move to final destination
                os.replace(temp_file, local_file)
            
            # Verify file was downloaded correctly
            if os.path.getsize(local_file) > 0:
                with stats_lock:
                    stats["files_downloaded"] += 1
                safe_print(f"[Thread {thread_id}] Downloaded {filename} ({os.path.getsize(local_file)/1024/1024:.1f} MB)")
                return True
            else:
                if os.path.exists(local_file):
                    os.remove(local_file)
                safe_print(f"[Thread {thread_id}] Download failed (empty file): {filename}")
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = 2 ** attempt
                safe_print(f"[Thread {thread_id}] Download error: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                safe_print(f"[Thread {thread_id}] Failed to download after {MAX_RETRIES} attempts: {filename}")
                # Clean up failed download
                if os.path.exists(f"{local_file}.part"):
                    os.remove(f"{local_file}.part")
                with stats_lock:
                    stats["errors"] += 1
                return False
    
    return False

def tile_covers_india(tile_url, retries=MAX_RETRIES):
    """Check if a tile covers India by examining coordinates of sample TIFs."""
    if any(india_tile in tile_url for india_tile in INDIA_TILE_IDS):
        safe_print(f"Known India tile detected: {tile_url}")
        return True
        
    safe_print(f"Searching for TIF files under {tile_url}")
    all_tifs = []
    
    def collect_tifs(url, max_depth=2, current_depth=0):
        if current_depth > max_depth:
            return
            
        contents = list_directory_contents(url)
        tifs = [f for f in contents["files"] if f.lower().endswith(".tif")]
        all_tifs.extend(tifs)
        
        for subdir in contents["dirs"]:
            collect_tifs(subdir, max_depth, current_depth + 1)
    
    collect_tifs(tile_url)
    
    if not all_tifs:
        safe_print(f"WARNING: No TIF files found under {tile_url}")
        for tile_id in INDIA_TILE_IDS:
            if tile_id in tile_url:
                safe_print(f"Known India tile ID {tile_id} in URL path, accepting")
                return True
        return False
    
    sample_tifs = all_tifs[:5]
    safe_print(f"Checking {len(sample_tifs)} sample TIFs from tile")
    
    # Use ThreadPoolExecutor instead for more reliable execution
    with ThreadPoolExecutor(max_workers=MAX_CHECK_THREADS) as executor:
        # Check each TIF directly
        future_to_tif = {executor.submit(check_tif_coverage, tif): tif for tif in sample_tifs}
        
        for future in concurrent.futures.as_completed(future_to_tif):
            tif = future_to_tif[future]
            try:
                covers = future.result()
                if covers:
                    safe_print(f"Found TIF covering India: {os.path.basename(tif)}")
                    return True
            except Exception as e:
                safe_print(f"Error checking TIF {os.path.basename(tif)}: {e}")
    
    safe_print(f"No TIFs covering India found in tile")
    return False

def download_hierarchy(url, local_base_path, depth=0, max_depth=3):
    """Recursively download TIF files from a URL hierarchy using parallel threads."""
    if depth > max_depth:
        return 0
    
    url_parts = url.rstrip("/").split("/")
    current_dir = url_parts[-1]
    
    local_path = os.path.join(local_base_path, current_dir)
    if os.path.exists(local_path) and os.listdir(local_path):
        local_path = os.path.join(local_base_path, f"{current_dir}_new")
        safe_print(f"Directory already exists, creating {local_path} instead")
    
    os.makedirs(local_path, exist_ok=True)
    
    contents = list_directory_contents(url)
    files_downloaded = 0
    
    tifs = [f for f in contents["files"] if f.lower().endswith(".tif")]
    if tifs:
        safe_print(f"Found {len(tifs)} TIFs at {url}, processing in parallel")
        
        # Use ThreadPoolExecutor instead
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            future_to_file = {
                executor.submit(check_and_download_tif, tif_url, os.path.join(local_path, os.path.basename(tif_url))): 
                tif_url for tif_url in tifs
            }
            
            for future in concurrent.futures.as_completed(future_to_file):
                tif_url = future_to_file[future]
                try:
                    success = future.result()
                    if success:
                        files_downloaded += 1
                except Exception as e:
                    safe_print(f"Error processing {os.path.basename(tif_url)}: {e}")
                    with stats_lock:
                        stats["errors"] += 1
    
    subdirs = contents["dirs"]
    if subdirs:
        # Process subdirectories in parallel
        with ThreadPoolExecutor(max_workers=max(2, MAX_THREADS//4)) as executor:
            future_to_dir = {}
            for subdir in subdirs:
                future = executor.submit(download_hierarchy, subdir, local_path, depth + 1, max_depth)
                future_to_dir[future] = subdir
            
            for future in concurrent.futures.as_completed(future_to_dir):
                try:
                    subdir_files = future.result()
                    files_downloaded += subdir_files
                except Exception as e:
                    safe_print(f"Error in recursive processing: {e}")
                    with stats_lock:
                        stats["errors"] += 1
    
    return files_downloaded

if __name__ == "__main__":
    try:
        # Process only WRB property
        prop = 'wrb'
        safe_print(f"\n{'='*80}\nProcessing {prop} (MostProbable)...\n{'='*80}")
        prop_path = f"{prop}/MostProbable"
        current_url = urljoin(BASE_URL, prop_path + "/")
        dest_folder = f"/workspace/soilgrids_selected/{prop_path}"
        
        os.makedirs(dest_folder, exist_ok=True)
        
        # Verify we can access the URL before proceeding
        try:
            test_response = session.get(current_url, timeout=TIMEOUT)
            test_response.raise_for_status()
            safe_print(f"Successfully verified access to {current_url}")
        except Exception as e:
            safe_print(f"WARNING: Could not verify access to {current_url}: {e}")
            safe_print("Will attempt to continue anyway...")
        
        # Get tile directories
        safe_print(f"Listing tile directories at {current_url}")
        tile_urls = list_tile_directories(current_url)
        if not tile_urls:
            raise Exception(f"No tile directories found at {current_url}")
        
        safe_print(f"Found {len(tile_urls)} tile directories")
        
        for tile_url in tile_urls:
            tile_name = os.path.basename(tile_url.rstrip("/"))
            with stats_lock:
                stats["total_tiles_checked"] += 1
            
            safe_print(f"Checking if tile {tile_name} covers India...")
            if not tile_covers_india(tile_url):
                safe_print(f"[SKIP] {prop_path}/{tile_name}")
                continue
            
            with stats_lock:
                stats["tiles_covering_india"] += 1
            safe_print(f"[KEEP] {prop_path}/{tile_name}")
            
            safe_print(f"Downloading hierarchy for {tile_name}...")
            local_tile_dir = os.path.join(dest_folder, tile_name)
            download_hierarchy(tile_url, dest_folder)
                
    except Exception as e:
        safe_print(f"Critical error in main process: {str(e)}")
        stats["errors"] += 1
    finally:
        # Print final stats
        safe_print("\n" + "="*80)
        safe_print("Download Summary:")
        safe_print(f"Total tiles checked: {stats['total_tiles_checked']}")
        safe_print(f"Tiles covering India: {stats['tiles_covering_india']}")
        safe_print(f"Total TIFs checked individually: {stats['total_tifs_checked']}")
        safe_print(f"TIFs covering India: {stats['tifs_covering_india']}")
        safe_print(f"Files downloaded: {stats['files_downloaded']}")
        safe_print(f"Errors encountered: {stats['errors']}")
        safe_print("Done.")
