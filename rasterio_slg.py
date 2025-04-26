#!/usr/bin/env python3
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── CONFIG ────────────────────────────────────────────────────────────────
GDAL_TRANSLATE = r"C:\OSGeo4W\bin\gdal_translate.exe"
BASE_URL       = "https://files.isric.org/soilgrids/latest/data"
VRT_ROOT       = "soilgrids_raw"
TIF_ROOT       = "soilgrids_tifs"
MAX_WORKERS    = 8
TIMEOUT_SEC    = 600   # 10 minutes per layer

# ─── BUILD TASK LIST ───────────────────────────────────────────────────────
tasks = []
for prop in os.listdir(VRT_ROOT):
    prop_dir = os.path.join(VRT_ROOT, prop)
    if not os.path.isdir(prop_dir):
        continue
    if prop == "wrb":
        tasks.append((prop, "MostProbable.vrt"))
    else:
        for fn in os.listdir(prop_dir):
            if fn.endswith("_mean.vrt"):
                tasks.append((prop, fn))

print(f"Queued {len(tasks)} layers for translation → {MAX_WORKERS} threads, {TIMEOUT_SEC}s timeout each")

# ─── WORKER ─────────────────────────────────────────────────────────────────
def translate_task(prop, vrt_name):
    key = f"{prop}/{vrt_name}"
    out_dir = os.path.join(TIF_ROOT, prop)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(vrt_name)[0]
    out_tif = os.path.join(out_dir, base + ".tif")

    if os.path.exists(out_tif):
        return f"[SKIP ] {key}"

    remote_vrt = f"{BASE_URL}/{prop}/{vrt_name}"
    vsipath    = f"/vsicurl/{remote_vrt}"
    cmd        = [GDAL_TRANSLATE, "-co", "COMPRESS=LZW", vsipath, out_tif]

    # Log start
    print(f"[START] {key}", flush=True)
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=TIMEOUT_SEC
        )
        return f"[DONE ] {key}"
    except subprocess.TimeoutExpired:
        return f"[TIMEOUT] {key} (> {TIMEOUT_SEC}s)"
    except subprocess.CalledProcessError as e:
        errmsg = e.stderr.decode(errors='ignore').splitlines()[-1]
        return f"[ERROR] {key}: {errmsg}"

# ─── EXECUTE IN PARALLEL ───────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(TIF_ROOT, exist_ok=True)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        future_to_task = {
            exe.submit(translate_task, prop, vrt): (prop, vrt)
            for prop, vrt in tasks
        }
        for fut in as_completed(future_to_task):
            # As soon as each finishes, print its result
            print(fut.result())
    print("All tasks complete.")
