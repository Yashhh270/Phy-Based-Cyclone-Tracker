import os
import time
import cdsapi
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------- settings ----------------
ERA5_DIR = "./era5_local"          # change to any folder with ~20 GB free
MAX_CONCURRENT = 3                 # lower to 3 if you see 429 / too-many-requests errors
MAX_RETRIES = 3

# highest-value years first (most NI training windows), so partial progress is still useful
YEARS = [1996,1992,2019,2005,2018,2013,2015,1991,2008,2016,1998,1997,2004,1984,2003,
         2014,2006,1995,2021,2022,1985,1982,2011,2023,2024,2001,2017,1989,2000,1988,
         2020,2025,1981,2009,1999,1990,1987,2002,1980,2007,1994,2010,2012,1993,1983,1986,
         1979]   # 1979 included because some storms starting in late Dec 1979 carry 1979 timestamps

os.makedirs(ERA5_DIR, exist_ok=True)


# ---------------- download one month ----------------
def download_month(year, month):
    outfile = os.path.join(ERA5_DIR, f"{year}_{month:02d}.nc")
    tmpfile = outfile + ".part"

    if os.path.exists(outfile):
        return (year, month, "skipped", None)

    client = cdsapi.Client(quiet=True)   # one client per thread avoids shared-state issues

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client.retrieve(
                "reanalysis-era5-pressure-levels",
                {
                    "product_type": "reanalysis",
                    "variable": ["u_component_of_wind", "v_component_of_wind"],
                    "pressure_level": ["500", "700"],
                    "year": str(year),
                    "month": f"{month:02d}",
                    "day": [f"{d:02d}" for d in range(1, 32)],
                    "time": ["00:00", "06:00", "12:00", "18:00"],
                    "area": [45, -180, -45, 180],
                    "grid": [1.0, 1.0],
                    "format": "netcdf",
                },
                tmpfile,
            )
            os.rename(tmpfile, outfile)     # only becomes "real" once fully downloaded
            return (year, month, "downloaded", None)
        except Exception as e:
            if os.path.exists(tmpfile):
                os.remove(tmpfile)          # never leave a truncated file behind
            if attempt == MAX_RETRIES:
                return (year, month, "failed", str(e))
            time.sleep(20 * attempt)        # back off before retrying


# ---------------- run everything ----------------
if __name__ == "__main__":
    tasks = [(y, m) for y in YEARS for m in range(1, 13)
             if not os.path.exists(os.path.join(ERA5_DIR, f"{y}_{m:02d}.nc"))]

    print(f"{len(tasks)} monthly files to download "
          f"({len(YEARS) * 12 - len(tasks)} already present)")

    done, failed = 0, []
    start = time.time()

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
        futures = [pool.submit(download_month, y, m) for (y, m) in tasks]
        for fut in as_completed(futures):
            y, m, status, err = fut.result()
            done += 1
            if status == "failed":
                failed.append((y, m))
                print(f"[{done}/{len(tasks)}] FAILED {y}-{m:02d}: {err}")
            else:
                elapsed = (time.time() - start) / 60
                print(f"[{done}/{len(tasks)}] {status} {y}-{m:02d}   ({elapsed:.1f} min elapsed)")

    print("\n===== SUMMARY =====")
    print(f"Failed: {len(failed)}")
    for y, m in failed:
        print(f"  {y}-{m:02d}")
    if failed:
        print("Re-run the script. It skips finished files and retries only the failed ones.")
    else:
        print("All files downloaded. Ready for extraction.")