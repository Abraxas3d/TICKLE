#!/usr/bin/env python3
"""
nexrad_replay.py -- TICKLE Step 4: replay archived NEXRAD against the corridor

Pulls NEXRAD Level II volumes for a date/time window from the AWS open
archive (bucket unidata-nexrad-level2, anonymous access), extracts reflectivity
inside the corridor's forward-scatter ellipse and altitude band, and writes a
timeline: max dBZ, echo area, and echo-top proxy per volume scan. Overlays
any 10 GHz corridor QSOs from the contest database on the plot.

This is the label generator's skeleton: "estimated scatter viability vs time"
from radar physics, no QSOs required.

Usage (examples):
  # The Parker rover mystery -- scatter or tropo?
  python3 nexrad_replay.py --date 2021-09-18 --start 14 --end 23 --radars KYUX

  # Did the backyard GOES archive watch a live corridor storm?
  python3 nexrad_replay.py --date 2023-09-01 --start 00 --end 23 \\
      --radars KYUX KIWA

Defaults: corridor = San Miguel Mtn (32.6976,-116.9330) <-> DM23XQ center
(33.6875,-114.0417); altitude band 3-15 km MSL (proxy until Step 2's floor
maps exist -- pass --floor-npz step2_outputs/floor_sanmiguel__parker.npz to
use the real common-volume floor).

Outputs in ./step4_outputs/: <date>_metrics.csv and <date>_timeline.png

Requires: numpy, matplotlib, boto3, arm-pyart (all in the TICKLE venv).
Downloads cache to /Users/w5nyv/TICKLE/data/nexrad_cache/ (override with
--cache); volumes are 5-15 MB each, so a full day on one radar is roughly
1-3 GB on first pull (estimate), then cached.
"""

import argparse, math, os, sqlite3, sys, warnings
import numpy as np

DEFAULT_CACHE = "/Users/w5nyv/TICKLE/data/nexrad_cache"
DEFAULT_QSODB = "/Users/w5nyv/TICKLE/data/contest_qsos.db"
BUCKET = "unidata-nexrad-level2"   # archive moved here July 2025; legacy
                                   # noaa-nexrad-level2 deprecated 2025-09-01

SITE_A = (32.6976, -116.9330)   # San Miguel Mtn (DM12MQ)
SITE_B = (33.6875, -114.0417)   # DM23XQ subsquare center (Parker/Bouse)

# ----------------------------------------------------------------- geometry
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - lon1)
    dp = p2 - p1
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

class FloorModel:
    """Common-volume floor: Step 2 npz if given, else flat altitude band."""
    def __init__(self, npz_path=None, lo_km=3.0):
        self.lo = lo_km * 1000.0
        self.grid = None
        if npz_path:
            z = np.load(npz_path)
            self.grid = (z["lat"], z["lon"], z["floor_m"])
            print(f"using Step 2 floor map {npz_path}")

    def floor_m(self, lat, lon):
        if self.grid is None:
            return np.full(np.shape(lat), self.lo)
        lats, lons, fl = self.grid
        i = np.clip(np.searchsorted(lats, lat) - 1, 0, len(lats)-1)
        j = np.clip(np.searchsorted(lons, lon) - 1, 0, len(lons)-1)
        return fl[i, j]

def corridor_metrics(radar, A=SITE_A, B=SITE_B, ellipse_pad_km=100.0,
                     floor=None, top_m=16000.0, refl_field="reflectivity"):
    """Reflectivity stats inside the forward-scatter ellipse + height band."""
    D = float(haversine_km(A[0], A[1], np.array([B[0]]), np.array([B[1]]))[0])
    lat = radar.gate_latitude["data"].ravel()
    lon = radar.gate_longitude["data"].ravel()
    alt = radar.gate_altitude["data"].ravel()          # m MSL
    z = radar.fields[refl_field]["data"]
    z = np.ma.filled(z, np.nan).ravel()

    dA = haversine_km(A[0], A[1], lat, lon)
    dB = haversine_km(B[0], B[1], lat, lon)
    fl = floor.floor_m(lat, lon) if floor else 3000.0
    sel = ((dA + dB) <= (D + ellipse_pad_km)) & (alt >= fl) & (alt <= top_m) \
          & np.isfinite(z)
    n = int(sel.sum())
    if n == 0:
        return dict(n_gates=0, max_dbz=np.nan, area30=0.0, area40=0.0,
                    top30_km=np.nan, cen_lat=np.nan, cen_lon=np.nan)
    zs, lats, lons, alts = z[sel], lat[sel], lon[sel], alt[sel]
    # crude per-gate area weight: range-dependent, but for a timeline the
    # gate COUNT above threshold (scaled) is a serviceable area proxy [estimate]
    gate_km2 = 1.0
    hot = zs >= 30.0
    m = dict(
        n_gates=n,
        max_dbz=float(np.nanmax(zs)),
        area30=float(hot.sum() * gate_km2),
        area40=float((zs >= 40.0).sum() * gate_km2),
        top30_km=float(alts[hot].max()/1000.0) if hot.any() else np.nan,
        cen_lat=float(lats[hot].mean()) if hot.any() else np.nan,
        cen_lon=float(lons[hot].mean()) if hot.any() else np.nan,
    )
    return m

# ----------------------------------------------------------------- AWS I/O
def list_volumes(date, radar):
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    prefix = f"{date[:4]}/{date[5:7]}/{date[8:10]}/{radar}/"
    keys = []
    tok = None
    while True:
        kw = dict(Bucket=BUCKET, Prefix=prefix, MaxKeys=1000)
        if tok: kw["ContinuationToken"] = tok
        resp = s3.list_objects_v2(**kw)
        keys += [o["Key"] for o in resp.get("Contents", [])]
        if not resp.get("IsTruncated"): break
        tok = resp["NextContinuationToken"]
    return s3, sorted(k for k in keys if not k.endswith("_MDM"))

def fetch(s3, key, cache):
    local = os.path.join(cache, key.replace("/", "_"))
    os.makedirs(cache, exist_ok=True)
    if not os.path.exists(local):
        s3.download_file(BUCKET, key, local)
    return local

def key_hour(key):
    """.../KYUX20210918_155500_V06 -> 15.9166 (fractional hour UTC)."""
    stem = os.path.basename(key)
    t = stem.split("_")[1]
    return int(t[:2]) + int(t[2:4])/60 + int(t[4:6])/3600

# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--start", type=float, default=0, help="UTC hour")
    ap.add_argument("--end", type=float, default=24, help="UTC hour")
    ap.add_argument("--radars", nargs="+", default=["KYUX"])
    ap.add_argument("--siteA", default=None, help="lat,lon (default San Miguel)")
    ap.add_argument("--siteB", default=None, help="lat,lon (default DM23XQ)")
    ap.add_argument("--floor-npz", default=None,
                    help="Step 2 floor map for this pair (optional)")
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--qso-db", default=DEFAULT_QSODB)
    ap.add_argument("--out", default="step4_outputs")
    a = ap.parse_args()

    A = tuple(map(float, a.siteA.split(","))) if a.siteA else SITE_A
    B = tuple(map(float, a.siteB.split(","))) if a.siteB else SITE_B
    floor = FloorModel(a.floor_npz)
    os.makedirs(a.out, exist_ok=True)

    import pyart
    warnings.filterwarnings("ignore")

    rows = []
    for radar_id in a.radars:
        s3, keys = list_volumes(a.date, radar_id)
        keys = [k for k in keys if a.start <= key_hour(k) <= a.end]
        print(f"{radar_id}: {len(keys)} volumes in window")
        for i, key in enumerate(keys):
            try:
                local = fetch(s3, key, os.path.join(a.cache, a.date))
                radar = pyart.io.read_nexrad_archive(local)
                fieldname = ("reflectivity" if "reflectivity" in radar.fields
                             else sorted(radar.fields)[0])
                m = corridor_metrics(radar, A, B, floor=floor,
                                     refl_field=fieldname)
                m.update(radar=radar_id, hour=key_hour(key), key=key)
                rows.append(m)
                print(f"  [{i+1}/{len(keys)}] {os.path.basename(key)}  "
                      f"max {m['max_dbz']:.0f} dBZ  "
                      f"area40 {m['area40']:.0f}  top {m['top30_km'] if m['top30_km']==m['top30_km'] else float('nan'):.1f} km",
                      flush=True)
                del radar
            except Exception as e:
                print(f"  skip {key}: {e}")

    if not rows:
        sys.exit("no volumes processed")

    # ------------------------------------------------------------- outputs
    import csv as csvmod
    csv_path = os.path.join(a.out, f"{a.date}_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csvmod.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # corridor QSOs in window for overlay
    qso_hours = []
    try:
        qc = sqlite3.connect(a.qso_db)
        def _soc(la, lo): return 32.0 <= la <= 34.5 and -118.5 <= lo <= -115.8
        def _azd(la, lo): return 31.5 <= la <= 35.5 and -115.8 <= lo <= -110.5
        seen = set()
        for ts, la1, lo1, la2, lo2, c1, c2 in qc.execute(
            "SELECT ts, lat1, lon1, lat2, lon2, call1, call2 FROM qsos "
            "WHERE band_ghz BETWEEN 9.5 AND 11 AND ts LIKE ? "
            "AND lat1 IS NOT NULL AND lat2 IS NOT NULL", (a.date + "%",)):
            if not ((_soc(la1, lo1) and _azd(la2, lo2)) or
                    (_soc(la2, lo2) and _azd(la1, lo1))):
                continue
            key = (ts, tuple(sorted((c1, c2))))
            if key in seen:
                continue
            seen.add(key)
            h = int(ts[11:13]) + int(ts[14:16]) / 60
            if a.start <= h <= a.end:
                qso_hours.append(h)
    except sqlite3.OperationalError as e:
        print(f"(QSO overlay skipped: {e})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    for radar_id in a.radars:
        rr = [r for r in rows if r["radar"] == radar_id]
        ax.plot([r["hour"] for r in rr], [r["max_dbz"] for r in rr],
                ".-", label=f"{radar_id} max dBZ in corridor")
    ax.axhline(30, ls="--", alpha=0.5)
    ax.axhline(40, ls="--", alpha=0.5)
    ax.text(a.start, 30.5, "30 dBZ (marginal)", fontsize=8)
    ax.text(a.start, 40.5, "40 dBZ (workable, rule-of-thumb)", fontsize=8)
    for h in qso_hours:
        ax.axvline(h, color="r", alpha=0.6)
    if qso_hours:
        ax.plot([], [], color="r", label="10 GHz QSO (any corridor pair)")
    ax.set_xlabel("UTC hour"); ax.set_ylabel("max reflectivity (dBZ)")
    ax.set_title(f"{a.date}  corridor reflectivity vs 10 GHz activity\n"
                 f"A=({A[0]:.3f},{A[1]:.3f})  B=({B[0]:.3f},{B[1]:.3f})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    png = os.path.join(a.out, f"{a.date}_timeline.png")
    fig.savefig(png, dpi=130)
    print(f"\nwrote {csv_path} and {png}")

    hot = [r for r in rows if r["max_dbz"] == r["max_dbz"] and r["max_dbz"] >= 40]
    if hot:
        h0, h1 = min(r["hour"] for r in hot), max(r["hour"] for r in hot)
        print(f"VERDICT: >=40 dBZ echoes in the corridor volume from "
              f"~{h0:.1f}z to ~{h1:.1f}z -- rain scatter was ON the menu.")
    else:
        print("VERDICT: no >=40 dBZ corridor echoes in this window -- "
              "contacts here were NOT convective rain scatter "
              "(tropo or other modes suspected).")

if __name__ == "__main__":
    main()
