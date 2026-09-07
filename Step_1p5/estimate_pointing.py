#!/usr/bin/env python3
"""
estimate_pointing.py -- Recover mesoscale sector pointing history (Step 1.5)

For every CH13 frame in the manifest, extracts the map-overlay line mask,
groups consecutive frames into "dwell segments" (same pointing), and
chamfer-matches one representative frame per segment against Natural Earth
coastline/state borders projected through the GOES-18 fixed-grid projection.

Writes results into the manifest database:
    pointing_segments(product, seg_start, seg_end, n_frames, rep_path,
                      score, x0_rad, y0_rad, center_lat, center_lon,
                      covers_corridor)

Then reports what fraction of the archive covers the SD->AZ corridor.

Usage:
    pip3 install numpy scipy pillow
    python3 estimate_pointing.py goes18_manifest.db
    python3 estimate_pointing.py goes18_manifest.db --tx 32.84,-117.25 --rx 33.45,-112.07
    python3 estimate_pointing.py goes18_manifest.db --limit 2000      # trial run
    python3 estimate_pointing.py --paths img1.jpg img2.jpg            # spot-check mode

Resumable: segments already matched are skipped on rerun.

ASSUMPTION (labeled per your standing rule): satellite longitude is taken as
the GOES-West nominal -137.0 deg. If GOES-18 was stationed at -137.2 during
part of the archive, absolute centers shift by a few km; the corridor
coverage verdicts are insensitive to this. Registration accuracy against the
two validated samples was ~1-2 pixels (2-4 km); treat centers as +/- ~5 km.
"""

import argparse, json, os, sqlite3, sys, time, urllib.request
from datetime import datetime
import numpy as np
from PIL import Image
from scipy import ndimage

# ----------------------------- projection ---------------------------------
REQ, RPOL = 6378137.0, 6356752.31414
H = 42164160.0
LON0 = np.deg2rad(-137.0)   # assumed nominal GOES-West longitude
E2 = (REQ**2 - RPOL**2) / REQ**2
RES_CH13 = 56e-6            # rad/pixel for 2-km HRIT imagery

def latlon_to_scan(lat_deg, lon_deg):
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    phi_c = np.arctan((RPOL**2 / REQ**2) * np.tan(lat))
    rc = RPOL / np.sqrt(1 - E2 * np.cos(phi_c)**2)
    dl = lon - LON0
    sx = H - rc * np.cos(phi_c) * np.cos(dl)
    sy = -rc * np.cos(phi_c) * np.sin(dl)
    sz = rc * np.sin(phi_c)
    vis = (H * (H - sx)) >= (sy**2 + (REQ**2 / RPOL**2) * sz**2)
    r = np.sqrt(sx**2 + sy**2 + sz**2)
    return np.arcsin(-sy / r), np.arctan(sz / sx), vis

def scan_to_latlon(x, y):
    a = np.sin(x)**2 + np.cos(x)**2 * (np.cos(y)**2 + (REQ**2/RPOL**2) * np.sin(y)**2)
    b = -2 * H * np.cos(x) * np.cos(y)
    c = H**2 - REQ**2
    disc = b*b - 4*a*c
    if disc < 0: return None, None
    rs = (-b - np.sqrt(disc)) / (2*a)
    sx, sy, sz = rs*np.cos(x)*np.cos(y), -rs*np.sin(x), rs*np.cos(x)*np.sin(y)
    lat = np.arctan((REQ**2/RPOL**2) * sz / np.sqrt((H-sx)**2 + sy**2))
    lon = LON0 - np.arctan(sy / (H - sx))
    return float(np.rad2deg(lat)), float(np.rad2deg(lon))

# --------------------------- border reference -----------------------------
NE_BASE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
NE_FILES = ["ne_50m_coastline.geojson", "ne_50m_admin_1_states_provinces_lines.geojson"]

def load_border_points(cache_dir="ne_cache", step_km=4.0):
    os.makedirs(cache_dir, exist_ok=True)
    pts = []
    for name in NE_FILES:
        local = os.path.join(cache_dir, name)
        if not os.path.exists(local):
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(NE_BASE + name, local)
        gj = json.load(open(local))
        for feat in gj["features"]:
            g = feat["geometry"]
            lines = g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]
            for line in lines:
                arr = np.asarray(line, dtype=float)
                if len(arr) < 2: continue
                out = [arr[0]]
                for a, b in zip(arr[:-1], arr[1:]):
                    d = np.hypot((b[0]-a[0]) * 111.0 * np.cos(np.deg2rad((a[1]+b[1])/2)),
                                 (b[1]-a[1]) * 111.0)
                    n = max(1, int(d / step_km))
                    for k in range(1, n+1):
                        out.append(a + (b-a) * k / n)
                pts.append(np.asarray(out))
    ll = np.vstack(pts)
    x, y, vis = latlon_to_scan(ll[:, 1], ll[:, 0])
    m = vis & np.isfinite(x) & np.isfinite(y)
    return np.column_stack([x[m], y[m]])

# --------------------------- mask + matching ------------------------------
def extract_overlay_mask(path):
    try:
        im = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    except Exception:
        return None, None
    if im.shape != (500, 500):
        return None, None
    tophat = im - ndimage.grey_opening(im, size=(3, 3))
    mask = tophat > 28
    mask[:3, :] = mask[-3:, :] = False
    mask[:, :3] = mask[:, -3:] = False
    return mask, im.shape

def make_scorer(mask, hp, wp, res, bx, by):
    dt = ndimage.distance_transform_edt(~mask)
    mn = max(int(mask.sum()), 1)
    def score(cx, cy):
        sel = (np.abs(bx - cx) < wp/2*res) & (np.abs(by - cy) < hp/2*res)
        if sel.sum() < 150: return -1.0
        col = ((bx[sel] - cx) / res + wp/2).astype(np.int32)
        row = ((cy - by[sel]) / res + hp/2).astype(np.int32)
        ok = (col >= 0) & (col < wp) & (row >= 0) & (row < hp)
        if ok.sum() < 150: return -1.0
        lin = np.unique(row[ok] * wp + col[ok])
        fwd = float(np.mean(np.exp(-dt[lin // wp, lin % wp] / 2.0)))
        R = np.zeros((hp, wp), dtype=bool)
        R[lin // wp, lin % wp] = True
        R = ndimage.binary_dilation(R, iterations=2)
        bwd = float((mask & R).sum()) / mn
        return fwd * bwd
    return score

def match_sector(mask, shape, borders, res=RES_CH13):
    hp, wp = shape
    bx, by = borders[:, 0], borders[:, 1]
    f = 4
    m4 = mask.reshape(hp//f, f, wp//f, f).any(axis=(1, 3))
    sc = make_scorer(m4, hp//f, wp//f, res*f, bx, by)
    best = (-2.0, 0.0, 0.0)
    for cx in np.arange(-0.16, 0.16, 0.003):
        for cy in np.arange(-0.16, 0.16, 0.003):
            s = sc(cx, cy)
            if s > best[0]: best = (s, cx, cy)
    scf = make_scorer(mask, hp, wp, res, bx, by)
    for stepr in (0.0006, 0.0001, 0.00003):
        s0, cx0, cy0 = best
        best = (scf(cx0, cy0), cx0, cy0)
        for cx in np.arange(cx0 - stepr*5, cx0 + stepr*5, stepr):
            for cy in np.arange(cy0 - stepr*5, cy0 + stepr*5, stepr):
                s = scf(cx, cy)
                if s > best[0]: best = (s, cx, cy)
    return best

# --------------------------- corridor test --------------------------------
def corridor_scan_bbox(tx, rx, margin_km=75.0):
    """Bounding box (scan angles) around the great-circle-ish corridor,
    from sample points along the line between endpoints plus margin."""
    lats = np.linspace(tx[0], rx[0], 25)
    lons = np.linspace(tx[1], rx[1], 25)
    dlat = margin_km / 111.0
    pts_lat, pts_lon = [], []
    for la, lo in zip(lats, lons):
        dlon = margin_km / (111.0 * np.cos(np.deg2rad(la)))
        pts_lat += [la - dlat, la + dlat, la, la]
        pts_lon += [lo, lo, lo - dlon, lo + dlon]
    x, y, vis = latlon_to_scan(np.array(pts_lat), np.array(pts_lon))
    return float(x.min()), float(x.max()), float(y.min()), float(y.max())

def frame_covers(cx, cy, bbox, hp=500, wp=500, res=RES_CH13):
    x_lo, x_hi = cx - wp/2*res, cx + wp/2*res
    y_lo, y_hi = cy - hp/2*res, cy + hp/2*res
    return (bbox[0] >= x_lo and bbox[1] <= x_hi and
            bbox[2] >= y_lo and bbox[3] <= y_hi)

# ------------------------------- driver -----------------------------------
SEG_SCHEMA = """
CREATE TABLE IF NOT EXISTS pointing_segments (
    id INTEGER PRIMARY KEY,
    product TEXT, seg_start TEXT, seg_end TEXT, n_frames INTEGER,
    rep_path TEXT, score REAL, x0_rad REAL, y0_rad REAL,
    center_lat REAL, center_lon REAL, covers_corridor INTEGER
);
CREATE INDEX IF NOT EXISTS idx_seg ON pointing_segments (product, seg_start);
"""

def iou(a, b):
    inter = (a & b).sum()
    union = (a | b).sum()
    return inter / union if union else 1.0

def process(db_path, tx, rx, margin_km, limit=None, min_score=0.10):
    borders = load_border_points()
    print(f"border reference points: {len(borders):,}")
    bbox = corridor_scan_bbox(tx, rx, margin_km)
    conn = sqlite3.connect(db_path)
    conn.executescript(SEG_SCHEMA)
    cur = conn.cursor()

    for product in ("M1", "M2"):
        done_upto = cur.execute(
            "SELECT MAX(seg_end) FROM pointing_segments WHERE product=?", (product,)
        ).fetchone()[0]
        q = ("SELECT path, start_time FROM files WHERE parsed=1 AND channel=13 "
             "AND product=? ")
        args = [product]
        if done_upto:
            q += "AND start_time > ? "
            args.append(done_upto)
            print(f"{product}: resuming after {done_upto}")
        q += "ORDER BY start_time"
        if limit: q += f" LIMIT {int(limit)}"
        rows = cur.execute(q, args).fetchall()
        print(f"{product}: {len(rows):,} frames to scan")

        seg_frames, prev_mask = [], None   # seg_frames: (path, time, mask, npx)
        n_seg = 0
        t0 = time.time()

        def finalize(frames):
            nonlocal n_seg
            if not frames: return
            # representative = most overlay pixels (least cloud-obscured)
            cand = sorted(frames, key=lambda r: -r[3])[:5]
            best = (-2.0, 0.0, 0.0); rep = cand[0]
            for c in cand:
                s, cx, cy = match_sector(c[2], (500, 500), borders)
                if s > best[0]: best, rep = (s, cx, cy), c
                if s >= 0.5: break
            s, cx, cy = best
            lat, lon = scan_to_latlon(cx, cy)
            cov = frame_covers(cx, cy, bbox) if s >= min_score else 0
            cur.execute(
                "INSERT INTO pointing_segments (product, seg_start, seg_end, n_frames,"
                " rep_path, score, x0_rad, y0_rad, center_lat, center_lon, covers_corridor)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (product, frames[0][1], frames[-1][1], len(frames), rep[0],
                 s, cx, cy, lat, lon, int(cov)))
            conn.commit()
            n_seg += 1

        for i, (path, t) in enumerate(rows):
            mask, shape = extract_overlay_mask(path)
            if mask is None: continue
            npx = int(mask.sum())
            if prev_mask is not None and iou(mask, prev_mask) < 0.35:
                finalize(seg_frames); seg_frames = []
            seg_frames.append((path, t, mask, npx))
            prev_mask = mask
            if i % 500 == 0 and i:
                rate = i / (time.time() - t0)
                print(f"  {product} {i:,}/{len(rows):,} frames, {n_seg} segments, "
                      f"{rate:.0f} f/s, ETA {(len(rows)-i)/rate/60:.0f} min", flush=True)
        finalize(seg_frames)
        print(f"{product}: {n_seg} segments matched")

    report(conn, tx, rx)
    conn.close()

def report(conn, tx, rx):
    cur = conn.cursor()
    print("\n" + "=" * 66)
    print(f"CORRIDOR COVERAGE  ({tx} -> {rx})")
    print("=" * 66)
    for product in ("M1", "M2"):
        tot = cov = low = 0
        for s, e, n, sc, c in cur.execute(
            "SELECT seg_start, seg_end, n_frames, score, covers_corridor "
            "FROM pointing_segments WHERE product=?", (product,)):
            tot += n
            if sc < 0.10: low += n
            elif c: cov += n
        if tot:
            print(f"{product}: {tot:,} frames | corridor-covering: {cov:,} "
                  f"({100*cov/tot:.1f}%) | unmatchable (score<0.10): {low:,}")
    print("\nTop dwell targets (where NOAA pointed the sectors):")
    for prod, lat, lon, n in cur.execute(
        "SELECT product, ROUND(center_lat), ROUND(center_lon), SUM(n_frames) nf "
        "FROM pointing_segments WHERE score>=0.10 AND center_lat IS NOT NULL "
        "GROUP BY product, "
        "ROUND(center_lat), ROUND(center_lon) ORDER BY nf DESC LIMIT 12"):
        print(f"  {prod} ~({lat:+.0f},{lon:+.0f}): {n:,} frames")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db", nargs="?", help="manifest database from build_manifest.py")
    ap.add_argument("--tx", default="32.84,-117.25", help="lat,lon of near endpoint")
    ap.add_argument("--rx", default="33.45,-112.07", help="lat,lon of far endpoint")
    ap.add_argument("--margin", type=float, default=75.0, help="corridor margin, km")
    ap.add_argument("--limit", type=int, help="max frames per sector (trial runs)")
    ap.add_argument("--paths", nargs="+", help="spot-check specific CH13 jpgs, no db")
    a = ap.parse_args()
    tx = tuple(float(v) for v in a.tx.split(","))
    rx = tuple(float(v) for v in a.rx.split(","))
    if a.paths:
        borders = load_border_points()
        bbox = corridor_scan_bbox(tx, rx, a.margin)
        for p in a.paths:
            mask, shape = extract_overlay_mask(p)
            if mask is None:
                print(f"{p}: not a 500x500 CH13 frame"); continue
            s, cx, cy = match_sector(mask, shape, borders)
            lat, lon = scan_to_latlon(cx, cy)
            print(f"{os.path.basename(p)}: score {s:.3f} center "
                  f"lat {lat:+.2f}, lon {lon:+.2f} "
                  f"covers corridor: {frame_covers(cx, cy, bbox)}")
        return
    if not a.db: sys.exit("provide the manifest db (or --paths for spot checks)")
    process(a.db, tx, rx, a.margin, a.limit)

if __name__ == "__main__":
    main()
