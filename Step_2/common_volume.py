#!/usr/bin/env python3
"""
common_volume.py -- TICKLE Step 2: path geometry & common scattering volume

For each site: builds a 360-degree terrain horizon profile from SRTM 1-arcsec
elevation data (AWS Terrain Tiles "skadi" mirror, anonymous HTTPS, no key).
For each site pair: computes, on a lat/lon grid over the corridor, the minimum
altitude visible from BOTH ends (the common-volume floor), with 4/3-Earth
refraction. A storm cell is a usable mirror where its top pokes above this
floor.

Outputs (in ./step2_outputs/):
    horizon_<site>.npz          az + horizon angle profile per site
    floor_<near>__<far>.npz     lat/lon grid + floor height (m MSL)
    floor_<near>__<far>.png     contour map
    summary.txt                 verdict table

Usage:
    python3 common_volume.py                 # full matrix
    python3 common_volume.py --pairs delmar:parker sanmiguel:parker
    python3 common_volume.py --list          # show configured sites

Requires: numpy, matplotlib (in the TICKLE venv).
First run downloads ~15-25 SRTM tiles (rough estimate: 10-25 MB each,
cached in ./dem_tiles/ so later runs are offline).

Reference cell tops for reading the verdicts (typical literature values,
not measurements): moderate monsoon convection ~10-12 km MSL, strong
organized cells ~13-16 km MSL.
"""

import argparse, gzip, math, os, sys, urllib.request
import numpy as np

# ------------------------------------------------------------------ sites
# mode "fixed": use given lat/lon exactly (operational sites -- YOUR spots).
# mode "grid_high": highest SRTM point within the 6-char subsquare
#   (contest sites -- estimates where label-generating operators stood;
#    accessibility irrelevant, this is label geometry only).
# agl_m: antenna height above ground.
SITES = {
    "delmar":    dict(name="Del Mar / Carmel Valley home (DM12jw)",
                      mode="fixed", lat=32.9375, lon=-117.2083, agl_m=10.0),
    "sanmiguel": dict(name="San Miguel Mtn (DM12MQ community site)",
                      mode="fixed", lat=32.6976, lon=-116.9330, agl_m=5.0),
    "palomar":   dict(name="Palomar cabin (DM13nh)",
                      mode="fixed", lat=33.3125, lon=-116.8750, agl_m=5.0),
    "parker":    dict(name="Parker/Bouse area (DM23XQ contest spot)",
                      mode="grid_high", grid="DM23XQ", agl_m=5.0),
    "phoenix":   dict(name="Phoenix west metro (DM33, representative)",
                      mode="fixed", lat=33.40, lon=-112.30, agl_m=5.0),
}
PAIRS = [("delmar", "parker"), ("delmar", "phoenix"),
         ("sanmiguel", "parker"), ("sanmiguel", "phoenix"),
         ("palomar", "parker"), ("palomar", "phoenix")]

RE = 6371000.0 * 4.0 / 3.0        # effective Earth radius, 4/3 refraction
HORIZON_RANGE_M = 150e3           # terrain farther than this can't raise mask
AZ_STEP_DEG = 0.5
RANGE_STEP_M = 200.0
GRID = dict(lat0=31.0, lat1=36.0, lon0=-118.6, lon1=-110.4, step=0.05)
DEFAULT_DEM_CACHE = "/Users/w5nyv/TICKLE/data/dem_tiles"

# ------------------------------------------------------------------ maidenhead
def grid_to_box(g):
    """6-char grid -> (lat_lo, lat_hi, lon_lo, lon_hi) of subsquare."""
    g = g.upper()
    lon = (ord(g[0]) - 65) * 20 - 180 + int(g[2]) * 2 + (ord(g[4]) - 65) * (2/24)
    lat = (ord(g[1]) - 65) * 10 - 90 + int(g[3]) * 1 + (ord(g[5]) - 65) * (1/24)
    return lat, lat + 1/24, lon, lon + 2/24

# ------------------------------------------------------------------ DEM
TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi/{ns}{lat:02d}/{ns}{lat:02d}{ew}{lon:03d}.hgt.gz"

class Dem:
    """SRTM 1-arcsec tiles, lazily downloaded, memory-mapped."""
    def __init__(self, cache="DEFAULT_DEM_CACHE"):
        self.cache = cache
        os.makedirs(cache, exist_ok=True)
        self.tiles = {}

    def _tile(self, ilat, ilon):
        ilat, ilon = int(ilat), int(ilon)   # numpy ints -> native (np.bool_ can't index tuples)
        key = (ilat, ilon)
        if key in self.tiles:
            return self.tiles[key]
        ns, ew = ("N", "S")[ilat < 0], ("E", "W")[ilon < 0]
        stem = f"{ns}{abs(ilat):02d}{ew}{abs(ilon):03d}"
        hgt = os.path.join(self.cache, stem + ".hgt")
        if not os.path.exists(hgt):
            url = TILE_URL.format(ns=ns, lat=abs(ilat), ew=ew, lon=abs(ilon))
            print(f"  downloading {stem} ...", flush=True)
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    raw = gzip.decompress(r.read())
                with open(hgt, "wb") as f:
                    f.write(raw)
            except Exception as e:
                sys.exit(f"DEM download failed for {stem}: {e}\n"
                         f"(url: {url})")
        n = int(math.sqrt(os.path.getsize(hgt) // 2))
        arr = np.memmap(hgt, dtype=">i2", mode="r", shape=(n, n))
        self.tiles[key] = arr
        return arr

    def elev(self, lat, lon):
        """Vectorized nearest-sample elevation (m). Handles voids as 0."""
        lat = np.atleast_1d(np.asarray(lat, dtype=float))
        lon = np.atleast_1d(np.asarray(lon, dtype=float))
        out = np.zeros(lat.shape, dtype=float)
        ilat = np.floor(lat).astype(int)
        ilon = np.floor(lon).astype(int)
        for key in set(zip(ilat.ravel(), ilon.ravel())):
            m = (ilat == key[0]) & (ilon == key[1])
            arr = self._tile(*key)
            n = arr.shape[0]
            row = np.clip(((key[0] + 1 - lat[m]) * (n - 1)).round().astype(int), 0, n-1)
            col = np.clip(((lon[m] - key[1]) * (n - 1)).round().astype(int), 0, n-1)
            v = arr[row, col].astype(float)
            v[v < -1000] = 0.0            # SRTM voids
            out[m] = v
        return out

# ------------------------------------------------------------------ geodesy
def dest_point(lat, lon, bearing_deg_, dist_m):
    """Destination on sphere. Vectorized over dist_m."""
    R = 6371000.0
    br = math.radians(bearing_deg_)
    p1 = math.radians(lat); l1 = math.radians(lon)
    dr = np.asarray(dist_m) / R
    p2 = np.arcsin(np.sin(p1)*np.cos(dr) + np.cos(p1)*np.sin(dr)*math.cos(br))
    l2 = l1 + np.arctan2(math.sin(br)*np.sin(dr)*np.cos(p1),
                         np.cos(dr) - np.sin(p1)*np.sin(p2))
    return np.degrees(p2), np.degrees(l2)

def dist_bearing(lat1, lon1, lat2, lon2):
    """Vectorized over point-2 arrays. Returns (m, deg)."""
    R = 6371000.0
    p1, p2 = math.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    a = np.sin((p2-p1)/2)**2 + math.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    d = 2*R*np.arcsin(np.sqrt(a))
    y = np.sin(dl)*np.cos(p2)
    x = math.cos(p1)*np.sin(p2) - math.sin(p1)*np.cos(p2)*np.cos(dl)
    return d, (np.degrees(np.arctan2(y, x)) + 360) % 360

# ------------------------------------------------------------------ core
def resolve_site(key, dem):
    s = SITES[key]
    if s["mode"] == "fixed":
        lat, lon = s["lat"], s["lon"]
    else:
        la0, la1, lo0, lo1 = grid_to_box(s["grid"])
        lats = np.linspace(la0, la1, 160)
        lons = np.linspace(lo0, lo1, 260)
        LA, LO = np.meshgrid(lats, lons, indexing="ij")
        E = dem.elev(LA.ravel(), LO.ravel()).reshape(LA.shape)
        i = np.unravel_index(np.argmax(E), E.shape)
        lat, lon = float(LA[i]), float(LO[i])
        print(f"  {key}: highest point in {s['grid']}: "
              f"({lat:.4f},{lon:.4f}) {E[i]:.0f} m  [estimated stand-in "
              f"for historical operating position]")
    ground = float(dem.elev(lat, lon)[0])
    return dict(key=key, name=s["name"], lat=lat, lon=lon,
                ground_m=ground, obs_m=ground + s["agl_m"])

def horizon_profile(site, dem):
    """Max terrain elevation angle (deg) per azimuth, 4/3-Earth."""
    azs = np.arange(0, 360, AZ_STEP_DEG)
    dists = np.arange(RANGE_STEP_M, HORIZON_RANGE_M, RANGE_STEP_M)
    prof = np.empty(azs.shape)
    for i, az in enumerate(azs):
        la, lo = dest_point(site["lat"], site["lon"], az, dists)
        h = dem.elev(la, lo)
        ang = np.degrees(np.arctan2(h - site["obs_m"] - dists**2/(2*RE), dists))
        prof[i] = max(float(ang.max()), np.degrees(math.atan2(-math.sqrt(
            2*max(site["obs_m"], 1.0)/RE), 1)))  # never below smooth-earth horizon
    return azs, prof

def floor_map(siteA, siteB, azA, profA, azB, profB):
    g = GRID
    lats = np.arange(g["lat0"], g["lat1"], g["step"])
    lons = np.arange(g["lon0"], g["lon1"], g["step"])
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    floors = np.empty(LA.shape)
    dA, brA = dist_bearing(siteA["lat"], siteA["lon"], LA, LO)
    dB, brB = dist_bearing(siteB["lat"], siteB["lon"], LA, LO)
    thA = np.interp(brA.ravel(), azA, profA, period=360).reshape(brA.shape)
    thB = np.interp(brB.ravel(), azB, profB, period=360).reshape(brB.shape)
    hA = siteA["obs_m"] + dA*np.tan(np.radians(thA)) + dA**2/(2*RE)
    hB = siteB["obs_m"] + dB*np.tan(np.radians(thB)) + dB**2/(2*RE)
    floors = np.maximum(hA, hB)
    return lats, lons, floors, dA, dB

def verdict(siteA, siteB, lats, lons, floors, dA, dB):
    D, _ = dist_bearing(siteA["lat"], siteA["lon"],
                        np.array([siteB["lat"]]), np.array([siteB["lon"]]))
    D = float(D[0])
    ellipse = (dA + dB) <= (D + 100e3)          # forward-scatter-relevant zone
    f_in = np.where(ellipse, floors, np.inf)
    i = np.unravel_index(np.argmin(f_in), f_in.shape)
    best = float(f_in[i])
    return dict(path_km=D/1e3, best_floor_km=best/1e3,
                best_lat=float(lats[i[0]]), best_lon=float(lons[i[1]]),
                mirror_dA_km=float(dA[i]/1e3), mirror_dB_km=float(dB[i]/1e3))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*",
                    help="near:far keys, e.g. delmar:parker (default: full matrix)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dem-cache", default=DEFAULT_DEM_CACHE)
    a = ap.parse_args()
    if a.list:
        for k, s in SITES.items():
            print(f"{k:10s} {s['name']}")
        return
    pairs = ([tuple(p.split(":")) for p in a.pairs] if a.pairs else PAIRS)

    os.makedirs("step2_outputs", exist_ok=True)
    dem = Dem(a.dem_cache)
    sites, profiles = {}, {}
    for key in sorted({k for p in pairs for k in p}):
        print(f"resolving site {key} ...")
        sites[key] = resolve_site(key, dem)
        print(f"  {sites[key]['name']}: ({sites[key]['lat']:.4f},"
              f"{sites[key]['lon']:.4f}) ground {sites[key]['ground_m']:.0f} m,"
              f" antenna {sites[key]['obs_m']:.0f} m MSL")
        print(f"  building horizon profile ...")
        azs, prof = horizon_profile(sites[key], dem)
        profiles[key] = (azs, prof)
        np.savez(f"step2_outputs/horizon_{key}.npz", az=azs, horizon_deg=prof)
        east = prof[(azs >= 45) & (azs <= 135)]
        print(f"  eastern horizon (az 45-135): min {east.min():+.2f} deg, "
              f"median {np.median(east):+.2f} deg, max {east.max():+.2f} deg")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lines = ["pair                       path_km  best_floor_km  mirror@ (lat,lon)  dA_km  dB_km"]
    for nk, fk in pairs:
        A, B = sites[nk], sites[fk]
        lats, lons, floors, dA, dB = floor_map(A, B, *profiles[nk], *profiles[fk])
        v = verdict(A, B, lats, lons, floors, dA, dB)
        np.savez(f"step2_outputs/floor_{nk}__{fk}.npz",
                 lat=lats, lon=lons, floor_m=floors)
        fig, ax = plt.subplots(figsize=(10, 6))
        levels = [0, 2000, 4000, 6000, 8000, 10000, 12000, 16000, 25000]
        cs = ax.contourf(lons, lats, np.clip(floors, 0, 25000),
                         levels=levels, cmap="viridis_r")
        fig.colorbar(cs, label="common-volume floor (m MSL)")
        ax.plot([A["lon"], B["lon"]], [A["lat"], B["lat"]], "r.-")
        ax.annotate(nk, (A["lon"], A["lat"]), color="r")
        ax.annotate(fk, (B["lon"], B["lat"]), color="r")
        ax.plot(v["best_lon"], v["best_lat"], "w*", ms=14)
        ax.set_title(f"{A['name']}  <->  {B['name']}\n"
                     f"path {v['path_km']:.0f} km, best floor "
                     f"{v['best_floor_km']:.1f} km MSL at white star")
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        fig.tight_layout()
        fig.savefig(f"step2_outputs/floor_{nk}__{fk}.png", dpi=130)
        plt.close(fig)
        lines.append(f"{nk:>10s} -> {fk:<10s}  {v['path_km']:7.0f}"
                     f"  {v['best_floor_km']:13.1f}"
                     f"  ({v['best_lat']:.2f},{v['best_lon']:.2f})"
                     f"  {v['mirror_dA_km']:5.0f}  {v['mirror_dB_km']:5.0f}")
        print(lines[-1])

    lines.append("")
    lines.append("Reading the floor: a storm cell can act as the mirror where its")
    lines.append("top exceeds the floor. Typical monsoon cell tops (literature-")
    lines.append("typical values, not measured here): moderate ~10-12 km MSL,")
    lines.append("strong organized ~13-16 km MSL. Floors under ~8 km: routinely")
    lines.append("workable. 8-12 km: needs vigorous convection. >16 km: no.")
    with open("step2_outputs/summary.txt", "w") as f:
        f.write("\n".join(lines))
    print("\nwrote step2_outputs/summary.txt and per-pair maps")

if __name__ == "__main__":
    main()
