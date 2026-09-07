#!/usr/bin/env python3
"""
parse_contest_logs.py -- ARRL 10 GHz & Up Cabrillo logs -> QSO database (Step 3)

Walks a directory of Cabrillo files (.log/.cbr/.txt), parses headers and QSO
lines tolerantly (real-world Cabrillo is messy), converts Maidenhead grids to
lat/lon, computes path distance/bearing, and writes everything to SQLite.
Then reports: per-year totals, distance distribution, and the San Diego <->
Arizona corridor subset with timestamps (the future NEXRAD cross-reference).

Usage:
    python3 parse_contest_logs.py /path/to/logs_dir
    python3 parse_contest_logs.py /path/to/logs_dir --db contest_qsos.db

Lines that fail to parse are written to unparsed_lines.txt for inspection --
send a sample back if that file is large and the parser will be extended.

Requires: standard library only.
"""

import argparse, math, os, re, sqlite3, sys
from datetime import datetime

# ---------------------------------------------------------------- maidenhead
def grid_to_latlon(grid):
    """6-char (or 4-char) Maidenhead grid -> (lat, lon) of square center."""
    g = grid.strip().upper()[:6]      # extended (8-char) grids -> 6-char square
    if not re.fullmatch(r"[A-R]{2}[0-9]{2}([A-X]{2})?", g):
        return None, None
    lon = (ord(g[0]) - ord("A")) * 20 - 180
    lat = (ord(g[1]) - ord("A")) * 10 - 90
    lon += int(g[2]) * 2
    lat += int(g[3]) * 1
    if len(g) == 6:
        lon += (ord(g[4]) - ord("A")) * (2 / 24) + (1 / 24)
        lat += (ord(g[5]) - ord("A")) * (1 / 24) + (1 / 48)
    else:
        lon += 1.0
        lat += 0.5
    return lat, lon

def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * r * math.asin(math.sqrt(a))

def bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1)*math.sin(p2) - math.sin(p1)*math.cos(p2)*math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360

# ---------------------------------------------------------------- parsing
# QSO line, tolerant. Canonical ARRL 10 GHz & Up form:
#   QSO: 10G PH 2019-08-17 1830 W6XXX DM12JR K7YYY DM33XX
# Seen in the wild: band as 10G/10GHZ/10368/10.368G/2.3G/LIGHT, dates with
# slashes, extra whitespace, lowercase grids, trailing junk.
QSO_RE = re.compile(
    r"^QSO:\s+(?P<band>\S+)\s+(?P<mode>\S+)\s+"
    r"(?P<date>\d{4}[-/]\d{2}[-/]\d{2})\s+(?P<time>\d{3,4})\s+"
    r"(?P<call1>\S+)\s+(?P<grid1>[A-Ra-r]{2}\d{2}(?:[A-Xa-x]{2}(?:\d{2})?)?)\s+"
    r"(?P<call2>\S+)\s+(?P<grid2>[A-Ra-r]{2}\d{2}(?:[A-Xa-x]{2}(?:\d{2})?)?)",
    re.IGNORECASE,
)

def norm_band(tok):
    """Map band token to GHz (float) where possible."""
    t = tok.upper().replace("GHZ", "G").rstrip("G")
    try:
        v = float(t)
        if v > 1000:            # given in MHz, e.g. 10368
            v /= 1000.0
        return v
    except ValueError:
        return None             # e.g. LIGHT

HEADER_KEYS = ("CALLSIGN", "CATEGORY-OPERATOR", "LOCATION", "CLUB", "GRID-LOCATOR")

def parse_file(path):
    header = {}
    qsos, bad = [], []
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except OSError as e:
        return header, qsos, [f"(unreadable: {e})"]
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        u = s.upper()
        if u.startswith("QSO:"):
            m = QSO_RE.match(s)
            if not m:
                bad.append(s)
                continue
            d = m.groupdict()
            date = d["date"].replace("/", "-")
            hhmm = d["time"].zfill(4)
            try:
                ts = datetime.strptime(f"{date} {hhmm}", "%Y-%m-%d %H%M")
            except ValueError:
                bad.append(s)
                continue
            qsos.append({
                "band_raw": d["band"], "band_ghz": norm_band(d["band"]),
                "mode": d["mode"].upper(), "ts": ts.isoformat(),
                "call1": d["call1"].upper(), "grid1": d["grid1"].upper(),
                "call2": d["call2"].upper(), "grid2": d["grid2"].upper(),
            })
        else:
            for k in HEADER_KEYS:
                if u.startswith(k + ":"):
                    header[k] = s.split(":", 1)[1].strip()
    return header, qsos, bad

# ---------------------------------------------------------------- database
SCHEMA = """
CREATE TABLE IF NOT EXISTS qsos (
    id INTEGER PRIMARY KEY,
    src_file TEXT, log_call TEXT,
    band_raw TEXT, band_ghz REAL, mode TEXT, ts TEXT,
    call1 TEXT, grid1 TEXT, lat1 REAL, lon1 REAL,
    call2 TEXT, grid2 TEXT, lat2 REAL, lon2 REAL,
    dist_km REAL, bearing12 REAL
);
CREATE INDEX IF NOT EXISTS idx_q_ts ON qsos (ts);
CREATE INDEX IF NOT EXISTS idx_q_band ON qsos (band_ghz);
CREATE TABLE IF NOT EXISTS logs (
    src_file TEXT PRIMARY KEY, log_call TEXT, category TEXT,
    location TEXT, club TEXT, n_qsos INTEGER, n_bad INTEGER
);
"""

# corridor definition: one end near coastal Southern California,
# other end in the AZ desert. Deliberately generous; refine in queries later.
def in_socal(lat, lon):  return 32.0 <= lat <= 34.5 and -118.5 <= lon <= -115.8
def in_azdes(lat, lon):  return 31.5 <= lat <= 35.5 and -115.8 <= lon <= -110.5

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logdir", help="directory containing Cabrillo files")
    ap.add_argument("--db", default="contest_qsos.db")
    a = ap.parse_args()

    conn = sqlite3.connect(a.db)
    conn.executescript(SCHEMA)
    cur = conn.cursor()

    n_files = n_q = 0
    all_bad = []
    for root, _, files in os.walk(a.logdir):
        for name in files:
            if not name.lower().endswith((".log", ".cbr", ".txt", ".cab")):
                continue
            path = os.path.join(root, name)
            try:                      # skip HTML index pages saved as .log
                head = open(path, "rb").read(512).lstrip().lower()
                if head.startswith((b"<!doctype", b"<html", b"<?xml")) or b"<html" in head:
                    print(f"  skipping HTML file: {name}")
                    continue
            except OSError:
                continue
            header, qsos, bad = parse_file(path)
            log_call = header.get("CALLSIGN", "?")
            cur.execute("INSERT OR REPLACE INTO logs VALUES (?,?,?,?,?,?,?)",
                        (path, log_call, header.get("CATEGORY-OPERATOR"),
                         header.get("LOCATION"), header.get("CLUB"),
                         len(qsos), len(bad)))
            for q in qsos:
                la1, lo1 = grid_to_latlon(q["grid1"])
                la2, lo2 = grid_to_latlon(q["grid2"])
                dist = brg = None
                if la1 is not None and la2 is not None:
                    dist = haversine_km(la1, lo1, la2, lo2)
                    brg = bearing_deg(la1, lo1, la2, lo2)
                cur.execute(
                    "INSERT INTO qsos (src_file, log_call, band_raw, band_ghz,"
                    " mode, ts, call1, grid1, lat1, lon1, call2, grid2, lat2,"
                    " lon2, dist_km, bearing12)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (path, log_call, q["band_raw"], q["band_ghz"], q["mode"],
                     q["ts"], q["call1"], q["grid1"], la1, lo1,
                     q["call2"], q["grid2"], la2, lo2, dist, brg))
            all_bad += [(name, b) for b in bad]
            n_files += 1
            n_q += len(qsos)
    conn.commit()

    if all_bad:
        with open("unparsed_lines.txt", "w") as f:
            for name, b in all_bad:
                f.write(f"{name}: {b}\n")

    print(f"parsed {n_files} logs, {n_q:,} QSO lines "
          f"({len(all_bad)} unparsed -> unparsed_lines.txt)")

    # -------------------------------------------------------------- report
    print("\nQSOs by year (10 GHz band only, i.e. 9.5-11 GHz tokens):")
    for y, n in cur.execute(
        "SELECT substr(ts,1,4), COUNT(*) FROM qsos "
        "WHERE band_ghz BETWEEN 9.5 AND 11 GROUP BY 1 ORDER BY 1"):
        print(f"  {y}: {n:,}")

    print("\n10 GHz distance distribution (unique QSO lines, both logs count):")
    for lo, hi in [(0,50),(50,100),(100,200),(200,300),(300,400),(400,500),(500,2000)]:
        n = cur.execute(
            "SELECT COUNT(*) FROM qsos WHERE band_ghz BETWEEN 9.5 AND 11 "
            "AND dist_km >= ? AND dist_km < ?", (lo, hi)).fetchone()[0]
        print(f"  {lo:>4}-{hi:<4} km: {n:,}")

    print("\nSoCal <-> AZ-desert corridor QSOs at 10 GHz:")
    rows = cur.execute(
        "SELECT ts, call1, grid1, lat1, lon1, call2, grid2, lat2, lon2, dist_km "
        "FROM qsos WHERE band_ghz BETWEEN 9.5 AND 11 AND lat1 IS NOT NULL "
        "ORDER BY ts").fetchall()
    corridor = []
    seen = set()
    for ts, c1, g1, la1, lo1, c2, g2, la2, lo2, d in rows:
        pair = ((in_socal(la1, lo1) and in_azdes(la2, lo2)) or
                (in_socal(la2, lo2) and in_azdes(la1, lo1)))
        if not pair:
            continue
        key = (ts, *sorted([c1, c2]))     # dedupe both-sides-logged QSOs
        if key in seen:
            continue
        seen.add(key)
        corridor.append((ts, c1, g1, c2, g2, d))
    print(f"  {len(corridor):,} unique corridor QSOs")
    for ts, c1, g1, c2, g2, d in corridor[:20]:
        print(f"    {ts}  {c1} ({g1}) <-> {c2} ({g2})  {d:.0f} km")
    if len(corridor) > 20:
        print(f"    ... and {len(corridor)-20:,} more (all in {a.db})")

    print("\nCorridor QSOs by UTC hour (convective clue -- monsoon cells "
          "peak local afternoon/evening = ~20:00-05:00 UTC):")
    hours = {}
    for ts, *_ in corridor:
        h = int(ts[11:13]); hours[h] = hours.get(h, 0) + 1
    for h in sorted(hours):
        print(f"  {h:02d}z: {'#' * hours[h]} {hours[h]}")

    conn.close()

if __name__ == "__main__":
    main()
