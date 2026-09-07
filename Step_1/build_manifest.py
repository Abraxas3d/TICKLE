#!/usr/bin/env python3
"""
build_manifest.py -- GOES-18 archive manifest builder (Step 1)

Walks a drive/directory, inventories EVERY file, parses GOES ABI L1b
netCDF filenames (and goestools-style decoded product names), stores
everything in a SQLite database, then prints a summary report and a
gap analysis per (sector, channel).

Reads FILENAMES ONLY -- never opens file contents -- so a 226 GB
archive scans in minutes.

Usage:
    python3 build_manifest.py /Volumes/YOUR_DRIVE_NAME
    python3 build_manifest.py /Volumes/YOUR_DRIVE_NAME --db goes18_manifest.db

Requires: Python 3.8+, standard library only (sqlite3 is built in).
"""

import argparse
import csv
import os
import re
import sqlite3
import statistics
import sys
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------------
# Filename parsers
# ---------------------------------------------------------------------------

# Standard GOES-R series ABI L1b/L2 netCDF naming, e.g.:
# OR_ABI-L1b-RadC-M6C13_G18_s20231001800205_e20231001802578_c20231001803022.nc
ABI_NC_RE = re.compile(
    r"^OR_(?P<instrument>[A-Z]+)-"
    r"(?P<level>L1b|L2)-"
    r"(?P<product>[A-Za-z0-9]+?)"          # RadC, RadF, RadM1, RadM2, CMIPC...
    r"-M(?P<mode>\d+)"
    r"(?:C(?P<channel>\d{2}))?"            # channel absent on some L2 products
    r"_G(?P<satellite>\d{2})"
    r"_s(?P<start>\d{13,14})"
    r"_e(?P<end>\d{13,14})"
    r"_c(?P<created>\d{13,14})"
    r"\.nc$"
)

# goestools/goesproc decoded imagery, e.g.:
# GOES18_FD_CH13_20231010T180020Z.png  /  GOES17_M1_CH02_...
GOESTOOLS_RE = re.compile(
    r"^GOES(?P<satellite>\d{2})_"
    r"(?P<product>[A-Za-z0-9]+)_"
    r"CH(?P<channel>\d{2})_"
    r"(?P<start>\d{8}T\d{6}Z?)"
    r"\.(?P<ext>png|jpg|jpeg|gif|tif|tiff)$",
    re.IGNORECASE,
)


def parse_abi_timestamp(ts: str):
    """Parse sYYYYJJJHHMMSSt (year, day-of-year, time, tenths) to UTC datetime."""
    try:
        year = int(ts[0:4])
        doy = int(ts[4:7])
        hh, mm, ss = int(ts[7:9]), int(ts[9:11]), int(ts[11:13])
        base = datetime(year, 1, 1, hh, mm, ss, tzinfo=timezone.utc)
        return base + timedelta(days=doy - 1)
    except (ValueError, IndexError):
        return None


def parse_goestools_timestamp(ts: str):
    try:
        return datetime.strptime(ts.rstrip("Zz"), "%Y%m%dT%H%M%S").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None


def parse_filename(name: str):
    """Return a dict of parsed GOES fields, or None if the name is unrecognized."""
    m = ABI_NC_RE.match(name)
    if m:
        d = m.groupdict()
        return {
            "scheme": "abi_netcdf",
            "instrument": d["instrument"],
            "level": d["level"],
            "product": d["product"],
            "scan_mode": d["mode"],
            "channel": int(d["channel"]) if d["channel"] else None,
            "satellite": "G" + d["satellite"],
            "start_time": parse_abi_timestamp(d["start"]),
            "end_time": parse_abi_timestamp(d["end"]),
            "created_time": parse_abi_timestamp(d["created"]),
        }
    m = GOESTOOLS_RE.match(name)
    if m:
        d = m.groupdict()
        return {
            "scheme": "goestools_image",
            "instrument": "ABI",
            "level": None,
            "product": d["product"],
            "scan_mode": None,
            "channel": int(d["channel"]),
            "satellite": "G" + d["satellite"],
            "start_time": parse_goestools_timestamp(d["start"]),
            "end_time": None,
            "created_time": None,
        }
    return None


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    filename TEXT NOT NULL,
    extension TEXT,
    size_bytes INTEGER,
    mtime_utc TEXT,
    parsed INTEGER NOT NULL DEFAULT 0,   -- 1 if a GOES pattern matched
    scheme TEXT,                          -- abi_netcdf | goestools_image
    instrument TEXT,
    level TEXT,
    product TEXT,                         -- RadC, RadF, RadM1, FD, M1, ...
    scan_mode TEXT,
    channel INTEGER,
    satellite TEXT,
    start_time TEXT,                      -- ISO 8601 UTC scan start
    end_time TEXT,
    created_time TEXT
);
CREATE INDEX IF NOT EXISTS idx_files_prod_chan_start
    ON files (product, channel, start_time);
CREATE INDEX IF NOT EXISTS idx_files_parsed ON files (parsed);
"""


def iso(dt):
    return dt.isoformat() if dt else None


def scan_drive(root: str, conn: sqlite3.Connection, batch_size: int = 2000):
    cur = conn.cursor()
    n_seen = 0
    batch = []
    skipped_dirs = {".Trashes", ".Spotlight-V100", ".fseventsd", "System Volume Information"}

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skipped_dirs]
        for name in filenames:
            if name.startswith("."):
                continue  # .DS_Store, ._AppleDouble files, etc.
            full = os.path.join(dirpath, name)
            try:
                st = os.stat(full)
            except OSError:
                continue
            ext = os.path.splitext(name)[1].lower()
            parsed = parse_filename(name)
            row = (
                full,
                name,
                ext,
                st.st_size,
                datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
                1 if parsed else 0,
                parsed["scheme"] if parsed else None,
                parsed["instrument"] if parsed else None,
                parsed["level"] if parsed else None,
                parsed["product"] if parsed else None,
                parsed["scan_mode"] if parsed else None,
                parsed["channel"] if parsed else None,
                parsed["satellite"] if parsed else None,
                iso(parsed["start_time"]) if parsed else None,
                iso(parsed["end_time"]) if parsed else None,
                iso(parsed["created_time"]) if parsed else None,
            )
            batch.append(row)
            n_seen += 1
            if len(batch) >= batch_size:
                cur.executemany(
                    "INSERT OR REPLACE INTO files "
                    "(path, filename, extension, size_bytes, mtime_utc, parsed, "
                    " scheme, instrument, level, product, scan_mode, channel, "
                    " satellite, start_time, end_time, created_time) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    batch,
                )
                conn.commit()
                batch.clear()
                print(f"  ...{n_seen:,} files scanned", end="\r", flush=True)
    if batch:
        cur.executemany(
            "INSERT OR REPLACE INTO files "
            "(path, filename, extension, size_bytes, mtime_utc, parsed, "
            " scheme, instrument, level, product, scan_mode, channel, "
            " satellite, start_time, end_time, created_time) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            batch,
        )
        conn.commit()
    print(f"  ...{n_seen:,} files scanned")
    return n_seen


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def human_bytes(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def summary_report(conn: sqlite3.Connection):
    cur = conn.cursor()
    print("\n" + "=" * 70)
    print("ARCHIVE SUMMARY")
    print("=" * 70)

    total, size = cur.execute(
        "SELECT COUNT(*), COALESCE(SUM(size_bytes),0) FROM files"
    ).fetchone()
    parsed, psize = cur.execute(
        "SELECT COUNT(*), COALESCE(SUM(size_bytes),0) FROM files WHERE parsed=1"
    ).fetchone()
    print(f"Total files:        {total:,}  ({human_bytes(size)})")
    print(f"Parsed as GOES:     {parsed:,}  ({human_bytes(psize)})")
    print(f"Unrecognized:       {total - parsed:,}")

    print("\nFiles by extension:")
    for ext, n, s in cur.execute(
        "SELECT COALESCE(extension,'(none)'), COUNT(*), SUM(size_bytes) "
        "FROM files GROUP BY extension ORDER BY SUM(size_bytes) DESC LIMIT 15"
    ):
        print(f"  {ext:<10} {n:>10,}   {human_bytes(s or 0)}")

    if parsed:
        lo, hi = cur.execute(
            "SELECT MIN(start_time), MAX(start_time) FROM files "
            "WHERE parsed=1 AND start_time IS NOT NULL"
        ).fetchone()
        print(f"\nScan time range:    {lo}  ->  {hi}")

        print("\nParsed files by satellite / product / channel:")
        for sat, prod, ch, n in cur.execute(
            "SELECT satellite, product, channel, COUNT(*) FROM files WHERE parsed=1 "
            "GROUP BY satellite, product, channel ORDER BY satellite, product, channel"
        ):
            chs = f"C{ch:02d}" if ch is not None else "--"
            print(f"  {sat}  {prod:<8} {chs}   {n:>8,} files")

        # Suspiciously small files (possible truncated/corrupt captures):
        # flag files < 25% of the median size for their (product, channel) group.
        print("\nSuspiciously small files (<25% of group median size):")
        n_flagged = 0
        for prod, ch in cur.execute(
            "SELECT DISTINCT product, channel FROM files WHERE parsed=1"
        ).fetchall():
            sizes = [
                r[0]
                for r in cur.execute(
                    "SELECT size_bytes FROM files WHERE parsed=1 "
                    "AND product IS ? AND channel IS ?",
                    (prod, ch),
                )
            ]
            if len(sizes) < 5:
                continue
            med = statistics.median(sizes)
            small = cur.execute(
                "SELECT COUNT(*) FROM files WHERE parsed=1 "
                "AND product IS ? AND channel IS ? AND size_bytes < ?",
                (prod, ch, med * 0.25),
            ).fetchone()[0]
            if small:
                n_flagged += small
                chs = f"C{ch:02d}" if ch is not None else "--"
                print(f"  {prod} {chs}: {small} files well below median ({human_bytes(med)})")
        if not n_flagged:
            print("  none found")

    unrec = cur.execute(
        "SELECT filename FROM files WHERE parsed=0 ORDER BY size_bytes DESC LIMIT 10"
    ).fetchall()
    if unrec:
        print("\nSample unrecognized filenames (largest first) -- if these look")
        print("like GOES data, send these names back so a parser can be added:")
        for (name,) in unrec:
            print(f"  {name}")


def gap_report(conn: sqlite3.Connection, gap_csv_path: str, factor: float = 1.5):
    """Per (product, channel): empirical cadence and gaps > factor * median."""
    cur = conn.cursor()
    print("\n" + "=" * 70)
    print(f"GAP ANALYSIS  (gap = interval > {factor}x median cadence)")
    print("=" * 70)

    groups = cur.execute(
        "SELECT product, channel FROM files "
        "WHERE parsed=1 AND start_time IS NOT NULL "
        "GROUP BY product, channel HAVING COUNT(*) >= 10 "
        "ORDER BY product, channel"
    ).fetchall()

    all_gaps = []
    for prod, ch in groups:
        times = [
            datetime.fromisoformat(r[0])
            for r in cur.execute(
                "SELECT DISTINCT start_time FROM files WHERE parsed=1 "
                "AND product IS ? AND channel IS ? ORDER BY start_time",
                (prod, ch),
            )
        ]
        deltas = [
            (b - a).total_seconds() for a, b in zip(times, times[1:])
        ]
        if not deltas:
            continue
        med = statistics.median(deltas)
        threshold = med * factor
        gaps = [
            (a, b, (b - a).total_seconds())
            for a, b in zip(times, times[1:])
            if (b - a).total_seconds() > threshold
        ]
        missing_time = sum(g[2] - med for g in gaps)
        span = (times[-1] - times[0]).total_seconds()
        coverage = 100.0 * (1 - missing_time / span) if span > 0 else 0.0
        chs = f"C{ch:02d}" if ch is not None else "--"
        print(
            f"\n{prod} {chs}: {len(times):,} scans, "
            f"median cadence {med/60:.1f} min, "
            f"{len(gaps)} gaps, coverage ~{coverage:.1f}%"
        )
        for a, b, secs in sorted(gaps, key=lambda g: -g[2])[:5]:
            print(f"    gap {timedelta(seconds=int(secs))}   {a.isoformat()} -> {b.isoformat()}")
        if len(gaps) > 5:
            print(f"    ...and {len(gaps)-5} more (full list in CSV)")
        for a, b, secs in gaps:
            all_gaps.append([prod, chs, a.isoformat(), b.isoformat(), int(secs)])

    if all_gaps:
        with open(gap_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["product", "channel", "gap_start_utc", "gap_end_utc", "gap_seconds"])
            w.writerows(all_gaps)
        print(f"\nFull gap list written to: {gap_csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Build a SQLite manifest of a GOES-18 archive.")
    ap.add_argument("root", help="Path to the archive, e.g. /Volumes/GOES_DRIVE")
    ap.add_argument("--db", default="goes18_manifest.db", help="Output SQLite file")
    ap.add_argument("--gap-csv", default="goes18_gaps.csv", help="Output gap-list CSV")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"Not a directory: {args.root}")

    conn = sqlite3.connect(args.db)
    conn.executescript(SCHEMA)

    print(f"Scanning {args.root} (filenames only, no file contents)...")
    t0 = datetime.now()
    scan_drive(args.root, conn)
    print(f"Scan finished in {datetime.now() - t0}")

    summary_report(conn)
    gap_report(conn, args.gap_csv)

    conn.close()
    print(f"\nManifest database: {args.db}")
    print("Query it any time, e.g.:")
    print(f"  sqlite3 {args.db} \"SELECT COUNT(*) FROM files WHERE channel=13;\"")


if __name__ == "__main__":
    main()
