import os
import re
import time
from urllib.parse import urljoin, parse_qs, urlparse
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://contests.arrl.org/"
ENTRY_URL = "https://contests.arrl.org/publiclogs.php?cn=10g"
OUTPUT_DIR = "arrl_10g_logs"

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0"
})

def download_all_logs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Fetching contest year list from {ENTRY_URL}...")
    resp = session.get(ENTRY_URL)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    
    # 1. Discover all year links for the 10 GHz contest
    year_links = {}
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True)
        if "eid=15" in href and "iid=" in href:
            if re.match(r"^\d{4}$", text):
                year_links[text] = urljoin(BASE_URL, href)

    if not year_links:
        # Fallback: Scrape the year page directly if redirected
        print("Using current page as the active year view...")
        year_links["current"] = ENTRY_URL

    print(f"Found {len(year_links)} years: {list(year_links.keys())}")

    # 2. Iterate through each year
    for year, y_url in year_links.items():
        year_dir = os.path.join(OUTPUT_DIR, year)
        os.makedirs(year_dir, exist_ok=True)
        print(f"\nProcessing Year {year} ({y_url})...")
        
        y_resp = session.get(y_url)
        y_soup = BeautifulSoup(y_resp.text, "html.parser")
        
        # 3. Find all callsign log links
        log_links = []
        for a in y_soup.find_all("a", href=True):
            href = a["href"]
            call = a.get_text(strip=True)
            # Match links pointing to individual call log entries
            if ("call=" in href or "eid=15" in href) and re.match(r"^[A-Z0-9/]+$", call):
                log_links.append((call, urljoin(BASE_URL, href)))
        
        print(f"Discovered {len(log_links)} logs for {year}.")
        
        # 4. Download each log file
        for call, log_url in log_links:
            safe_call = re.sub(r"[^A-Za-z0-9_-]", "_", call)
            target_path = os.path.join(year_dir, f"{safe_call}.log")
            
            if os.path.exists(target_path):
                continue
                
            try:
                log_resp = session.get(log_url)
                if log_resp.status_code == 200:
                    # Logs displayed inside <pre> tags or raw text
                    log_soup = BeautifulSoup(log_resp.text, "html.parser")
                    pre_tag = log_soup.find("pre")
                    content = pre_tag.get_text() if pre_tag else log_resp.text
                    
                    with open(target_path, "w", encoding="utf-8", errors="replace") as f:
                        f.write(content)
                    print(f"  Downloaded: {year}/{safe_call}.log")
                else:
                    print(f"  Failed ({log_resp.status_code}): {call}")
            except Exception as e:
                print(f"  Error fetching {call}: {e}")
                
            # Respectful rate limiting
            time.sleep(0.5)

if __name__ == "__main__":
    download_all_logs()

