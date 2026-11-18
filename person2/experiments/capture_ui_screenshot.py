"""Start the real API (Laptop 1) and UI (Laptop 2) on this machine, run one trip from the UI, and save screenshots
of the result for report section 7. Both servers are stopped afterwards.

    python person2/experiments/capture_ui_screenshot.py [--trip 00000015]

Uses Playwright with the bundled Chromium or, if that is not installed, the system Edge browser.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def wait_http(url: str, timeout: float = 120) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=3)
            return
        except Exception:
            time.sleep(1)
    raise TimeoutError(url)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trip", default=None, help="trip to select in the UI (default: the first one listed)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "figures")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"], cwd=ROOT)
    ui = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "ui/app.py", "--server.headless", "true",
                           "--server.port", "8501", "--browser.gatherUsageStats", "false"], cwd=ROOT)
    try:
        wait_http("http://127.0.0.1:8000/health")
        wait_http("http://127.0.0.1:8501")
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception:
                browser = p.chromium.launch(channel="msedge")
            page = browser.new_page(viewport={"width": 1400, "height": 1000})
            page.goto("http://127.0.0.1:8501", wait_until="networkidle")
            page.wait_for_selector("text=Evaluate Route", timeout=60000)
            if a.trip:
                page.locator("div[data-baseweb='select']").first.click()
                page.get_by_role("option", name=a.trip).click()
            page.screenshot(path=str(a.out_dir / "ui_trip_list.png"), full_page=True)
            page.get_by_role("button", name="Evaluate Route").click()
            page.wait_for_selector("text=processed successfully", timeout=180000)
            page.wait_for_timeout(4000)  # let the map finish rendering
            page.screenshot(path=str(a.out_dir / "ui_evaluation_result.png"), full_page=True)
            browser.close()
        print("wrote", a.out_dir / "ui_trip_list.png", "and", a.out_dir / "ui_evaluation_result.png")
    finally:
        for proc in (ui, api):
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
