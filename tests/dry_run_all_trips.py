"""
Dry-run script for Person 4: Evaluates all 10 student testing trips end-to-end.
Prints formatted CMD logs for evaluators and generates a consolidated summary.
"""
import time
import json
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi.testclient import TestClient
import pandas as pd

from api.main import app
from api.config import settings

def dry_run():
    client = TestClient(app)
    trips_resp = client.get("/trips")
    trip_ids = trips_resp.json()["trips"]
    
    print("\n" + "="*80)
    print("  PATHSNAP EVALUATION BENCHMARK: DRY RUN OF ALL 10 RESERVED TRIPS")
    print("="*80)
    print(f"Server Target: Laptop 1 (Local TestClient)")
    print(f"Total Trips to Evaluate: {len(trip_ids)}\n")
    
    results = []
    
    for i, trip_id in enumerate(trip_ids, start=1):
        t0 = time.perf_counter()
        res = client.post(f"/predict/{trip_id}")
        rtt = (time.perf_counter() - t0) * 1000
        
        if res.status_code == 200:
            data = res.json()
            metrics = data["metrics"]
            timing = data["timing"]
            exec_mode = data.get("execution_mode", "unknown")
            
            row = {
                "trip_id": trip_id,
                "status": "PASS",
                "execution_mode": exec_mode,
                "observations": metrics["total_observations"],
                "top1_acc_pct": round(metrics["pointwise_top1_accuracy"] * 100, 2),
                "continuity_pct": round(metrics["route_continuity"] * 100, 2),
                "server_time_ms": timing["total_latency_ms"],
                "client_rtt_ms": round(rtt, 2)
            }
            results.append(row)
            print(f"[{i:02d}/10] Trip {trip_id} | SUCCESS ({exec_mode}) | Obs: {row['observations']:4d} | "
                  f"Top-1: {row['top1_acc_pct']:5.2f}% | Continuity: {row['continuity_pct']:5.2f}% | "
                  f"Server Latency: {row['server_time_ms']:6.2f} ms")
        else:
            print(f"[{i:02d}/10] Trip {trip_id} | FAILED with HTTP {res.status_code}")
            results.append({"trip_id": trip_id, "status": "FAIL"})

    print("\n" + "="*80)
    print("  DRY RUN SUMMARY REPORT")
    print("="*80)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    avg_latency = df["server_time_ms"].mean() if "server_time_ms" in df else 0
    avg_acc = df["top1_acc_pct"].mean() if "top1_acc_pct" in df else 0
    avg_cont = df["continuity_pct"].mean() if "continuity_pct" in df else 0
    
    print("-" * 80)
    print(f"Trips Evaluated: {len(results)}/10 Passed")
    print(f"Mean Server Latency: {avg_latency:.2f} ms")
    print(f"Mean Observation Accuracy: {avg_acc:.2f}%")
    print(f"Mean Route Continuity: {avg_cont:.2f}%")
    print("=" * 80 + "\n")
    
    # Save JSON summary
    summary_path = settings.LOG_DIR / "dry_run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to {summary_path}")

if __name__ == "__main__":
    dry_run()
