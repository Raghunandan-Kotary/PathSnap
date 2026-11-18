# PathSnap: Two-Laptop Setup & Deployment Guide

This guide ensures reliable communication between Laptop 1 (FastAPI Server) and Laptop 2 (Evaluator UI) during live evaluation.

---

## 1. Networking Prerequisites

Both laptops must be on the same local network subnet.

### Recommended Approaches:
- **Approach 1 (Preferred / Most Reliable)**: Turn on a personal mobile Wi-Fi hotspot on a smartphone and connect both laptops to it. This isolates the connection from restrictive campus/university Wi-Fi port-blocking or client isolation.
- **Approach 2**: Connect both laptops to the same campus Wi-Fi network.

---

## 2. Laptop 1 Configuration (FastAPI Server & Model Host)

1. Find Laptop 1's local IP address:
   ```cmd
   # Windows
   ipconfig
   # Note down IPv4 Address (e.g. 192.168.1.105)
   ```
2. Open Firewall Port 8000 (if prompted by Windows Defender):
   - When launching `uvicorn`, click **Allow access** on Private networks.
   - Alternatively run PowerShell as Administrator:
     ```powershell
     New-NetFirewallRule -DisplayName "PathSnap FastAPI" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
     ```
3. Run the API:
   ```cmd
   deploy\run_laptop1_api.bat
   # Or directly:
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
4. Confirm server output in terminal:
   `INFO: Uvicorn running on http://0.0.0.0:8000`

---

## 3. Laptop 2 Configuration (Evaluator UI)

1. Test reachability from Laptop 2 to Laptop 1:
   ```cmd
   # Ping Laptop 1
   ping 192.168.1.105
   ```
2. Start the Evaluator UI:
   ```cmd
   deploy\run_laptop2_ui.bat
   # Or directly:
   streamlit run ui/app.py
   ```
3. Open the browser to `http://localhost:8501`.
4. In the sidebar, set the **API Server URL** to `http://<Laptop-1-IP>:8000` (e.g. `http://192.168.1.105:8000`).
5. Click **Test API Connection** to verify green status.
6. The evaluator selects any trip and clicks **Evaluate Route**.

---

## 4. Evaluator Verification Checkpoints
- Evaluators will inspect the command terminal on Laptop 1.
- Every evaluation request will display timestamped logging:
  `[TIMESTAMP] INFO: Evaluator request for Trip ID 00000040 | Processing Time: ...ms | Top-1 Accuracy: ...%`
