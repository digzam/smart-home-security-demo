# Smart Home Security Demo 🏠🔒

> **Software-Only Prototype** — University Interim Demonstration  
> The laptop webcam temporarily replaces the ESP32-CAM + PIR hardware while the physical integration is still under development.

---

## What This Is

A working desktop prototype of a Smart IoT Home Security System.  
It demonstrates that the **core software pipeline** functions correctly before hardware integration.

```
Laptop Webcam
      ↓
YOLOv8n Human Detection
      ↓
Visitor Event Logic (cooldown — no duplicate records)
      ↓
FastAPI Backend  +  SQLite Database
      ↓
Local Web Dashboard  →  http://127.0.0.1:8000
```

Future integration simply swaps the webcam for an ESP32-CAM posting images to `POST /api/visits` — the backend stays the same.

---

## Files

```
smart-home-security-demo/
├── app.py           ← FastAPI server + webcam loop + detection + all API routes
├── database.py      ← SQLite helpers (init, create, read, delete visits)
├── index.html       ← Live dashboard (MJPEG feed + visit table + image viewer)
├── requirements.txt ← 4 Python packages
└── README.md        ← This file
```

Images are saved to:
```
images/
└── visit_0001.jpg
    visit_0042.jpg
    ...
```

Database file: `security.db` (created automatically on first run).

---

## Installation

### 1. Python environment (Python 3.10+ recommended)

```bash
# Create a virtual environment (recommended)
python -m venv venv

# Activate it  (Windows)
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **YOLOv8 model download** — On the first run, `ultralytics` automatically downloads `yolov8n.pt` (~6 MB). This requires a one-time internet connection. After that, the app runs fully offline.

---

## Running the App

```bash
python app.py
```

Then open your browser at:

```
http://127.0.0.1:8000
```

You will see:
- **Live webcam feed** with bounding boxes drawn around any detected person
- **System status panel** (camera state, humans in frame, current visit ID)
- **Visit history table** (auto-refreshes every 2 seconds)

Press **Ctrl+C** in the terminal to stop the server.

---

## Demonstration Workflow

1. Run `python app.py` and open the dashboard.
2. The system displays **"Waiting for visitor…"**
3. Walk in front of the webcam.
4. YOLOv8n detects you → a green bounding box appears.
5. The system creates **Visit #1**:
   - Saves a JPEG to `images/visit_0001.jpg`
   - Inserts a record in `security.db`
   - Dashboard table updates within 2 seconds
6. Stay in view → **no duplicate visits** are created (cooldown active).
7. Step away → cooldown timer counts down → system returns to **"Waiting for visitor…"**
8. Click **View** next to any visit to see the captured image.

---

## Configuration

All settings are at the top of `app.py` — no config files needed:

| Setting | Default | Description |
|---|---|---|
| `CAMERA_INDEX` | `0` | Webcam index (try `1` if default doesn't work) |
| `DETECTION_CONFIDENCE` | `0.50` | Minimum YOLO confidence to count as a person |
| `EVENT_COOLDOWN_SECS` | `30` | Seconds before a new visit can start |
| `IMAGE_SAVE_DIR` | `images/` | Folder for captured JPEGs |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Dashboard HTML |
| `GET` | `/video_feed` | Live MJPEG webcam stream |
| `GET` | `/api/status` | Camera + detection status |
| `GET` | `/api/visits` | All visits (newest first) |
| `GET` | `/api/visits/{id}` | Single visit record |
| `GET` | `/api/visits/{id}/image` | Captured visitor JPEG |
| `POST` | `/api/visits` | Create visit from uploaded image _(future ESP32-CAM)_ |
| `PATCH` | `/api/visits/{id}` | Update visit status |
| `DELETE` | `/api/visits` | Clear all records + images |
| `POST` | `/api/camera/start` | Start webcam |
| `POST` | `/api/camera/stop` | Stop webcam |

Full API documentation: see `API.md`.

---

## Database

**File:** `security.db` (SQLite, created automatically)

**Table:** `visits`

| Column | Type | Description |
|---|---|---|
| `visit_id` | INTEGER PK | Auto-incremented unique ID |
| `timestamp` | DATETIME | When the visit was detected |
| `image_path` | TEXT | Path to the saved JPEG |
| `human_detected` | INTEGER | Always 1 (reserved for future use) |
| `human_count` | INTEGER | Number of people in the frame |
| `processing_status` | TEXT | `pending` / `reviewed` |
| `face_detected` | INTEGER | Reserved for future face recognition |
| `recognized_person` | TEXT | Reserved for future face recognition |
| `face_confidence` | REAL | Reserved for future face recognition |

---

## Future ESP32-CAM Integration

The backend is already compatible with hardware integration.

**Current (this prototype):**
```
Laptop Webcam  →  internal detection loop  →  database
```

**Future (hardware):**
```
ESP32-CAM  →  POST /api/visits (JPEG body)  →  same backend  →  same database
```

The `POST /api/visits` endpoint is implemented and ready.  
No backend changes are needed to support the ESP32-CAM.

See `INTEGRATION.md` for a step-by-step migration guide.

---

## Troubleshooting

**Webcam not opening:**
- Make sure no other app (Teams, Zoom) is using the camera.
- Try changing `CAMERA_INDEX = 1` in `app.py`.

**YOLOv8 download stuck:**
- The model downloads from the internet on first run. Ensure you have connectivity.
- After download, the app runs fully offline.

**Port 8000 in use:**
- Change `PORT = 8001` (or any free port) at the top of `app.py`.

**Low detection accuracy:**
- Lower `DETECTION_CONFIDENCE = 0.35` for better sensitivity in dim lighting.
- Ensure the person is fully visible in the frame.
