"""
app.py — Smart Home Security Demo
===================================
All-in-one:
  • FastAPI server + REST endpoints
  • Laptop webcam capture (background thread)
  • YOLOv8n human detection
  • Visitor event logic with cooldown (no duplicate records)
  • MJPEG live stream for the browser dashboard
  • Serves index.html dashboard

Run:
    python app.py
    Open: http://127.0.0.1:8000
"""

import cv2
import threading
import time
import logging
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
import uvicorn
from ultralytics import YOLO

import database

# ─── Configuration ────────────────────────────────────────────────────────────
# Edit these values to change behaviour — no hardcoded paths or secrets.

CAMERA_INDEX         = 0        # 0 = default laptop webcam; try 1 if nothing shows
DETECTION_CONFIDENCE = 0.50     # YOLO confidence threshold (0.0–1.0)
EVENT_COOLDOWN_SECS  = 30       # seconds with no detection before a new visit can start
IMAGE_SAVE_DIR       = Path("images")   # folder for captured visitor JPEGs
YOLO_MODEL           = "yolov8n.pt"     # downloaded automatically on first run (~6 MB)
PERSON_CLASS_ID      = 0                # COCO class 0 = person

HOST = "127.0.0.1"
PORT = 8000

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="Smart Home Security Demo")

# ─── Shared state ─────────────────────────────────────────────────────────────
# All mutable globals below are only ever written inside state_lock,
# so FastAPI threads always read a consistent snapshot.

state_lock       = threading.Lock()
current_humans   = 0             # people visible in the latest frame
current_visit_id = None          # visit_id of the active/most-recent visit
system_status    = "Waiting for visitor…"
last_visit_time  = 0.0           # epoch seconds when the last visit was created

# Latest JPEG bytes for the MJPEG stream (written by camera thread)
frame_lock   = threading.Lock()
latest_frame = None              # bytes | None

# Camera thread lifecycle
camera_running = False
camera_thread  = None

# YOLO model (loaded once on startup)
model = None


# ─── Startup / shutdown ───────────────────────────────────────────────────────

@app.on_event("startup")
def on_startup():
    global model
    IMAGE_SAVE_DIR.mkdir(exist_ok=True)
    database.init_db()
    log.info("Loading YOLOv8n — first run downloads ~6 MB, please wait…")
    model = YOLO(YOLO_MODEL)
    log.info("YOLOv8n ready.")
    _start_camera()


@app.on_event("shutdown")
def on_shutdown():
    _stop_camera()


# ─── Camera / detection loop (background thread) ──────────────────────────────

def _camera_loop():
    """
    Runs forever in a daemon thread:
      1. Read frames from the webcam.
      2. Run YOLOv8n (persons only).
      3. Apply visit-event logic with cooldown.
      4. Save the annotated frame for the MJPEG stream.
    """
    global current_humans, current_visit_id, system_status
    global last_visit_time, latest_frame

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        log.error(
            f"❌  Cannot open camera (index={CAMERA_INDEX}). "
            "Check that your webcam is connected and not used by another app."
        )
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    log.info(f"✅  Webcam opened (index={CAMERA_INDEX}). Dashboard → http://{HOST}:{PORT}")

    while camera_running:
        ok, frame = cap.read()
        if not ok:
            log.warning("Dropped frame — retrying…")
            time.sleep(0.05)
            continue

        # ── YOLOv8 inference (persons only, no verbose output) ────────────
        results     = model(frame, conf=DETECTION_CONFIDENCE,
                            classes=[PERSON_CLASS_ID], verbose=False)
        boxes       = results[0].boxes
        human_count = len(boxes)

        # ── Annotate frame with bounding boxes ────────────────────────────
        annotated = frame.copy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 230, 0), 2)
            label_y = max(y1 - 8, 16)
            cv2.putText(annotated, f"PERSON  {conf:.2f}",
                        (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 0), 2)

        # ── HUD overlay ───────────────────────────────────────────────────
        cv2.putText(annotated, f"Humans detected: {human_count}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)

        # ── Visit-event logic ─────────────────────────────────────────────
        # Rule: create ONE visit when a person appears.
        #       Ignore subsequent frames until cooldown expires.
        #
        # Pattern: check state under lock → do DB/disk I/O outside lock
        #          → update state under lock again.
        # This prevents the lock from blocking FastAPI status reads during
        # slow I/O operations.
        now = time.time()
        create_new_visit = False

        with state_lock:
            current_humans  = human_count
            time_since_last = now - last_visit_time

            if human_count > 0 and time_since_last >= EVENT_COOLDOWN_SECS:
                # Guard the timestamp immediately so a second frame cannot
                # race past this check before the DB write finishes.
                last_visit_time = now
                create_new_visit = True
            elif human_count > 0:
                system_status = f"Active visit  |  Visit #{current_visit_id}  (cooldown)"
            else:
                if time_since_last >= EVENT_COOLDOWN_SECS:
                    system_status = "Waiting for visitor…"
                else:
                    remaining = int(EVENT_COOLDOWN_SECS - time_since_last)
                    system_status = (
                        f"Visit #{current_visit_id} ended  |"
                        f"  Next visit ready in {remaining}s"
                    )

        if create_new_visit:
            # ── Heavy I/O happens OUTSIDE the lock ────────────────────────
            ts       = datetime.now()
            visit_id = database.create_visit(ts, "", human_count)

            img_path = IMAGE_SAVE_DIR / f"visit_{visit_id:04d}.jpg"
            cv2.imwrite(str(img_path), frame)               # raw frame (no HUD)
            database.update_image_path(visit_id, str(img_path))

            # Update shared state under lock once work is done
            with state_lock:
                current_visit_id = visit_id
                system_status    = f"🚨 VISITOR DETECTED  |  Visit #{visit_id}"

            log.info(
                f"\n{'─'*50}\n"
                f"  VISITOR DETECTED\n"
                f"  Visit ID  : {visit_id}\n"
                f"  Humans    : {human_count}\n"
                f"  Image     : {img_path}\n"
                f"{'─'*50}"
            )

        # ── Encode frame as JPEG for MJPEG stream ─────────────────────────
        _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with frame_lock:
            latest_frame = jpeg.tobytes()

        time.sleep(0.033)   # ~30 fps cap

    cap.release()
    log.info("Camera released.")


def _start_camera():
    global camera_running, camera_thread
    if not camera_running:
        camera_running = True
        camera_thread  = threading.Thread(
            target=_camera_loop, daemon=True, name="camera-loop"
        )
        camera_thread.start()


def _stop_camera():
    global camera_running
    camera_running = False
    log.info("Camera stop requested.")


# ─── MJPEG stream generator ───────────────────────────────────────────────────

def _frame_generator():
    """Yield MJPEG boundary frames from the latest annotated webcam frame."""
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    while True:
        with frame_lock:
            frame = latest_frame
        if frame:
            yield boundary + frame + b"\r\n"
        time.sleep(0.033)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/video_feed")
def video_feed():
    """Live MJPEG stream embedded in the dashboard."""
    return StreamingResponse(
        _frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serve the main dashboard HTML page."""
    return Path("index.html").read_text(encoding="utf-8")


# ── Status ────────────────────────────────────────────────────────────────────

@app.get("/api/status")
def api_status():
    """Return current system state (used by dashboard auto-refresh)."""
    with state_lock:
        return {
            "camera":           "running" if camera_running else "stopped",
            "humans_detected":  current_humans,
            "current_visit_id": current_visit_id,
            "status":           system_status,
        }


# ── Visits ────────────────────────────────────────────────────────────────────

@app.get("/api/visits")
def api_get_visits():
    """Return all visits, newest first."""
    return database.get_visits()


@app.get("/api/visits/{visit_id}")
def api_get_visit(visit_id: int):
    """Return a single visit record."""
    visit = database.get_visit(visit_id)
    if not visit:
        return Response(status_code=404, content="Visit not found")
    return visit


@app.get("/api/visits/{visit_id}/image")
def api_visit_image(visit_id: int):
    """Serve the captured JPEG for a visit."""
    visit = database.get_visit(visit_id)
    if not visit:
        return Response(status_code=404, content="Visit not found")
    img = Path(visit["image_path"])
    if not img.exists():
        return Response(status_code=404, content="Image file not found on disk")
    return FileResponse(str(img), media_type="image/jpeg")


@app.post("/api/visits")
async def api_create_visit(request: Request):
    """
    Accept a raw JPEG body and create a visit record.

    This endpoint is reserved for FUTURE ESP32-CAM integration:
        ESP32-CAM  →  POST /api/visits (body = JPEG bytes)  →  same backend

    The webcam prototype creates visits internally (via _camera_loop),
    but having this endpoint keeps the backend contract stable.
    """
    body = await request.body()
    if not body:
        return Response(status_code=400, content="Empty image body")

    ts       = datetime.now()
    visit_id = database.create_visit(ts, "", 1)
    img_path = IMAGE_SAVE_DIR / f"visit_{visit_id:04d}.jpg"
    img_path.write_bytes(body)
    database.update_image_path(visit_id, str(img_path))

    log.info(f"[POST /api/visits] Created visit #{visit_id} from uploaded image.")
    return {"visit_id": visit_id, "image_path": str(img_path)}


@app.patch("/api/visits/{visit_id}")
async def api_update_visit(visit_id: int, request: Request):
    """Update the processing_status field of a visit (e.g. pending → reviewed)."""
    body  = await request.json()
    visit = database.get_visit(visit_id)
    if not visit:
        return Response(status_code=404, content="Visit not found")
    # Only status updates for now; extend as needed
    new_status = body.get("processing_status")
    if new_status:
        import sqlite3
        with sqlite3.connect(database.DB_PATH) as conn:
            conn.execute(
                "UPDATE visits SET processing_status = ? WHERE visit_id = ?",
                (new_status, visit_id),
            )
            conn.commit()
    return database.get_visit(visit_id)


@app.delete("/api/visits")
def api_clear_history():
    """
    Delete ALL visit records from the database AND their image files from disk.
    Requires explicit confirmation from the dashboard UI before this is called.
    """
    global current_visit_id
    deleted = database.delete_all_visits()
    with state_lock:
        current_visit_id = None
    return {"message": f"Cleared {deleted} visit(s) and their images."}


# ── Camera controls ───────────────────────────────────────────────────────────

@app.post("/api/camera/start")
def api_camera_start():
    _start_camera()
    return {"message": "Camera started"}


@app.post("/api/camera/stop")
def api_camera_stop():
    _stop_camera()
    return {"message": "Camera stopped"}


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Smart Home Security Demo")
    print("  Software Prototype — Laptop Webcam Mode")
    print("=" * 55)
    print(f"  Dashboard  →  http://{HOST}:{PORT}")
    print(f"  Camera     →  index {CAMERA_INDEX}")
    print(f"  Cooldown   →  {EVENT_COOLDOWN_SECS}s between visits")
    print(f"  Confidence →  {DETECTION_CONFIDENCE}")
    print("=" * 55 + "\n")
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
