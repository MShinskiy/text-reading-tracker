from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from MonitorTracking import MonitorTracker

app = FastAPI(title="Eye Tracker")

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

tracker: Optional[MonitorTracker] = None


def _get_tracker() -> MonitorTracker:
    global tracker
    if tracker is None:
        tracker = MonitorTracker(workspace_width=1, workspace_height=1)
    return tracker


async def _decode_upload_to_bgr(file: UploadFile) -> np.ndarray:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty frame upload")

    encoded = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image frame")
    return frame


async def _decode_uploads_to_bgr(files: list[UploadFile]) -> list[np.ndarray]:
    if not files:
        raise HTTPException(status_code=400, detail="No frame uploads")
    return [await _decode_upload_to_bgr(file) for file in files]


def _reading_rect(
    left: Optional[float],
    top: Optional[float],
    width: Optional[float],
    height: Optional[float],
) -> Optional[dict[str, float]]:
    if left is None or top is None or width is None or height is None:
        return None
    return {
        "left": float(left),
        "top": float(top),
        "width": float(width),
        "height": float(height),
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/api/frame")
async def process_frame(
    frame: UploadFile = File(...),
):
    frame_bgr = await _decode_upload_to_bgr(frame)
    gaze_tracker = _get_tracker()
    result = gaze_tracker.process_frame(frame_bgr)
    return JSONResponse(result.to_dict())


@app.post("/api/calibrate")
async def calibrate(
    frame: UploadFile = File(...),
    workspace_width: int = Form(...),
    workspace_height: int = Form(...),
):
    # Backward-compatible single-center calibration endpoint.
    frame_bgr = await _decode_upload_to_bgr(frame)
    gaze_tracker = _get_tracker()
    gaze_tracker.configure_workspace(workspace_width, workspace_height)
    result = gaze_tracker.calibrate_eyes_and_monitor(frame_bgr)
    return JSONResponse(result.to_dict())


@app.post("/api/calibrate-point")
async def calibrate_point(
    frame: UploadFile = File(...),
    workspace_width: int = Form(...),
    workspace_height: int = Form(...),
    device_pixel_ratio: float = Form(1.0),
    target_x_norm: Optional[float] = Form(None),
    target_y_norm: Optional[float] = Form(None),
    target_x_px: Optional[float] = Form(None),
    target_y_px: Optional[float] = Form(None),
    reading_rect_left: Optional[float] = Form(None),
    reading_rect_top: Optional[float] = Form(None),
    reading_rect_width: Optional[float] = Form(None),
    reading_rect_height: Optional[float] = Form(None),
    physical_workspace_width_cm: Optional[float] = Form(None),
    physical_workspace_height_cm: Optional[float] = Form(None),
    card_width_px: Optional[float] = Form(None),
):
    frame_bgr = await _decode_upload_to_bgr(frame)
    gaze_tracker = _get_tracker()
    gaze_tracker.configure_workspace(
        workspace_width,
        workspace_height,
        device_pixel_ratio=device_pixel_ratio,
        reading_rect_px=_reading_rect(
            reading_rect_left,
            reading_rect_top,
            reading_rect_width,
            reading_rect_height,
        ),
        physical_width_cm=physical_workspace_width_cm,
        physical_height_cm=physical_workspace_height_cm,
        card_width_px=card_width_px,
    )
    result = gaze_tracker.add_calibration_point(
        frame_bgr,
        target_x_norm=target_x_norm,
        target_y_norm=target_y_norm,
        target_x_px=target_x_px,
        target_y_px=target_y_px,
    )
    return JSONResponse(result.to_dict())


@app.post("/api/calibrate-point-batch")
async def calibrate_point_batch(
    frames: list[UploadFile] = File(...),
    workspace_width: int = Form(...),
    workspace_height: int = Form(...),
    device_pixel_ratio: float = Form(1.0),
    target_x_norm: Optional[float] = Form(None),
    target_y_norm: Optional[float] = Form(None),
    target_x_px: Optional[float] = Form(None),
    target_y_px: Optional[float] = Form(None),
    reading_rect_left: Optional[float] = Form(None),
    reading_rect_top: Optional[float] = Form(None),
    reading_rect_width: Optional[float] = Form(None),
    reading_rect_height: Optional[float] = Form(None),
    physical_workspace_width_cm: Optional[float] = Form(None),
    physical_workspace_height_cm: Optional[float] = Form(None),
    card_width_px: Optional[float] = Form(None),
    min_valid_samples: int = Form(3),
):
    frames_bgr = await _decode_uploads_to_bgr(frames)
    gaze_tracker = _get_tracker()
    gaze_tracker.configure_workspace(
        workspace_width,
        workspace_height,
        device_pixel_ratio=device_pixel_ratio,
        reading_rect_px=_reading_rect(
            reading_rect_left,
            reading_rect_top,
            reading_rect_width,
            reading_rect_height,
        ),
        physical_width_cm=physical_workspace_width_cm,
        physical_height_cm=physical_workspace_height_cm,
        card_width_px=card_width_px,
    )
    result = gaze_tracker.add_calibration_point_batch(
        frames_bgr,
        target_x_norm=target_x_norm,
        target_y_norm=target_y_norm,
        target_x_px=target_x_px,
        target_y_px=target_y_px,
        min_valid_samples=min_valid_samples,
    )
    return JSONResponse(result.to_dict())


@app.post("/api/calibrate-center")
async def calibrate_center(
    frame: UploadFile = File(...),
    workspace_width: int = Form(...),
    workspace_height: int = Form(...),
):
    frame_bgr = await _decode_upload_to_bgr(frame)
    gaze_tracker = _get_tracker()
    gaze_tracker.configure_workspace(workspace_width, workspace_height)
    result = gaze_tracker.calibrate_screen_center(frame_bgr)
    return JSONResponse(result.to_dict())


@app.post("/api/reset")
def reset():
    if tracker is not None:
        tracker.reset_calibration()
    return {"ok": True}


@app.on_event("shutdown")
def shutdown_event() -> None:
    if tracker is not None:
        tracker.close()
