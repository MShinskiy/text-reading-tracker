"""
Server-side gaze tracking library extracted from the MonitorTracking script.

The caller, for example a local web server, owns frame acquisition and workspace size.
The original owns only MediaPipe landmark detection, calibration state, gaze estimation,
and projection to workspace coordinates.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.transform import Rotation as Rscipy


# === Landmark indices ===
LEFT_IRIS_IDX = 468
RIGHT_IRIS_IDX = 473
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

# === Nose-only landmark indices (for stable up/down eye sphere tracking) ===
# These landmarks are near the nose and are less affected by lateral head movement
NOSE_INDICES = [
    4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241,
    461, 125, 354, 218, 438, 195, 167, 393, 165, 391,
    3, 248,
]


@dataclass
class GazeTrackerConfig:
    """
    Configuration
    """

    # Physical/virtual workspace assumptions
    face_to_monitor_cm: float = 60.0
    # for 27' monitor with 16:9 aspect ratio
    monitor_width_cm: float = 60.0
    monitor_height_cm: float = 33.5

    # Smoothing/default ray settings
    filter_length: int = 15         # default 10
    gaze_length: float = 350.0

    # Angle-to-screen fallback mapping
    yaw_degrees: float = 15.0       # 5 * 3
    pitch_degrees: float = 5.0      # 2.0 * 2.5

    # Eye-sphere assumption
    base_eye_radius: float = 20.0

    # MediaPipe settings
    max_num_faces: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # Face-distance helper defaults
    real_eye_width_cm: float = 9.5      # assume between eye dist
    horizontal_fov_deg: float = 78.0    # from webcam specs


@dataclass
class WorkspaceCalibration:
    """Browser/session geometry captured once during calibration."""

    width_px: int
    height_px: int
    device_pixel_ratio: float = 1.0
    reading_rect_px: Optional[Dict[str, float]] = None
    physical_width_cm: Optional[float] = None
    physical_height_cm: Optional[float] = None
    card_width_px: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width_px": self.width_px,
            "height_px": self.height_px,
            "device_pixel_ratio": self.device_pixel_ratio,
            "reading_rect_px": self.reading_rect_px,
            "physical_width_cm": self.physical_width_cm,
            "physical_height_cm": self.physical_height_cm,
            "card_width_px": self.card_width_px,
        }


@dataclass
class GazeResult:
    """
    Result returned by process_frame()
    """

    face_detected: bool
    calibrated: bool
    valid: bool
    x: Optional[int] = None
    y: Optional[int] = None
    x_norm: Optional[float] = None
    y_norm: Optional[float] = None
    raw_yaw: Optional[float] = None
    raw_pitch: Optional[float] = None
    gaze_origin: Optional[List[float]] = None
    gaze_direction: Optional[List[float]] = None
    monitor_intersection: Optional[List[float]] = None
    monitor_coordinates: Optional[Tuple[float, float]] = None
    reason: Optional[str] = None
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "face_detected": self.face_detected,
            "calibrated": self.calibrated,
            "valid": self.valid,
            "x": self.x,
            "y": self.y,
            "x_norm": self.x_norm,
            "y_norm": self.y_norm,
            "raw_yaw": self.raw_yaw,
            "raw_pitch": self.raw_pitch,
            "gaze_origin": self.gaze_origin,
            "gaze_direction": self.gaze_direction,
            "monitor_intersection": self.monitor_intersection,
            "monitor_coordinates": self.monitor_coordinates,
            "reason": self.reason,
            "debug": self.debug,
        }


@dataclass
class _FrameState:
    """
    Internal per-frame values used by calibration and prediction
    """

    frame_shape: Tuple[int, int, int]
    face_landmarks: Sequence[Any]
    head_center: np.ndarray
    R_final: np.ndarray
    nose_points_3d: np.ndarray
    iris_3d_left: np.ndarray
    iris_3d_right: np.ndarray
    sphere_world_l: Optional[np.ndarray] = None
    sphere_world_r: Optional[np.ndarray] = None
    scaled_radius_l: Optional[float] = None
    scaled_radius_r: Optional[float] = None
    raw_combined_direction: Optional[np.ndarray] = None
    avg_combined_direction: Optional[np.ndarray] = None


class MonitorTracker:
    """
    Stateful gaze original

    A local web server should create one instance, send decoded frames to process_frame(),
    and call calibration methods when the browser UI asks the user to look at a target
    """

    def __init__(self, workspace_width: int, workspace_height: int, config: Optional[GazeTrackerConfig] = None) -> None:
        self.config = config or GazeTrackerConfig()
        self.workspace_width = int(workspace_width)
        self.workspace_height = int(workspace_height)
        self.workspace_calibration: Optional[WorkspaceCalibration] = None

        # Calibration offsets for yaw/pitch fallback mapping
        self.calibration_offset_yaw = 0.0
        self.calibration_offset_pitch = 0.0

        # Eye-sphere tracking state
        self.left_sphere_locked = False
        self.left_sphere_local_offset: Optional[np.ndarray] = None
        self.left_calibration_nose_scale: Optional[float] = None

        self.right_sphere_locked = False
        self.right_sphere_local_offset: Optional[np.ndarray] = None
        self.right_calibration_nose_scale: Optional[float] = None

        # 3D monitor plane state
        self.monitor_corners: Optional[List[np.ndarray]] = None
        self.monitor_center_w: Optional[np.ndarray] = None
        self.monitor_normal_w: Optional[np.ndarray] = None
        self.units_per_cm: Optional[float] = None

        # Stored gaze markers on the monitor plane as (a,b) normalized coordinates.
        self.gaze_markers: List[Tuple[float, float]] = []

        # Point calibration correction.
        # Each sample is ((raw_x_norm, raw_y_norm), (target_x_norm, target_y_norm)).
        # The transform maps raw normalized coordinates to calibrated normalized
        # coordinates. With 9 samples the frontend uses an affine baseline plus local
        # residual correction so edge fixes do not bend the whole screen diagonally.
        self.calibration_samples: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        self.calibration_transform: Optional[np.ndarray] = None
        self.calibration_model: Optional[str] = None
        self.calibration_residuals: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

        # Smoothing buffer
        self.combined_gaze_directions: deque[np.ndarray] = deque(maxlen=self.config.filter_length)

        # Reference matrices to stabilize PCA eigenvector signs
        self.R_ref_nose: List[Optional[np.ndarray]] = [None]

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=self.config.max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )

    def close(self) -> None:
        if self._face_mesh is not None:
            self._face_mesh.close()

    def set_workspace(self, width: int, height: int) -> None:
        """Update the virtual workspace dimensions"""
        self.workspace_width = int(width)
        self.workspace_height = int(height)

    def configure_workspace(
        self,
        width: int,
        height: int,
        device_pixel_ratio: float = 1.0,
        reading_rect_px: Optional[Dict[str, float]] = None,
        physical_width_cm: Optional[float] = None,
        physical_height_cm: Optional[float] = None,
        card_width_px: Optional[float] = None,
    ) -> None:
        """
        Store the browser/session coordinate space once during calibration.

        Frame processing can then return coordinates in this stored workspace without
        the frontend sending viewport dimensions on every frame.
        """
        width = max(1, int(round(width)))
        height = max(1, int(round(height)))

        workspace_changed = (
            width != self.workspace_width
            or height != self.workspace_height
        )
        physical_changed = (
            physical_width_cm is not None
            and abs(float(physical_width_cm) - self.config.monitor_width_cm) > 1e-6
        ) or (
            physical_height_cm is not None
            and abs(float(physical_height_cm) - self.config.monitor_height_cm) > 1e-6
        )

        if (workspace_changed or physical_changed) and (self.calibrated or self.monitor_calibrated or self.calibration_samples):
            self.reset_calibration()

        self.set_workspace(width, height)

        if physical_width_cm is not None and physical_width_cm > 0:
            self.config.monitor_width_cm = float(physical_width_cm)
        if physical_height_cm is not None and physical_height_cm > 0:
            self.config.monitor_height_cm = float(physical_height_cm)

        self.workspace_calibration = WorkspaceCalibration(
            width_px=width,
            height_px=height,
            device_pixel_ratio=max(0.1, float(device_pixel_ratio or 1.0)),
            reading_rect_px=reading_rect_px,
            physical_width_cm=(
                float(physical_width_cm)
                if physical_width_cm is not None and physical_width_cm > 0
                else None
            ),
            physical_height_cm=(
                float(physical_height_cm)
                if physical_height_cm is not None and physical_height_cm > 0
                else None
            ),
            card_width_px=(
                float(card_width_px)
                if card_width_px is not None and card_width_px > 0
                else None
            ),
        )

    def reset_calibration(self) -> None:
        self.calibration_offset_yaw = 0.0
        self.calibration_offset_pitch = 0.0

        self.left_sphere_locked = False
        self.left_sphere_local_offset = None
        self.left_calibration_nose_scale = None

        self.right_sphere_locked = False
        self.right_sphere_local_offset = None
        self.right_calibration_nose_scale = None

        self.monitor_corners = None
        self.monitor_center_w = None
        self.monitor_normal_w = None
        self.units_per_cm = None
        self.gaze_markers.clear()
        self.calibration_samples.clear()
        self.calibration_transform = None
        self.calibration_model = None
        self.calibration_residuals.clear()
        self.combined_gaze_directions.clear()
        self.R_ref_nose = [None]

    @property
    def calibrated(self) -> bool:
        return bool(self.left_sphere_locked and self.right_sphere_locked)

    @property
    def monitor_calibrated(self) -> bool:
        return self.monitor_corners is not None and self.monitor_center_w is not None

    @property
    def point_calibrated(self) -> bool:
        return self.calibration_transform is not None and len(self.calibration_samples) >= 9

    def process_frame(self, frame_bgr: np.ndarray) -> GazeResult:
        """
        Process one BGR frame and return workspace coordinates if calibrated
        """
        state = self._analyze_frame(frame_bgr)
        if state is None:
            return GazeResult(face_detected=False, calibrated=self.calibrated, valid=False, reason="no_face_detected")

        if not self.calibrated:
            return GazeResult(face_detected=True, calibrated=False, valid=False, reason="eye_spheres_not_calibrated", debug=self._state_debug(state))

        if state.avg_combined_direction is None:
            return GazeResult(face_detected=True, calibrated=True, valid=False, reason="gaze_direction_unavailable", debug=self._state_debug(state))

        # Preserve original yaw/pitch-to-screen mapping as fallback and compatibility output.
        screen_x, screen_y, raw_yaw, raw_pitch = self._convert_gaze_to_screen_coordinates(
            state.avg_combined_direction,
            self.calibration_offset_yaw,
            self.calibration_offset_pitch
        )

        result = GazeResult(
            face_detected=True,
            calibrated=True,
            valid=True,
            x=screen_x,
            y=screen_y,
            x_norm=screen_x / max(self.workspace_width, 1),
            y_norm=screen_y / max(self.workspace_height, 1),
            raw_yaw=raw_yaw,
            raw_pitch=raw_pitch,
            gaze_direction=state.avg_combined_direction.tolist(),
            debug=self._state_debug(state)
        )

        # Keep computation result
        plane_hit = self._intersect_gaze_with_monitor(state)
        if plane_hit is not None:
            point, a, b, inside = plane_hit
            result.monitor_intersection = point.tolist()
            result.monitor_coordinates = (a, b)
            result.gaze_origin = (
                ((state.sphere_world_l + state.sphere_world_r) * 0.5).tolist()
                if state.sphere_world_l is not None and state.sphere_world_r is not None
                else None
            )

            if inside:
                # Raw monitor-plane coordinates. The axis flip is preserved from the
                # previous web version because it matched the browser coordinate space.
                raw_x_norm = 1.0 - a
                raw_y_norm = 1.0 - b
                x_norm, y_norm = self._apply_calibration_transform(raw_x_norm, raw_y_norm)

                x = int(round(x_norm * self.workspace_width))
                y = int(round(y_norm * self.workspace_height))

                result.x = max(0, min(x, self.workspace_width - 1))
                result.y = max(0, min(y, self.workspace_height - 1))
                result.x_norm = max(0.0, min(x_norm, 1.0))
                result.y_norm = max(0.0, min(y_norm, 1.0))
                result.valid = True
                result.reason = None
                result.debug["raw_x_norm"] = raw_x_norm
                result.debug["raw_y_norm"] = raw_y_norm
                result.debug["point_calibrated"] = self.point_calibrated
            else:
                # Keep fallback x/y but report that the plane projection is outside.
                result.valid = False
                result.reason = "gaze_outside_monitor_plane"

        return result

    def calibrate_eyes_and_monitor(self, frame_bgr: np.ndarray) -> GazeResult:
        """
        Locks both eye spheres and creates the 3D monitor plane from the current frame.
        The caller should invoke this when the user is looking at the workspace center.
        """
        state = self._analyze_frame(frame_bgr, update_smoothing=False)
        if state is None:
            return GazeResult(face_detected=False, calibrated=False, valid=False, reason="no_face_detected")

        h, w = state.frame_shape[:2]
        current_nose_scale = compute_scale(state.nose_points_3d)

        camera_dir_world = np.array([0, 0, 1], dtype=float)
        camera_dir_local = state.R_final.T @ camera_dir_world

        # Lock LEFT eye
        self.left_sphere_local_offset = state.R_final.T @ (state.iris_3d_left - state.head_center)
        self.left_sphere_local_offset += self.config.base_eye_radius * camera_dir_local
        self.left_calibration_nose_scale = current_nose_scale
        self.left_sphere_locked = True

        # Lock RIGHT eye
        self.right_sphere_local_offset = state.R_final.T @ (state.iris_3d_right - state.head_center)
        self.right_sphere_local_offset += self.config.base_eye_radius * camera_dir_local
        self.right_calibration_nose_scale = current_nose_scale
        self.right_sphere_locked = True

        # Create 3D monitor plane at calibration, matching the original script
        sphere_world_l_calib = state.head_center + state.R_final @ self.left_sphere_local_offset
        sphere_world_r_calib = state.head_center + state.R_final @ self.right_sphere_local_offset

        left_dir = state.iris_3d_left - sphere_world_l_calib
        right_dir = state.iris_3d_right - sphere_world_r_calib
        if np.linalg.norm(left_dir) > 1e-9:
            left_dir /= np.linalg.norm(left_dir)
        if np.linalg.norm(right_dir) > 1e-9:
            right_dir /= np.linalg.norm(right_dir)

        forward_hint = (left_dir + right_dir) * 0.5
        if np.linalg.norm(forward_hint) > 1e-9:
            forward_hint /= np.linalg.norm(forward_hint)
        else:
            forward_hint = None

        gaze_origin = (sphere_world_l_calib + sphere_world_r_calib) / 2
        gaze_dir = forward_hint

        self.monitor_corners, self.monitor_center_w, self.monitor_normal_w, self.units_per_cm = create_monitor_plane(
            state.head_center,
            state.R_final,
            state.face_landmarks,
            w,
            h,
            face_to_monitor_cm=self.config.face_to_monitor_cm,
            monitor_width_cm=self.config.monitor_width_cm,
            monitor_height_cm=self.config.monitor_height_cm,
            forward_hint=forward_hint,
            gaze_origin=gaze_origin,
            gaze_dir=gaze_dir
        )

        # Clear smoothing after calibration to avoid mixing pre/post-calibration samples.
        self.combined_gaze_directions.clear()

        return self.process_frame(frame_bgr)


    def add_calibration_point(
        self,
        frame_bgr: np.ndarray,
        target_x_norm: Optional[float] = None,
        target_y_norm: Optional[float] = None,
        target_x_px: Optional[float] = None,
        target_y_px: Optional[float] = None,
    ) -> GazeResult:
        """
        Add one calibration sample for a known target position.

        The first sample should be the center target. If eye/monitor calibration has
        not happened yet, this method bootstraps the existing eye-sphere and monitor
        calibration from that frame. Then it records the current raw monitor-plane
        projection and fits a correction from raw coordinates to target coordinates.
        """
        return self.add_calibration_point_batch(
            [frame_bgr],
            target_x_norm=target_x_norm,
            target_y_norm=target_y_norm,
            target_x_px=target_x_px,
            target_y_px=target_y_px,
            min_valid_samples=1,
        )

    def add_calibration_point_batch(
        self,
        frames_bgr: Sequence[np.ndarray],
        target_x_norm: Optional[float] = None,
        target_y_norm: Optional[float] = None,
        target_x_px: Optional[float] = None,
        target_y_px: Optional[float] = None,
        min_valid_samples: int = 3,
    ) -> GazeResult:
        """
        Add one robust calibration sample from several frames at the same target.

        The raw gaze coordinates are aggregated with a median/outlier-trimmed mean
        before fitting the calibration transform.
        """
        if target_x_px is not None and target_y_px is not None:
            target_x_norm = float(target_x_px) / max(self.workspace_width, 1)
            target_y_norm = float(target_y_px) / max(self.workspace_height, 1)

        if target_x_norm is None or target_y_norm is None:
            return GazeResult(
                face_detected=False,
                calibrated=self.calibrated,
                valid=False,
                reason="missing_calibration_target",
            )

        target_x_norm = float(np.clip(target_x_norm, 0.0, 1.0))
        target_y_norm = float(np.clip(target_y_norm, 0.0, 1.0))

        frames = [frame for frame in frames_bgr if frame is not None]
        if not frames:
            return GazeResult(
                face_detected=False,
                calibrated=self.calibrated,
                valid=False,
                reason="missing_calibration_frames",
            )

        if not self.calibrated or not self.monitor_calibrated:
            bootstrap = self.calibrate_eyes_and_monitor(frames[0])
            if not bootstrap.face_detected or not self.monitor_calibrated:
                bootstrap.reason = bootstrap.reason or "monitor_calibration_failed"
                return bootstrap

        raw_points: List[Tuple[float, float]] = []
        last_state: Optional[_FrameState] = None
        failure_counts: Dict[str, int] = {}
        last_point: Optional[np.ndarray] = None
        last_monitor_coordinates: Optional[Tuple[float, float]] = None

        for frame in frames:
            state = self._analyze_frame(frame, update_smoothing=False)
            if state is None:
                failure_counts["no_face_detected"] = failure_counts.get("no_face_detected", 0) + 1
                continue

            last_state = state
            if state.raw_combined_direction is None:
                failure_counts["gaze_direction_unavailable"] = failure_counts.get("gaze_direction_unavailable", 0) + 1
                continue

            # Use instantaneous directions for calibration so old targets do not
            # leak into the current target's sample.
            state.avg_combined_direction = state.raw_combined_direction
            plane_hit = self._intersect_gaze_with_monitor(state)
            if plane_hit is None:
                failure_counts["monitor_intersection_unavailable"] = failure_counts.get("monitor_intersection_unavailable", 0) + 1
                continue

            point, a, b, inside = plane_hit
            last_point = point
            last_monitor_coordinates = (a, b)
            if not inside:
                failure_counts["gaze_outside_monitor_plane"] = failure_counts.get("gaze_outside_monitor_plane", 0) + 1
                continue

            raw_points.append((1.0 - a, 1.0 - b))

        min_required = max(1, int(min_valid_samples))
        if len(raw_points) < min_required:
            reason = "insufficient_calibration_samples"
            if failure_counts:
                reason = max(failure_counts.items(), key=lambda item: item[1])[0]
            return GazeResult(
                face_detected=last_state is not None,
                calibrated=self.calibrated,
                valid=False,
                reason=reason,
                monitor_intersection=last_point.tolist() if last_point is not None else None,
                monitor_coordinates=last_monitor_coordinates,
                debug=(
                    {
                        **self._state_debug(last_state),
                        "valid_calibration_frames": len(raw_points),
                        "requested_calibration_frames": len(frames),
                        "calibration_failures": failure_counts,
                    }
                    if last_state is not None
                    else {
                        "valid_calibration_frames": len(raw_points),
                        "requested_calibration_frames": len(frames),
                        "calibration_failures": failure_counts,
                    }
                ),
            )

        raw_x_norm, raw_y_norm, kept_count = _robust_average_2d(raw_points)
        self.calibration_samples.append(((raw_x_norm, raw_y_norm), (target_x_norm, target_y_norm)))
        self._fit_calibration_transform()
        self.combined_gaze_directions.clear()

        result = self.process_frame(frames[-1])
        result.debug["calibration_point_added"] = True
        result.debug["calibration_count"] = len(self.calibration_samples)
        result.debug["calibration_target"] = [target_x_norm, target_y_norm]
        result.debug["calibration_target_px"] = [
            float(target_x_px) if target_x_px is not None else target_x_norm * self.workspace_width,
            float(target_y_px) if target_y_px is not None else target_y_norm * self.workspace_height,
        ]
        result.debug["calibration_raw"] = [raw_x_norm, raw_y_norm]
        result.debug["calibration_raw_frame_count"] = len(raw_points)
        result.debug["calibration_raw_kept_count"] = kept_count
        result.debug["calibration_model"] = self.calibration_model
        result.debug["point_calibrated"] = self.point_calibrated
        result.debug["workspace_calibration"] = (
            self.workspace_calibration.to_dict()
            if self.workspace_calibration is not None
            else None
        )
        return result

    def clear_point_calibration(self) -> None:
        """Clear only the point-correction layer while keeping eye/monitor calibration."""
        self.calibration_samples.clear()
        self.calibration_transform = None
        self.calibration_model = None
        self.calibration_residuals.clear()

    def _fit_calibration_transform(self) -> None:
        if len(self.calibration_samples) < 3:
            self.calibration_transform = None
            self.calibration_model = None
            self.calibration_residuals.clear()
            return

        raw = np.array([[rx, ry, 1.0] for (rx, ry), _target in self.calibration_samples], dtype=float)
        target = np.array([[tx, ty] for _raw, (tx, ty) in self.calibration_samples], dtype=float)

        # coeff is 3x2. Transpose to 2x3 so [x, y] = transform @ [raw_x, raw_y, 1].
        coeff, *_ = np.linalg.lstsq(raw, target, rcond=None)
        self.calibration_transform = coeff.T
        self.calibration_model = "affine"
        self.calibration_residuals.clear()

        if len(self.calibration_samples) >= 9:
            predicted = raw @ coeff
            self.calibration_residuals = [
                ((float(rx), float(ry)), (float(tx - px), float(ty - py)))
                for ((rx, ry), (tx, ty)), (px, py) in zip(self.calibration_samples, predicted)
            ]
            self.calibration_model = "affine_local"

    def _apply_calibration_transform(self, raw_x_norm: float, raw_y_norm: float) -> Tuple[float, float]:
        if self.calibration_transform is None:
            return (
                float(np.clip(raw_x_norm, 0.0, 1.0)),
                float(np.clip(raw_y_norm, 0.0, 1.0)),
            )

        features = np.array([raw_x_norm, raw_y_norm, 1.0], dtype=float)
        corrected = self.calibration_transform @ features
        if self.calibration_model == "affine_local" and self.calibration_residuals:
            corrected += _local_residual_correction(
                raw_x_norm,
                raw_y_norm,
                self.calibration_residuals,
            )

        return (
            float(np.clip(corrected[0], 0.0, 1.0)),
            float(np.clip(corrected[1], 0.0, 1.0)),
        )

    def calibrate_screen_center(self, frame_bgr: np.ndarray) -> GazeResult:
        """
        Sets yaw/pitch offsets so the current gaze maps to the center of the workspace.
        This mainly affects the original yaw/pitch fallback mapping.
        """
        state = self._analyze_frame(frame_bgr, update_smoothing=False)
        if state is None:
            return GazeResult(face_detected=False, calibrated=self.calibrated, valid=False, reason="no_face_detected")
        if not self.calibrated or state.raw_combined_direction is None:
            return GazeResult(face_detected=True, calibrated=False, valid=False, reason="eye_spheres_not_calibrated", debug=self._state_debug(state),)

        _, _, raw_yaw, raw_pitch = self._convert_gaze_to_screen_coordinates(
            state.raw_combined_direction,
            0.0,
            0.0,
        )
        self.calibration_offset_yaw = 0.0 - raw_yaw
        self.calibration_offset_pitch = 0.0 - raw_pitch
        return self.process_frame(frame_bgr)

    def add_marker(self, frame_bgr: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Adds and returns a normalized monitor-plane marker (a,b), or None if unavailable
        """
        result = self.process_frame(frame_bgr)
        if result.monitor_coordinates is None or not result.valid:
            return None
        self.gaze_markers.append(result.monitor_coordinates)
        return result.monitor_coordinates

    def update_face_distance_from_frame(
        self,
        frame_bgr: np.ndarray,
        focal_length_px: Optional[float] = None,
        camera_width_px: Optional[int] = None,
        horizontal_fov_deg: Optional[float] = None,
        real_eye_width_cm: Optional[float] = None
    ) -> Optional[float]:
        """
        Estimates face distance and stores it as config.face_to_monitor_cm
        """
        distance = get_face_distance_cm(
            frame_bgr,
            focal_length_px=focal_length_px,
            real_eye_width_cm=real_eye_width_cm or self.config.real_eye_width_cm,
            camera_width_px=camera_width_px or frame_bgr.shape[1],
            horizontal_fov_deg=horizontal_fov_deg or self.config.horizontal_fov_deg,
        )
        if distance is not None:
            self.config.face_to_monitor_cm = float(distance)
        return distance

    def _analyze_frame(self, frame_bgr: np.ndarray, update_smoothing: bool = True) -> Optional[_FrameState]:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0].landmark
        left_iris = face_landmarks[LEFT_IRIS_IDX]
        right_iris = face_landmarks[RIGHT_IRIS_IDX]

        head_center, R_final, nose_points_3d = compute_head_pose(face_landmarks, NOSE_INDICES, self.R_ref_nose, w, h)

        iris_3d_left = np.array([left_iris.x * w, left_iris.y * h, left_iris.z * w])
        iris_3d_right = np.array([right_iris.x * w, right_iris.y * h, right_iris.z * w])

        state = _FrameState(
            frame_shape=frame_bgr.shape,
            face_landmarks=face_landmarks,
            head_center=head_center,
            R_final=R_final,
            nose_points_3d=nose_points_3d,
            iris_3d_left=iris_3d_left,
            iris_3d_right=iris_3d_right,
        )

        if self.calibrated:
            current_nose_scale = compute_scale(nose_points_3d)

            scale_ratio_l = (
                current_nose_scale / self.left_calibration_nose_scale
                if self.left_calibration_nose_scale
                else 1.0
            )
            scale_ratio_r = (
                current_nose_scale / self.right_calibration_nose_scale
                if self.right_calibration_nose_scale
                else 1.0
            )

            state.sphere_world_l = head_center + R_final @ (
                self.left_sphere_local_offset * scale_ratio_l
            )
            state.sphere_world_r = head_center + R_final @ (
                self.right_sphere_local_offset * scale_ratio_r
            )
            state.scaled_radius_l = self.config.base_eye_radius * scale_ratio_l
            state.scaled_radius_r = self.config.base_eye_radius * scale_ratio_r

            left_gaze_dir = iris_3d_left - state.sphere_world_l
            right_gaze_dir = iris_3d_right - state.sphere_world_r
            if np.linalg.norm(left_gaze_dir) > 1e-9 and np.linalg.norm(right_gaze_dir) > 1e-9:
                left_gaze_dir /= np.linalg.norm(left_gaze_dir)
                right_gaze_dir /= np.linalg.norm(right_gaze_dir)
                raw_combined_direction = (left_gaze_dir + right_gaze_dir) / 2
                raw_combined_direction /= np.linalg.norm(raw_combined_direction)
                state.raw_combined_direction = raw_combined_direction

                if update_smoothing:
                    self.combined_gaze_directions.append(raw_combined_direction)
                    avg_combined_direction = np.mean(self.combined_gaze_directions, axis=0)
                else:
                    avg_combined_direction = raw_combined_direction

                if np.linalg.norm(avg_combined_direction) > 1e-9:
                    avg_combined_direction /= np.linalg.norm(avg_combined_direction)
                    state.avg_combined_direction = avg_combined_direction

        return state

    def _convert_gaze_to_screen_coordinates(self, combined_gaze_direction: np.ndarray, calibration_offset_yaw: float, calibration_offset_pitch: float) -> Tuple[int, int, float, float]:
        """
        Convert 3D gaze direction vector to 2D screen coordinates
        """
        # Reference forward direction (camera looking straight ahead)
        reference_forward = np.array([0, 0, -1], dtype=float)  # Z-axis into the screen

        # Normalize the gaze direction
        avg_direction = combined_gaze_direction / np.linalg.norm(combined_gaze_direction)

        # Horizontal (yaw) angle from reference (project onto XZ plane)
        xz_proj = np.array([avg_direction[0], 0, avg_direction[2]], dtype=float)
        xz_proj /= np.linalg.norm(xz_proj)
        yaw_rad = math.acos(np.clip(np.dot(reference_forward, xz_proj), -1.0, 1.0))
        if avg_direction[0] < 0:
            yaw_rad = -yaw_rad # left is negative

        # Vertical (pitch) angle from reference (project onto YZ plane)
        yz_proj = np.array([0, avg_direction[1], avg_direction[2]], dtype=float)
        yz_proj /= np.linalg.norm(yz_proj)
        pitch_rad = math.acos(np.clip(np.dot(reference_forward, yz_proj), -1.0, 1.0))
        if avg_direction[1] > 0:
            pitch_rad = -pitch_rad  # up is positive

        # Convert to degrees and re-center around 0
        yaw_deg = np.degrees(yaw_rad)
        pitch_deg = np.degrees(pitch_rad)

        # Convert left rotations to 0-180 (from old script logic)
        if yaw_deg < 0:
            yaw_deg = -yaw_deg
        elif yaw_deg > 0:
            yaw_deg = -yaw_deg

        # yaw is now converted to -90 (looking directly left) to +90 (looking directly right), wrt camera
        # pitch is now converted to +90 (looking straight up) and -90 (looking straight down), wrt camera

        raw_yaw_deg = float(yaw_deg)
        raw_pitch_deg = float(pitch_deg)

        # Apply calibration offsets
        yaw_deg += calibration_offset_yaw
        pitch_deg += calibration_offset_pitch

        # Map to full screen resolution
        screen_x = int(
            ((yaw_deg + self.config.yaw_degrees) / (2 * self.config.yaw_degrees))
            * self.workspace_width
        )
        screen_y = int(
            ((self.config.pitch_degrees - pitch_deg) / (2 * self.config.pitch_degrees))
            * self.workspace_height
        )

        margin_x = min(10, max(0, self.workspace_width - 1))
        margin_y = min(10, max(0, self.workspace_height - 1))
        # Clamp screen position to monitor bounds
        screen_x = max(margin_x, min(screen_x, max(margin_x, self.workspace_width - margin_x)))
        screen_y = max(margin_y, min(screen_y, max(margin_y, self.workspace_height - margin_y)))

        return screen_x, screen_y, raw_yaw_deg, raw_pitch_deg

    def _intersect_gaze_with_monitor(self, state: _FrameState) -> Optional[Tuple[np.ndarray, float, float, bool]]:
        if (
            self.monitor_corners is None
            or self.monitor_center_w is None
            or self.monitor_normal_w is None
            or state.avg_combined_direction is None
            or state.sphere_world_l is None
            or state.sphere_world_r is None
        ):
            return None

        O = (state.sphere_world_l + state.sphere_world_r) * 0.5
        D = _normalize(state.avg_combined_direction)
        C = np.asarray(self.monitor_center_w, dtype=float)
        N = _normalize(np.asarray(self.monitor_normal_w, dtype=float))

        denom = float(np.dot(N, D))
        if abs(denom) < 1e-6:
            return None

        t = float(np.dot(N, (C - O)) / denom)
        if t <= 0.0:
            return None

        P = O + t * D
        p0, p1, _p2, p3 = [np.asarray(p, dtype=float) for p in self.monitor_corners]
        u = p1 - p0
        v = p3 - p0
        wv = P - p0

        u_len2 = float(np.dot(u, u))
        v_len2 = float(np.dot(v, v))
        if u_len2 <= 1e-9 or v_len2 <= 1e-9:
            return None

        a = float(np.dot(wv, u) / u_len2)
        b = float(np.dot(wv, v) / v_len2)
        inside = 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0
        return P, a, b, inside

    def _state_debug(self, state: _FrameState) -> Dict[str, Any]:
        return {
            "head_center": state.head_center.tolist(),
            "units_per_cm": self.units_per_cm,
            "monitor_calibrated": self.monitor_calibrated,
            "left_sphere_locked": self.left_sphere_locked,
            "right_sphere_locked": self.right_sphere_locked,
            "gaze_markers": list(self.gaze_markers),
            "calibration_count": len(self.calibration_samples),
            "point_calibrated": self.point_calibrated,
            "calibration_transform": (
                self.calibration_transform.tolist()
                if self.calibration_transform is not None
                else None
            ),
            "calibration_model": self.calibration_model,
            "calibration_residual_count": len(self.calibration_residuals),
            "workspace": (
                self.workspace_calibration.to_dict()
                if self.workspace_calibration is not None
                else {
                    "width_px": self.workspace_width,
                    "height_px": self.workspace_height,
                }
            ),
        }


# alias
GazeTracker = MonitorTracker

def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _robust_average_2d(points: Sequence[Tuple[float, float]]) -> Tuple[float, float, int]:
    values = np.asarray(points, dtype=float)
    if len(values) == 1:
        return float(values[0, 0]), float(values[0, 1]), 1

    median = np.median(values, axis=0)
    distances = np.linalg.norm(values - median, axis=1)

    if len(values) >= 5:
        keep_count = max(3, int(math.ceil(len(values) * 0.75)))
        kept = values[np.argsort(distances)[:keep_count]]
    elif len(values) >= 3:
        kept = values[np.argsort(distances)[: len(values) - 1]]
    else:
        kept = values

    averaged = np.mean(kept, axis=0)
    return float(averaged[0]), float(averaged[1]), int(len(kept))


def _local_residual_correction(
    x: float,
    y: float,
    residuals: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]],
) -> np.ndarray:
    if not residuals:
        return np.zeros(2, dtype=float)

    point = np.array([x, y], dtype=float)
    raw_points = np.array([raw for raw, _residual in residuals], dtype=float)
    residual_values = np.array([residual for _raw, residual in residuals], dtype=float)
    distances = np.linalg.norm(raw_points - point, axis=1)
    nearest_count = min(4, len(residuals))
    nearest_indices = np.argsort(distances)[:nearest_count]

    nearest_distances = distances[nearest_indices]
    nearest_residuals = residual_values[nearest_indices]
    sigma = 0.22
    weights = np.exp(-((nearest_distances / sigma) ** 2))
    total_weight = float(np.sum(weights))
    if total_weight <= 1e-9:
        return np.zeros(2, dtype=float)

    return np.average(nearest_residuals, axis=0, weights=weights)


def _focal_px(width: float, fov_deg: float) -> float:
    return 0.5 * width / math.tan(math.radians(fov_deg) * 0.5)


def compute_scale(points_3d: np.ndarray) -> float:
    """Use average pairwise distance for robustness"""
    n = len(points_3d)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += float(np.linalg.norm(points_3d[i] - points_3d[j]))
            count += 1
    return total / count if count > 0 else 1.0


def compute_head_pose(
    face_landmarks: Sequence[Any],
    indices: Sequence[int],
    ref_matrix_container: List[Optional[np.ndarray]],
    w: int,
    h: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute stabilized coordinate frame from nose-region landmarks
    """
    points_3d = np.array(
        [[face_landmarks[i].x * w, face_landmarks[i].y * h, face_landmarks[i].z * w]
         for i in indices],
        dtype=float
    )

    center = np.mean(points_3d, axis=0)
    centered = points_3d - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvecs = eigvecs[:, np.argsort(-eigvals)]

    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] *= -1

    r = Rscipy.from_matrix(eigvecs)
    roll, pitch, yaw = r.as_euler("zyx", degrees=False)
    R_final = Rscipy.from_euler("zyx", [roll, pitch, yaw]).as_matrix()

    if ref_matrix_container[0] is None:
        ref_matrix_container[0] = R_final.copy()
    else:
        R_ref = ref_matrix_container[0]
        for i in range(3):
            if np.dot(R_final[:, i], R_ref[:, i]) < 0:
                R_final[:, i] *= -1

    return center, R_final, points_3d


def create_monitor_plane(
    head_center: np.ndarray,
    R_final: np.ndarray,
    face_landmarks: Sequence[Any],
    w: int,
    h: int,
    face_to_monitor_cm: float = 60.0,
    monitor_width_cm: float = 60.0,
    monitor_height_cm: float = 33.5,
    forward_hint: Optional[np.ndarray] = None,
    gaze_origin: Optional[np.ndarray] = None,
    gaze_dir: Optional[np.ndarray] = None
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, float]:
    """
    Build a monitor plane in world units.
    """
    try:
        lm_chin = face_landmarks[152]
        lm_fore = face_landmarks[10]
        chin_w = np.array([lm_chin.x * w, lm_chin.y * h, lm_chin.z * w], dtype=float)
        fore_w = np.array([lm_fore.x * w, lm_fore.y * h, lm_fore.z * w], dtype=float)
        face_h_units = np.linalg.norm(fore_w - chin_w)
        units_per_cm = face_h_units / 15.0
    except Exception:
        units_per_cm = 5.0

    half_w = (monitor_width_cm * 0.5) * units_per_cm
    half_h = (monitor_height_cm * 0.5) * units_per_cm

    head_forward = -R_final[:, 2]
    if forward_hint is not None:
        head_forward = _normalize(forward_hint)

    if gaze_origin is not None and gaze_dir is not None:
        gaze_dir = _normalize(gaze_dir)
        plane_point = head_center + head_forward * (face_to_monitor_cm * units_per_cm)
        plane_normal = head_forward

        denom = np.dot(plane_normal, gaze_dir)
        if abs(denom) > 1e-6:
            t = np.dot(plane_normal, plane_point - gaze_origin) / denom
            center_w = gaze_origin + t * gaze_dir
        else:
            center_w = head_center + head_forward * (face_to_monitor_cm * units_per_cm)
    else:
        center_w = head_center + head_forward * (face_to_monitor_cm * units_per_cm)

    world_up = np.array([0, -1, 0], dtype=float)
    head_right = np.cross(world_up, head_forward)
    head_right /= np.linalg.norm(head_right)
    head_up = np.cross(head_forward, head_right)
    head_up /= np.linalg.norm(head_up)

    p0 = center_w - head_right * half_w - head_up * half_h
    p1 = center_w + head_right * half_w - head_up * half_h
    p2 = center_w + head_right * half_w + head_up * half_h
    p3 = center_w - head_right * half_w + head_up * half_h

    normal_w = head_forward / (np.linalg.norm(head_forward) + 1e-9)
    return [p0, p1, p2, p3], center_w, normal_w, units_per_cm

# === Face Distance Helpers ===

def _get_eye_pixel_distance_from_landmarks(frame_bgr: np.ndarray, face_mesh: Any) -> Optional[float]:
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    face = results.multi_face_landmarks[0]
    left = face.landmark[LEFT_EYE_OUTER]
    right = face.landmark[RIGHT_EYE_OUTER]

    x1, y1 = left.x * w, left.y * h
    x2, y2 = right.x * w, right.y * h
    return math.hypot(x2 - x1, y2 - y1)


def calibrate_focal_length_from_frame(frame_bgr: np.ndarray, known_distance_cm: float, real_eye_width_cm: float = 9.5) -> Optional[float]:
    """Return focal length in pixels from one frame taken at a known distance"""
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    try:
        pixel_eye_distance = _get_eye_pixel_distance_from_landmarks(frame_bgr, face_mesh)
    finally:
        face_mesh.close()

    if pixel_eye_distance is None or pixel_eye_distance <= 0:
        return None
    return (known_distance_cm * pixel_eye_distance) / real_eye_width_cm


def get_face_distance_cm(frame_bgr: np.ndarray,
    focal_length_px: Optional[float] = None,
    real_eye_width_cm: float = 9.5,
    camera_width_px: Optional[int] = None,
    horizontal_fov_deg: Optional[float] = None
) -> Optional[float]:
    """
    Estimate face distance in cm from a frame
    Provide either focal_length_px directly, or camera_width_px + horizontal_fov_deg
    """
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    try:
        pixel_eye_distance = _get_eye_pixel_distance_from_landmarks(frame_bgr, face_mesh)
    finally:
        face_mesh.close()

    if pixel_eye_distance is None or pixel_eye_distance <= 0:
        return None

    if focal_length_px is None:
        if camera_width_px is None or horizontal_fov_deg is None:
            return None
        focal_length_px = (camera_width_px / 2.0) / math.tan(
            math.radians(horizontal_fov_deg / 2.0)
        )

    return (real_eye_width_cm * focal_length_px) / pixel_eye_distance
