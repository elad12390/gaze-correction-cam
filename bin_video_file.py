#!/usr/bin/env python3

"""
Video File Gaze Correction (headless, cloud-friendly).

Reads a video file frame-by-frame, applies gaze correction, writes a new video.
No webcam, no GUI, no virtual camera — pure file I/O. Designed for batch
processing in CI/CD, Docker containers, and cloud workers.

Usage:
    python bin_video_file.py --input in.mp4 --output out.mp4
    python bin_video_file.py -i in.mp4 -o out.mp4 --backend mediapipe
    python bin_video_file.py -i in.mp4 -o out.mp4 --no-progress
    python bin_video_file.py -i in.mp4 -o out.mp4 --camera-offset 0 -21 -1 \\
        --focal-length 650 --ipd 6.3

Output format (--codec):
    mp4v   default, broad compatibility
    avc1   H.264 (smaller, requires ffmpeg/x264 in opencv build)
    XVID   AVI

Exit codes:
    0  success
    1  bad arguments / missing files
    2  no faces detected on any frame (output video produced anyway)
    3  inference error
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
from tqdm import tqdm

from displayers.face_predictor import (
    EyeExtractionConfig,
    create_face_predictor,
)
from model_managers.gaze_corrector_v1 import GazeCorrector


################################################################################
# CLI
################################################################################


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Headless gaze correction for video files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    io_grp = parser.add_argument_group("I/O")
    io_grp.add_argument("-i", "--input", required=True, help="Input video file")
    io_grp.add_argument("-o", "--output", required=True, help="Output video file")
    io_grp.add_argument(
        "--codec",
        default="mp4v",
        help="FourCC codec for output (default: mp4v). Try avc1 for smaller H.264 files if ffmpeg/x264 is available.",
    )
    io_grp.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar (use in CI logs).",
    )

    model_grp = parser.add_argument_group("Model")
    model_grp.add_argument(
        "--backend",
        default="dlib",
        choices=["dlib", "mediapipe"],
        help="Face detection backend (default: dlib)",
    )
    model_grp.add_argument(
        "--config",
        default="./model_managers/gaze_corrector_v1_01.yaml",
        help="Path to gaze corrector YAML config",
    )
    model_grp.add_argument(
        "--db-path",
        default="./user_settings.db",
        help="SQLite path for camera settings (default: ./user_settings.db)",
    )
    model_grp.add_argument(
        "--setting-name",
        default="video_file_default",
        help="Camera setting name in DB (default: video_file_default)",
    )

    cal_grp = parser.add_argument_group(
        "Camera calibration (overrides DB on this run)"
    )
    cal_grp.add_argument(
        "--camera-offset",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Camera offset in cm relative to screen center (default: from DB or 0 -21 -1)",
    )
    cal_grp.add_argument(
        "--focal-length",
        type=float,
        help="Focal length in pixels (default: from DB or 650)",
    )
    cal_grp.add_argument(
        "--ipd",
        type=float,
        help="Inter-pupillary distance in cm (default: from DB or 6.3)",
    )

    debug_grp = parser.add_argument_group("Debug")
    debug_grp.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Process only N frames then stop (0 = all frames). Useful for smoke tests.",
    )
    debug_grp.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help="Skip the first N frames (default: 0).",
    )

    return parser.parse_args(argv)


################################################################################
# Pipeline
################################################################################


@contextmanager
def video_capture(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {path}")
    try:
        yield cap
    finally:
        cap.release()


@contextmanager
def video_writer(path: str, fourcc: int, fps: float, size: tuple[int, int]):
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output video for writing: {path}")
    try:
        yield writer
    finally:
        writer.release()


def apply_calibration_overrides(
    corrector: GazeCorrector, args: argparse.Namespace
) -> None:
    """Apply CLI calibration overrides to the corrector (skips DB save side-effects when None)."""
    if args.camera_offset is not None:
        corrector.set_camera_offset(*args.camera_offset)
    if args.focal_length is not None:
        corrector.set_focal_length(args.focal_length)
    if args.ipd is not None:
        corrector.set_ipd(args.ipd)


def process_video(args: argparse.Namespace) -> int:
    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[init] backend={args.backend} config={args.config}", file=sys.stderr)
    predictor = create_face_predictor(args.backend)
    corrector = GazeCorrector(
        config_path=args.config,
        db_path=args.db_path,
        setting_name=args.setting_name,
    )
    apply_calibration_overrides(corrector, args)
    eye_cfg = EyeExtractionConfig()

    fourcc = cv2.VideoWriter_fourcc(*args.codec)

    frames_total_processed = 0
    frames_with_face = 0

    try:
        with video_capture(str(in_path)) as cap:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            if args.max_frames > 0:
                total_frames = (
                    min(total_frames, args.max_frames) if total_frames > 0 else args.max_frames
                )

            print(
                f"[input] {width}x{height} @ {fps:.2f}fps  "
                f"frames={total_frames if total_frames > 0 else 'unknown'}",
                file=sys.stderr,
            )
            print(f"[output] {out_path} codec={args.codec}", file=sys.stderr)

            with video_writer(str(out_path), fourcc, fps, (width, height)) as writer:
                # Skip frames if requested (verbatim copy of skipped frames is wasteful;
                # we just discard them so the output starts at frame N).
                for _ in range(args.skip_frames):
                    ok, _ = cap.read()
                    if not ok:
                        break

                pbar = (
                    None
                    if args.no_progress
                    else tqdm(
                        total=total_frames if total_frames > 0 else None,
                        unit="frame",
                        dynamic_ncols=True,
                    )
                )

                start = time.perf_counter()
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    face_data_list = predictor.list_eye_data(frame, eye_cfg)
                    out_frame = frame
                    if face_data_list:
                        # Only correct first face (matches single_window behavior)
                        try:
                            out_frame = corrector.apply_correction(
                                frame, face_data_list[0], (width, height)
                            )
                            frames_with_face += 1
                        except Exception as e:  # noqa: BLE001 — boundary catch
                            print(
                                f"[warn] frame {frames_total_processed}: {e}",
                                file=sys.stderr,
                            )
                            return 3

                    writer.write(out_frame)
                    frames_total_processed += 1

                    if pbar is not None:
                        pbar.update(1)

                    if args.max_frames and frames_total_processed >= args.max_frames:
                        break

                if pbar is not None:
                    pbar.close()

                elapsed = time.perf_counter() - start
                effective_fps = (
                    frames_total_processed / elapsed if elapsed > 0 else 0.0
                )
                print(
                    f"[done] {frames_total_processed} frames "
                    f"({frames_with_face} with face) in {elapsed:.1f}s "
                    f"({effective_fps:.1f} fps)",
                    file=sys.stderr,
                )
    finally:
        try:
            corrector.close()
        except Exception:  # noqa: BLE001
            pass

    if frames_total_processed > 0 and frames_with_face == 0:
        print(
            "[warn] no faces detected in any frame — output video is unmodified copy",
            file=sys.stderr,
        )
        return 2

    return 0


################################################################################
# Entry point
################################################################################


def main() -> int:
    args = parse_args()
    try:
        return process_video(args)
    except KeyboardInterrupt:
        print("[interrupted]", file=sys.stderr)
        return 130
    except Exception as e:  # noqa: BLE001 — top-level crash boundary
        print(f"[fatal] {type(e).__name__}: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
