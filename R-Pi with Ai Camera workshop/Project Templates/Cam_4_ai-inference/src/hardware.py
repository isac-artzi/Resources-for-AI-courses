"""The ONLY module in this project that talks to real camera/NPU hardware.

Run on the Pi with the AI Camera attached:

    python -m src.hardware

Loads an IMX500 object-detection model, runs a few inference passes to
let auto-exposure and the model settle, then saves one frame with
bounding boxes drawn on it to ./captures/inference.jpg.

Adapted from Raspberry Pi's own reference example, simplified to a
single-shot capture instead of a live preview loop:
https://github.com/raspberrypi/picamera2/blob/main/examples/imx500/imx500_object_detection_demo.py
"""
from __future__ import annotations

import os
from pathlib import Path

from src.detections import filter_by_confidence, format_detection_label, label_for_category, resolve_model_path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "captures"
MODEL_DIR = "/usr/share/imx500-models"
# This is the model Raspberry Pi's own demo defaults to. If it's not on
# your Pi's image, resolve_model_path() below falls back to whatever
# .rpk file IS present.
PREFERRED_MODEL = "imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
CONFIDENCE_THRESHOLD = 0.55
IOU_THRESHOLD = 0.65
MAX_DETECTIONS = 10
WARMUP_FRAMES = 15  # let auto-exposure and the model settle before we trust a result


def parse_frame_detections(imx500, picam2, intrinsics, metadata: dict) -> list[dict]:
    """Turn one frame's raw NPU output into a list of
    {"label": category_index_as_str, "confidence": float, "box": (x, y, w, h)}
    dicts, using the postprocessing path this model's own intrinsics
    declare (SSD-style by default, or nanodet-style if the model says so).

    This mirrors Raspberry Pi's own parse_detections() but returns plain
    dicts instead of a custom Detection class, so it's consistent with
    the rest of this series.
    """
    from picamera2.devices.imx500 import postprocess_nanodet_detection
    from picamera2.devices.imx500.postprocess import scale_boxes

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    if np_outputs is None:
        return []

    input_w, input_h = imx500.get_input_size()

    if intrinsics.postprocess == "nanodet":
        boxes, scores, classes = postprocess_nanodet_detection(
            outputs=np_outputs[0], conf=CONFIDENCE_THRESHOLD,
            iou_thres=IOU_THRESHOLD, max_out_dets=MAX_DETECTIONS,
        )[0]
        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
    else:
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if intrinsics.bbox_normalization:
            boxes = boxes / input_h
        if intrinsics.bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]

    detections = []
    for box, score, category in zip(boxes, scores, classes):
        # imx500.convert_inference_coords maps the model's own coordinate
        # space onto the actual ISP output frame -- this is the step my
        # first draft got wrong by trying to do the math manually.
        x, y, w, h = imx500.convert_inference_coords(box, metadata, picam2)
        detections.append({"label": str(int(category)), "confidence": float(score), "box": (x, y, w, h)})

    return filter_by_confidence(detections, CONFIDENCE_THRESHOLD)


def run() -> None:
    import cv2
    from picamera2 import Picamera2
    from picamera2.devices import IMX500
    from picamera2.devices.imx500 import NetworkIntrinsics

    OUTPUT_DIR.mkdir(exist_ok=True)

    available = os.listdir(MODEL_DIR) if os.path.isdir(MODEL_DIR) else []
    model_path = resolve_model_path(PREFERRED_MODEL, MODEL_DIR, available)
    print(f"Loading model: {model_path}")

    # IMX500 must be constructed before Picamera2 -- it pushes the model
    # onto the sensor's own firmware first.
    imx500 = IMX500(model_path)
    imx500.show_network_fw_progress_bar()

    intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
    if not imx500.network_intrinsics:
        intrinsics.task = "object detection"
    intrinsics.update_with_defaults()

    if intrinsics.task != "object detection":
        raise RuntimeError(
            f"{model_path} is a '{intrinsics.task}' model, not object detection. "
            "Pick a detection model from /usr/share/imx500-models/ instead."
        )

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12
    )
    picam2.start(config, show_preview=False)

    detections = []
    for _ in range(WARMUP_FRAMES):
        metadata = picam2.capture_metadata()
        detections = parse_frame_detections(imx500, picam2, intrinsics, metadata)

    frame = picam2.capture_array()
    picam2.stop()

    labels = intrinsics.labels or []
    for det in detections:
        x, y, w, h = det["box"]
        label = label_for_category(labels, int(det["label"]))
        text = format_detection_label(label, det["confidence"])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, text, (x, max(y - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out_path = OUTPUT_DIR / "inference.jpg"
    cv2.imwrite(str(out_path), frame)
    print(f"{len(detections)} detection(s) above {CONFIDENCE_THRESHOLD} confidence")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    run()
