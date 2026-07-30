"""cam-clip-record-playback -- record a short clip and read it back
frame-by-frame, establishing the video I/O pattern reused in Topic 6's
motion-tracking assignment.

* ``clip``      -- pure functions: duration/index/seek-plan math. No
                    hardware import -- fully testable off-Pi.
* ``hardware``  -- the ONLY module that imports picamera2 (recording) and
                    cv2.VideoCapture (playback).
"""
