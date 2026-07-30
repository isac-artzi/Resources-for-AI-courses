"""cam-stream-fps -- measure real-world throughput of the camera stream.

* ``fps``       -- pure functions: turn a list of frame timestamps into
                    FPS numbers. No hardware import -- fully testable off-Pi.
* ``hardware``  -- the ONLY module that imports picamera2. Opens a live
                    stream at a few resolutions and times real frames.
"""
