"""cam-frame-to-numpy -- get comfortable with the camera<->numpy/OpenCV
boundary before Topic 1's assignments need it.

* ``convert``   -- pure functions operating on numpy arrays: shape/dtype
                    validation, a frame summary, and a channel-order swap.
                    No hardware import -- fully testable off-Pi with
                    synthetic arrays.
* ``hardware``  -- the ONLY module that imports picamera2.
"""
