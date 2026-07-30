"""cam-live-preview -- a live video window on the Pi's own screen, using
picamera2's built-in QTGL preview (the same mechanism behind the demos
the AI Camera ships with).

* ``timing``    -- pure functions: format a countdown/elapsed display.
                    No hardware import -- fully testable off-Pi.
* ``hardware``  -- the ONLY module that imports picamera2. Opens a live
                    preview window and keeps it open for a set duration.
"""
