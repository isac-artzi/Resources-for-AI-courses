"""cam-ai-inference -- run a demo model on the AI Camera's on-sensor IMX500
NPU and draw the results, so students see the on-chip inference path
(distinct from running a model on the Pi's own CPU).

* ``detections`` -- pure functions: parse/filter/scale raw detection
                     results, no hardware import -- testable off-Pi.
* ``hardware``   -- the ONLY module that imports picamera2's IMX500 devtools.
"""
