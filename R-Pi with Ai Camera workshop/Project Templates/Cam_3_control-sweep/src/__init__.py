"""cam-control-sweep -- sweep exposure/gain/white-balance and see the effect.

* ``sweep``     -- pure functions: generate sweep values, plan a contact-
                    sheet grid layout. No hardware import -- testable off-Pi.
* ``hardware``  -- the ONLY module that imports picamera2 (and cv2 for
                    assembling the output grid image).
"""
