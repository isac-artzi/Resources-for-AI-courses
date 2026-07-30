"""cam-hello -- the "hello world" of the Pi AI Camera fundamentals series.

The package is split so the parts that DON'T need a physical camera are
easy to unit-test on any machine:

* ``metadata``  -- pure functions: format capture metadata, build filenames.
                    No hardware import here -- fully testable off-Pi.
* ``hardware``  -- the ONLY module that imports picamera2. Talks to the
                    real sensor, calls into ``metadata`` for the parts
                    that don't need hardware.
"""
