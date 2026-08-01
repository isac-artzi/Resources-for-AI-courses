"""
Build step 3 says "print and record the observation and action space in the
README". This is that command:

    python -m envs

Paste the output into the README. It exists as a module rather than as a
docstring instruction because a number you copied from the documentation and a
number your installed Gymnasium actually reports are not the same kind of
evidence.
"""

from __future__ import annotations

from envs import describe

if __name__ == "__main__":
    print(describe())
