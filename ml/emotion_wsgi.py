"""
WSGI entry for the emotion-only API (Keras HDF5).

  gunicorn -b 0.0.0.0:5001 emotion_wsgi:app

Configuration is applied on import (env vars and parse_known_args-compatible CLI).
"""

from emotion_server import app  # noqa: F401
