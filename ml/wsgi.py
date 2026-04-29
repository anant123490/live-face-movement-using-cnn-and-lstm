"""
WSGI entrypoint for production servers (e.g. Gunicorn).

Run on EC2:
  gunicorn -c gunicorn.conf.py wsgi:app
"""

from app import app as app  # noqa: F401

