"""
WSGI entrypoint for production servers (e.g. Gunicorn).

Run on EC2 (from the ml/ directory):
  gunicorn -c gunicorn.conf.py wsgi:app

parse_known_args() ignores Gunicorn's own CLI flags; app config still comes from
env vars and optional flags you pass before ``wsgi:app``.
"""

from app import app as app  # noqa: F401
from app import configure_app, parse_args

configure_app(parse_args())

