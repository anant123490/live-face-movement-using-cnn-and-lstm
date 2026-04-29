#!/bin/bash
set -e

# SageMaker sets the container port to 8080 by default for inference endpoints.
exec gunicorn -w 1 -b 0.0.0.0:8080 flask_app:app
