import os

# Bind to localhost by default; put Nginx in front for public traffic.
bind = os.getenv("BIND", "127.0.0.1:5000")

# Keep 1 worker because the app maintains per-client tracking/smoothing state in memory.
workers = int(os.getenv("WEB_CONCURRENCY", "1"))

# Inference can take time on CPU.
timeout = int(os.getenv("TIMEOUT", "180"))

accesslog = "-"
errorlog = "-"

