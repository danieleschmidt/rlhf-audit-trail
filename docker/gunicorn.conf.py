"""
Gunicorn configuration for RLHF Audit Trail production deployment.
Optimized for performance, reliability, and observability.
"""

import multiprocessing
import os
from pathlib import Path

# =============================================================================
# Server Socket
# =============================================================================

# The socket to bind
bind = f"{os.environ.get('RLHF_AUDIT_HOST', '0.0.0.0')}:{os.environ.get('RLHF_AUDIT_PORT', '8000')}"

# The number of pending connections
backlog = 2048

# =============================================================================
# Worker Processes  
# =============================================================================

# The number of worker processes for handling requests
workers = int(os.environ.get('RLHF_AUDIT_WORKERS', multiprocessing.cpu_count() * 2 + 1))

# The type of workers to run
worker_class = "uvicorn.workers.UvicornWorker"

# The maximum number of simultaneous clients per worker
worker_connections = 1000

# The maximum number of requests a worker will process before restarting
max_requests = int(os.environ.get('GUNICORN_MAX_REQUESTS', '1000'))

# Randomize max_requests to prevent thundering herd
max_requests_jitter = int(os.environ.get('GUNICORN_MAX_REQUESTS_JITTER', '50'))

# Workers silent for more than this many seconds are killed and restarted
timeout = int(os.environ.get('GUNICORN_TIMEOUT', '120'))

# Timeout for graceful workers restart
graceful_timeout = int(os.environ.get('GUNICORN_GRACEFUL_TIMEOUT', '30'))

# The number of seconds to wait for requests on a Keep-Alive connection
keepalive = int(os.environ.get('GUNICORN_KEEPALIVE', '5'))

# =============================================================================
# Logging
# =============================================================================

# The log level
loglevel = os.environ.get('RLHF_AUDIT_LOG_LEVEL', 'info').lower()

# The Access log file to write to
accesslog = os.environ.get('GUNICORN_ACCESS_LOG', '/app/logs/gunicorn_access.log')

# The Error log file to write to  
errorlog = os.environ.get('GUNICORN_ERROR_LOG', '/app/logs/gunicorn_error.log')

# The access log format
access_log_format = (
    '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s '
    '"%(f)s" "%(a)s" %(D)s %(p)s'
)

# Capture stdout/stderr in log files
capture_output = True

# Enable stdio inheritance
enable_stdio_inheritance = True

# =============================================================================
# Process Naming
# =============================================================================

# A base to use with setproctitle for process naming
proc_name = "rlhf-audit-trail"

# =============================================================================
# Server Mechanics
# =============================================================================

# Daemonize the Gunicorn process
daemon = False

# The path to a pid file to write
pidfile = os.environ.get('GUNICORN_PID_FILE', '/tmp/gunicorn.pid')

# A filename to use for the PID file
user = None
group = None

# Redirect stdout/stderr to errorlog
#capture_output = True

# The granularity of Error log outputs
#spew = False

# Disable redirect access logs to syslog
disable_redirect_access_to_syslog = True

# =============================================================================
# SSL
# =============================================================================

# SSL keyfile path
keyfile = os.environ.get('GUNICORN_SSL_KEYFILE')

# SSL certificate file path  
certfile = os.environ.get('GUNICORN_SSL_CERTFILE')

# SSL certificate authority file path
ca_certs = os.environ.get('GUNICORN_SSL_CA_CERTS')

# Whether client certificate is required
cert_reqs = int(os.environ.get('GUNICORN_SSL_CERT_REQS', '0'))

# SSL ciphers to use
ciphers = os.environ.get('GUNICORN_SSL_CIPHERS', 'HIGH:!aNULL:!eNULL:!EXPORT:!DES:!RC4:!MD5:!PSK:!SRP:!CAMELLIA')

# =============================================================================
# Security
# =============================================================================

# Limit request line size
limit_request_line = int(os.environ.get('GUNICORN_LIMIT_REQUEST_LINE', '4094'))

# Limit the allowed size of an HTTP request header field
limit_request_field_size = int(os.environ.get('GUNICORN_LIMIT_REQUEST_FIELD_SIZE', '8190'))

# Limit the number of headers in a request
limit_request_fields = int(os.environ.get('GUNICORN_LIMIT_REQUEST_FIELDS', '100'))

# =============================================================================
# Application Specific Settings  
# =============================================================================

# The application module and callable
# This is set via command line, but we can provide a default
wsgi_application = None

# Environment variables to pass to workers
raw_env = [
    f'PYTHONPATH=/app/src',
    f'RLHF_AUDIT_ENV={os.environ.get("RLHF_AUDIT_ENV", "production")}',
    f'RLHF_AUDIT_LOG_LEVEL={os.environ.get("RLHF_AUDIT_LOG_LEVEL", "INFO")}',
]

# =============================================================================
# Hooks
# =============================================================================

def on_starting(server):
    """Called just before the master process is initialized."""
    server.log.info("Starting RLHF Audit Trail server")
    
    # Ensure log directory exists
    log_dir = Path('/app/logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Log configuration
    server.log.info(f"Workers: {workers}")
    server.log.info(f"Worker class: {worker_class}")
    server.log.info(f"Bind: {bind}")
    server.log.info(f"Timeout: {timeout}s")
    server.log.info(f"Max requests: {max_requests}")


def on_reload(server):
    """Called to recycle workers during a reload via SIGHUP."""
    server.log.info("Reloading RLHF Audit Trail server")


def when_ready(server):
    """Called just after the server is started."""
    server.log.info("RLHF Audit Trail server is ready. Accepting connections")


def worker_int(worker):
    """Called just after a worker exited on SIGINT or SIGQUIT."""
    worker.log.info(f"Worker {worker.pid} received INT or QUIT signal")


def pre_fork(server, worker):
    """Called just before a worker is forked."""
    server.log.debug(f"About to fork worker {worker.age}")


def post_fork(server, worker):
    """Called just after a worker has been forked."""
    server.log.info(f"Worker spawned (pid: {worker.pid})")


def post_worker_init(worker):
    """Called just after a worker has initialized the application."""
    worker.log.info(f"Worker {worker.pid} initialized successfully")


def worker_abort(worker):
    """Called when a worker received the SIGABRT signal."""
    worker.log.warning(f"Worker {worker.pid} received SIGABRT signal")


def pre_exec(server):
    """Called just before a new master process is forked."""
    server.log.info("Forked child, re-executing")


def pre_request(worker, req):
    """Called just before a worker processes the request."""
    # Only log debug info in development
    if os.environ.get('RLHF_AUDIT_ENV') == 'development':
        worker.log.debug(f"{worker.pid} - Processing request: {req.uri}")


def post_request(worker, req, environ, resp):
    """Called after a worker processes the request."""
    # Log slow requests
    request_time = resp.response_length if hasattr(resp, 'response_length') else 0
    if request_time > 5.0:  # Log requests taking more than 5 seconds
        worker.log.warning(f"Slow request: {req.uri} took {request_time:.2f}s")


def child_exit(server, worker):
    """Called just after a worker has been reaped."""
    server.log.info(f"Worker {worker.pid} exited")


def worker_exit(server, worker):
    """Called just after a worker has been exited."""
    server.log.info(f"Worker {worker.pid} exit")


def nworkers_changed(server, new_value, old_value):
    """Called just after num_workers has been changed."""
    server.log.info(f"Number of workers changed from {old_value} to {new_value}")


def on_exit(server):
    """Called just before exiting Gunicorn."""
    server.log.info("RLHF Audit Trail server shutting down")


# =============================================================================
# Development vs Production Settings
# =============================================================================

if os.environ.get('RLHF_AUDIT_ENV') == 'development':
    # Development settings
    workers = 1
    reload = True
    loglevel = 'debug'
    timeout = 0  # Disable timeout in development
    
elif os.environ.get('RLHF_AUDIT_ENV') == 'production':
    # Production settings
    preload_app = True  # Load application code before forking workers
    max_requests = 1000  # Restart workers after handling this many requests
    max_requests_jitter = 100
    
    # Enable detailed access logging in production
    access_log_format = (
        '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s '
        '"%(f)s" "%(a)s" %(D)s %(p)s "%(L)s"'
    )

# =============================================================================
# Health Check Integration
# =============================================================================

def post_worker_init(worker):
    """Initialize worker with health check capabilities."""
    worker.log.info(f"Worker {worker.pid} initialized")
    
    # Set worker-specific environment variables
    os.environ['WORKER_PID'] = str(worker.pid)
    
    # Initialize application monitoring
    try:
        # Import and initialize monitoring if available
        import sys
        sys.path.insert(0, '/app/src')
        
        # Any worker-specific initialization can go here
        worker.log.debug("Worker monitoring initialized")
        
    except Exception as e:
        worker.log.warning(f"Failed to initialize worker monitoring: {e}")