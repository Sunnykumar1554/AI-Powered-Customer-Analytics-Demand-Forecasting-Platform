"""
RetailPulse – Prometheus Metrics Module
Exposes application-level counters and histograms for production monitoring.

Usage:
    from src.metrics import track_request, track_forecast, track_churn, track_inventory

    # In Streamlit page handlers:
    track_request()              # Increment total request counter
    track_forecast()             # Increment forecast request counter
    track_churn(count=5)         # Record batch of churn predictions
    track_inventory()            # Increment inventory check counter

Metrics Endpoint:
    The Prometheus metrics are served at /metrics when using the start_metrics_server() function.
"""

from prometheus_client import Counter, Histogram, start_http_server, REGISTRY
import time
import functools

# ──────────────────────────────────────────────────
# Counters
# ──────────────────────────────────────────────────

REQUESTS_TOTAL = Counter(
    'retailpulse_requests_total',
    'Total number of dashboard page requests',
    ['page']
)

FORECAST_REQUESTS = Counter(
    'retailpulse_forecast_requests_total',
    'Total number of forecast generation requests',
    ['model']
)

CHURN_PREDICTIONS = Counter(
    'retailpulse_churn_predictions_total',
    'Total number of churn predictions made'
)

INVENTORY_REQUESTS = Counter(
    'retailpulse_inventory_requests_total',
    'Total number of inventory optimization requests'
)

# ──────────────────────────────────────────────────
# Histograms
# ──────────────────────────────────────────────────

REQUEST_LATENCY = Histogram(
    'retailpulse_request_duration_seconds',
    'Time spent processing requests',
    ['page'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

MODEL_INFERENCE_LATENCY = Histogram(
    'retailpulse_model_inference_seconds',
    'Time spent on model inference',
    ['model'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

# ──────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────

def track_request(page: str = "home"):
    """Increment the total request counter for a given page."""
    REQUESTS_TOTAL.labels(page=page).inc()


def track_forecast(model: str = "hybrid"):
    """Increment the forecast request counter."""
    FORECAST_REQUESTS.labels(model=model).inc()


def track_churn(count: int = 1):
    """Increment the churn prediction counter."""
    CHURN_PREDICTIONS.inc(count)


def track_inventory():
    """Increment the inventory optimization request counter."""
    INVENTORY_REQUESTS.inc()


def measure_latency(page: str):
    """
    Decorator to measure and record request latency for a page.

    Usage:
        @measure_latency("forecasting")
        def run_forecast():
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start
                REQUEST_LATENCY.labels(page=page).observe(duration)
        return wrapper
    return decorator


def measure_inference(model: str):
    """
    Decorator to measure and record model inference latency.

    Usage:
        @measure_inference("prophet")
        def predict():
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start
                MODEL_INFERENCE_LATENCY.labels(model=model).observe(duration)
        return wrapper
    return decorator


# ──────────────────────────────────────────────────
# Metrics Server
# ──────────────────────────────────────────────────

def start_metrics_server(port: int = 8000):
    """
    Start a Prometheus metrics HTTP server on the specified port.
    Call this once at application startup.

    The metrics will be available at:
        http://localhost:{port}/metrics
    """
    try:
        start_http_server(port)
        print(f"📡 Prometheus metrics server started on port {port}")
    except OSError as e:
        print(f"⚠️ Metrics server could not start on port {port}: {e}")


# ──────────────────────────────────────────────────
# Standalone test
# ──────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick self-test: start server and push sample metrics
    start_metrics_server(8000)

    # Simulate some metrics
    track_request("home")
    track_request("forecasting")
    track_forecast("prophet")
    track_forecast("lstm")
    track_forecast("hybrid")
    track_churn(10)
    track_inventory()

    print("✅ Sample metrics pushed. Visit http://localhost:8000/metrics")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Metrics server stopped.")
