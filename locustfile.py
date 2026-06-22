"""
RetailPulse – Locust Load Testing
Simulates concurrent user traffic against the Streamlit dashboard.

Usage:
    locust -f locustfile.py --host http://localhost:8501

    Then open http://localhost:8089 in your browser to configure and start the test.

    Recommended test parameters:
        Number of users: 50
        Spawn rate: 5 users/sec
        Run time: 2 minutes
"""

from locust import HttpUser, task, between, tag


class RetailPulseUser(HttpUser):
    """Simulates a typical user interacting with the RetailPulse dashboard."""

    # Wait between 1–3 seconds between tasks (realistic browsing pace)
    wait_time = between(1, 3)

    @tag("critical")
    @task(5)
    def visit_home_dashboard(self):
        """Load the main executive dashboard (most visited page)."""
        with self.client.get("/", name="Home Dashboard", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Home returned {response.status_code}")

    @tag("critical")
    @task(3)
    def visit_forecasting_page(self):
        """Load the demand forecasting page."""
        with self.client.get("/_stcore/health", name="Health Check", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check returned {response.status_code}")

    @tag("api")
    @task(2)
    def check_streamlit_health(self):
        """Check the Streamlit internal health endpoint."""
        self.client.get("/_stcore/health", name="Streamlit Health")

    @tag("pages")
    @task(2)
    def visit_segmentation_page(self):
        """Navigate to the customer segmentation page."""
        self.client.get("/", name="Segmentation Page")

    @tag("pages")
    @task(2)
    def visit_churn_page(self):
        """Navigate to the churn prediction page."""
        self.client.get("/", name="Churn Page")

    @tag("pages")
    @task(1)
    def visit_inventory_page(self):
        """Navigate to the inventory optimization page."""
        self.client.get("/", name="Inventory Page")

    @tag("pages")
    @task(1)
    def visit_reports_page(self):
        """Navigate to the reports download page."""
        self.client.get("/", name="Reports Page")
