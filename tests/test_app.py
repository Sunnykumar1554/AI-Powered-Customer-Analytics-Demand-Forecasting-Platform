"""
RetailPulse – Smoke Tests
Validates core imports, path resolution, and model file presence.
"""

import os
import sys
import importlib

# ── Resolve project root ──
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)


# ── Test 1: Core package imports ──
class TestImports:
    """Ensure all critical Python packages are importable."""

    def test_import_streamlit(self):
        importlib.import_module("streamlit")

    def test_import_pandas(self):
        importlib.import_module("pandas")

    def test_import_numpy(self):
        importlib.import_module("numpy")

    def test_import_sklearn(self):
        importlib.import_module("sklearn")

    def test_import_plotly(self):
        importlib.import_module("plotly")

    def test_import_xgboost(self):
        importlib.import_module("xgboost")


# ── Test 2: Project structure ──
class TestProjectStructure:
    """Verify essential directories and files exist."""

    def test_streamlit_app_exists(self):
        assert os.path.isfile(os.path.join(ROOT_DIR, "streamlit_app.py"))

    def test_requirements_exists(self):
        assert os.path.isfile(os.path.join(ROOT_DIR, "requirements.txt"))

    def test_dockerfile_exists(self):
        assert os.path.isfile(os.path.join(ROOT_DIR, "Dockerfile"))

    def test_pages_directory_exists(self):
        assert os.path.isdir(os.path.join(ROOT_DIR, "pages"))

    def test_src_directory_exists(self):
        assert os.path.isdir(os.path.join(ROOT_DIR, "src"))

    def test_data_directory_exists(self):
        assert os.path.isdir(os.path.join(ROOT_DIR, "data"))

    def test_models_directory_exists(self):
        assert os.path.isdir(os.path.join(ROOT_DIR, "models"))


# ── Test 3: Model files ──
class TestModelFiles:
    """Check that trained model artifacts are present."""

    MODELS_DIR = os.path.join(ROOT_DIR, "models")

    def test_prophet_model(self):
        assert os.path.isfile(os.path.join(self.MODELS_DIR, "prophet_model.pkl"))

    def test_lstm_model(self):
        assert os.path.isfile(os.path.join(self.MODELS_DIR, "lstm_model.pth"))

    def test_churn_model(self):
        assert os.path.isfile(os.path.join(self.MODELS_DIR, "churn_model.pkl"))

    def test_xgboost_model(self):
        assert os.path.isfile(os.path.join(self.MODELS_DIR, "best_xgboost.pkl"))

    def test_hybrid_model(self):
        assert os.path.isfile(os.path.join(self.MODELS_DIR, "hybrid_model.pkl"))


# ── Test 4: Kubernetes manifests ──
class TestKubernetesFiles:
    """Verify Kubernetes deployment files are present and valid YAML."""

    K8S_DIR = os.path.join(ROOT_DIR, "k8s")

    def test_deployment_yaml_exists(self):
        assert os.path.isfile(os.path.join(self.K8S_DIR, "deployment.yaml"))

    def test_service_yaml_exists(self):
        assert os.path.isfile(os.path.join(self.K8S_DIR, "service.yaml"))


# ── Test 5: CI/CD workflow ──
class TestCICDWorkflow:
    """Verify GitHub Actions workflow is present."""

    def test_ci_workflow_exists(self):
        path = os.path.join(ROOT_DIR, ".github", "workflows", "ci.yml")
        assert os.path.isfile(path)


# ── Test 6: Dashboard pages ──
class TestDashboardPages:
    """Ensure all Streamlit pages are present."""

    PAGES_DIR = os.path.join(ROOT_DIR, "pages")

    def test_forecasting_page(self):
        assert os.path.isfile(os.path.join(self.PAGES_DIR, "1_Forecasting.py"))

    def test_segmentation_page(self):
        assert os.path.isfile(os.path.join(self.PAGES_DIR, "2_Segmentation.py"))

    def test_churn_page(self):
        assert os.path.isfile(os.path.join(self.PAGES_DIR, "3_Churn.py"))

    def test_inventory_page(self):
        assert os.path.isfile(os.path.join(self.PAGES_DIR, "4_Inventory.py"))

    def test_reports_page(self):
        assert os.path.isfile(os.path.join(self.PAGES_DIR, "5_Reports.py"))
