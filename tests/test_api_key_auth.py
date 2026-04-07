import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import Request

import main as app_main
from app.api.routes.projects import router as projects_router
from app.core.dependencies import inspect_api_key, require_api_key
from app.core.errors import AuthenticationError


class ApiKeyDependencyTests(unittest.TestCase):
    def test_inspect_api_key_is_optional_when_backend_has_no_configured_key(self) -> None:
        result = inspect_api_key(
            x_api_key=None,
            settings=SimpleNamespace(api_key=None),
        )

        self.assertFalse(result["required"])
        self.assertTrue(result["valid"])
        self.assertFalse(result["provided"])
        self.assertEqual(result["header_name"], "X-API-Key")

    def test_require_api_key_rejects_missing_key_when_backend_requires_it(self) -> None:
        with patch("app.core.dependencies.get_settings", return_value=SimpleNamespace(api_key="secret")):
            with self.assertRaises(AuthenticationError) as context:
                require_api_key(x_api_key=None)

        self.assertIn("API key", str(context.exception))

    def test_require_api_key_accepts_matching_header(self) -> None:
        with patch("app.core.dependencies.get_settings", return_value=SimpleNamespace(api_key="secret")):
            require_api_key(x_api_key="secret")

    def test_projects_router_declares_api_key_dependency(self) -> None:
        dependency_names = [
            getattr(dependency.dependency, "__name__", "")
            for dependency in projects_router.dependencies
        ]
        self.assertIn("require_api_key", dependency_names)

    def test_health_route_declares_api_key_dependency(self) -> None:
        health_route = next(route for route in app_main.app.routes if route.path == "/health")
        dependency_names = [
            getattr(dependency.call, "__name__", "")
            for dependency in health_route.dependant.dependencies
        ]
        self.assertIn("require_api_key", dependency_names)


class HealthEndpointTests(unittest.TestCase):
    @staticmethod
    def _build_request() -> Request:
        scope = {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/health",
            "raw_path": b"/health",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 54321),
            "server": ("backend-pc", 8000),
        }
        return Request(scope)

    def test_health_reports_network_and_auth_metadata(self) -> None:
        fake_settings = SimpleNamespace(api_key="secret", storage_root=Path("data/projects"))
        fake_network = {
            "hostname": "backend-pc",
            "fqdn": "backend-pc.local",
            "advertised_urls": ["http://backend-pc:8000"],
            "preferred_base_url": "http://backend-pc:8000",
            "observed_base_url": "http://backend-pc:8000",
        }

        with patch.object(app_main, "settings", fake_settings), patch.object(
            app_main,
            "get_processing_service",
            return_value=SimpleNamespace(engine_name="colmap"),
        ), patch.object(app_main, "build_health_network_info", return_value=fake_network):
            payload = app_main.health_check(request=self._build_request())

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["engine"], "colmap")
        self.assertEqual(payload["network"]["preferred_base_url"], "http://backend-pc:8000")
        self.assertTrue(payload["auth"]["required"])
        self.assertEqual(payload["auth"]["header_name"], "X-API-Key")


if __name__ == "__main__":
    unittest.main()
