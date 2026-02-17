"""Operational Profile loader and management.

This module provides utilities for loading and managing OperationalProfiles
that encode user-specific operational knowledge.

The profile is the key differentiator:
- Generated before execution
- Set up by the user
- Enforced at runtime
- External source of truth, not just context
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.governance.models import (
    DatabaseConfig,
    NetworkConfig,
    OperationalProfile,
    PathConfig,
    Procedure,
    ServiceConfig,
)


class ProfileLoader:
    """Loads and manages operational profiles from configuration files."""

    DEFAULT_PROFILE_PATHS = [
        "./.openclaw/profile.json",
        "./operational-profile.json",
        "~/.config/openclaw/profile.json",
        "~/.openclaw/profile.json",
    ]

    def __init__(self, config_dir: str | None = None):
        """Initialize the profile loader.

        Args:
            config_dir: Optional directory to search for profiles.
        """
        self._config_dir = config_dir
        self._profiles: dict[str, OperationalProfile] = {}
        self._default_profile: OperationalProfile | None = None

    def load_from_file(self, path: str) -> OperationalProfile:
        """Load an operational profile from a JSON file.

        Args:
            path: Path to the profile JSON file.

        Returns:
            Loaded OperationalProfile.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file is invalid JSON or missing required fields.
        """
        expanded_path = os.path.expanduser(path)
        file_path = Path(expanded_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Profile not found: {path}")

        try:
            data = json.loads(file_path.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in profile: {e}") from e

        return self._parse_profile(data)

    def load_from_dict(self, data: dict[str, Any]) -> OperationalProfile:
        """Load an operational profile from a dictionary.

        Args:
            data: Profile data as a dictionary.

        Returns:
            Loaded OperationalProfile.
        """
        return self._parse_profile(data)

    def _parse_profile(self, data: dict[str, Any]) -> OperationalProfile:
        """Parse profile data into an OperationalProfile model.

        Args:
            data: Raw profile data.

        Returns:
            Parsed OperationalProfile.
        """
        # Parse nested configs
        paths_data = data.get("paths", {})
        paths = PathConfig(
            working_dirs=paths_data.get("working_dirs", []),
            protected_paths=paths_data.get("protected_paths", []),
            sensitive_patterns=paths_data.get("sensitive_patterns", []),
            scratch_dirs=paths_data.get("scratch_dirs", []),
        )

        db_data = data.get("database", {})
        database = DatabaseConfig(
            protected_tables=db_data.get("protected_tables", []),
            safe_tables=db_data.get("safe_tables", []),
            require_where_clause=db_data.get("require_where_clause", True),
            max_affected_rows=db_data.get("max_affected_rows", 1000),
            production_indicators=db_data.get(
                "production_indicators", ["prod", "production", "live"]
            ),
        )

        services_data = data.get("services", {})
        services = ServiceConfig(
            protected_services=services_data.get("protected_services", []),
            safe_services=services_data.get("safe_services", []),
            forbidden_commands=services_data.get("forbidden_commands", []),
            confirm_commands=services_data.get("confirm_commands", []),
        )

        network_data = data.get("network", {})
        network = NetworkConfig(
            allowed_domains=network_data.get("allowed_domains", []),
            blocked_domains=network_data.get("blocked_domains", []),
            internal_patterns=network_data.get("internal_patterns", []),
        )

        # Parse procedures
        procedures = []
        for proc_data in data.get("procedures", []):
            procedures.append(
                Procedure(
                    name=proc_data["name"],
                    description=proc_data.get("description", ""),
                    required_steps=proc_data.get("required_steps", []),
                    preconditions=proc_data.get("preconditions", []),
                    postconditions=proc_data.get("postconditions", []),
                    rollback_steps=proc_data.get("rollback_steps", []),
                )
            )

        return OperationalProfile(
            profile_id=data.get("profile_id", "default"),
            name=data.get("name", "Default Profile"),
            description=data.get("description"),
            version=data.get("version", "1.0.0"),
            environment=data.get("environment", "development"),
            project_root=data.get("project_root"),
            paths=paths,
            database=database,
            services=services,
            network=network,
            procedures=procedures,
            global_constraints=data.get("global_constraints", []),
            maintenance_windows=data.get("maintenance_windows", []),
            blocked_hours=data.get("blocked_hours", []),
            notify_on_high_risk=data.get("notify_on_high_risk", True),
            notify_channels=data.get("notify_channels", []),
        )

    def discover_profile(self) -> OperationalProfile | None:
        """Discover and load profile from standard locations.

        Searches for profiles in order of precedence:
        1. Current directory (./.openclaw/profile.json)
        2. Current directory (./operational-profile.json)
        3. User config (~/.config/openclaw/profile.json)
        4. User home (~/.openclaw/profile.json)

        Returns:
            OperationalProfile if found, None otherwise.
        """
        for path in self.DEFAULT_PROFILE_PATHS:
            expanded = os.path.expanduser(path)
            if os.path.exists(expanded):
                try:
                    return self.load_from_file(expanded)
                except (ValueError, FileNotFoundError):
                    continue

        return None

    def get_default_profile(self) -> OperationalProfile:
        """Get a minimal default profile for safety.

        Returns:
            A conservative default profile that blocks dangerous operations.
        """
        if self._default_profile:
            return self._default_profile

        self._default_profile = OperationalProfile(
            profile_id="default-safe",
            name="Default Safe Profile",
            description="Conservative defaults when no profile is configured",
            environment="unknown",
            paths=PathConfig(
                protected_paths=[
                    "~",
                    "~/",
                    "/",
                    "/etc",
                    "/var",
                    "/usr",
                    "/bin",
                    "/sbin",
                    "~/.ssh",
                    "~/.aws",
                    "~/.config",
                ],
                sensitive_patterns=[
                    "*.env",
                    "*secret*",
                    "*password*",
                    "*.pem",
                    "*.key",
                ],
            ),
            database=DatabaseConfig(
                protected_tables=["users", "accounts", "payments", "orders"],
                require_where_clause=True,
                max_affected_rows=100,
            ),
            services=ServiceConfig(
                forbidden_commands=[
                    "rm -rf /",
                    "rm -rf ~",
                    "rm -rf /*",
                    "rm -rf ~/",
                ],
                confirm_commands=[
                    "rm -rf",
                    "rm -r",
                    "reboot",
                    "shutdown",
                    "kill",
                    "pkill",
                ],
            ),
            network=NetworkConfig(
                blocked_domains=[
                    "*.pastebin.com",
                    "*.requestbin.com",
                    "*.webhook.site",
                ],
            ),
            global_constraints=[
                "Require confirmation for all destructive operations",
                "Never modify system directories",
                "Never expose secrets",
            ],
            notify_on_high_risk=True,
        )

        return self._default_profile

    def register_profile(self, profile: OperationalProfile) -> None:
        """Register a profile for later retrieval by ID.

        Args:
            profile: The profile to register.
        """
        self._profiles[profile.profile_id] = profile

    def get_profile(self, profile_id: str) -> OperationalProfile | None:
        """Get a registered profile by ID.

        Args:
            profile_id: The profile ID to look up.

        Returns:
            The profile if found, None otherwise.
        """
        return self._profiles.get(profile_id)


# Singleton instance for convenience
_default_loader: ProfileLoader | None = None


def get_profile_loader() -> ProfileLoader:
    """Get the default profile loader instance."""
    global _default_loader
    if _default_loader is None:
        _default_loader = ProfileLoader()
    return _default_loader


def load_profile(path: str | None = None) -> OperationalProfile:
    """Convenience function to load a profile.

    Args:
        path: Optional path to profile. If not provided, discovers from standard locations.

    Returns:
        Loaded profile or default safe profile.
    """
    loader = get_profile_loader()

    if path:
        return loader.load_from_file(path)

    discovered = loader.discover_profile()
    if discovered:
        return discovered

    return loader.get_default_profile()
