"""Role-based permission matrix for the dashboard."""

from __future__ import annotations

from typing import Dict, Set

# Mapping of role names to the set of permissions they grant. Permissions
# correspond to tab identifiers or named actions within the application.
ROLE_PERMISSIONS: Dict[str, Set[str]] = {
    "viewer": {
        "overview",
        "comparison",
        "ad-hoc",
        "cycle-detail",
        "eis",
        "document-status",
        "missing-data",
        "doe-heatmap",
        "flags",
    },
    "operator": {
        "overview",
        "new-material",
        "data-import",
        "export",
        "comparison",
        "ad-hoc",
        "cycle-detail",
        "eis",
        "document-status",
        "missing-data",
        "doe-heatmap",
        "flags",
    },
    "admin": {
        "overview",
        "new-material",
        "data-import",
        "export",
        "import-jobs",
        "comparison",
        "advanced-analysis",
        "ad-hoc",
        "cycle-detail",
        "eis",
        "document-status",
        "missing-data",
        "doe-heatmap",
        "trait-filter",
        "flags",
    },
}

__all__ = ["ROLE_PERMISSIONS"]
