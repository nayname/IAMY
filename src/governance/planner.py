"""Plan generation for the governance layer.

This module provides the PlanGenerator class for:
- Building ordered PlannedActions from classified intent
- Extracting resource access patterns
- Calculating risk assessments
- Generating complete ExecutionPlans
- Incorporating user-specific operational knowledge (OperationalProfile)

The execution plan is:
- Generated before execution
- Set up by the user (via operational profile)
- Enforced at runtime

The plan encodes user-specific operational knowledge - paths, procedures,
constraints, configs - and serves as an external guiding source of truth
for execution, not just a non-binding context.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any, List, Dict

from src.governance.models import (
    EnhancedExecutionPlan,
    ExecutionPlan,
    Intent,
    IntentCategory,
    OperationalProfile,
    PlannedAction,
    ResourceAccess,
    RiskAssessment,
    RiskLevel,
    ToolCall,
)
from src.governance.profile import load_profile, ProfileLoader

EXECUTION_PLAN_PROMPT = """
...
"""


class PlanGenerator:
    """Generates execution plans from classified intent.

    Uses patterns config for risk calculation and resource extraction.
    Incorporates operational profiles for user-specific constraints.

    The execution plan encodes user-specific operational knowledge:
    - Paths: working directories, protected paths, sensitive patterns
    - Procedures: standard operating procedures for common tasks
    - Constraints: what's never allowed
    - Configs: environment-specific settings
    """

    # Base risk scores by category
    BASE_RISK: dict[IntentCategory, int] = {
        IntentCategory.FILE_READ: 10,
        IntentCategory.FILE_WRITE: 30,
        IntentCategory.FILE_DELETE: 50,
        IntentCategory.NETWORK_REQUEST: 30,
        IntentCategory.CODE_EXECUTION: 70,
        IntentCategory.SKILL_INVOCATION: 20,
        IntentCategory.SYSTEM_COMMAND: 70,
        IntentCategory.UNKNOWN: 40,
    }

    # Risk level thresholds
    RISK_THRESHOLDS: list[tuple[int, RiskLevel]] = [
        (80, RiskLevel.CRITICAL),
        (60, RiskLevel.HIGH),
        (40, RiskLevel.MEDIUM),
        (20, RiskLevel.LOW),
        (0, RiskLevel.INFO),
    ]

    # Patterns for extracting resources
    PATH_KEYS = {"path", "file", "filepath", "filename", "directory", "dir"}
    URL_KEYS = {"url", "uri", "endpoint", "href"}

    def __init__(
        self,
        patterns_path: str,
        llm,
        profile_path: str | None = None,
    ) -> None:
        """Initialize the plan generator.

        Args:
            patterns_path: Path to the intent-patterns.json config file.
            llm: LLM instance for plan generation.
            profile_path: Optional path to operational profile.
        """
        self._patterns_path = patterns_path
        self._risk_multipliers: dict[str, float] = {}
        self._tool_categories: dict[str, str] = {}  # tool -> category
        self._load_config()
        self.llm = llm

        # Load operational profile
        self._profile_loader = ProfileLoader()
        self._operational_profile: OperationalProfile | None = None
        if profile_path:
            try:
                self._operational_profile = self._profile_loader.load_from_file(profile_path)
            except (FileNotFoundError, ValueError):
                # Fall back to discovery
                self._operational_profile = self._profile_loader.discover_profile()
        else:
            self._operational_profile = self._profile_loader.discover_profile()

    @property
    def operational_profile(self) -> OperationalProfile | None:
        """Get the current operational profile."""
        return self._operational_profile

    def set_operational_profile(self, profile: OperationalProfile) -> None:
        """Set the operational profile for plan generation.

        Args:
            profile: The operational profile to use.
        """
        self._operational_profile = profile

    def load_profile(self, path: str) -> None:
        """Load an operational profile from a file.

        Args:
            path: Path to the profile JSON file.
        """
        self._operational_profile = self._profile_loader.load_from_file(path)


    def generate(
            self,
            *,
            user_message: str,
            context: Dict[str, Any],
            session_id: str | None = None,
            ttl_minutes: int = 10,
            profile: OperationalProfile | None = None,
    ) -> Dict[str, Any]:
        """
        Generate a FULL ExecutionPlan artifact with operational knowledge.

        The plan encodes user-specific operational knowledge - paths, procedures,
        constraints, configs - and serves as an external guiding source of truth
        for execution, not just a non-binding context.

        Args:
            user_message: The user's request.
            context: Additional context for plan generation.
            session_id: Optional session ID for binding.
            ttl_minutes: Plan time-to-live in minutes.
            profile: Optional operational profile (uses instance profile if not provided).

        Returns:
            dict conforming to execution-plan.json schema with operational knowledge.
        """
        # Use provided profile or fall back to instance profile
        active_profile = profile or self._operational_profile

        # ---------- 1. Build context with operational knowledge ----------
        enriched_context = dict(context)
        if active_profile:
            enriched_context["operationalProfile"] = {
                "environment": active_profile.environment,
                "projectRoot": active_profile.project_root,
                "workingDirs": active_profile.paths.working_dirs,
                "protectedPaths": active_profile.paths.protected_paths,
                "protectedTables": active_profile.database.protected_tables,
                "forbiddenCommands": active_profile.services.forbidden_commands,
                "globalConstraints": active_profile.global_constraints,
                "procedures": [
                    {"name": p.name, "description": p.description}
                    for p in active_profile.procedures
                ],
            }

        # ---------- 2. Ask LLM to generate a COMPLETE plan ----------
        raw = self.llm.complete(
            prompt=EXECUTION_PLAN_PROMPT.format(
                user_message=user_message,
                context=json.dumps(enriched_context, indent=2),
            ),
            temperature=0,
        )

        try:
            plan: Dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError("Planner did not return valid JSON") from e

        # ---------- 3. System-owned authoritative fields ----------
        now = datetime.utcnow()

        plan["version"] = "1.0.0"
        plan["planId"] = str(uuid.uuid4())
        plan["createdAt"] = now.isoformat() + "Z"
        plan["expiresAt"] = (now + timedelta(minutes=ttl_minutes)).isoformat() + "Z"

        if session_id is not None:
            plan["sessionId"] = session_id

        # ---------- 4. Normalize intent ----------
        intent = plan.setdefault("intent", {})
        intent.setdefault("userMessage", user_message)

        # Enforce enums defensively
        intent["category"] = intent.get("category", "mixed")
        intent["riskLevel"] = intent.get("riskLevel", "medium")

        # ---------- 5. Normalize operations ----------
        operations = plan.get("operations")
        if not operations:
            raise ValueError("ExecutionPlan must contain operations")

        for idx, op in enumerate(operations, start=1):
            op.setdefault("id", f"op-{idx:03d}")
            op.setdefault("parallel", False)
            op.setdefault("maxInvocations", 1)
            op.setdefault("requiresConfirmation", False)

            # Defensive: ensure allow exists
            if "allow" not in op:
                raise ValueError(f"Operation {op['id']} missing allow rules")

        # ---------- 6. Normalize global constraints ----------
        constraints = plan.setdefault("constraints", {})
        constraints.setdefault("allowUnplanned", False)
        constraints.setdefault("requireSequential", False)
        constraints.setdefault("maxTotalOperations", len(operations))
        constraints.setdefault("maxDurationMs", 300000)

        # ---------- 7. Merge operational profile constraints ----------
        if active_profile:
            # Add forbidden paths from profile
            forbidden_paths = constraints.setdefault("forbiddenPaths", [])
            forbidden_paths.extend(active_profile.paths.protected_paths)
            constraints["forbiddenPaths"] = list(set(forbidden_paths))

            # Add forbidden commands from profile
            forbidden_commands = constraints.setdefault("forbiddenCommands", [])
            forbidden_commands.extend(active_profile.services.forbidden_commands)
            constraints["forbiddenCommands"] = list(set(forbidden_commands))

            # Add profile reference to metadata
            plan.setdefault("metadata", {})["operationalProfileId"] = active_profile.profile_id
            plan["metadata"]["operationalProfileVersion"] = active_profile.version
            plan["metadata"]["environment"] = active_profile.environment

            # Store full profile summary for audit
            plan["operationalKnowledge"] = {
                "profileId": active_profile.profile_id,
                "profileName": active_profile.name,
                "environment": active_profile.environment,
                "workingDirs": active_profile.paths.working_dirs,
                "protectedPaths": active_profile.paths.protected_paths[:10],  # Limit for size
                "protectedTables": active_profile.database.protected_tables,
                "globalConstraints": active_profile.global_constraints,
                "appliedProcedure": self._find_matching_procedure(
                    user_message, active_profile
                ),
            }

        # ---------- 8. Metadata enrichment ----------
        metadata = plan.setdefault("metadata", {})
        metadata.setdefault(
            "generatedBy",
            getattr(self.llm, "model_name", "unknown"),
        )
        metadata.setdefault(
            "qualityScore",
            self._estimate_quality(plan),
        )

        return plan

    def _find_matching_procedure(
        self, user_message: str, profile: OperationalProfile
    ) -> Dict[str, Any] | None:
        """Find a procedure that matches the user's intent.

        Args:
            user_message: The user's request.
            profile: The operational profile to search.

        Returns:
            Matching procedure summary or None.
        """
        user_lower = user_message.lower()

        for proc in profile.procedures:
            # Simple keyword matching - could be enhanced with LLM
            proc_keywords = proc.name.lower().replace("_", " ").split()
            if any(kw in user_lower for kw in proc_keywords):
                return {
                    "name": proc.name,
                    "description": proc.description,
                    "requiredSteps": proc.required_steps,
                    "preconditions": proc.preconditions,
                }

        return None


    def _estimate_quality(self, plan: Dict[str, Any]) -> int:
        """
        Crude heuristic: more explicit plans score higher.
        """
        score = 100

        for op in plan.get("operations", []):
            if not op.get("deny"):
                score -= 5
            if not op.get("requires"):
                score -= 3

        if plan.get("constraints", {}).get("allowUnplanned"):
            score -= 15

        return max(0, min(100, score))

    def _load_config(self) -> None:
        """Load configuration from patterns file."""
        path = Path(self._patterns_path)
        if path.exists():
            config = json.loads(path.read_text())
            self._risk_multipliers = config.get("risk_multipliers", {})

            # Build reverse mapping: tool -> category
            for category, tools in config.get("tool_categories", {}).items():
                for tool in tools:
                    self._tool_categories[tool.lower()] = category

    def _categorize_tool(self, tool_name: str) -> IntentCategory:
        """Get the category for a tool name."""
        category_str = self._tool_categories.get(tool_name.lower())
        if category_str:
            try:
                return IntentCategory(category_str)
            except ValueError:
                pass
        return IntentCategory.UNKNOWN

    def _extract_resources(self, tool_call: ToolCall) -> list[ResourceAccess]:
        """Extract resource access patterns from a tool call.

        Args:
            tool_call: The tool call to analyze.

        Returns:
            List of ResourceAccess objects for resources accessed.
        """
        resources: list[ResourceAccess] = []
        tool_lower = tool_call.name.lower()

        # Determine operation from tool name
        if "read" in tool_lower or "get" in tool_lower or "list" in tool_lower:
            operation = "read"
        elif "write" in tool_lower or "create" in tool_lower or "save" in tool_lower:
            operation = "write"
        elif "delete" in tool_lower or "remove" in tool_lower:
            operation = "delete"
        elif "http" in tool_lower or "fetch" in tool_lower or "api" in tool_lower:
            operation = "fetch"
        elif "execute" in tool_lower or "run" in tool_lower:
            operation = "execute"
        else:
            operation = "access"

        # Extract resources from arguments
        self._extract_from_dict(tool_call.arguments, resources, operation)

        return resources

    def _extract_from_dict(
        self,
        data: dict[str, Any],
        resources: list[ResourceAccess],
        operation: str,
    ) -> None:
        """Recursively extract resources from a dictionary."""
        for key, value in data.items():
            key_lower = key.lower()

            if isinstance(value, str):
                # Check for file paths
                if key_lower in self.PATH_KEYS or self._looks_like_path(value):
                    resources.append(
                        ResourceAccess(type="file", path=value, operation=operation)
                    )
                # Check for URLs
                elif key_lower in self.URL_KEYS or self._looks_like_url(value):
                    resources.append(
                        ResourceAccess(type="url", path=value, operation=operation)
                    )

            elif isinstance(value, dict):
                self._extract_from_dict(value, resources, operation)

            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        self._extract_from_dict(item, resources, operation)
                    elif isinstance(item, str):
                        if self._looks_like_path(item):
                            resources.append(
                                ResourceAccess(type="file", path=item, operation=operation)
                            )
                        elif self._looks_like_url(item):
                            resources.append(
                                ResourceAccess(type="url", path=item, operation=operation)
                            )

    def _looks_like_path(self, value: str) -> bool:
        """Check if a string looks like a file path."""
        return value.startswith("/") or value.startswith("./") or "\\" in value

    def _looks_like_url(self, value: str) -> bool:
        """Check if a string looks like a URL."""
        return bool(re.match(r"^https?://", value, re.IGNORECASE))

    def _build_actions(self, intent: Intent) -> list[PlannedAction]:
        """Build ordered PlannedActions from intent.

        Args:
            intent: The classified intent.

        Returns:
            List of PlannedAction objects with sequential ordering.
        """
        actions: list[PlannedAction] = []

        for sequence, tool_call in enumerate(intent.tool_calls):
            category = self._categorize_tool(tool_call.name)
            resources = self._extract_resources(tool_call)

            # Calculate risk score
            base_risk = self.BASE_RISK.get(category, 40)
            multiplier = self._risk_multipliers.get(category.value, 1.0)
            risk_score = min(int(base_risk * multiplier), 100)

            actions.append(
                PlannedAction(
                    sequence=sequence,
                    tool_call=tool_call,
                    category=category,
                    resources=resources,
                    risk_score=risk_score,
                )
            )

        return actions

    def _assess_risk(self, actions: list[PlannedAction]) -> RiskAssessment:
        """Calculate risk assessment for a set of actions.

        Args:
            actions: List of planned actions to assess.

        Returns:
            RiskAssessment with overall score, level, and factors.
        """
        if not actions:
            return RiskAssessment(
                overall_score=0,
                level=RiskLevel.INFO,
                factors=[],
                mitigations=[],
            )

        # Calculate overall score (max of individual scores, with small additive factor)
        max_score = max(a.risk_score for a in actions)
        additive = min(len(actions) - 1, 5) * 2  # Small bonus for multiple actions
        overall_score = min(max_score + additive, 100)

        # Determine risk level
        level = RiskLevel.INFO
        for threshold, risk_level in self.RISK_THRESHOLDS:
            if overall_score >= threshold:
                level = risk_level
                break

        # Identify risk factors
        factors: list[str] = []
        categories_seen = set()
        for action in actions:
            if action.category not in categories_seen:
                categories_seen.add(action.category)
                if action.category != IntentCategory.UNKNOWN:
                    factors.append(action.category.value)

        # Suggest mitigations
        mitigations: list[str] = []
        if level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            mitigations.append("requires_approval")
        if IntentCategory.CODE_EXECUTION in categories_seen:
            mitigations.append("sandbox_execution")
        if IntentCategory.NETWORK_REQUEST in categories_seen:
            mitigations.append("url_allowlist")

        return RiskAssessment(
            overall_score=overall_score,
            level=level,
            factors=factors,
            mitigations=mitigations,
        )

    def generate(
        self,
        intent: Intent,
        request_body: dict[str, Any],
        session_id: str | None = None,
    ) -> ExecutionPlan:
        """Generate an execution plan from classified intent.

        Args:
            intent: The classified intent for the request.
            request_body: The original request body for hashing.
            session_id: Optional session ID for tracking.

        Returns:
            A complete ExecutionPlan with actions and risk assessment.
        """
        # Generate unique plan ID
        plan_id = str(uuid.uuid4())

        # Compute request hash
        request_json = json.dumps(request_body, sort_keys=True)
        request_hash = hashlib.sha256(request_json.encode()).hexdigest()

        # Build actions
        actions = self._build_actions(intent)

        # Assess risk
        risk_assessment = self._assess_risk(actions)

        return ExecutionPlan(
            plan_id=plan_id,
            session_id=session_id,
            request_hash=request_hash,
            actions=actions,
            risk_assessment=risk_assessment,
        )
