"""Governance layer Pydantic models.

This module defines all data models for the governance layer including:
- Intent classification (IntentCategory, Intent, IntentSignal)
- Plan generation (ExecutionPlan, PlannedAction, RiskAssessment)
- Policy validation (PolicyRule, PolicyViolation, ValidationResult)
- Approval flow (ApprovalRequest, ApprovalStatus)
- Session management (Session)
- Token handling (PlanToken)
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.models import RiskLevel, Severity

# --- Enums ---


class IntentCategory(str, Enum):
    """Categories for classifying tool call intent."""

    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    NETWORK_REQUEST = "network_request"
    CODE_EXECUTION = "code_execution"
    SKILL_INVOCATION = "skill_invocation"
    SYSTEM_COMMAND = "system_command"
    UNKNOWN = "unknown"


class GovernanceDecision(str, Enum):
    """Possible governance decisions for a request."""

    ALLOW = "allow"
    BLOCK = "block"
    REQUIRE_APPROVAL = "require_approval"
    RATE_LIMITED = "rate_limited"


class ApprovalStatus(str, Enum):
    """Status of an approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PolicyType(str, Enum):
    """Types of governance policies."""

    ACTION = "action"
    RESOURCE = "resource"
    SEQUENCE = "sequence"
    RATE = "rate"
    CONTEXT = "context"


class PolicyEffect(str, Enum):
    """Effect of a policy rule."""

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


# --- Core Models ---


class ToolCall(BaseModel):
    """Represents a tool call extracted from a request."""

    model_config = ConfigDict(frozen=True)

    name: str
    arguments: dict[str, object]
    id: str | None = None


class IntentSignal(BaseModel):
    """A signal contributing to intent classification."""

    model_config = ConfigDict(frozen=True)

    category: IntentCategory
    confidence: float = Field(ge=0.0, le=1.0)
    source: str
    details: str | None = None


class Intent(BaseModel):
    """Classified intent for a request."""

    model_config = ConfigDict(frozen=True)

    primary_category: IntentCategory
    signals: list[IntentSignal]
    tool_calls: list[ToolCall]
    confidence: float = Field(ge=0.0, le=1.0)


class ResourceAccess(BaseModel):
    """Represents a resource being accessed by a tool call."""

    model_config = ConfigDict(frozen=True)

    type: str  # "file", "url", "api", etc.
    path: str
    operation: str  # "read", "write", "delete", "fetch", etc.


class PlannedAction(BaseModel):
    """A single action in an execution plan."""

    model_config = ConfigDict(frozen=True)

    sequence: int = Field(ge=0)
    tool_call: ToolCall
    category: IntentCategory
    resources: list[ResourceAccess]
    risk_score: int = Field(ge=0, le=100)


class RiskAssessment(BaseModel):
    """Risk assessment for an execution plan."""

    model_config = ConfigDict(frozen=True)

    overall_score: int = Field(ge=0, le=100)
    level: RiskLevel
    factors: list[str]
    mitigations: list[str]


class ExecutionPlan(BaseModel):
    """An auditable execution plan for a request."""

    model_config = ConfigDict(frozen=True)

    plan_id: str
    session_id: str | None
    request_hash: str = Field(min_length=64, max_length=64)
    actions: list[PlannedAction]
    risk_assessment: RiskAssessment


class PlanToken(BaseModel):
    """A signed token for plan verification."""

    model_config = ConfigDict(frozen=True)

    plan_id: str
    issued_at: str
    expires_at: str
    signature: str


# --- Policy Models ---


class PolicyRule(BaseModel):
    """A governance policy rule."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    type: PolicyType
    effect: PolicyEffect
    conditions: dict[str, object]
    priority: int = 0


class PolicyViolation(BaseModel):
    """A policy violation detected during validation."""

    model_config = ConfigDict(frozen=True)

    rule_id: str
    severity: Severity
    action_sequence: int | None
    message: str


class ValidationResult(BaseModel):
    """Result of policy validation."""

    model_config = ConfigDict(frozen=True)

    valid: bool
    violations: list[PolicyViolation]
    decision: GovernanceDecision
    approval_required: bool


# --- Approval Models ---


class ApprovalRequest(BaseModel):
    """A request for human approval."""

    model_config = ConfigDict(frozen=True)

    approval_id: str
    plan_id: str
    requester_id: str
    status: ApprovalStatus
    requested_at: str
    expires_at: str
    violations: list[PolicyViolation] = Field(default_factory=list)
    original_request: dict[str, object] | None = None
    acknowledgment: str | None = None
    reason: str | None = None


# --- Session Models ---


class Session(BaseModel):
    """Session tracking for multi-turn conversations."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    created_at: str
    last_activity: str
    action_count: int = Field(ge=0)
    risk_accumulator: int = Field(ge=0)


class ExecutionMode(str, Enum):
    """How the plan should be executed."""

    # Governance drives execution, calls tools directly
    GOVERNANCE_DRIVEN = "governance_driven"

    # Plan is injected into LLM context, LLM executes
    AGENT_GUIDED = "agent_guided"

    # Hybrid: governance executes, LLM consulted for decisions
    HYBRID = "hybrid"


class StepStatus(str, Enum):
    """Status of a single execution step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"  # Blocked by governance
    AWAITING_APPROVAL = "awaiting_approval"
    RECOVERED = "recovered"  # Failed but recovered


class RecoveryStrategy(str, Enum):
    """Strategy for handling step failures."""

    FAIL_FAST = "fail_fast"  # Stop execution immediately
    RETRY = "retry"  # Retry the same step
    SKIP = "skip"  # Skip and continue
    ALTERNATIVE = "alternative"  # Try alternative step
    REPLAN = "replan"  # Generate new sub-plan
    HUMAN_INTERVENTION = "human_intervention"  # Wait for human


class StepResult(BaseModel):
    """Outcome of executing a single step."""

    model_config = ConfigDict(frozen=True)

    sequence: int = Field(ge=0)
    status: StepStatus
    started_at: str
    completed_at: str | None = None
    duration_ms: int | None = Field(default=None, ge=0)

    # Tool execution details
    tool_name: str
    tool_args: dict[str, Any]
    tool_result: Any | None = None
    error: str | None = None

    # Governance checks
    governance_decision: GovernanceDecision | None = None
    governance_reason: str | None = None

    # Recovery details
    retry_count: int = Field(default=0, ge=0)
    recovery_action: RecoveryStrategy | None = None


class ExecutionContext(BaseModel):
    """Runtime context passed through execution."""

    model_config = ConfigDict(frozen=True)

    plan_id: str
    session_id: str
    user_id: str
    token: str

    # Execution configuration
    mode: ExecutionMode = ExecutionMode.GOVERNANCE_DRIVEN
    max_retries: int = Field(default=3, ge=0)
    timeout_seconds: int = Field(default=300, ge=1)
    fail_on_governance_block: bool = True

    # User-provided operational knowledge
    constraints: list[str] = Field(default_factory=list)
    preferences: dict[str, Any] = Field(default_factory=dict)


class EnhancedExecutionPlan(BaseModel):
    """Execution plan with conditionals, recovery, and operational knowledge.

    Extends the base ExecutionPlan with:
    - Conditional branches
    - Recovery paths
    - User-provided constraints
    - Execution mode preferences
    """

    model_config = ConfigDict(frozen=True)

    # Base plan
    base_plan: ExecutionPlan

    # Enhanced execution semantics
    # conditionals: list[ConditionalBranch] = Field(default_factory=list)
    # recovery_paths: list[RecoveryPath] = Field(default_factory=list)

    # User operational knowledge
    description: str | None = None  # Human-readable description
    constraints: list[str] = Field(default_factory=list)  # Must-have constraints
    preferences: list[str] = Field(default_factory=list)  # Nice-to-have preferences

    # Execution configuration
    default_mode: ExecutionMode = ExecutionMode.GOVERNANCE_DRIVEN
    allow_mode_override: bool = False
    require_preview: bool = True  # User must preview before execution

    @property
    def plan_id(self) -> str:
        return self.base_plan.plan_id

    @property
    def actions(self) -> list[PlannedAction]:
        return self.base_plan.actions

    def validate_action_against_profile(self, action: PlannedAction) -> list[str]:
        """Validate a planned action against the operational profile.

        Returns list of violation messages (empty if valid).
        """
        violations = []

        if not self.operational_profile:
            return violations

        profile = self.operational_profile

        # Check path constraints
        for resource in action.resources:
            if resource.type == "file":
                if not profile.is_path_allowed(resource.path):
                    violations.append(
                        f"Path '{resource.path}' is not allowed by operational profile"
                    )

        # Check command constraints if this is a system command
        if action.category == IntentCategory.SYSTEM_COMMAND:
            # Extract command from tool call arguments
            cmd = action.tool_call.arguments.get("command", "")
            if isinstance(cmd, str) and profile.is_command_forbidden(cmd):
                violations.append(
                    f"Command is forbidden by operational profile"
                )

        # Check network constraints
        for resource in action.resources:
            if resource.type == "url":
                # Extract domain from URL
                import re
                match = re.match(r"https?://([^/]+)", resource.path)
                if match:
                    domain = match.group(1)
                    if not profile.is_domain_allowed(domain):
                        violations.append(
                            f"Domain '{domain}' is not allowed by operational profile"
                        )

        return violations


class ExecutionState(BaseModel):
    """Full state machine for plan execution."""

    plan_id: str
    session_id: str
    context: ExecutionContext

    # Current position
    current_sequence: int = Field(default=0, ge=0)
    status: StepStatus = StepStatus.PENDING

    # History
    step_results: list[StepResult] = Field(default_factory=list)

    # Timestamps
    started_at: str | None = None
    completed_at: str | None = None

    # Summary
    total_steps: int = Field(ge=0)
    completed_steps: int = Field(default=0, ge=0)
    failed_steps: int = Field(default=0, ge=0)
    skipped_steps: int = Field(default=0, ge=0)

    def is_complete(self) -> bool:
        """Check if execution is complete."""
        return self.current_sequence >= self.total_steps or self.status in (
            StepStatus.COMPLETED,
            StepStatus.FAILED,
            StepStatus.BLOCKED,
        )

    def get_progress(self) -> float:
        """Get execution progress as percentage."""
        if self.total_steps == 0:
            return 100.0
        return (self.completed_steps / self.total_steps) * 100


class RecoveryPath(BaseModel):
    """Recovery path for a failed step."""

    model_config = ConfigDict(frozen=True)

    trigger_step: int  # Which step this recovers from
    trigger_errors: list[str] = Field(default_factory=list)  # Error patterns that trigger this
    strategy: RecoveryStrategy

    # For RETRY strategy
    max_retries: int = Field(default=3, ge=1)
    backoff_ms: int = Field(default=1000, ge=0)

    # For ALTERNATIVE strategy
    alternative_steps: list[PlannedAction] = Field(default_factory=list)

    # For REPLAN strategy
    replan_constraints: list[str] = Field(default_factory=list)


class DatabaseConfig(BaseModel):
    """Database-specific operational constraints."""

    model_config = ConfigDict(frozen=True)

    # Protected tables - never DELETE/DROP/TRUNCATE
    protected_tables: list[str] = Field(
        default_factory=list,
        description="Tables that must never be modified destructively (e.g., users, orders)"
    )

    # Safe tables - can be modified without confirmation
    safe_tables: list[str] = Field(
        default_factory=list,
        description="Tables safe for modifications (e.g., sessions, tmp_*, cache_*)"
    )

    # Required clauses - enforce safety patterns
    require_where_clause: bool = Field(
        default=True,
        description="DELETE/UPDATE must have WHERE clause"
    )

    max_affected_rows: int | None = Field(
        default=1000,
        description="Max rows affected before requiring confirmation"
    )

    # Environment detection
    production_indicators: list[str] = Field(
        default_factory=lambda: ["prod", "production", "live"],
        description="Patterns that indicate production database"
    )


class NetworkConfig(BaseModel):
    """Network/API operational constraints."""

    model_config = ConfigDict(frozen=True)

    # Allowed domains - whitelist for outbound requests
    allowed_domains: list[str] = Field(
        default_factory=list,
        description="Domains allowed for outbound requests"
    )

    # Blocked domains - never connect
    blocked_domains: list[str] = Field(
        default_factory=lambda: [
            "*.pastebin.com",
            "*.requestbin.com",
            "*.webhook.site",
        ],
        description="Domains to block (potential exfiltration)"
    )

    # Internal-only patterns
    internal_patterns: list[str] = Field(
        default_factory=lambda: [
            "*.internal.*",
            "*.local",
            "localhost",
            "127.0.0.1",
        ],
        description="Patterns indicating internal/safe endpoints"
    )


class OperationalProfile(BaseModel):
    """User-specific operational knowledge that guides and constrains execution.

    This is the core differentiator: the plan encodes user-specific operational
    knowledge - paths, procedures, constraints, configs - and serves as an
    external guiding source of truth for execution, not just non-binding context.

    The profile is:
    - Generated before execution
    - Set up by the user
    - Enforced at runtime
    """

    model_config = ConfigDict(frozen=True)

    # Profile metadata
    profile_id: str = Field(description="Unique profile identifier")
    name: str = Field(description="Human-readable profile name")
    description: str | None = Field(
        default=None,
        description="What this profile is for"
    )
    version: str = Field(default="1.0.0", description="Profile version")

    # Environment context
    environment: str = Field(
        default="development",
        description="Environment type: development, staging, production"
    )
    project_root: str | None = Field(
        default=None,
        description="Root directory for this project"
    )

    # Operational knowledge components
    paths: PathConfig = Field(
        default_factory=PathConfig,
        description="Path-related constraints and allowances"
    )
    database: DatabaseConfig = Field(
        default_factory=DatabaseConfig,
        description="Database-related constraints"
    )
    services: ServiceConfig = Field(
        default_factory=ServiceConfig,
        description="Service/infrastructure constraints"
    )
    network: NetworkConfig = Field(
        default_factory=NetworkConfig,
        description="Network/API constraints"
    )

    # Standard operating procedures
    procedures: list[Procedure] = Field(
        default_factory=list,
        description="Defined procedures for common tasks"
    )

    # Global constraints (always enforced)
    global_constraints: list[str] = Field(
        default_factory=list,
        description="Constraints that always apply regardless of task"
    )

    # Timing constraints
    maintenance_windows: list[str] = Field(
        default_factory=list,
        description="Time windows when destructive operations are allowed (cron format)"
    )
    blocked_hours: list[str] = Field(
        default_factory=list,
        description="Hours when high-risk operations are blocked (e.g., '09:00-18:00')"
    )

    # Notification settings
    notify_on_high_risk: bool = Field(
        default=True,
        description="Send notification for high-risk operations"
    )
    notify_channels: list[str] = Field(
        default_factory=list,
        description="Channels for notifications (e.g., 'slack:#ops', 'email:team@')"
    )

    def get_procedure(self, name: str) -> Procedure | None:
        """Get a procedure by name."""
        for proc in self.procedures:
            if proc.name == name:
                return proc
        return None

    def is_path_allowed(self, path: str) -> bool:
        """Check if a path is within working directories and not protected."""
        import fnmatch

        # Check if protected
        for protected in self.paths.protected_paths:
            if fnmatch.fnmatch(path, protected) or path.startswith(protected):
                return False

        # Check if in working dirs (if defined)
        if self.paths.working_dirs:
            for working in self.paths.working_dirs:
                if path.startswith(working):
                    return True
            return False

        return True

    def is_table_protected(self, table: str) -> bool:
        """Check if a database table is protected."""
        import fnmatch

        for protected in self.database.protected_tables:
            if fnmatch.fnmatch(table, protected):
                return True
        return False

    def is_command_forbidden(self, command: str) -> bool:
        """Check if a command is forbidden."""
        import fnmatch

        command_lower = command.lower()
        for forbidden in self.services.forbidden_commands:
            if fnmatch.fnmatch(command_lower, forbidden.lower()):
                return True
            if forbidden.lower() in command_lower:
                return True
        return False

    def is_domain_allowed(self, domain: str) -> bool:
        """Check if a domain is allowed for outbound requests."""
        import fnmatch

        # Check blocked first
        for blocked in self.network.blocked_domains:
            if fnmatch.fnmatch(domain, blocked):
                return False

        # If whitelist defined, check it
        if self.network.allowed_domains:
            for allowed in self.network.allowed_domains:
                if fnmatch.fnmatch(domain, allowed):
                    return True
            return False

        return True


class PathConfig(BaseModel):
    """Path configuration for operational context."""

    model_config = ConfigDict(frozen=True)

    # Working directories - where operations are allowed
    working_dirs: list[str] = Field(
        default_factory=list,
        description="Directories where operations are permitted (e.g., ~/code/project)"
    )

    # Protected paths - never touch these
    protected_paths: list[str] = Field(
        default_factory=list,
        description="Paths that must never be modified (e.g., ~/.ssh, /etc)"
    )

    # Sensitive patterns - require extra confirmation
    sensitive_patterns: list[str] = Field(
        default_factory=list,
        description="Glob patterns for sensitive files (e.g., *.env, *secret*)"
    )

    # Temp/scratch directories - safe for temporary operations
    scratch_dirs: list[str] = Field(
        default_factory=list,
        description="Directories safe for temporary files"
    )


class Procedure(BaseModel):
    """A standard operating procedure for a specific task type."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Procedure name (e.g., 'credential_rotation')")
    description: str = Field(description="What this procedure does")

    # Required steps in order
    required_steps: list[str] = Field(
        default_factory=list,
        description="Steps that must be executed in order"
    )

    # Pre-conditions that must be true
    preconditions: list[str] = Field(
        default_factory=list,
        description="Conditions to verify before starting"
    )

    # Post-conditions to verify
    postconditions: list[str] = Field(
        default_factory=list,
        description="Conditions to verify after completion"
    )

    # Rollback steps if something fails
    rollback_steps: list[str] = Field(
        default_factory=list,
        description="Steps to execute on failure"
    )


class ServiceConfig(BaseModel):
    """Service/infrastructure operational constraints."""

    model_config = ConfigDict(frozen=True)

    # Protected services - require confirmation to restart/modify
    protected_services: list[str] = Field(
        default_factory=list,
        description="Services that require confirmation (e.g., postgres, nginx)"
    )

    # Safe services - can be restarted freely
    safe_services: list[str] = Field(
        default_factory=list,
        description="Services safe to restart (e.g., redis, memcached)"
    )

    # Forbidden commands - never execute
    forbidden_commands: list[str] = Field(
        default_factory=lambda: [
            "rm -rf /",
            "rm -rf ~",
            "rm -rf /*",
            ":(){ :|:& };:",  # Fork bomb
            "> /dev/sda",
            "mkfs.",
            "dd if=",
        ],
        description="Commands that must never be executed"
    )

    # Commands requiring confirmation
    confirm_commands: list[str] = Field(
        default_factory=lambda: [
            "reboot",
            "shutdown",
            "systemctl restart",
            "service * restart",
            "kill -9",
            "pkill",
        ],
        description="Commands requiring explicit confirmation"
    )

