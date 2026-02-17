"""Governance layer for openclaw-secure-stack.

This module provides pre-execution governance including:
- Intent classification
- Plan generation with operational profiles
- Policy validation
- Approval flow
- Execution enforcement

The execution plan encodes user-specific operational knowledge:
- Paths: working directories, protected paths, sensitive patterns
- Procedures: standard operating procedures for common tasks
- Constraints: what's never allowed
- Configs: environment-specific settings

The plan is:
- Generated before execution
- Set up by the user (via operational profile)
- Enforced at runtime
"""

from src.governance.approver import (
    ApprovalExpiredError,
    ApprovalGate,
    ApprovalNotFoundError,
    ApproverMismatchError,
)
from src.governance.classifier import IntentClassifier
from src.governance.db import GovernanceDB
from src.governance.enforcer import EnforcementResult, GovernanceEnforcer
from src.governance.middleware import EvaluationResult, GovernanceMiddleware
from src.governance.models import (
    ApprovalRequest,
    ApprovalStatus,
    DatabaseConfig,
    EnhancedExecutionPlan,
    ExecutionContext,
    ExecutionMode,
    ExecutionPlan,
    ExecutionState,
    GovernanceDecision,
    Intent,
    IntentCategory,
    IntentSignal,
    NetworkConfig,
    OperationalProfile,
    PathConfig,
    PlannedAction,
    PlanToken,
    PolicyEffect,
    PolicyRule,
    PolicyType,
    PolicyViolation,
    Procedure,
    RecoveryPath,
    RecoveryStrategy,
    ResourceAccess,
    RiskAssessment,
    RiskLevel,
    ServiceConfig,
    Session,
    StepResult,
    StepStatus,
    ToolCall,
    ValidationResult,
)
from src.governance.planner import PlanGenerator
from src.governance.profile import ProfileLoader, get_profile_loader, load_profile
from src.governance.session import SessionManager
from src.governance.store import PlanNotFoundError, PlanStore, TokenVerificationResult
from src.governance.validator import PolicyValidator

__all__ = [
    # Exceptions
    "ApprovalExpiredError",
    "ApprovalNotFoundError",
    "ApproverMismatchError",
    "PlanNotFoundError",
    # Components
    "ApprovalGate",
    "GovernanceDB",
    "GovernanceEnforcer",
    "GovernanceMiddleware",
    "IntentClassifier",
    "PlanGenerator",
    "PlanStore",
    "PolicyValidator",
    "ProfileLoader",
    "SessionManager",
    # Profile helpers
    "get_profile_loader",
    "load_profile",
    # Result types
    "EnforcementResult",
    "EvaluationResult",
    "TokenVerificationResult",
    # Models - Core
    "ApprovalRequest",
    "ApprovalStatus",
    "ExecutionPlan",
    "EnhancedExecutionPlan",
    "ExecutionContext",
    "ExecutionMode",
    "ExecutionState",
    "GovernanceDecision",
    "Intent",
    "IntentCategory",
    "IntentSignal",
    "PlannedAction",
    "PlanToken",
    "PolicyEffect",
    "PolicyRule",
    "PolicyType",
    "PolicyViolation",
    "ResourceAccess",
    "RiskAssessment",
    "RiskLevel",
    "Session",
    "StepResult",
    "StepStatus",
    "ToolCall",
    "ValidationResult",
    # Models - Operational Profile
    "DatabaseConfig",
    "NetworkConfig",
    "OperationalProfile",
    "PathConfig",
    "Procedure",
    "RecoveryPath",
    "RecoveryStrategy",
    "ServiceConfig",
]
