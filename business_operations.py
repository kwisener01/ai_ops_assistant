"""
business_operations.py
=======================

Business operational procedures and workflow management for Fynix Systems.
This module defines standard operating procedures (SOPs) for key business
processes including sales, project delivery, client support, and team management.

Key Components:
- SOP (Standard Operating Procedure) management
- Workflow tracking and automation
- Process compliance monitoring
- Operational efficiency metrics
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import pandas as pd


class ProcessCategory(Enum):
    """Categories of business processes."""
    SALES = "sales"
    DELIVERY = "delivery"
    SUPPORT = "support"
    OPERATIONS = "operations"
    HR = "hr"
    FINANCE = "finance"


class ProcessStatus(Enum):
    """Status of process execution."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class Priority(Enum):
    """Priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ProcessStep:
    """Individual step in a business process."""

    step_id: str
    name: str
    description: str
    responsible: str  # Role or person responsible
    estimated_duration: float  # Hours
    dependencies: List[str] = field(default_factory=list)  # IDs of prerequisite steps
    status: ProcessStatus = ProcessStatus.NOT_STARTED
    actual_duration: Optional[float] = None
    completed_date: Optional[_dt.date] = None
    notes: List[str] = field(default_factory=list)

    def is_blocked(self, completed_steps: List[str]) -> bool:
        """Check if step is blocked by incomplete dependencies."""
        return not all(dep in completed_steps for dep in self.dependencies)

    def complete(self, duration: Optional[float] = None) -> None:
        """Mark step as completed."""
        self.status = ProcessStatus.COMPLETED
        self.completed_date = _dt.date.today()
        if duration is not None:
            self.actual_duration = duration


@dataclass
class SOP:
    """Standard Operating Procedure definition."""

    sop_id: str
    name: str
    category: ProcessCategory
    description: str
    steps: List[ProcessStep]
    owner: str
    version: str = "1.0"
    last_updated: _dt.date = field(default_factory=_dt.date.today)
    compliance_required: bool = True

    def get_total_estimated_duration(self) -> float:
        """Calculate total estimated duration."""
        return sum(step.estimated_duration for step in self.steps)

    def get_completion_percentage(self) -> float:
        """Calculate percentage of completed steps."""
        if not self.steps:
            return 0.0
        completed = sum(1 for step in self.steps if step.status == ProcessStatus.COMPLETED)
        return (completed / len(self.steps)) * 100

    def get_next_steps(self) -> List[ProcessStep]:
        """Get list of steps ready to be executed."""
        completed_ids = [s.step_id for s in self.steps if s.status == ProcessStatus.COMPLETED]
        return [
            step for step in self.steps
            if step.status == ProcessStatus.NOT_STARTED and not step.is_blocked(completed_ids)
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize SOP to dictionary."""
        return {
            "sop_id": self.sop_id,
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "steps": [
                {
                    "step_id": s.step_id,
                    "name": s.name,
                    "description": s.description,
                    "responsible": s.responsible,
                    "estimated_duration": s.estimated_duration,
                    "dependencies": s.dependencies,
                    "status": s.status.value,
                    "actual_duration": s.actual_duration,
                    "completed_date": s.completed_date.isoformat() if s.completed_date else None,
                    "notes": s.notes,
                }
                for s in self.steps
            ],
            "owner": self.owner,
            "version": self.version,
            "last_updated": self.last_updated.isoformat(),
            "compliance_required": self.compliance_required,
        }


@dataclass
class WorkflowInstance:
    """An active instance of a workflow based on an SOP."""

    instance_id: str
    sop: SOP
    start_date: _dt.date
    target_completion_date: _dt.date
    actual_completion_date: Optional[_dt.date] = None
    priority: Priority = Priority.MEDIUM
    client_name: Optional[str] = None
    project_name: Optional[str] = None
    assigned_team: List[str] = field(default_factory=list)
    status: ProcessStatus = ProcessStatus.NOT_STARTED

    def is_overdue(self) -> bool:
        """Check if workflow is past target completion date."""
        if self.status == ProcessStatus.COMPLETED:
            return False
        return _dt.date.today() > self.target_completion_date

    def get_time_remaining(self) -> int:
        """Get days remaining until target completion."""
        if self.status == ProcessStatus.COMPLETED:
            return 0
        delta = self.target_completion_date - _dt.date.today()
        return max(0, delta.days)


class OperationalManager:
    """Manages business operations, SOPs, and workflow instances.

    This class provides centralized management of operational procedures,
    tracks workflow execution, ensures compliance, and generates operational
    efficiency reports.
    """

    def __init__(self) -> None:
        self._sops: Dict[str, SOP] = {}
        self._workflows: Dict[str, WorkflowInstance] = {}
        self._init_standard_sops()

    def _init_standard_sops(self) -> None:
        """Initialize standard SOPs for Fynix Systems."""
        # Sales Process SOP
        sales_sop = create_sales_process_sop()
        self.add_sop(sales_sop)

        # Project Delivery SOP
        delivery_sop = create_project_delivery_sop()
        self.add_sop(delivery_sop)

        # Client Support SOP
        support_sop = create_client_support_sop()
        self.add_sop(support_sop)

        # Client Onboarding SOP
        onboarding_sop = create_client_onboarding_sop()
        self.add_sop(onboarding_sop)

    # ------------------------------------------------------------------
    # SOP Management
    # ------------------------------------------------------------------
    def add_sop(self, sop: SOP) -> None:
        """Add or update an SOP."""
        self._sops[sop.sop_id] = sop

    def get_sop(self, sop_id: str) -> Optional[SOP]:
        """Retrieve an SOP by ID."""
        return self._sops.get(sop_id)

    def get_sops_by_category(self, category: ProcessCategory) -> List[SOP]:
        """Get all SOPs for a specific category."""
        return [sop for sop in self._sops.values() if sop.category == category]

    def list_all_sops(self) -> List[Dict[str, Any]]:
        """List all SOPs with summary information."""
        return [
            {
                "sop_id": sop.sop_id,
                "name": sop.name,
                "category": sop.category.value,
                "owner": sop.owner,
                "steps_count": len(sop.steps),
                "estimated_duration": sop.get_total_estimated_duration(),
            }
            for sop in self._sops.values()
        ]

    # ------------------------------------------------------------------
    # Workflow Management
    # ------------------------------------------------------------------
    def create_workflow(self, sop_id: str, instance_id: str,
                       target_days: int = 30,
                       priority: Priority = Priority.MEDIUM,
                       client_name: Optional[str] = None,
                       project_name: Optional[str] = None) -> WorkflowInstance:
        """Create a new workflow instance from an SOP."""
        sop = self.get_sop(sop_id)
        if not sop:
            raise ValueError(f"SOP {sop_id} not found")

        # Create a deep copy of the SOP for this workflow
        import copy
        sop_copy = copy.deepcopy(sop)

        workflow = WorkflowInstance(
            instance_id=instance_id,
            sop=sop_copy,
            start_date=_dt.date.today(),
            target_completion_date=_dt.date.today() + _dt.timedelta(days=target_days),
            priority=priority,
            client_name=client_name,
            project_name=project_name,
        )

        self._workflows[instance_id] = workflow
        return workflow

    def get_workflow(self, instance_id: str) -> Optional[WorkflowInstance]:
        """Retrieve a workflow instance."""
        return self._workflows.get(instance_id)

    def complete_workflow_step(self, instance_id: str, step_id: str,
                               duration: Optional[float] = None,
                               notes: Optional[str] = None) -> None:
        """Mark a workflow step as completed."""
        workflow = self.get_workflow(instance_id)
        if not workflow:
            raise ValueError(f"Workflow {instance_id} not found")

        step = next((s for s in workflow.sop.steps if s.step_id == step_id), None)
        if not step:
            raise ValueError(f"Step {step_id} not found in workflow")

        step.complete(duration)
        if notes:
            step.notes.append(notes)

        # Check if all steps are completed
        if workflow.sop.get_completion_percentage() == 100:
            workflow.status = ProcessStatus.COMPLETED
            workflow.actual_completion_date = _dt.date.today()

    def get_active_workflows(self) -> List[WorkflowInstance]:
        """Get all workflows that are not completed."""
        return [
            wf for wf in self._workflows.values()
            if wf.status != ProcessStatus.COMPLETED
        ]

    def get_overdue_workflows(self) -> List[WorkflowInstance]:
        """Get all workflows that are past their target completion date."""
        return [wf for wf in self.get_active_workflows() if wf.is_overdue()]

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def generate_operational_report(self) -> Dict[str, Any]:
        """Generate comprehensive operational efficiency report."""
        active = self.get_active_workflows()
        overdue = self.get_overdue_workflows()
        completed = [wf for wf in self._workflows.values() if wf.status == ProcessStatus.COMPLETED]

        report = {
            "summary": {
                "total_workflows": len(self._workflows),
                "active_workflows": len(active),
                "completed_workflows": len(completed),
                "overdue_workflows": len(overdue),
                "on_time_completion_rate": self._calculate_on_time_rate(completed),
            },
            "by_priority": {
                Priority.CRITICAL.value: len([w for w in active if w.priority == Priority.CRITICAL]),
                Priority.HIGH.value: len([w for w in active if w.priority == Priority.HIGH]),
                Priority.MEDIUM.value: len([w for w in active if w.priority == Priority.MEDIUM]),
                Priority.LOW.value: len([w for w in active if w.priority == Priority.LOW]),
            },
            "sop_utilization": {
                sop_id: len([w for w in self._workflows.values() if w.sop.sop_id == sop_id])
                for sop_id in self._sops.keys()
            },
            "overdue_details": [
                {
                    "instance_id": wf.instance_id,
                    "sop_name": wf.sop.name,
                    "client": wf.client_name,
                    "days_overdue": (_dt.date.today() - wf.target_completion_date).days,
                }
                for wf in overdue
            ],
        }

        return report

    def _calculate_on_time_rate(self, completed_workflows: List[WorkflowInstance]) -> float:
        """Calculate percentage of workflows completed on time."""
        if not completed_workflows:
            return 0.0

        on_time = sum(
            1 for wf in completed_workflows
            if wf.actual_completion_date and wf.actual_completion_date <= wf.target_completion_date
        )

        return (on_time / len(completed_workflows)) * 100

    def export_to_csv(self, file_path: str) -> None:
        """Export all workflows to CSV."""
        data = []
        for wf in self._workflows.values():
            data.append({
                "instance_id": wf.instance_id,
                "sop_name": wf.sop.name,
                "category": wf.sop.category.value,
                "status": wf.status.value,
                "priority": wf.priority.value,
                "client": wf.client_name,
                "project": wf.project_name,
                "start_date": wf.start_date.isoformat(),
                "target_date": wf.target_completion_date.isoformat(),
                "completion_percentage": wf.sop.get_completion_percentage(),
                "is_overdue": wf.is_overdue(),
            })

        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)


# ------------------------------------------------------------------
# Standard SOP Definitions
# ------------------------------------------------------------------

def create_sales_process_sop() -> SOP:
    """Create standard sales process SOP."""
    steps = [
        ProcessStep("s1", "Lead Qualification", "Qualify incoming lead", "Sales", 1.0),
        ProcessStep("s2", "Discovery Call", "Conduct discovery call with prospect", "Sales", 2.0, ["s1"]),
        ProcessStep("s3", "Needs Analysis", "Analyze client needs and requirements", "Sales", 3.0, ["s2"]),
        ProcessStep("s4", "Solution Design", "Design tailored AI automation solution", "Solution Architect", 8.0, ["s3"]),
        ProcessStep("s5", "Proposal Creation", "Create and send proposal", "Sales", 4.0, ["s4"]),
        ProcessStep("s6", "Proposal Review", "Review proposal with client", "Sales", 2.0, ["s5"]),
        ProcessStep("s7", "Negotiation", "Negotiate terms and pricing", "Sales", 3.0, ["s6"]),
        ProcessStep("s8", "Contract Signing", "Execute contract", "Sales", 1.0, ["s7"]),
    ]

    return SOP(
        sop_id="sales_001",
        name="AI Automation Sales Process",
        category=ProcessCategory.SALES,
        description="Standard sales process from lead to closed deal",
        steps=steps,
        owner="Head of Sales",
    )


def create_project_delivery_sop() -> SOP:
    """Create standard project delivery SOP."""
    steps = [
        ProcessStep("d1", "Kickoff Meeting", "Conduct project kickoff with client", "Project Manager", 2.0),
        ProcessStep("d2", "Requirements Gathering", "Document detailed requirements", "Business Analyst", 16.0, ["d1"]),
        ProcessStep("d3", "Technical Design", "Create technical architecture", "Tech Lead", 16.0, ["d2"]),
        ProcessStep("d4", "Development Sprint 1", "First development iteration", "Engineering Team", 80.0, ["d3"]),
        ProcessStep("d5", "Client Review 1", "First client review and feedback", "Project Manager", 4.0, ["d4"]),
        ProcessStep("d6", "Development Sprint 2", "Second development iteration", "Engineering Team", 80.0, ["d5"]),
        ProcessStep("d7", "Testing & QA", "Comprehensive testing", "QA Team", 40.0, ["d6"]),
        ProcessStep("d8", "UAT", "User acceptance testing", "Project Manager", 16.0, ["d7"]),
        ProcessStep("d9", "Deployment", "Deploy to production", "DevOps", 8.0, ["d8"]),
        ProcessStep("d10", "Training", "Train client team", "Project Manager", 8.0, ["d9"]),
        ProcessStep("d11", "Handoff", "Project handoff and documentation", "Project Manager", 4.0, ["d10"]),
    ]

    return SOP(
        sop_id="delivery_001",
        name="AI Automation Project Delivery",
        category=ProcessCategory.DELIVERY,
        description="Standard project delivery process",
        steps=steps,
        owner="VP of Engineering",
    )


def create_client_support_sop() -> SOP:
    """Create standard client support SOP."""
    steps = [
        ProcessStep("sup1", "Ticket Creation", "Create support ticket", "Support", 0.25),
        ProcessStep("sup2", "Initial Assessment", "Assess issue severity and category", "Support", 0.5, ["sup1"]),
        ProcessStep("sup3", "Issue Investigation", "Investigate and diagnose issue", "Support Engineer", 2.0, ["sup2"]),
        ProcessStep("sup4", "Solution Implementation", "Implement fix or workaround", "Support Engineer", 4.0, ["sup3"]),
        ProcessStep("sup5", "Client Verification", "Verify resolution with client", "Support", 1.0, ["sup4"]),
        ProcessStep("sup6", "Documentation", "Document issue and solution", "Support", 0.5, ["sup5"]),
        ProcessStep("sup7", "Ticket Closure", "Close support ticket", "Support", 0.25, ["sup6"]),
    ]

    return SOP(
        sop_id="support_001",
        name="Client Support Ticket Resolution",
        category=ProcessCategory.SUPPORT,
        description="Standard process for handling client support requests",
        steps=steps,
        owner="Head of Support",
    )


def create_client_onboarding_sop() -> SOP:
    """Create client onboarding SOP."""
    steps = [
        ProcessStep("on1", "Welcome Email", "Send welcome email and schedule kickoff", "Account Manager", 0.5),
        ProcessStep("on2", "Account Setup", "Set up client account and access", "Operations", 2.0, ["on1"]),
        ProcessStep("on3", "Kickoff Call", "Conduct onboarding kickoff call", "Account Manager", 2.0, ["on2"]),
        ProcessStep("on4", "Documentation Sharing", "Share relevant documentation", "Account Manager", 1.0, ["on3"]),
        ProcessStep("on5", "Training Session", "Conduct initial training", "Success Manager", 4.0, ["on4"]),
        ProcessStep("on6", "Success Plan", "Create client success plan", "Success Manager", 3.0, ["on5"]),
        ProcessStep("on7", "30-Day Check-in", "Schedule 30-day check-in", "Account Manager", 1.0, ["on6"]),
    ]

    return SOP(
        sop_id="onboarding_001",
        name="Client Onboarding Process",
        category=ProcessCategory.OPERATIONS,
        description="Standard process for onboarding new clients",
        steps=steps,
        owner="VP of Customer Success",
    )
