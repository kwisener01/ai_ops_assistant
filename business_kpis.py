"""
business_kpis.py
=================

KPI tracking and monitoring system for Fynix Systems AI automation business.
This module provides comprehensive tracking of business metrics, operational
efficiency, client satisfaction, and financial performance.

Key Performance Indicators tracked:
- Revenue metrics (MRR, ARR, growth rate)
- Client metrics (acquisition, retention, satisfaction, LTV)
- Project metrics (delivery time, success rate, utilization)
- Team metrics (productivity, capacity, efficiency)
- Operational metrics (response time, automation rate, uptime)
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import pandas as pd
import numpy as np


class KPICategory(Enum):
    """Categories of business KPIs."""
    REVENUE = "revenue"
    CLIENT = "client"
    PROJECT = "project"
    TEAM = "team"
    OPERATIONAL = "operational"


class KPITrend(Enum):
    """Trend direction for KPI."""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"


@dataclass
class KPI:
    """Individual Key Performance Indicator."""

    kpi_id: str
    name: str
    category: KPICategory
    value: float
    target: float
    unit: str  # e.g., "$", "%", "days", "count"
    date: _dt.date
    description: Optional[str] = None
    trend: Optional[KPITrend] = None

    def performance_ratio(self) -> float:
        """Calculate performance as ratio of actual to target."""
        if self.target == 0:
            return 0.0
        return self.value / self.target

    def is_on_target(self, threshold: float = 0.9) -> bool:
        """Check if KPI meets target threshold."""
        return self.performance_ratio() >= threshold

    def to_dict(self) -> Dict[str, Any]:
        """Serialize KPI to dictionary."""
        return {
            "kpi_id": self.kpi_id,
            "name": self.name,
            "category": self.category.value,
            "value": self.value,
            "target": self.target,
            "unit": self.unit,
            "date": self.date.isoformat(),
            "description": self.description,
            "trend": self.trend.value if self.trend else None,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KPI":
        """Deserialize KPI from dictionary."""
        return cls(
            kpi_id=d["kpi_id"],
            name=d["name"],
            category=KPICategory(d["category"]),
            value=float(d["value"]),
            target=float(d["target"]),
            unit=d["unit"],
            date=_dt.date.fromisoformat(d["date"]),
            description=d.get("description"),
            trend=KPITrend(d["trend"]) if d.get("trend") else None,
        )


@dataclass
class RevenueMetrics:
    """Monthly Recurring Revenue and Annual metrics."""

    mrr: float
    arr: float
    growth_rate: float  # Month-over-month percentage
    new_revenue: float
    expansion_revenue: float
    churn_revenue: float
    date: _dt.date

    def net_new_mrr(self) -> float:
        """Calculate net new MRR."""
        return self.new_revenue + self.expansion_revenue - self.churn_revenue


@dataclass
class ClientMetrics:
    """Client acquisition, retention and satisfaction metrics."""

    total_clients: int
    new_clients: int
    churned_clients: int
    retention_rate: float  # Percentage
    satisfaction_score: float  # 0-10 scale
    nps: Optional[float]  # Net Promoter Score
    avg_client_ltv: float  # Lifetime value
    date: _dt.date


@dataclass
class ProjectMetrics:
    """Project delivery and success metrics."""

    total_projects: int
    completed_projects: int
    in_progress_projects: int
    avg_delivery_time: float  # Days
    on_time_delivery_rate: float  # Percentage
    success_rate: float  # Percentage
    team_utilization: float  # Percentage
    date: _dt.date


class KPIDashboard:
    """Central KPI tracking and reporting dashboard for Fynix Systems.

    This class manages all business KPIs, tracks trends over time, and
    generates comprehensive performance reports for stakeholders.
    """

    def __init__(self) -> None:
        self._kpis: Dict[str, KPI] = {}
        self._revenue_history: List[RevenueMetrics] = []
        self._client_history: List[ClientMetrics] = []
        self._project_history: List[ProjectMetrics] = []

    # ------------------------------------------------------------------
    # KPI Management
    # ------------------------------------------------------------------
    def add_kpi(self, kpi: KPI) -> None:
        """Add or update a KPI."""
        self._kpis[kpi.kpi_id] = kpi

    def get_kpi(self, kpi_id: str) -> Optional[KPI]:
        """Retrieve a KPI by ID."""
        return self._kpis.get(kpi_id)

    def get_kpis_by_category(self, category: KPICategory) -> List[KPI]:
        """Get all KPIs for a specific category."""
        return [kpi for kpi in self._kpis.values() if kpi.category == category]

    def get_underperforming_kpis(self, threshold: float = 0.9) -> List[KPI]:
        """Get KPIs that are below target threshold."""
        return [kpi for kpi in self._kpis.values() if not kpi.is_on_target(threshold)]

    # ------------------------------------------------------------------
    # Revenue Tracking
    # ------------------------------------------------------------------
    def add_revenue_metrics(self, metrics: RevenueMetrics) -> None:
        """Record monthly revenue metrics."""
        self._revenue_history.append(metrics)

        # Update KPI dashboard
        self.add_kpi(KPI(
            kpi_id="mrr",
            name="Monthly Recurring Revenue",
            category=KPICategory.REVENUE,
            value=metrics.mrr,
            target=metrics.mrr * 1.1,  # 10% growth target
            unit="$",
            date=metrics.date,
            description="Monthly recurring revenue"
        ))

        self.add_kpi(KPI(
            kpi_id="growth_rate",
            name="Revenue Growth Rate",
            category=KPICategory.REVENUE,
            value=metrics.growth_rate,
            target=15.0,  # 15% monthly growth target
            unit="%",
            date=metrics.date,
            description="Month-over-month revenue growth"
        ))

    def get_revenue_trend(self, months: int = 6) -> pd.DataFrame:
        """Get revenue trends for the last N months."""
        recent = sorted(self._revenue_history, key=lambda x: x.date)[-months:]
        return pd.DataFrame([
            {
                "date": m.date,
                "mrr": m.mrr,
                "arr": m.arr,
                "growth_rate": m.growth_rate,
                "net_new_mrr": m.net_new_mrr()
            }
            for m in recent
        ])

    # ------------------------------------------------------------------
    # Client Tracking
    # ------------------------------------------------------------------
    def add_client_metrics(self, metrics: ClientMetrics) -> None:
        """Record monthly client metrics."""
        self._client_history.append(metrics)

        # Update KPI dashboard
        self.add_kpi(KPI(
            kpi_id="client_retention",
            name="Client Retention Rate",
            category=KPICategory.CLIENT,
            value=metrics.retention_rate,
            target=95.0,  # 95% retention target
            unit="%",
            date=metrics.date,
            description="Percentage of clients retained month-over-month"
        ))

        self.add_kpi(KPI(
            kpi_id="client_satisfaction",
            name="Client Satisfaction Score",
            category=KPICategory.CLIENT,
            value=metrics.satisfaction_score,
            target=8.5,  # 8.5/10 target
            unit="/10",
            date=metrics.date,
            description="Average client satisfaction score"
        ))

        if metrics.nps is not None:
            self.add_kpi(KPI(
                kpi_id="nps",
                name="Net Promoter Score",
                category=KPICategory.CLIENT,
                value=metrics.nps,
                target=50.0,  # NPS of 50 is excellent
                unit="",
                date=metrics.date,
                description="Net Promoter Score"
            ))

    def get_client_trend(self, months: int = 6) -> pd.DataFrame:
        """Get client trends for the last N months."""
        recent = sorted(self._client_history, key=lambda x: x.date)[-months:]
        return pd.DataFrame([
            {
                "date": m.date,
                "total_clients": m.total_clients,
                "new_clients": m.new_clients,
                "churned_clients": m.churned_clients,
                "retention_rate": m.retention_rate,
                "satisfaction_score": m.satisfaction_score,
                "nps": m.nps,
            }
            for m in recent
        ])

    # ------------------------------------------------------------------
    # Project Tracking
    # ------------------------------------------------------------------
    def add_project_metrics(self, metrics: ProjectMetrics) -> None:
        """Record monthly project metrics."""
        self._project_history.append(metrics)

        # Update KPI dashboard
        self.add_kpi(KPI(
            kpi_id="on_time_delivery",
            name="On-Time Delivery Rate",
            category=KPICategory.PROJECT,
            value=metrics.on_time_delivery_rate,
            target=90.0,  # 90% on-time delivery
            unit="%",
            date=metrics.date,
            description="Percentage of projects delivered on time"
        ))

        self.add_kpi(KPI(
            kpi_id="project_success_rate",
            name="Project Success Rate",
            category=KPICategory.PROJECT,
            value=metrics.success_rate,
            target=95.0,  # 95% success rate
            unit="%",
            date=metrics.date,
            description="Percentage of projects completed successfully"
        ))

        self.add_kpi(KPI(
            kpi_id="team_utilization",
            name="Team Utilization Rate",
            category=KPICategory.PROJECT,
            value=metrics.team_utilization,
            target=85.0,  # 85% utilization target
            unit="%",
            date=metrics.date,
            description="Percentage of team capacity utilized"
        ))

    def get_project_trend(self, months: int = 6) -> pd.DataFrame:
        """Get project trends for the last N months."""
        recent = sorted(self._project_history, key=lambda x: x.date)[-months:]
        return pd.DataFrame([
            {
                "date": m.date,
                "total_projects": m.total_projects,
                "completed_projects": m.completed_projects,
                "avg_delivery_time": m.avg_delivery_time,
                "on_time_delivery_rate": m.on_time_delivery_rate,
                "success_rate": m.success_rate,
                "team_utilization": m.team_utilization,
            }
            for m in recent
        ])

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """Convert all KPIs to a DataFrame."""
        if not self._kpis:
            return pd.DataFrame()
        return pd.DataFrame([kpi.to_dict() for kpi in self._kpis.values()])

    def generate_kpi_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive KPI summary."""
        df = self.to_dataframe()
        if df.empty:
            return {"status": "No KPIs tracked"}

        summary = {
            "total_kpis": len(df),
            "on_target": sum(1 for kpi in self._kpis.values() if kpi.is_on_target()),
            "underperforming": len(self.get_underperforming_kpis()),
            "by_category": df.groupby("category")["value"].count().to_dict(),
            "avg_performance": df.apply(lambda row: row["value"] / row["target"] if row["target"] > 0 else 0, axis=1).mean(),
        }

        return summary

    def save_to_file(self, file_path: str) -> None:
        """Save all KPI data to CSV."""
        df = self.to_dataframe()
        df.to_csv(file_path, index=False)

    def load_from_file(self, file_path: str) -> None:
        """Load KPI data from CSV."""
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            kpi = KPI.from_dict(row.to_dict())
            self.add_kpi(kpi)


def create_standard_kpis(date: Optional[_dt.date] = None) -> KPIDashboard:
    """Create a KPI dashboard with standard business metrics initialized.

    This helper creates a dashboard with typical AI automation company KPIs
    pre-configured with industry-standard targets.
    """
    dashboard = KPIDashboard()
    date = date or _dt.date.today()

    # Revenue KPIs
    revenue_kpis = [
        KPI("mrr", "Monthly Recurring Revenue", KPICategory.REVENUE, 0, 50000, "$", date,
            "Monthly recurring revenue from active clients"),
        KPI("arr", "Annual Recurring Revenue", KPICategory.REVENUE, 0, 600000, "$", date,
            "Annualized recurring revenue"),
        KPI("growth_rate", "Revenue Growth Rate", KPICategory.REVENUE, 0, 15, "%", date,
            "Month-over-month revenue growth"),
    ]

    # Client KPIs
    client_kpis = [
        KPI("total_clients", "Total Clients", KPICategory.CLIENT, 0, 20, "count", date,
            "Total number of active clients"),
        KPI("retention_rate", "Client Retention Rate", KPICategory.CLIENT, 0, 95, "%", date,
            "Percentage of clients retained"),
        KPI("csat", "Client Satisfaction", KPICategory.CLIENT, 0, 8.5, "/10", date,
            "Average client satisfaction score"),
        KPI("nps", "Net Promoter Score", KPICategory.CLIENT, 0, 50, "", date,
            "Net promoter score"),
        KPI("cac", "Client Acquisition Cost", KPICategory.CLIENT, 0, 5000, "$", date,
            "Cost to acquire a new client"),
    ]

    # Project KPIs
    project_kpis = [
        KPI("delivery_time", "Average Delivery Time", KPICategory.PROJECT, 0, 30, "days", date,
            "Average project delivery time"),
        KPI("on_time_rate", "On-Time Delivery Rate", KPICategory.PROJECT, 0, 90, "%", date,
            "Percentage of projects delivered on time"),
        KPI("success_rate", "Project Success Rate", KPICategory.PROJECT, 0, 95, "%", date,
            "Percentage of projects completed successfully"),
        KPI("utilization", "Team Utilization", KPICategory.PROJECT, 0, 85, "%", date,
            "Percentage of team capacity utilized"),
    ]

    # Team KPIs
    team_kpis = [
        KPI("productivity", "Team Productivity", KPICategory.TEAM, 0, 100, "score", date,
            "Team productivity index"),
        KPI("capacity", "Available Capacity", KPICategory.TEAM, 0, 160, "hours", date,
            "Available team hours per month"),
    ]

    # Operational KPIs
    operational_kpis = [
        KPI("response_time", "Average Response Time", KPICategory.OPERATIONAL, 0, 2, "hours", date,
            "Average response time to client requests"),
        KPI("uptime", "System Uptime", KPICategory.OPERATIONAL, 0, 99.9, "%", date,
            "Percentage of time systems are operational"),
        KPI("automation_rate", "Automation Rate", KPICategory.OPERATIONAL, 0, 80, "%", date,
            "Percentage of processes automated"),
    ]

    # Add all KPIs to dashboard
    for kpi in revenue_kpis + client_kpis + project_kpis + team_kpis + operational_kpis:
        dashboard.add_kpi(kpi)

    return dashboard
