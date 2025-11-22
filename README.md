# Fynix Systems - AI Operations Assistant

> **Comprehensive business operations, KPI tracking, and workflow management system for Fynix Systems**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

The **AI Operations Assistant** is a comprehensive business management system designed specifically for **Fynix Systems** (https://fynix.systems/), an AI automation company. This system provides:

- **🌐 Web Dashboard** - Interactive Streamlit web interface with real-time visualization
- **📊 KPI Tracking & Monitoring** - Real-time tracking of business metrics across revenue, clients, projects, and operations
- **📋 Standard Operating Procedures (SOPs)** - Predefined workflows for sales, delivery, support, and onboarding
- **🔄 Workflow Management** - Active workflow tracking with dependency management and progress monitoring
- **📈 Performance Analytics** - Comprehensive reporting and trend analysis
- **🎯 Issue Tracking** - Root cause analysis using 5-Whys and Ishikawa diagrams
- **📑 Automated Reporting** - PDF report generation with insights and recommendations

## 🌐 Live Demo

**Access the web dashboard:** (Coming soon - deploy to https://work-assist.streamlit.app/)

**Run locally:**
```bash
streamlit run streamlit_app.py
```

---

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [System Components](#system-components)
5. [Usage Examples](#usage-examples)
6. [Business Procedures](#business-procedures)
7. [KPI Reference](#kpi-reference)
8. [Architecture](#architecture)
9. [Contributing](#contributing)
10. [Support](#support)

---

## Features

### 🎯 Business KPI Tracking

Track and monitor key performance indicators across five categories:

- **Revenue Metrics**: MRR, ARR, growth rate, revenue composition
- **Client Metrics**: Retention, satisfaction (CSAT), NPS, LTV, CAC
- **Project Metrics**: On-time delivery, success rate, team utilization
- **Team Metrics**: Productivity, capacity, efficiency
- **Operational Metrics**: Response time, uptime, automation rate

### 📋 Standard Operating Procedures

Pre-built SOPs for core business processes:

- **Sales Process** (8 steps, ~24 hours)
- **Project Delivery** (11 steps, ~274 hours)
- **Client Support** (7 steps, ~8 hours)
- **Client Onboarding** (7 steps, ~14 days)

### 🔄 Workflow Management

- Create workflow instances from SOPs
- Track step completion and dependencies
- Monitor progress and identify blockers
- Flag overdue workflows automatically
- Generate operational efficiency reports

### 📊 Analytics & Reporting

- Trend analysis for revenue, clients, and projects
- Performance ratio calculations
- Underperforming KPI identification
- Comprehensive PDF reports
- Fishbone (Ishikawa) diagrams for root cause analysis

### 🛠️ Issue Tracking (Legacy Manufacturing Module)

The original manufacturing operations module is included and provides:

- Quality and downtime issue tracking
- 5-Whys root cause analysis
- Corrective action management
- Pattern detection for recurring issues
- CSV/Excel data import/export

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/kwisener01/ai_ops_assistant.git
cd ai_ops_assistant

# Install required packages
pip install -r requirements.txt
```

### Launch Web Dashboard

```bash
# Start the Streamlit web interface
streamlit run streamlit_app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Required Packages

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
fpdf>=1.7.2
pdfplumber>=0.9.0
python-pptx>=0.6.21
openpyxl>=3.1.0
streamlit>=1.28.0
plotly>=5.17.0
```

---

## Quick Start

### 0. Web Dashboard (Recommended)

```bash
# Launch the interactive web interface
streamlit run streamlit_app.py
```

The web dashboard provides:
- 📊 Interactive KPI dashboards
- 🔄 Workflow management interface
- 📈 Real-time charts and analytics
- ➕ Easy data entry forms
- 📥 Data export capabilities

### 1. KPI Tracking (Python API)

```python
from business_kpis import KPIDashboard, RevenueMetrics, ClientMetrics
import datetime as dt

# Create dashboard
dashboard = KPIDashboard()

# Track revenue metrics
revenue = RevenueMetrics(
    mrr=45000,
    arr=540000,
    growth_rate=12.5,
    new_revenue=5000,
    expansion_revenue=2000,
    churn_revenue=1500,
    date=dt.date.today()
)
dashboard.add_revenue_metrics(revenue)

# Track client metrics
clients = ClientMetrics(
    total_clients=18,
    new_clients=3,
    churned_clients=1,
    retention_rate=94.4,
    satisfaction_score=8.7,
    nps=52,
    avg_client_ltv=220000,
    date=dt.date.today()
)
dashboard.add_client_metrics(clients)

# Get underperforming KPIs
underperforming = dashboard.get_underperforming_kpis()
for kpi in underperforming:
    print(f"{kpi.name}: {kpi.value}{kpi.unit} (Target: {kpi.target}{kpi.unit})")
```

### 2. Workflow Management

```python
from business_operations import OperationalManager, Priority

# Initialize manager
ops = OperationalManager()

# Create a sales workflow
workflow = ops.create_workflow(
    sop_id="sales_001",
    instance_id="deal_acme_corp",
    target_days=21,
    priority=Priority.HIGH,
    client_name="Acme Corporation",
    project_name="Warehouse Automation"
)

# Complete steps
ops.complete_workflow_step("deal_acme_corp", "s1", duration=0.5, notes="Qualified - great fit")
ops.complete_workflow_step("deal_acme_corp", "s2", duration=1.5, notes="Discovery call completed")

# Check progress
print(f"Completion: {workflow.sop.get_completion_percentage():.1f}%")
print(f"Next steps: {[s.name for s in workflow.sop.get_next_steps()]}")
```

### 3. Issue Tracking (Manufacturing Module)

```python
from ai_ops_assistant import OpsAssistant
import datetime as dt

# Create assistant
assistant = OpsAssistant()

# Add issue
issue_id = assistant.add_issue(
    "Client API integration failing",
    category="Technical",
    date_raised=dt.date.today()
)

# Conduct 5-Whys analysis
assistant.ask_why(issue_id, "Why is the API failing?", "Authentication token expired")
assistant.ask_why(issue_id, "Why did the token expire?", "Token refresh not implemented")
assistant.ask_why(issue_id, "Why wasn't refresh implemented?", "Missed in initial scope")
assistant.finalize_root_cause(issue_id)

# Add corrective action
assistant.add_corrective_action(issue_id, "Implement token refresh mechanism")
assistant.update_status(issue_id, "Closed")

# Generate report
assistant.generate_report("issue_report.pdf")
```

---

## System Components

### 1. `business_kpis.py` - KPI Tracking System

**Classes:**
- `KPI` - Individual KPI with value, target, and performance ratio
- `RevenueMetrics` - Monthly revenue tracking
- `ClientMetrics` - Client performance tracking
- `ProjectMetrics` - Project delivery tracking
- `KPIDashboard` - Central KPI management and reporting

**Key Methods:**
- `add_kpi()` - Add/update a KPI
- `get_underperforming_kpis()` - Identify KPIs below target
- `add_revenue_metrics()` - Record monthly revenue
- `get_revenue_trend()` - Revenue trends over time
- `generate_kpi_summary()` - Comprehensive KPI report

### 2. `business_operations.py` - Operational Management

**Classes:**
- `ProcessStep` - Individual workflow step with dependencies
- `SOP` - Standard Operating Procedure definition
- `WorkflowInstance` - Active workflow execution
- `OperationalManager` - Workflow and SOP management

**Key Methods:**
- `create_workflow()` - Start new workflow from SOP
- `complete_workflow_step()` - Mark step complete
- `get_active_workflows()` - List active workflows
- `get_overdue_workflows()` - Identify delayed workflows
- `generate_operational_report()` - Operational efficiency report

### 3. `ai_ops_assistant.py` - Issue Tracking (Legacy)

**Classes:**
- `Issue` - Quality/downtime issue
- `OpsAssistant` - Issue management and analysis

**Key Methods:**
- `add_issue()` - Create new issue
- `ask_why()` - Record 5-Whys analysis
- `finalize_root_cause()` - Set root cause
- `draw_fishbone()` - Generate Ishikawa diagram
- `generate_report()` - Create PDF report

---

## Business Procedures

See **[BUSINESS_PROCEDURES.md](BUSINESS_PROCEDURES.md)** for comprehensive documentation of:

- Standard Operating Procedures (SOPs)
- Key Performance Indicators (KPIs)
- Process workflows and timelines
- Success criteria and targets
- Operational metrics and reporting

---

## KPI Reference

### Revenue KPIs

| KPI | Target | Description |
|-----|--------|-------------|
| MRR | $50K+ | Monthly Recurring Revenue |
| ARR | $600K+ | Annual Recurring Revenue |
| Growth Rate | 15%+ | Month-over-month revenue growth |
| Average Deal Size | $50K+ | Average contract value |

### Client KPIs

| KPI | Target | Description |
|-----|--------|-------------|
| Total Clients | 20+ | Number of active clients |
| Retention Rate | 95%+ | Client retention percentage |
| CSAT | 8.5/10 | Client satisfaction score |
| NPS | 50+ | Net Promoter Score |
| LTV | $200K+ | Client Lifetime Value |

### Project KPIs

| KPI | Target | Description |
|-----|--------|-------------|
| On-Time Delivery | 90%+ | Projects delivered on schedule |
| Success Rate | 95%+ | Projects completed successfully |
| Avg Delivery Time | 60-75 days | Average project duration |
| Team Utilization | 85% | Percentage of capacity used |

### Operational KPIs

| KPI | Target | Description |
|-----|--------|-------------|
| Response Time | < 2 hours | Average support response time |
| Uptime | 99.9%+ | System availability |
| Automation Rate | 80%+ | Processes automated |

---

## Architecture

```
ai_ops_assistant/
│
├── streamlit_app.py           # 🌐 Web dashboard (main entry point)
├── business_kpis.py           # KPI tracking system
├── business_operations.py     # SOP and workflow management
├── ai_ops_assistant.py        # Legacy issue tracking module
├── demo_business_system.py    # CLI demo script
├── BUSINESS_PROCEDURES.md     # Comprehensive procedures doc
├── DEPLOYMENT.md              # Deployment guide for Streamlit Cloud
├── requirements.txt           # Python dependencies
├── .streamlit/                # Streamlit configuration
│   ├── config.toml
│   └── secrets.toml
└── README.md                  # This file
```

### Data Flow

```
┌─────────────────┐
│  Business Data  │
└────────┬────────┘
         │
         ├──────────────┐
         │              │
┌────────▼─────────┐    │
│  KPI Dashboard   │    │
│  - Revenue       │    │
│  - Clients       │    │
│  - Projects      │    │
└──────────────────┘    │
                        │
              ┌─────────▼──────────┐
              │ Operational Mgr    │
              │ - SOPs             │
              │ - Workflows        │
              │ - Compliance       │
              └─────────┬──────────┘
                        │
              ┌─────────▼──────────┐
              │  Reports & Alerts  │
              │  - PDF Reports     │
              │  - KPI Dashboards  │
              │  - Notifications   │
              └────────────────────┘
```

---

## Usage Examples

### Example 1: Monthly Business Review

```python
from business_kpis import create_standard_kpis
from business_operations import OperationalManager
import datetime as dt

# Initialize systems
dashboard = create_standard_kpis()
ops = OperationalManager()

# Record this month's metrics
# ... (add revenue, client, project metrics)

# Generate reports
kpi_summary = dashboard.generate_kpi_summary()
ops_report = ops.generate_operational_report()

print("=== Monthly Business Review ===")
print(f"Total KPIs: {kpi_summary['total_kpis']}")
print(f"On Target: {kpi_summary['on_target']}")
print(f"Underperforming: {kpi_summary['underperforming']}")
print(f"\nActive Workflows: {ops_report['summary']['active_workflows']}")
print(f"Overdue Workflows: {ops_report['summary']['overdue_workflows']}")
```

### Example 2: Sales Pipeline Tracking

```python
# Track multiple sales opportunities
deals = [
    ("deal_acme", "Acme Corp", Priority.HIGH),
    ("deal_initech", "Initech", Priority.MEDIUM),
    ("deal_hooli", "Hooli", Priority.HIGH),
]

for deal_id, company, priority in deals:
    workflow = ops.create_workflow(
        sop_id="sales_001",
        instance_id=deal_id,
        target_days=21,
        priority=priority,
        client_name=company
    )

# Monitor pipeline
active = ops.get_active_workflows()
print(f"Active Deals: {len(active)}")
for wf in active:
    print(f"- {wf.client_name}: {wf.sop.get_completion_percentage():.0f}% complete")
```

### Example 3: Project Delivery Tracking

```python
# Create project workflow
project = ops.create_workflow(
    sop_id="delivery_001",
    instance_id="proj_warehouse_automation",
    target_days=75,
    priority=Priority.CRITICAL,
    client_name="Acme Corp",
    project_name="Warehouse Automation System"
)

# As project progresses, complete steps
ops.complete_workflow_step("proj_warehouse_automation", "d1", duration=2.0)
ops.complete_workflow_step("proj_warehouse_automation", "d2", duration=18.0)

# Check status
if project.is_overdue():
    print(f"⚠️  Project is overdue by {project.get_time_remaining()} days")
else:
    print(f"✅ Project on track, {project.get_time_remaining()} days remaining")
```

---

## Contributing

We welcome contributions! Here's how to get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (when available)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Update documentation for new features
- Test thoroughly before submitting

---

## Support

### Documentation

- **Business Procedures:** See [BUSINESS_PROCEDURES.md](BUSINESS_PROCEDURES.md)
- **API Reference:** See inline docstrings in each module
- **Examples:** See [Usage Examples](#usage-examples) section

### Contact

- **Operations:** ops@fynix.systems
- **Technical Support:** support@fynix.systems
- **Sales:** sales@fynix.systems
- **Website:** https://fynix.systems/

### Issues & Bugs

Please report issues on GitHub: https://github.com/kwisener01/ai_ops_assistant/issues

---

## License

This project is proprietary software of Fynix Systems.

---

## Acknowledgments

- Built for **Fynix Systems** (https://fynix.systems/)
- Inspired by Dan Sullivan's strategic thinking frameworks
- Incorporates industry best practices for AI automation businesses

---

## Roadmap

### Phase 1 (Current)
- ✅ KPI tracking system
- ✅ SOP and workflow management
- ✅ Business procedures documentation
- ✅ Basic reporting
- ✅ **Streamlit web dashboard**
- ✅ **Interactive data visualization**

### Phase 2 (Next)
- ⬜ Automated alerts and notifications
- ⬜ Integration with HubSpot/Zendesk
- ⬜ Advanced analytics and predictions
- ⬜ Data persistence (database integration)

### Phase 3 (Future)
- ⬜ Mobile app
- ⬜ AI-powered recommendations
- ⬜ Real-time collaboration features
- ⬜ Custom SOP builder

---

**Fynix Systems - Automating the Future with AI** 🚀

*Last Updated: January 22, 2025*
