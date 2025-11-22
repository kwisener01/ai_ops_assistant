# Fynix Systems - AI Operations Assistant

**Fynix Systems AI-Powered Operations Management Platform**

This repository contains the AI Operations Assistant developed specifically for Fynix Systems' manufacturing operations. The assistant is designed to streamline quality management, downtime analysis, and continuous improvement initiatives across our production facilities.

## Overview

The Fynix Systems AI Operations Assistant is a sophisticated tool that helps operations managers and production teams track issues, conduct root cause analysis, and generate comprehensive reports. By automating data collection, analysis, and reporting, the assistant enables our teams to focus on high-leverage activities and strategic improvements.

## Key Features

### 🔍 **Issue Tracking & Management**
- Comprehensive tracking of quality and downtime issues
- Persistent storage using CSV/Excel formats
- Detailed issue metadata including status, categories, and timelines
- Custom categorization aligned with Fynix Systems' operational framework

### 🎯 **Root Cause Analysis**
- Integrated 5-Whys methodology for systematic problem solving
- Ishikawa (fishbone) diagram visualization
- Support for the 6M framework (Man, Machine, Material, Method, Measurement, Environment)
- Structured approach to identify and document corrective actions

### 📊 **Analytics & Pattern Detection**
- Automated detection of recurring issues
- Category-based analysis and trending
- Mean time to resolution (MTTR) calculations
- Data-driven insights for continuous improvement

### 📈 **Automated Reporting**
- Professional PDF report generation
- Customizable date ranges and filters
- Visual dashboards with embedded charts
- Progress tracking using Gap and Gain methodology

### 📁 **Multi-Format Data Ingestion**
- Excel and CSV file import/export
- PDF text extraction for legacy data
- PowerPoint presentation parsing
- Flexible integration with existing Fynix Systems workflows

## Productivity Framework

The assistant incorporates proven strategic thinking frameworks:

- **Who Not How**: Identify delegation opportunities and collaborative solutions
- **The Gap and The Gain**: Track progress against past performance, not ideals
- **10× is Easier than 2×**: Focus on exponential improvements rather than incremental gains
- **Free Zone Frontier**: Explore collaborative opportunities beyond competitive constraints

## Installation

### Prerequisites

```bash
pip install pandas numpy matplotlib fpdf pdfplumber python-pptx openpyxl
```

### Quick Start

```python
from ai_ops_assistant import OpsAssistant

# Initialize the assistant
assistant = OpsAssistant()

# Load existing data
assistant.load_data("fynix_issues.xlsx")

# Add a new issue
issue_id = assistant.add_issue(
    "Hydraulic pressure fluctuation in Station 3",
    category="Machine"
)

# Conduct 5-Whys analysis
assistant.ask_why(issue_id, "Why is pressure fluctuating?", "Worn seal in valve assembly")
assistant.ask_why(issue_id, "Why is the seal worn?", "Exceeded service interval")
assistant.ask_why(issue_id, "Why was service interval exceeded?", "Maintenance schedule not updated")
assistant.finalize_root_cause(issue_id)

# Add corrective action
assistant.add_corrective_action(issue_id, "Update maintenance schedule and implement automated reminders")

# Generate report
assistant.generate_report("fynix_weekly_report.pdf")
```

## Usage Examples

### Running the Demo

```bash
python ai_ops_assistant.py
```

This will generate:
- `demo_fishbone.png` - Sample Ishikawa diagram
- `demo_report.pdf` - Sample operations report

### Integration with Fynix Systems

The assistant can be integrated into various Fynix Systems workflows:

- **Streamlit Dashboard**: Real-time operations monitoring
- **Scheduled Reporting**: Automated daily/weekly summary generation
- **Jupyter Notebooks**: Ad-hoc analysis and exploratory data analysis
- **CI/CD Pipeline**: Automated quality metrics tracking

## File Structure

```
ai_ops_assistant/
├── ai_ops_assistant.py    # Core module with OpsAssistant class
├── README.md              # This file
└── requirements.txt       # Python dependencies
```

## Data Format

Issues are stored with the following structure:

| Field | Type | Description |
|-------|------|-------------|
| issue_id | int | Unique identifier |
| description | str | Issue description |
| date_raised | date | When the issue was identified |
| status | str | Open/Closed |
| category | str | Issue category (6M framework) |
| root_cause | str | Identified root cause |
| corrective_actions | list | List of corrective actions |
| whys | list | 5-Whys question-answer pairs |

## Customization for Fynix Systems

This tool has been specifically tailored for Fynix Systems with:

- Custom report branding
- Integration with existing data formats
- Alignment with Fynix operational excellence standards
- Support for multi-facility deployments

## Support & Documentation

For questions or support related to the Fynix Systems AI Operations Assistant, please contact your operations manager or the continuous improvement team.

## License

Proprietary - Fynix Systems Internal Use Only

---

**Fynix Systems** - Driving Operational Excellence Through AI-Powered Insights
