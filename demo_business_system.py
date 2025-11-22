"""
demo_business_system.py
========================

Comprehensive demonstration of the Fynix Systems AI Operations Assistant.
This script shows how to use all components together for business operations,
KPI tracking, workflow management, and reporting.

Run this demo to see the system in action:
    python demo_business_system.py
"""

import datetime as dt
from business_kpis import (
    KPIDashboard, RevenueMetrics, ClientMetrics, ProjectMetrics,
    create_standard_kpis
)
from business_operations import OperationalManager, Priority


def demo_kpi_tracking():
    """Demonstrate KPI tracking and monitoring."""
    print("=" * 70)
    print("DEMO 1: KPI Tracking & Monitoring")
    print("=" * 70)

    # Create KPI dashboard
    dashboard = create_standard_kpis()

    # Record January metrics
    jan_revenue = RevenueMetrics(
        mrr=42000,
        arr=504000,
        growth_rate=10.5,
        new_revenue=4500,
        expansion_revenue=1800,
        churn_revenue=1300,
        date=dt.date(2025, 1, 1)
    )
    dashboard.add_revenue_metrics(jan_revenue)

    jan_clients = ClientMetrics(
        total_clients=15,
        new_clients=2,
        churned_clients=1,
        retention_rate=93.3,
        satisfaction_score=8.2,
        nps=48,
        avg_client_ltv=200000,
        date=dt.date(2025, 1, 1)
    )
    dashboard.add_client_metrics(jan_clients)

    jan_projects = ProjectMetrics(
        total_projects=8,
        completed_projects=6,
        in_progress_projects=2,
        avg_delivery_time=68,
        on_time_delivery_rate=83.3,
        success_rate=100.0,
        team_utilization=82.0,
        date=dt.date(2025, 1, 1)
    )
    dashboard.add_project_metrics(jan_projects)

    # Record February metrics (improved)
    feb_revenue = RevenueMetrics(
        mrr=48000,
        arr=576000,
        growth_rate=14.3,
        new_revenue=6500,
        expansion_revenue=2200,
        churn_revenue=700,
        date=dt.date(2025, 2, 1)
    )
    dashboard.add_revenue_metrics(feb_revenue)

    feb_clients = ClientMetrics(
        total_clients=18,
        new_clients=4,
        churned_clients=1,
        retention_rate=94.4,
        satisfaction_score=8.7,
        nps=52,
        avg_client_ltv=220000,
        date=dt.date(2025, 2, 1)
    )
    dashboard.add_client_metrics(feb_clients)

    feb_projects = ProjectMetrics(
        total_projects=10,
        completed_projects=7,
        in_progress_projects=3,
        avg_delivery_time=62,
        on_time_delivery_rate=91.4,
        success_rate=100.0,
        team_utilization=87.0,
        date=dt.date(2025, 2, 1)
    )
    dashboard.add_project_metrics(feb_projects)

    # Generate KPI summary
    print("\n📊 KPI Summary")
    print("-" * 70)
    summary = dashboard.generate_kpi_summary()
    print(f"Total KPIs Tracked: {summary['total_kpis']}")
    print(f"KPIs On Target: {summary['on_target']}")
    print(f"Underperforming KPIs: {summary['underperforming']}")
    print(f"Average Performance: {summary['avg_performance']:.1%}")

    # Show underperforming KPIs
    print("\n⚠️  Underperforming KPIs (< 90% of target)")
    print("-" * 70)
    underperforming = dashboard.get_underperforming_kpis()
    if underperforming:
        for kpi in underperforming:
            perf = kpi.performance_ratio()
            print(f"  • {kpi.name}: {kpi.value}{kpi.unit} / {kpi.target}{kpi.unit} ({perf:.1%})")
    else:
        print("  ✅ All KPIs are on target!")

    # Show revenue trend
    print("\n📈 Revenue Trend (Last 2 Months)")
    print("-" * 70)
    revenue_trend = dashboard.get_revenue_trend(months=2)
    for _, row in revenue_trend.iterrows():
        print(f"  {row['date']}: MRR ${row['mrr']:,.0f} | Growth {row['growth_rate']:.1f}% | Net New MRR ${row['net_new_mrr']:,.0f}")

    # Show client trend
    print("\n👥 Client Trend (Last 2 Months)")
    print("-" * 70)
    client_trend = dashboard.get_client_trend(months=2)
    for _, row in client_trend.iterrows():
        print(f"  {row['date']}: {row['total_clients']} clients | Retention {row['retention_rate']:.1f}% | CSAT {row['satisfaction_score']:.1f}/10")

    print("\n✅ KPI Tracking Demo Complete\n")


def demo_workflow_management():
    """Demonstrate workflow and SOP management."""
    print("=" * 70)
    print("DEMO 2: Workflow & SOP Management")
    print("=" * 70)

    # Initialize operational manager
    ops = OperationalManager()

    # List available SOPs
    print("\n📋 Available Standard Operating Procedures")
    print("-" * 70)
    sops = ops.list_all_sops()
    for sop in sops:
        print(f"  • {sop['name']} ({sop['category']})")
        print(f"    - Steps: {sop['steps_count']}")
        print(f"    - Estimated Duration: {sop['estimated_duration']:.0f} hours")
        print(f"    - Owner: {sop['owner']}")

    # Create sales workflows for three opportunities
    print("\n🎯 Creating Sales Workflows")
    print("-" * 70)

    deals = [
        ("deal_acme_corp", "Acme Corporation", "Warehouse Automation", Priority.HIGH),
        ("deal_initech", "Initech Inc", "Process Automation", Priority.MEDIUM),
        ("deal_hooli", "Hooli Technologies", "AI Customer Support", Priority.CRITICAL),
    ]

    for deal_id, company, project, priority in deals:
        workflow = ops.create_workflow(
            sop_id="sales_001",
            instance_id=deal_id,
            target_days=21,
            priority=priority,
            client_name=company,
            project_name=project
        )
        print(f"  ✓ Created: {company} - {project} (Priority: {priority.value})")

    # Progress through workflows at different stages
    print("\n⏩ Progressing Workflows")
    print("-" * 70)

    # Acme Corp - halfway through
    ops.complete_workflow_step("deal_acme_corp", "s1", duration=0.8, notes="Strong fit, budget confirmed")
    ops.complete_workflow_step("deal_acme_corp", "s2", duration=1.5, notes="Great discovery call")
    ops.complete_workflow_step("deal_acme_corp", "s3", duration=2.5, notes="Detailed needs analysis complete")
    ops.complete_workflow_step("deal_acme_corp", "s4", duration=7.0, notes="Solution designed, ROI looks great")
    print("  ✓ Acme Corp: 50% complete")

    # Initech - just started
    ops.complete_workflow_step("deal_initech", "s1", duration=0.5, notes="Qualified lead")
    ops.complete_workflow_step("deal_initech", "s2", duration=2.0, notes="Discovery call scheduled")
    print("  ✓ Initech: 25% complete")

    # Hooli - almost done
    for step in ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]:
        ops.complete_workflow_step("deal_hooli", step, duration=2.0)
    print("  ✓ Hooli: 87.5% complete (negotiation done, awaiting signature)")

    # Create a project delivery workflow
    print("\n🚀 Creating Project Delivery Workflow")
    print("-" * 70)

    project_wf = ops.create_workflow(
        sop_id="delivery_001",
        instance_id="proj_acme_warehouse",
        target_days=75,
        priority=Priority.HIGH,
        client_name="Acme Corporation",
        project_name="Warehouse Automation System"
    )
    print("  ✓ Created: Warehouse Automation System for Acme Corp")

    # Progress through first few steps
    ops.complete_workflow_step("proj_acme_warehouse", "d1", duration=2.0, notes="Kickoff successful")
    ops.complete_workflow_step("proj_acme_warehouse", "d2", duration=15.0, notes="Requirements documented")
    print("  ✓ Project: 18% complete")

    # Show active workflows
    print("\n📊 Active Workflows Status")
    print("-" * 70)
    active = ops.get_active_workflows()
    for wf in active:
        completion = wf.sop.get_completion_percentage()
        days_left = wf.get_time_remaining()
        status_icon = "🔴" if wf.is_overdue() else "🟢"
        print(f"  {status_icon} {wf.client_name} - {wf.project_name or wf.sop.name}")
        print(f"     Progress: {completion:.0f}% | Days Remaining: {days_left} | Priority: {wf.priority.value}")

    # Generate operational report
    print("\n📈 Operational Report")
    print("-" * 70)
    report = ops.generate_operational_report()
    print(f"  Total Workflows: {report['summary']['total_workflows']}")
    print(f"  Active: {report['summary']['active_workflows']}")
    print(f"  Completed: {report['summary']['completed_workflows']}")
    print(f"  Overdue: {report['summary']['overdue_workflows']}")
    print(f"  On-Time Completion Rate: {report['summary']['on_time_completion_rate']:.1f}%")

    print("\n  Workflows by Priority:")
    for priority, count in report['by_priority'].items():
        if count > 0:
            print(f"    • {priority.title()}: {count}")

    print("\n✅ Workflow Management Demo Complete\n")


def demo_integrated_system():
    """Demonstrate integrated use of KPIs and workflows."""
    print("=" * 70)
    print("DEMO 3: Integrated Business Operations")
    print("=" * 70)

    dashboard = create_standard_kpis()
    ops = OperationalManager()

    print("\n🎯 Monthly Business Review Simulation")
    print("-" * 70)

    # Record current month metrics
    current_revenue = RevenueMetrics(
        mrr=52000,
        arr=624000,
        growth_rate=16.7,
        new_revenue=8000,
        expansion_revenue=2500,
        churn_revenue=1500,
        date=dt.date.today()
    )
    dashboard.add_revenue_metrics(current_revenue)

    current_clients = ClientMetrics(
        total_clients=20,
        new_clients=3,
        churned_clients=1,
        retention_rate=95.0,
        satisfaction_score=8.8,
        nps=55,
        avg_client_ltv=240000,
        date=dt.date.today()
    )
    dashboard.add_client_metrics(current_clients)

    current_projects = ProjectMetrics(
        total_projects=12,
        completed_projects=10,
        in_progress_projects=2,
        avg_delivery_time=58,
        on_time_delivery_rate=95.0,
        success_rate=100.0,
        team_utilization=88.0,
        date=dt.date.today()
    )
    dashboard.add_project_metrics(current_projects)

    # Create some active workflows
    ops.create_workflow("sales_001", "deal_xyz", 21, Priority.HIGH, "XYZ Corp")
    ops.create_workflow("delivery_001", "proj_abc", 75, Priority.CRITICAL, "ABC Inc")
    ops.create_workflow("support_001", "ticket_123", 2, Priority.HIGH, "Acme Corp")

    # Generate comprehensive review
    print("\n📊 BUSINESS PERFORMANCE SUMMARY")
    print("-" * 70)

    kpi_summary = dashboard.generate_kpi_summary()
    ops_report = ops.generate_operational_report()

    print("\n💰 Revenue Performance:")
    print(f"  • MRR: ${current_revenue.mrr:,.0f} (Target: $50,000)")
    print(f"  • ARR: ${current_revenue.arr:,.0f} (Target: $600,000)")
    print(f"  • Growth Rate: {current_revenue.growth_rate:.1f}% (Target: 15%)")
    print(f"  • Net New MRR: ${current_revenue.net_new_mrr():,.0f}")
    print(f"  ✅ Revenue targets EXCEEDED!")

    print("\n👥 Client Performance:")
    print(f"  • Total Clients: {current_clients.total_clients} (Target: 20)")
    print(f"  • Retention Rate: {current_clients.retention_rate:.1f}% (Target: 95%)")
    print(f"  • CSAT Score: {current_clients.satisfaction_score:.1f}/10 (Target: 8.5)")
    print(f"  • NPS: {current_clients.nps} (Target: 50)")
    print(f"  ✅ Client targets MET!")

    print("\n🚀 Project Performance:")
    print(f"  • Completed Projects: {current_projects.completed_projects}")
    print(f"  • On-Time Delivery: {current_projects.on_time_delivery_rate:.1f}% (Target: 90%)")
    print(f"  • Success Rate: {current_projects.success_rate:.1f}% (Target: 95%)")
    print(f"  • Team Utilization: {current_projects.team_utilization:.1f}% (Target: 85%)")
    print(f"  ✅ Project targets EXCEEDED!")

    print("\n⚙️  Operational Performance:")
    print(f"  • Active Workflows: {ops_report['summary']['active_workflows']}")
    print(f"  • On-Time Completion: {ops_report['summary']['on_time_completion_rate']:.1f}%")
    print(f"  • Overdue Workflows: {ops_report['summary']['overdue_workflows']}")

    print("\n📈 Overall Assessment:")
    print(f"  • Total KPIs Tracked: {kpi_summary['total_kpis']}")
    print(f"  • KPIs On Target: {kpi_summary['on_target']}/{kpi_summary['total_kpis']}")
    print(f"  • Average Performance: {kpi_summary['avg_performance']:.1%}")

    if kpi_summary['avg_performance'] >= 1.0:
        print("\n  🎉 OUTSTANDING PERFORMANCE! All targets met or exceeded!")
    elif kpi_summary['avg_performance'] >= 0.9:
        print("\n  ✅ STRONG PERFORMANCE! Most targets achieved!")
    else:
        print("\n  ⚠️  IMPROVEMENT NEEDED - Review underperforming areas")

    print("\n✅ Integrated System Demo Complete\n")


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  FYNIX SYSTEMS - AI OPERATIONS ASSISTANT DEMO  ".center(68) + "║")
    print("║" + "  Comprehensive Business Operations Management  ".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\n")

    try:
        # Run all demos
        demo_kpi_tracking()
        input("Press Enter to continue to Workflow Management demo...")

        demo_workflow_management()
        input("Press Enter to continue to Integrated System demo...")

        demo_integrated_system()

        # Summary
        print("=" * 70)
        print("🎉 ALL DEMOS COMPLETE!")
        print("=" * 70)
        print("\nThe AI Operations Assistant provides:")
        print("  ✅ Real-time KPI tracking across all business metrics")
        print("  ✅ Standardized operational procedures (SOPs)")
        print("  ✅ Active workflow management with dependency tracking")
        print("  ✅ Performance analytics and trend analysis")
        print("  ✅ Automated reporting and alerts")
        print("\nNext Steps:")
        print("  1. Customize KPI targets for your business")
        print("  2. Adapt SOPs to your specific processes")
        print("  3. Integrate with your existing tools (CRM, PM, etc.)")
        print("  4. Set up automated reporting and dashboards")
        print("\nFor more information, see README.md and BUSINESS_PROCEDURES.md")
        print("\n" + "=" * 70 + "\n")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user. Exiting...")
    except Exception as e:
        print(f"\n\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
