"""
Fynix Systems - AI Operations Assistant
Streamlit Web Dashboard

Interactive web interface for business operations, KPI tracking,
and workflow management.

To run:
    streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import datetime as dt
from business_kpis import (
    KPIDashboard, KPI, KPICategory, RevenueMetrics,
    ClientMetrics, ProjectMetrics, create_standard_kpis
)
from business_operations import (
    OperationalManager, Priority, ProcessStatus
)
import plotly.express as px
import plotly.graph_objects as go


# Page configuration
st.set_page_config(
    page_title="Fynix Systems - Operations Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .kpi-good {
        color: #28a745;
        font-weight: bold;
    }
    .kpi-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .kpi-bad {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'dashboard' not in st.session_state:
    st.session_state.dashboard = create_standard_kpis()

if 'ops_manager' not in st.session_state:
    st.session_state.ops_manager = OperationalManager()

if 'initialized' not in st.session_state:
    # Add some demo data
    _init_demo_data()
    st.session_state.initialized = True


def _init_demo_data():
    """Initialize with demo data for demonstration."""
    dashboard = st.session_state.dashboard
    ops = st.session_state.ops_manager

    # Add historical revenue data
    for i in range(6):
        date = dt.date.today() - dt.timedelta(days=30 * (6 - i))
        revenue = RevenueMetrics(
            mrr=38000 + i * 2000,
            arr=(38000 + i * 2000) * 12,
            growth_rate=8 + i * 1.5,
            new_revenue=3000 + i * 500,
            expansion_revenue=1200 + i * 200,
            churn_revenue=1000 - i * 50,
            date=date
        )
        dashboard.add_revenue_metrics(revenue)

    # Add client metrics
    for i in range(6):
        date = dt.date.today() - dt.timedelta(days=30 * (6 - i))
        clients = ClientMetrics(
            total_clients=12 + i,
            new_clients=2 if i > 0 else 12,
            churned_clients=1 if i > 2 else 0,
            retention_rate=90 + i * 0.8,
            satisfaction_score=7.8 + i * 0.15,
            nps=40 + i * 2,
            avg_client_ltv=180000 + i * 5000,
            date=date
        )
        dashboard.add_client_metrics(clients)

    # Add project metrics
    for i in range(6):
        date = dt.date.today() - dt.timedelta(days=30 * (6 - i))
        projects = ProjectMetrics(
            total_projects=5 + i,
            completed_projects=4 + i - 1,
            in_progress_projects=1 + (i % 2),
            avg_delivery_time=75 - i * 2,
            on_time_delivery_rate=85 + i * 1.5,
            success_rate=93 + i * 0.5,
            team_utilization=75 + i * 2,
            date=date
        )
        dashboard.add_project_metrics(projects)

    # Create demo workflows
    ops.create_workflow("sales_001", "deal_acme", 21, Priority.HIGH, "Acme Corp", "Warehouse Automation")
    ops.create_workflow("sales_001", "deal_initech", 21, Priority.MEDIUM, "Initech Inc", "Process Automation")
    ops.create_workflow("delivery_001", "proj_warehouse", 75, Priority.CRITICAL, "Acme Corp", "Warehouse System")


def render_sidebar():
    """Render sidebar navigation."""
    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/1f77b4/ffffff?text=Fynix+Systems", use_container_width=True)
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["🏠 Dashboard", "📊 KPI Tracking", "🔄 Workflows", "📈 Analytics", "⚙️ Settings"],
            index=0
        )

        st.markdown("---")
        st.markdown("### Quick Stats")

        dashboard = st.session_state.dashboard
        kpi_summary = dashboard.generate_kpi_summary()

        st.metric("Total KPIs", kpi_summary['total_kpis'])
        st.metric("On Target", f"{kpi_summary['on_target']}/{kpi_summary['total_kpis']}")

        avg_perf = kpi_summary.get('avg_performance', 0)
        st.metric("Avg Performance", f"{avg_perf:.1%}")

        st.markdown("---")
        st.markdown("**Fynix Systems**")
        st.markdown("[fynix.systems](https://fynix.systems/)")

    return page


def render_dashboard():
    """Render main dashboard page."""
    st.markdown('<h1 class="main-header">🚀 Fynix Systems Operations Dashboard</h1>', unsafe_allow_html=True)

    dashboard = st.session_state.dashboard
    ops = st.session_state.ops_manager

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    # Get latest metrics
    revenue_trend = dashboard.get_revenue_trend(months=1)
    client_trend = dashboard.get_client_trend(months=1)
    project_trend = dashboard.get_project_trend(months=1)

    if not revenue_trend.empty:
        latest_mrr = revenue_trend.iloc[-1]['mrr']
        growth_rate = revenue_trend.iloc[-1]['growth_rate']

        with col1:
            st.metric(
                "Monthly Recurring Revenue",
                f"${latest_mrr:,.0f}",
                f"{growth_rate:.1f}% growth",
                delta_color="normal"
            )

    if not client_trend.empty:
        latest_clients = client_trend.iloc[-1]['total_clients']
        retention = client_trend.iloc[-1]['retention_rate']

        with col2:
            st.metric(
                "Total Clients",
                f"{latest_clients:.0f}",
                f"{retention:.1f}% retention",
                delta_color="normal"
            )

    if not project_trend.empty:
        on_time = project_trend.iloc[-1]['on_time_delivery_rate']
        success = project_trend.iloc[-1]['success_rate']

        with col3:
            st.metric(
                "On-Time Delivery",
                f"{on_time:.1f}%",
                f"{success:.1f}% success rate",
                delta_color="normal"
            )

    ops_report = ops.generate_operational_report()
    with col4:
        active = ops_report['summary']['active_workflows']
        overdue = ops_report['summary']['overdue_workflows']

        st.metric(
            "Active Workflows",
            f"{active}",
            f"{overdue} overdue" if overdue > 0 else "All on track",
            delta_color="inverse" if overdue > 0 else "off"
        )

    st.markdown("---")

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Revenue Trend (6 Months)")
        revenue_data = dashboard.get_revenue_trend(months=6)

        if not revenue_data.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=revenue_data['date'],
                y=revenue_data['mrr'],
                mode='lines+markers',
                name='MRR',
                line=dict(color='#1f77b4', width=3)
            ))
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="MRR ($)",
                hovermode='x unified',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("👥 Client Growth")
        client_data = dashboard.get_client_trend(months=6)

        if not client_data.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=client_data['date'],
                y=client_data['total_clients'],
                name='Total Clients',
                marker_color='#2ca02c'
            ))
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Number of Clients",
                hovermode='x unified',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

    # KPI Performance
    st.markdown("---")
    st.subheader("🎯 KPI Performance Overview")

    underperforming = dashboard.get_underperforming_kpis()

    if underperforming:
        st.warning(f"⚠️ {len(underperforming)} KPIs are below target")

        cols = st.columns(3)
        for idx, kpi in enumerate(underperforming[:6]):
            with cols[idx % 3]:
                perf = kpi.performance_ratio()
                st.markdown(f"""
                <div class="metric-card">
                    <strong>{kpi.name}</strong><br>
                    <span class="kpi-warning">{kpi.value}{kpi.unit}</span> / {kpi.target}{kpi.unit}<br>
                    <small>Performance: {perf:.1%}</small>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.success("✅ All KPIs are on target!")


def render_kpi_tracking():
    """Render KPI tracking page."""
    st.markdown('<h1 class="main-header">📊 KPI Tracking</h1>', unsafe_allow_html=True)

    dashboard = st.session_state.dashboard

    # Tabs for different KPI categories
    tab1, tab2, tab3, tab4 = st.tabs(["💰 Revenue", "👥 Clients", "🚀 Projects", "⚙️ Operations"])

    with tab1:
        st.subheader("Revenue Metrics")

        # Input form for new revenue data
        with st.expander("➕ Add Revenue Data"):
            col1, col2, col3 = st.columns(3)

            with col1:
                mrr = st.number_input("MRR ($)", min_value=0, value=50000, step=1000)
                growth_rate = st.number_input("Growth Rate (%)", min_value=-100.0, value=15.0, step=0.1)

            with col2:
                new_revenue = st.number_input("New Revenue ($)", min_value=0, value=5000, step=100)
                expansion_revenue = st.number_input("Expansion Revenue ($)", min_value=0, value=2000, step=100)

            with col3:
                churn_revenue = st.number_input("Churn Revenue ($)", min_value=0, value=1000, step=100)
                metric_date = st.date_input("Date", value=dt.date.today())

            if st.button("Add Revenue Metrics", type="primary"):
                revenue = RevenueMetrics(
                    mrr=mrr,
                    arr=mrr * 12,
                    growth_rate=growth_rate,
                    new_revenue=new_revenue,
                    expansion_revenue=expansion_revenue,
                    churn_revenue=churn_revenue,
                    date=metric_date
                )
                dashboard.add_revenue_metrics(revenue)
                st.success("Revenue metrics added successfully!")
                st.rerun()

        # Display revenue trend
        revenue_data = dashboard.get_revenue_trend(months=12)

        if not revenue_data.empty:
            st.subheader("Revenue Trend")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=revenue_data['date'],
                y=revenue_data['mrr'],
                mode='lines+markers',
                name='MRR',
                line=dict(color='#1f77b4', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=revenue_data['date'],
                y=revenue_data['net_new_mrr'],
                mode='lines+markers',
                name='Net New MRR',
                line=dict(color='#2ca02c', width=2)
            ))
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Amount ($)",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(revenue_data, use_container_width=True)

    with tab2:
        st.subheader("Client Metrics")

        # Input form for client data
        with st.expander("➕ Add Client Data"):
            col1, col2 = st.columns(2)

            with col1:
                total_clients = st.number_input("Total Clients", min_value=0, value=20, step=1)
                new_clients = st.number_input("New Clients", min_value=0, value=3, step=1)
                churned_clients = st.number_input("Churned Clients", min_value=0, value=1, step=1)
                retention_rate = st.number_input("Retention Rate (%)", min_value=0.0, max_value=100.0, value=95.0, step=0.1)

            with col2:
                csat = st.number_input("CSAT Score (0-10)", min_value=0.0, max_value=10.0, value=8.5, step=0.1)
                nps = st.number_input("NPS", min_value=-100, max_value=100, value=50, step=1)
                ltv = st.number_input("Avg Client LTV ($)", min_value=0, value=200000, step=10000)
                metric_date = st.date_input("Date", value=dt.date.today(), key="client_date")

            if st.button("Add Client Metrics", type="primary"):
                clients = ClientMetrics(
                    total_clients=total_clients,
                    new_clients=new_clients,
                    churned_clients=churned_clients,
                    retention_rate=retention_rate,
                    satisfaction_score=csat,
                    nps=nps,
                    avg_client_ltv=ltv,
                    date=metric_date
                )
                dashboard.add_client_metrics(clients)
                st.success("Client metrics added successfully!")
                st.rerun()

        # Display client trend
        client_data = dashboard.get_client_trend(months=12)

        if not client_data.empty:
            st.subheader("Client Growth & Retention")

            col1, col2 = st.columns(2)

            with col1:
                fig = px.line(client_data, x='date', y='total_clients',
                             title='Total Clients Over Time',
                             markers=True)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.line(client_data, x='date', y=['satisfaction_score', 'nps'],
                             title='CSAT & NPS Trends',
                             markers=True)
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(client_data, use_container_width=True)

    with tab3:
        st.subheader("Project Metrics")

        # Input form for project data
        with st.expander("➕ Add Project Data"):
            col1, col2 = st.columns(2)

            with col1:
                total_projects = st.number_input("Total Projects", min_value=0, value=12, step=1)
                completed = st.number_input("Completed Projects", min_value=0, value=10, step=1)
                in_progress = st.number_input("In Progress", min_value=0, value=2, step=1)

            with col2:
                avg_delivery = st.number_input("Avg Delivery Time (days)", min_value=0, value=65, step=1)
                on_time = st.number_input("On-Time Delivery (%)", min_value=0.0, max_value=100.0, value=90.0, step=0.1)
                success_rate = st.number_input("Success Rate (%)", min_value=0.0, max_value=100.0, value=95.0, step=0.1)

            utilization = st.slider("Team Utilization (%)", min_value=0, max_value=100, value=85)
            metric_date = st.date_input("Date", value=dt.date.today(), key="project_date")

            if st.button("Add Project Metrics", type="primary"):
                projects = ProjectMetrics(
                    total_projects=total_projects,
                    completed_projects=completed,
                    in_progress_projects=in_progress,
                    avg_delivery_time=avg_delivery,
                    on_time_delivery_rate=on_time,
                    success_rate=success_rate,
                    team_utilization=utilization,
                    date=metric_date
                )
                dashboard.add_project_metrics(projects)
                st.success("Project metrics added successfully!")
                st.rerun()

        # Display project trend
        project_data = dashboard.get_project_trend(months=12)

        if not project_data.empty:
            st.subheader("Project Performance")

            col1, col2 = st.columns(2)

            with col1:
                fig = px.line(project_data, x='date', y='on_time_delivery_rate',
                             title='On-Time Delivery Rate',
                             markers=True)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.line(project_data, x='date', y='team_utilization',
                             title='Team Utilization',
                             markers=True)
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(project_data, use_container_width=True)

    with tab4:
        st.subheader("Operational KPIs")

        st.info("📝 Operational KPIs can be customized based on your specific needs.")

        # Display all KPIs
        kpi_df = dashboard.to_dataframe()

        if not kpi_df.empty:
            st.dataframe(kpi_df, use_container_width=True)


def render_workflows():
    """Render workflow management page."""
    st.markdown('<h1 class="main-header">🔄 Workflow Management</h1>', unsafe_allow_html=True)

    ops = st.session_state.ops_manager

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Active Workflows", "Create Workflow", "Reports"])

    with tab1:
        st.subheader("Active Workflows")

        active_workflows = ops.get_active_workflows()

        if not active_workflows:
            st.info("No active workflows. Create one using the 'Create Workflow' tab.")
        else:
            for wf in active_workflows:
                with st.expander(f"{wf.client_name} - {wf.project_name or wf.sop.name}"):
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Progress", f"{wf.sop.get_completion_percentage():.0f}%")

                    with col2:
                        st.metric("Priority", wf.priority.value.upper())

                    with col3:
                        days_left = wf.get_time_remaining()
                        st.metric("Days Remaining", days_left)

                    with col4:
                        status = "🔴 Overdue" if wf.is_overdue() else "🟢 On Track"
                        st.metric("Status", status)

                    st.markdown("**Next Steps:**")
                    next_steps = wf.sop.get_next_steps()

                    if next_steps:
                        for step in next_steps:
                            st.markdown(f"- {step.name} ({step.estimated_duration:.1f}h) - {step.responsible}")
                    else:
                        st.success("All steps completed!")

                    # Show all steps
                    st.markdown("**All Steps:**")
                    for step in wf.sop.steps:
                        icon = "✅" if step.status == ProcessStatus.COMPLETED else "⏳"
                        st.markdown(f"{icon} {step.name} - {step.status.value}")

    with tab2:
        st.subheader("Create New Workflow")

        # List available SOPs
        sops = ops.list_all_sops()
        sop_options = {f"{sop['name']} ({sop['category']})": sop['sop_id'] for sop in sops}

        selected_sop = st.selectbox("Select SOP Template", list(sop_options.keys()))

        col1, col2 = st.columns(2)

        with col1:
            instance_id = st.text_input("Workflow ID", value=f"wf_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}")
            client_name = st.text_input("Client Name", value="")
            project_name = st.text_input("Project Name", value="")

        with col2:
            target_days = st.number_input("Target Completion (days)", min_value=1, value=30, step=1)
            priority = st.selectbox("Priority", ["LOW", "MEDIUM", "HIGH", "CRITICAL"])

        if st.button("Create Workflow", type="primary"):
            if client_name:
                sop_id = sop_options[selected_sop]
                priority_enum = Priority[priority]

                try:
                    ops.create_workflow(
                        sop_id=sop_id,
                        instance_id=instance_id,
                        target_days=target_days,
                        priority=priority_enum,
                        client_name=client_name,
                        project_name=project_name if project_name else None
                    )
                    st.success(f"Workflow '{instance_id}' created successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error creating workflow: {e}")
            else:
                st.error("Please enter a client name")

    with tab3:
        st.subheader("Operational Reports")

        report = ops.generate_operational_report()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Workflows", report['summary']['total_workflows'])

        with col2:
            st.metric("Active", report['summary']['active_workflows'])

        with col3:
            st.metric("Completed", report['summary']['completed_workflows'])

        with col4:
            st.metric("Overdue", report['summary']['overdue_workflows'])

        st.markdown("---")

        st.subheader("Workflows by Priority")
        priority_data = pd.DataFrame([
            {"Priority": k.title(), "Count": v}
            for k, v in report['by_priority'].items()
            if v > 0
        ])

        if not priority_data.empty:
            fig = px.pie(priority_data, values='Count', names='Priority',
                        title='Active Workflows by Priority')
            st.plotly_chart(fig, use_container_width=True)

        # Overdue details
        if report['overdue_details']:
            st.subheader("⚠️ Overdue Workflows")
            overdue_df = pd.DataFrame(report['overdue_details'])
            st.dataframe(overdue_df, use_container_width=True)


def render_analytics():
    """Render analytics page."""
    st.markdown('<h1 class="main-header">📈 Business Analytics</h1>', unsafe_allow_html=True)

    dashboard = st.session_state.dashboard

    # KPI Summary
    st.subheader("KPI Performance Summary")

    summary = dashboard.generate_kpi_summary()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total KPIs", summary['total_kpis'])

    with col2:
        on_target_pct = (summary['on_target'] / summary['total_kpis'] * 100) if summary['total_kpis'] > 0 else 0
        st.metric("On Target", f"{summary['on_target']}/{summary['total_kpis']}", f"{on_target_pct:.0f}%")

    with col3:
        avg_perf = summary.get('avg_performance', 0)
        st.metric("Avg Performance", f"{avg_perf:.1%}")

    st.markdown("---")

    # All KPIs table
    st.subheader("All KPIs")

    kpi_df = dashboard.to_dataframe()

    if not kpi_df.empty:
        # Add performance ratio
        kpi_df['performance'] = kpi_df.apply(
            lambda row: f"{(row['value'] / row['target'] * 100):.1f}%" if row['target'] > 0 else "N/A",
            axis=1
        )

        st.dataframe(
            kpi_df[['name', 'category', 'value', 'target', 'unit', 'performance', 'date']],
            use_container_width=True
        )

        # Download button
        csv = kpi_df.to_csv(index=False)
        st.download_button(
            label="📥 Download KPI Data (CSV)",
            data=csv,
            file_name=f"kpis_{dt.date.today()}.csv",
            mime="text/csv"
        )


def render_settings():
    """Render settings page."""
    st.markdown('<h1 class="main-header">⚙️ Settings</h1>', unsafe_allow_html=True)

    st.subheader("System Configuration")

    st.info("🚧 Settings page coming soon!")

    st.markdown("""
    **Planned Features:**
    - Custom KPI targets
    - Email notifications
    - Data export/import
    - User management
    - Theme customization
    """)

    st.markdown("---")

    st.subheader("About")
    st.markdown("""
    **Fynix Systems - AI Operations Assistant**

    Version: 1.0.0

    This dashboard provides comprehensive business operations management including:
    - Real-time KPI tracking
    - Workflow management
    - Performance analytics
    - Automated reporting

    For more information, visit [fynix.systems](https://fynix.systems/)
    """)

    if st.button("🔄 Reset Demo Data"):
        st.session_state.clear()
        st.success("Demo data reset! Refreshing page...")
        st.rerun()


# Main app
def main():
    """Main application entry point."""
    page = render_sidebar()

    if page == "🏠 Dashboard":
        render_dashboard()
    elif page == "📊 KPI Tracking":
        render_kpi_tracking()
    elif page == "🔄 Workflows":
        render_workflows()
    elif page == "📈 Analytics":
        render_analytics()
    elif page == "⚙️ Settings":
        render_settings()


if __name__ == "__main__":
    main()
