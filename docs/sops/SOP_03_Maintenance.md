# Standard Operating Procedure: Maintenance Department
## Transmission Assembly Plant

---

## SOP Information

| Field | Details |
|-------|---------|
| **SOP ID** | MAINT-001 |
| **Department** | Maintenance |
| **Effective Date** | 2025-11-22 |
| **Version** | 1.0 |
| **Owner** | Maintenance Manager |
| **Review Frequency** | Quarterly |

---

## 1. Purpose and Scope

### 1.1 Purpose
This SOP establishes maintenance procedures to maximize equipment reliability, minimize downtime, and ensure safe operation of all manufacturing equipment.

### 1.2 Scope
Applies to all maintenance personnel including technicians, electricians, and maintenance management.

---

## 2. Responsibilities

### 2.1 Maintenance Manager
- Maintenance strategy and planning
- Resource allocation
- Budget management
- Performance monitoring
- Contractor coordination

### 2.2 Maintenance Supervisor
- Daily work assignment
- Priority setting
- Parts and materials management
- Technician development
- Coordination with Production

### 2.3 Maintenance Technician
- Execute preventive and corrective maintenance
- Troubleshoot equipment issues
- Document all work performed
- Maintain clean and organized work areas
- Comply with safety procedures

### 2.4 Maintenance Planner
- Schedule preventive maintenance
- Develop work orders
- Parts procurement coordination
- Backlog management
- Performance metrics tracking

---

## 3. Types of Maintenance

### 3.1 Preventive Maintenance (PM)

**Definition**: Scheduled maintenance to prevent failures

**Frequency-Based PM:**
- Daily: Lubrication, visual inspections
- Weekly: Filter changes, adjustments
- Monthly: Detailed inspections, testing
- Quarterly: Major component servicing
- Annual: Equipment overhaul, certifications

**Usage-Based PM:**
- Hours of operation
- Cycles completed
- Units produced

**Procedure:**

**Step 1: PM Schedule Review**
- Check PM schedule in maintenance system
- Print work order and checklist
- Gather required parts and tools
- Review equipment manual if needed

**Step 2: Lockout/Tagout (LOTO)**
- Notify Production of pending shutdown
- Follow LOTO procedure (see Section 6)
- Verify zero energy state
- Post signage and barriers

**Step 3: PM Execution**
- Follow PM checklist step-by-step
- Inspect for wear, damage, abnormal conditions
- Perform measurements and tests
- Lubricate per lubrication chart
- Replace parts per PM schedule
- Clean equipment thoroughly

**Step 4: Documentation**
- Complete all checklist items
- Record readings and measurements
- Note any abnormal findings
- Photograph issues if applicable
- Sign and date work order

**Step 5: Testing and Startup**
- Remove LOTO devices
- Perform functional test
- Verify proper operation
- Return equipment to Production
- Update maintenance system

**Step 6: Follow-up**
- Enter data into AI Ops Assistant
- Create work orders for identified issues
- Update PM frequency if warranted
- Replenish spare parts used

### 3.2 Predictive Maintenance (PdM)

**Definition**: Condition-based maintenance using monitoring tools

**Techniques:**
- Vibration analysis
- Thermal imaging
- Oil analysis
- Ultrasonic testing
- Motor current analysis

**Procedure:**

**Step 1: Route Planning**
- Establish monitoring routes
- Identify critical equipment
- Determine monitoring frequency
- Baseline normal conditions

**Step 2: Data Collection**
- Use appropriate monitoring tool
- Follow manufacturer procedures
- Record readings in consistent locations
- Compare to baseline and trends
- Document environmental conditions

**Step 3: Analysis**
- Review data for anomalies
- Compare to alarm thresholds
- Trend analysis over time
- Consult equipment manuals
- Consult with specialists if needed

**Step 4: Action Decision**

| Condition | Action |
|-----------|--------|
| Normal | Continue monitoring |
| Caution (trending) | Increase monitoring frequency |
| Alert | Schedule corrective maintenance |
| Critical | Immediate shutdown and repair |

**Step 5: Documentation**
- Log findings in AI Ops Assistant
- Create work order if action needed
- Update equipment history
- Share information with team

### 3.3 Corrective Maintenance (CM)

**Definition**: Reactive maintenance in response to failure or issue

**Priority Levels:**

| Priority | Response Time | Examples |
|----------|---------------|----------|
| Emergency | Immediate | Safety hazard, production stoppage |
| Urgent | Within 2 hours | Significant production impact |
| Routine | Within 24 hours | Minor impact, workaround available |
| Planned | Scheduled | No current impact |

**Procedure:**

**Step 1: Work Request Receipt**
- Receive notification (call, email, system alert)
- Log in maintenance system immediately
- Assign work order number
- Determine priority level
- Estimate response time

**Step 2: Initial Assessment**
- Respond per priority timeline
- Interview equipment operator
- Observe equipment operation
- Review recent maintenance history
- Determine scope of issue

**Step 3: Troubleshooting**
- Review equipment manuals and drawings
- Perform diagnostic tests
- Isolate root cause
- Determine repair strategy
- Estimate downtime and parts needed

**Step 4: Repair Execution**
- Obtain authorization if extensive repair
- Implement LOTO if required
- Execute repair per plan
- Replace parts as needed
- Test repair thoroughly

**Step 5: Documentation**
- Complete work order with details
- Record parts used
- Document root cause
- Enter in AI Ops Assistant for tracking
- Update equipment history

**Step 6: Follow-up**
- Monitor equipment for 24-48 hours
- Verify issue resolved
- Identify preventive actions
- Update PM plans if applicable

### 3.4 Emergency Maintenance

**Triggers:**
- Safety hazard
- Fire or environmental release
- Complete production shutdown
- Critical infrastructure failure

**Procedure:**

**Step 1: Immediate Response**
- Respond immediately (drop current work)
- Assess safety risks
- Implement emergency procedures if needed
- Notify supervisor and management

**Step 2: Containment**
- Secure area, prevent injuries
- Stop further damage if possible
- Call for additional help if needed
- Implement temporary fix if safe

**Step 3: Permanent Repair**
- Develop repair plan
- Coordinate with Production on timing
- Execute repair with appropriate resources
- Verify complete resolution

**Step 4: Root Cause Analysis**
- Mandatory for all emergency situations
- Conduct 5-Whys analysis
- Create fishbone diagram
- Document in AI Ops Assistant
- Implement preventive measures

---

## 4. Work Order Management

### 4.1 Work Order Lifecycle

1. **Request**: Issue identified and logged
2. **Planning**: Scope, parts, schedule determined
3. **Scheduling**: Added to maintenance calendar
4. **Execution**: Work performed
5. **Documentation**: Work order completed
6. **Closure**: Review and approval
7. **Analysis**: Performance metrics updated

### 4.2 Work Order Information

**Required Fields:**
- Work order number (unique)
- Equipment ID and description
- Priority level
- Problem description
- Assigned technician
- Estimated hours
- Parts required
- Safety requirements
- LOTO required (yes/no)

### 4.3 Backlog Management

**Weekly Review:**
- Review all open work orders
- Re-prioritize as needed
- Address aging work orders (>30 days)
- Coordinate with Production for scheduling
- Identify resource constraints

**Target Metrics:**
- Emergency backlog: 0
- Urgent backlog: <5 work orders
- Routine backlog: <20 work orders
- Average age: <14 days

---

## 5. Spare Parts Management

### 5.1 Critical Spare Parts Inventory

**Categories:**
- **Critical**: Long lead time, production impact (12 months supply)
- **Essential**: Moderate lead time, frequent use (6 months supply)
- **Standard**: Short lead time, readily available (3 months supply)

### 5.2 Parts Replenishment

**Min/Max System:**
- Minimum stock level triggers reorder
- Maximum stock level prevents overstocking
- Automatic reorder for critical parts
- Manual review for non-critical parts

### 5.3 Parts Documentation

**For Each Part:**
- Part number (manufacturer and internal)
- Description and specifications
- Equipment compatibility list
- Supplier information
- Lead time
- Current stock level
- Location in parts room
- Cost

### 5.4 Parts Room Organization

**Best Practices:**
- Labeled bins and shelving
- Alphabetical or numerical arrangement
- First-in, first-out (FIFO) rotation
- Segregation by equipment type
- Climate control for sensitive parts
- Regular cycle counts (monthly)

---

## 6. Lockout/Tagout (LOTO) Procedure

### 6.1 When LOTO is Required

**Mandatory for:**
- Maintenance or repair of equipment
- Cleaning or unjamming
- Adjustment or setup of machinery
- Any work on energized equipment

### 6.2 Six-Step LOTO Procedure

**Step 1: Preparation**
- Identify all energy sources (electrical, pneumatic, hydraulic, mechanical, thermal)
- Identify all isolation points
- Notify affected personnel
- Gather LOTO devices and tools

**Step 2: Shutdown**
- Follow normal shutdown procedure
- Use equipment controls to shut down
- Allow equipment to coast to complete stop

**Step 3: Isolation**
- Operate isolation devices (disconnect switches, valves, etc.)
- Place in safe or off position
- Do not rely on control circuits

**Step 4: Lockout/Tagout Application**
- Apply personal lock and tag to each isolation point
- Only use assigned personal locks
- Tag must include: name, date, reason
- Each person applies their own lock (no exceptions)

**Step 5: Verify Zero Energy**
- Test start controls (should not operate)
- Use test equipment to verify no voltage
- Check for stored energy (springs, capacitors, elevated components)
- Bleed/block/dissipate stored energy
- Verify safe to work

**Step 6: Equipment Release**
- Remove all tools and reinstall guards
- Verify all personnel clear
- Remove locks and tags (only person who applied)
- Restore energy sources
- Test equipment operation
- Notify Production equipment is ready

### 6.3 Group Lockout

**When Required:**
- Multiple personnel working on same equipment
- Shift changes during extended work

**Procedure:**
- Supervisor applies master lock to lockbox
- Lockbox contains isolation device keys
- Each worker applies personal lock to lockbox
- Worker removes only their lock when complete
- Supervisor removes master lock when all personnel finished

---

## 7. Safety Procedures

### 7.1 Required PPE
- Safety glasses (ANSI Z87.1)
- Steel-toed boots
- Hearing protection (when in production areas)
- Gloves appropriate for task
- Arc flash PPE for electrical work (NFPA 70E)

### 7.2 Confined Space Entry
- Permit required for any confined space
- Atmospheric testing before entry
- Continuous monitoring during work
- Attendant stationed at entry point
- Rescue equipment available
- Communication method established

### 7.3 Working at Heights
- Fall protection required >4 feet
- Harness, lanyard, and anchor point
- Ladder inspection before each use
- Three points of contact on ladders
- Scaffolding inspected and tagged

### 7.4 Hot Work (Welding, Cutting, Grinding)
- Hot work permit required
- Fire watch assigned
- Fire extinguisher within 30 feet
- Combustibles removed or protected
- Area inspection 30 minutes after completion

---

## 8. Equipment Documentation

### 8.1 Equipment Records

**For Each Asset:**
- Equipment ID (unique identifier)
- Manufacturer and model
- Serial number
- Installation date
- Criticality rating
- Maintenance history
- Parts list and drawings
- Operator manuals
- Maintenance manuals
- Safety information

### 8.2 Equipment History Tracking

**Recorded Information:**
- All maintenance performed (PM and CM)
- Parts replaced
- Failures and root causes
- Downtime incidents
- Performance degradation trends
- Modifications or upgrades

**Analysis Uses:**
- Identify chronic issues
- Optimize PM frequencies
- Support capital replacement decisions
- Improve MTBF and MTTR
- Cost tracking and budgeting

---

## 9. Contractor Management

### 9.1 Contractor Selection
- Verify licenses and certifications
- Check insurance coverage
- Review safety record
- Obtain references
- Establish contract terms

### 9.2 Contractor On-Site Requirements

**Before Work Begins:**
- Site-specific safety orientation
- Review scope of work
- Provide facility access and keys
- Coordinate with Production schedule
- Establish communication protocol

**During Work:**
- Daily check-ins with supervisor
- Safety compliance monitoring
- Quality inspections
- Progress documentation

**After Completion:**
- Final inspection and acceptance
- Obtain as-built drawings if applicable
- Collect certifications and test reports
- Update equipment records
- Process payment

---

## 10. Lubrication Program

### 10.1 Lubrication Schedule

**Equipment Categories:**
- Gearboxes: Monthly oil level check, annual oil change
- Bearings: Weekly greasing (high-speed), monthly (low-speed)
- Chains and conveyors: Weekly lubrication
- Hydraulic systems: Daily level check, semi-annual oil change
- Pneumatic systems: Daily oil level check

### 10.2 Lubrication Procedure

**Step 1: Preparation**
- Review lubrication route sheet
- Gather correct lubricants
- Ensure grease gun and oil can clean
- Gather rags and safety equipment

**Step 2: Execution**
- Wipe grease fittings clean before greasing
- Apply grease until new grease appears or per manual
- Check oil levels, top off as needed
- Look for leaks or abnormal conditions
- Check for contamination

**Step 3: Documentation**
- Initial route sheet
- Note any abnormalities
- Create work order for issues found
- Replenish lubricants used

### 10.3 Lubricant Management
- Store in clean, dry area
- Label all containers clearly
- Use color-coded grease guns
- Prevent cross-contamination
- Dispose of waste oil properly (environmental compliance)

---

## 11. Performance Metrics

### 11.1 Daily Metrics
- Emergency work orders received
- Work orders completed
- Equipment downtime hours
- PM compliance (scheduled vs. completed)

### 11.2 Weekly Metrics
- Mean Time Between Failures (MTBF)
- Mean Time To Repair (MTTR)
- Overall Equipment Effectiveness (OEE)
- Planned vs. unplanned downtime ratio
- Work order backlog

### 11.3 Monthly Metrics
- PM compliance percentage
- Total maintenance cost
- Cost per unit produced
- Spare parts inventory turnover
- Contractor costs
- Equipment availability percentage
- Number of repeat failures

### 11.4 Key Performance Indicators (KPIs)

| KPI | Target | Measurement |
|-----|--------|-------------|
| PM Compliance | >95% | PM completed on time / PM scheduled |
| Equipment Uptime | >95% | Available time / Scheduled time |
| MTBF | >500 hours | Operating time / Number of failures |
| MTTR | <2 hours | Total repair time / Number of repairs |
| Planned Maintenance % | >80% | Planned hours / Total maintenance hours |

---

## 12. AI Operations Assistant Integration

### 12.1 Issue Logging
- Log all equipment failures
- Document root causes using 5-Whys
- Track corrective actions
- Monitor recurring issues

### 12.2 Predictive Analytics
- Identify failure patterns
- Optimize PM schedules
- Forecast spare parts needs
- Highlight chronic equipment issues

### 12.3 Reporting
- Generate maintenance performance reports
- Analyze downtime trends
- Calculate equipment reliability metrics
- Support data-driven decision making

---

## 13. Training Requirements

### 13.1 New Hire Training
- Safety orientation (8 hours)
- LOTO procedure (4 hours)
- Equipment familiarization (40 hours)
- Work order system (2 hours)
- AI Ops Assistant (2 hours)
- On-the-job training (160 hours)

### 13.2 Specialized Certifications
- Electrical (licensed electrician)
- Welding (AWS certification)
- Rigging and crane operation
- Confined space entry
- Arc flash training (NFPA 70E)
- Refrigeration (if applicable)

### 13.3 Continuing Education
- Annual safety refresher (4 hours)
- New equipment training
- Advanced troubleshooting techniques
- Manufacturer training on new equipment

---

## 14. Energy Efficiency

### 14.1 Energy Conservation Measures
- Repair compressed air leaks promptly
- Optimize HVAC schedules
- Use energy-efficient lighting
- Monitor motor loading and efficiency
- Implement variable frequency drives (VFDs) where applicable

### 14.2 Energy Audits
- Annual walk-through of facility
- Identify energy waste
- Benchmark energy usage
- Implement improvement projects
- Track energy cost per unit produced

---

## 15. Key Forms and Documents

- Preventive Maintenance Checklist
- Work Order Form
- Lockout/Tagout Log
- Equipment Maintenance History
- Spare Parts Request
- Contractor Work Permit
- Hot Work Permit
- Confined Space Entry Permit
- Lubrication Route Sheet

---

## Revision History

| Version | Date | Author | Description of Changes |
|---------|------|--------|------------------------|
| 1.0 | 2025-11-22 | AI Ops Team | Initial release |

---

## Approval Signatures

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Maintenance Manager | | | |
| Safety Manager | | | |
| Plant Manager | | | |

---

**Document Owner**: Maintenance Manager
**Next Review Date**: 2026-02-22
**Distribution**: All Maintenance Personnel
