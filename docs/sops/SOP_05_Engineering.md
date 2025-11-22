# Standard Operating Procedure: Engineering Department
## Transmission Assembly Plant

---

## SOP Information

| Field | Details |
|-------|---------|
| **SOP ID** | ENG-001 |
| **Department** | Engineering |
| **Effective Date** | 2025-11-22 |
| **Version** | 1.0 |
| **Owner** | Engineering Manager |
| **Review Frequency** | Quarterly |

---

## 1. Purpose and Scope

### 1.1 Purpose
This SOP defines engineering procedures for process improvement, problem solving, new product introduction, and technical support to manufacturing operations.

### 1.2 Scope
Applies to all engineering personnel including process engineers, manufacturing engineers, and technical specialists.

---

## 2. Responsibilities

### 2.1 Engineering Manager
- Strategic planning for engineering initiatives
- Project portfolio management
- Resource allocation
- Capital equipment justification
- Cross-functional coordination

### 2.2 Process Engineer
- Process optimization and improvement
- Cycle time reduction
- Yield improvement
- Standard work development
- Manufacturing process documentation

### 2.3 Manufacturing Engineer
- Production support and troubleshooting
- Equipment specification and installation
- Process validation
- Tooling and fixture design
- Ergonomics and safety improvements

### 2.4 Quality Engineer
- Process capability studies
- Root cause analysis facilitation
- Statistical process control (SPC) implementation
- Measurement system analysis
- Design of Experiments (DOE)

---

## 3. New Product Introduction (NPI)

### 3.1 NPI Process Phases

**Phase 1: Concept and Feasibility (Weeks 1-4)**
- Review customer requirements and specifications
- Assess manufacturing capability
- Identify gaps and required investments
- Create preliminary process flow
- Develop cost estimates
- Present feasibility report to management

**Phase 2: Design and Development (Weeks 5-12)**
- Develop detailed process flow
- Design tooling and fixtures
- Create work instructions
- Specify equipment requirements
- Conduct Design Failure Mode and Effects Analysis (DFMEA)
- Conduct Process FMEA (PFMEA)
- Create control plan

**Phase 3: Validation and Testing (Weeks 13-16)**
- Build prototype units
- Conduct process capability studies (Cpk targets: ≥1.67)
- Perform measurement system analysis (Gage R&R)
- Run pilot production
- Validate quality and performance
- Document lessons learned

**Phase 4: Production Launch (Weeks 17-20)**
- Train production personnel
- Conduct first article inspection
- Monitor closely for first 100 units
- Collect data and verify capability
- Transition to production team
- Hold post-launch review meeting

### 3.2 Engineering Change Management

**Change Request Process:**

**Step 1: Change Request Initiation**
- Complete Engineering Change Request (ECR) form
- Describe current state and proposed change
- Justify need (quality, cost, safety, customer requirement)
- Estimate impact on cost, schedule, and quality
- Obtain department manager approval

**Step 2: Impact Assessment**
- Engineering reviews all impacts:
  - Bill of materials (BOM) changes
  - Tooling or equipment modifications
  - Work instruction updates
  - Quality inspection changes
  - Supplier changes
  - Inventory implications (use-up, scrap, rework)

**Step 3: Cross-Functional Review**
- Present to Change Control Board (CCB)
- Members: Engineering, Production, Quality, Materials, Finance
- Discuss risks and mitigation plans
- Decision: Approve, Reject, or Request More Information

**Step 4: Implementation Planning**
- Create Engineering Change Order (ECO)
- Assign tasks and responsibilities
- Establish timeline
- Plan transition strategy (effective date, serial number cutover, etc.)
- Communicate to all stakeholders

**Step 5: Execution**
- Update drawings and documentation
- Revise BOM in system
- Update work instructions
- Train affected personnel
- Modify tooling/equipment as needed
- Implement changes per plan

**Step 6: Verification**
- Verify changes implemented correctly
- Conduct first article inspection
- Monitor production for issues
- Document effectiveness
- Close ECO

---

## 4. Continuous Improvement

### 4.1 Kaizen Events

**When to Use:**
- Specific problem or opportunity identified
- Cross-functional solution needed
- Implementation can occur within 1 week

**Kaizen Event Process:**

**Pre-Event (2 weeks before):**
- Define scope and objectives
- Select team (6-8 people, cross-functional)
- Gather baseline data
- Reserve conference room and resources

**Event Week (5 days):**

**Day 1: Current State**
- Team introduction and goal review
- Observe current process (Gemba walk)
- Create process map
- Collect data
- Identify waste and inefficiencies

**Day 2: Root Cause Analysis**
- Conduct 5-Whys analysis
- Create fishbone diagrams
- Identify root causes using AI Ops Assistant
- Prioritize issues to address

**Day 3: Ideation and Planning**
- Brainstorm solutions
- Evaluate ideas (impact vs. effort matrix)
- Select best solutions
- Create implementation plan

**Day 4: Implementation**
- Execute improvement actions
- Modify process, layout, or equipment
- Update documentation
- Test new process

**Day 5: Verification and Presentation**
- Collect data on improved process
- Calculate results (time savings, quality improvement, cost reduction)
- Prepare presentation
- Present to management and stakeholders
- Celebrate success

**Post-Event:**
- Sustain improvements (visual management, standard work)
- Monitor metrics for 30 days
- Document in AI Ops Assistant
- Share learnings organization-wide

### 4.2 Value Stream Mapping

**Purpose:** Identify and eliminate waste in end-to-end processes

**Process:**

**Step 1: Select Value Stream**
- Choose product family or process
- Define start and end points
- Form mapping team

**Step 2: Current State Map**
- Walk the process from start to finish
- Document each process step:
  - Cycle time (C/T)
  - Changeover time (C/O)
  - Uptime
  - Number of operators
  - Work in process (WIP)
  - Batch sizes
- Map information flow
- Calculate total lead time and value-added time

**Step 3: Analyze**
- Calculate value-added ratio (VAT / Total Lead Time)
- Identify waste categories:
  - Overproduction
  - Waiting
  - Transport
  - Over-processing
  - Inventory
  - Motion
  - Defects
  - Underutilized talent
- Prioritize improvement opportunities

**Step 4: Future State Map**
- Design ideal process flow
- Apply lean principles:
  - Continuous flow where possible
  - Pull systems
  - Level production
  - Quick changeovers
  - Error-proofing (Poka-Yoke)
- Set targets for improvement

**Step 5: Implementation Plan**
- Break into projects
- Assign owners and deadlines
- Sequence projects logically
- Execute per plan
- Track progress in AI Ops Assistant

---

## 5. Process Capability Studies

### 5.1 When Required
- New product launch
- New process or equipment
- After significant process change
- Customer request
- Quality issue investigation

### 5.2 Procedure

**Step 1: Planning**
- Identify critical characteristics
- Determine sample size (minimum 30 units, prefer 100+)
- Select measurement equipment (Gage R&R <10%)
- Ensure process is stable (no adjustments during study)

**Step 2: Data Collection**
- Produce parts under normal operating conditions
- Measure characteristic(s) for each unit
- Record all data with timestamps
- Note any special events or conditions

**Step 3: Analysis**
- Plot histogram of data
- Calculate mean (μ) and standard deviation (σ)
- Create control charts (X-bar and R chart)
- Assess process stability (in statistical control)

**Step 4: Capability Calculation**

**Process Capability Indices:**

**Cp (Potential Capability):**
```
Cp = (USL - LSL) / (6σ)
```
Measures capability if perfectly centered

**Cpk (Actual Capability):**
```
Cpk = min[(USL - μ) / (3σ), (μ - LSL) / (3σ)]
```
Accounts for centering

**Targets:**
- Cpk ≥ 1.67: Capable (desired)
- Cpk = 1.33 - 1.67: Acceptable
- Cpk < 1.33: Not capable, action required

**Step 5: Reporting**
- Document results in capability study report
- Include histogram, control charts, and calculations
- Recommend actions if not capable:
  - Center process
  - Reduce variation (process improvement)
  - Tighten specifications (with customer approval)
- Log in AI Ops Assistant

---

## 6. Root Cause Analysis and Problem Solving

### 6.1 8D Problem Solving

**When to Use:** Customer complaints, significant quality issues, recurring problems

**8 Disciplines:**

**D1: Form Team**
- Select cross-functional team
- Define roles and responsibilities
- Schedule regular meetings

**D2: Define Problem**
- Describe problem in quantifiable terms
- Who, What, When, Where, How, How Many
- Take photos or videos if applicable

**D3: Implement Interim Containment**
- Prevent problem from reaching customer
- Quarantine suspect inventory
- Implement 100% inspection if necessary
- Verify effectiveness of containment

**D4: Root Cause Analysis**
- Conduct 5-Whys analysis
- Create Ishikawa (fishbone) diagram using AI Ops Assistant
- Use data to verify root cause
- Determine escape point (where problem should have been caught)

**D5: Choose Permanent Corrective Actions**
- Brainstorm solutions
- Evaluate effectiveness and feasibility
- Select best solution(s)
- Plan implementation

**D6: Implement Corrective Actions**
- Execute plan
- Update procedures and work instructions
- Train personnel
- Verify implementation

**D7: Prevent Recurrence**
- Review similar processes for same issue
- Update FMEA
- Implement across organization (horizontal deployment)
- Update standards and best practices

**D8: Congratulate Team**
- Document lessons learned
- Share results with organization
- Recognize team contributions
- Celebrate success
- Close 8D report

**Documentation:**
- Use standardized 8D report template
- Log entire process in AI Ops Assistant
- Archive for future reference

### 6.2 Failure Mode and Effects Analysis (FMEA)

**Types:**
- **Design FMEA (DFMEA)**: Analyze product design
- **Process FMEA (PFMEA)**: Analyze manufacturing process

**PFMEA Procedure:**

**Step 1: Define Scope**
- Select process to analyze
- Assemble team
- Review process flow diagram

**Step 2: Identify Potential Failure Modes**
- For each process step, ask "What could go wrong?"
- List all possible failure modes

**Step 3: Identify Effects**
- For each failure mode, determine effect on customer/next operation
- Assess severity (1-10 scale, 10 = most severe)

**Step 4: Identify Causes**
- Determine potential causes of each failure mode
- Assess occurrence (1-10 scale, 10 = most frequent)

**Step 5: Identify Controls**
- List current process controls (detection methods)
- Assess detection (1-10 scale, 10 = least detectable)

**Step 6: Calculate Risk Priority Number (RPN)**
```
RPN = Severity × Occurrence × Detection
```

**Step 7: Prioritize Actions**
- Focus on high RPN items (typically RPN >100)
- Also address high severity regardless of RPN

**Step 8: Develop Action Plan**
- Identify actions to reduce RPN:
  - Reduce severity (design change)
  - Reduce occurrence (process improvement)
  - Improve detection (inspection, sensors)
- Assign responsibility and target date
- Implement actions
- Recalculate RPN

**Step 9: Documentation**
- Maintain FMEA as living document
- Update when process changes
- Review annually
- File in engineering records

---

## 7. Equipment and Tooling Management

### 7.1 Equipment Specification and Procurement

**Procedure:**

**Step 1: Requirements Definition**
- Define functional requirements
- Specify performance criteria (cycle time, accuracy, capacity)
- Identify safety requirements
- Establish budget constraints

**Step 2: Vendor Research**
- Identify potential suppliers
- Request proposals (RFP)
- Evaluate technical capability
- Check references

**Step 3: Evaluation and Selection**
- Compare proposals against requirements
- Assess total cost of ownership (purchase, installation, operating costs)
- Consider serviceability and spare parts availability
- Visit supplier facility if significant investment
- Select vendor

**Step 4: Purchase and Installation**
- Issue purchase order
- Coordinate delivery logistics
- Plan installation (utilities, rigging, space preparation)
- Conduct Factory Acceptance Test (FAT) at vendor site
- Oversee installation
- Conduct Site Acceptance Test (SAT)
- Train operators and maintenance

**Step 5: Documentation**
- Obtain manuals and drawings
- Create equipment file
- Update equipment inventory
- Establish preventive maintenance plan
- Log in AI Ops Assistant

### 7.2 Tooling and Fixture Design

**Design Process:**
1. Define functional requirements
2. Create concept sketches
3. Develop 3D CAD model
4. Perform tolerance stack-up analysis
5. Create detailed drawings
6. Build or procure tooling
7. Test and validate
8. Document and maintain

**Design Principles:**
- Error-proofing (Poka-Yoke): Make it impossible to assemble incorrectly
- Ergonomics: Minimize operator strain
- Repeatability: Ensure consistent positioning
- Quick changeover: Minimize setup time
- Durability: Suitable for production volume
- Maintainability: Easy to service and adjust

---

## 8. Standard Work and Work Instructions

### 8.1 Standard Work Development

**Components:**
1. **Takt Time**: Available time / Customer demand
2. **Work Sequence**: Optimized sequence of tasks
3. **Standard WIP**: Minimum in-process inventory needed

**Procedure:**

**Step 1: Observe Current Process**
- Time study of current method
- Video record if helpful
- Interview operators for insights

**Step 2: Develop Improved Method**
- Eliminate waste (non-value-added activities)
- Balance workload
- Ensure safety and quality
- Achieve takt time

**Step 3: Document**
- Create Standard Work Chart
- Specify work sequence
- Note critical quality checks
- Include safety points
- Photograph optimal positions

**Step 4: Validate**
- Test with operator
- Refine as needed
- Verify takt time achievable
- Ensure quality maintained

**Step 5: Implement**
- Train all operators
- Post at workstation
- Monitor adherence
- Continuous improvement

### 8.2 Work Instruction Creation

**Format:**
- Step-by-step procedure
- Clear, concise language
- Photos or diagrams for clarity
- Critical dimensions and tolerances
- Inspection requirements
- Safety warnings highlighted

**Approval and Control:**
- Reviewed by supervisor and quality
- Approved by department manager
- Version controlled (revision number and date)
- Distributed to affected personnel
- Old versions removed and archived

---

## 9. Ergonomics and Safety

### 9.1 Ergonomic Assessment

**When Required:**
- New workstation design
- New product introduction
- Operator discomfort reported
- Repetitive motion injury

**Assessment Process:**
1. Observe operator performing task
2. Identify risk factors:
   - Repetitive motions
   - Forceful exertions
   - Awkward postures
   - Contact stress
   - Vibration
3. Use ergonomic assessment tools (REBA, RULA)
4. Prioritize improvements
5. Implement changes (height adjustments, lift assists, tool modifications)
6. Re-assess and verify improvement

### 9.2 Machine Guarding

**Requirements:**
- All moving parts guarded
- Guards cannot be bypassed easily
- Maintain visibility for operation
- Allow access for maintenance
- Comply with OSHA standards

**Design:**
- Fixed guards (preferred when access not needed)
- Interlocked guards (stops machine when opened)
- Light curtains or laser scanners
- Two-hand controls
- Emergency stop buttons accessible

---

## 10. Data Collection and Analysis

### 10.1 Statistical Tools

**Common Tools:**
- **Control Charts**: Monitor process stability
- **Pareto Charts**: Identify vital few vs. trivial many
- **Scatter Diagrams**: Examine correlation between variables
- **Histograms**: Visualize distribution
- **Box Plots**: Compare multiple data sets

### 10.2 Design of Experiments (DOE)

**When to Use:** Optimize process with multiple variables

**Process:**
1. Define objective (quality improvement, cost reduction, etc.)
2. Identify factors (controllable variables)
3. Determine levels for each factor
4. Design experiment matrix (full factorial, fractional factorial, Taguchi)
5. Conduct experiments
6. Analyze results (ANOVA)
7. Identify significant factors and optimal settings
8. Verify results
9. Implement optimal process settings

---

## 11. Performance Metrics

### 11.1 Daily Metrics
- Production support requests
- Open engineering tasks
- Drawing releases

### 11.2 Weekly Metrics
- Project milestone completion
- Engineering change orders processed
- Process capability studies completed

### 11.3 Monthly Metrics
- Overall Equipment Effectiveness (OEE) improvement
- Cycle time reduction
- First-pass yield improvement
- Cost savings from improvements
- Kaizen events completed
- Engineering change order cycle time

### 11.4 Key Performance Indicators (KPIs)

| KPI | Target | Measurement |
|-----|--------|-------------|
| Project On-Time Completion | >90% | Projects completed on time / Total projects |
| Cost Savings from Improvements | >$100K/year | Sum of validated savings |
| Cycle Time Reduction | >10%/year | (Old C/T - New C/T) / Old C/T |
| Process Capability | Cpk ≥1.67 | Ongoing process monitoring |
| Engineering Change Lead Time | <30 days | ECO approval to implementation |

---

## 12. AI Operations Assistant Integration

### 12.1 Problem Solving
- Log all technical issues
- Conduct root cause analysis using 5-Whys
- Generate fishbone diagrams
- Track corrective actions to closure

### 12.2 Analytics
- Identify recurring issues for focused improvement
- Analyze patterns in equipment failures
- Prioritize projects based on impact
- Calculate ROI for improvement initiatives

### 12.3 Knowledge Management
- Document lessons learned from projects
- Share best practices
- Archive solutions for future reference

---

## 13. Training Requirements

### 13.1 New Engineer Training
- Plant orientation (1 week)
- Process overview (2 weeks)
- AI Ops Assistant system (4 hours)
- Root cause analysis techniques (8 hours)
- FMEA training (8 hours)
- Statistical tools (16 hours)
- Lean manufacturing principles (16 hours)

### 13.2 Ongoing Development
- Six Sigma Green Belt or Black Belt certification
- Advanced DOE training
- Project management courses
- Specialized equipment training
- Industry conferences and seminars

---

## 14. Key Forms and Documents

- Engineering Change Request (ECR)
- Engineering Change Order (ECO)
- Process FMEA
- Process Capability Study Report
- 8D Problem Solving Report
- Kaizen Event Summary
- Standard Work Chart
- Work Instruction Template
- Equipment Specification

---

## Revision History

| Version | Date | Author | Description of Changes |
|---------|------|--------|------------------------|
| 1.0 | 2025-11-22 | AI Ops Team | Initial release |

---

## Approval Signatures

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Engineering Manager | | | |
| Plant Manager | | | |

---

**Document Owner**: Engineering Manager
**Next Review Date**: 2026-02-22
**Distribution**: All Engineering Personnel
