# Standard Operating Procedure: Quality Control/Quality Assurance
## Transmission Assembly Plant

---

## SOP Information

| Field | Details |
|-------|---------|
| **SOP ID** | QC-001 |
| **Department** | Quality Control/Quality Assurance |
| **Effective Date** | 2025-11-22 |
| **Version** | 1.0 |
| **Owner** | Quality Manager |
| **Review Frequency** | Quarterly |

---

## 1. Purpose and Scope

### 1.1 Purpose
This SOP establishes quality control procedures to ensure all transmission assemblies meet or exceed customer specifications and quality standards.

### 1.2 Scope
Applies to all quality personnel including inspectors, technicians, and the quality management team.

---

## 2. Responsibilities

### 2.1 Quality Manager
- Overall quality system management
- Customer liaison for quality issues
- Root cause analysis leadership
- Continuous improvement initiatives
- Supplier quality oversight

### 2.2 Quality Engineer
- Process capability studies
- Measurement system analysis
- Quality data analysis
- SPC (Statistical Process Control) implementation
- Corrective action effectiveness verification

### 2.3 QC Inspector
- Incoming inspection
- In-process inspection
- Final inspection and testing
- Documentation and reporting
- Calibration verification

### 2.4 Quality Technician
- Sample testing
- Measurement and data recording
- Non-conformance tagging
- Laboratory equipment operation

---

## 3. Inspection Procedures

### 3.1 Incoming Inspection

**Frequency**: 100% of critical components, sample basis for others

**Procedure:**

**Step 1: Material Receipt**
- Verify packing slip matches purchase order
- Check for shipping damage
- Confirm quantity received
- Assign incoming inspection lot number

**Step 2: Visual Inspection**
- Check for obvious defects (scratches, dents, corrosion)
- Verify correct part number and revision
- Check packaging condition
- Review supplier documentation (certificates, test reports)

**Step 3: Dimensional Inspection**
- Use calibrated measuring equipment
- Measure critical dimensions per specification
- Sample size per AQL (Acceptable Quality Level) plan
- Record measurements on inspection form

**Step 4: Functional Testing (if applicable)**
- Perform tests per specification
- Document test results
- Verify performance criteria met

**Step 5: Disposition Decision**
- **Accept**: Stamp/tag as approved, release to inventory
- **Reject**: Red tag, segregate, notify Purchasing
- **Conditional Accept**: Document deviation, obtain approval
- **Hold**: Additional testing or information needed

**Step 6: Documentation**
- Complete Incoming Inspection Report
- Enter results in AI Ops Assistant
- File certificates with purchase order
- Update inventory system

### 3.2 In-Process Inspection

**Frequency**: Per control plan, typically at key process steps

**Procedure:**

**Step 1: Inspection Trigger**
- First piece after changeover (mandatory)
- Hourly sample during production
- After any process adjustment
- Random sampling per control plan

**Step 2: Sampling**
- Randomly select units from production
- Minimum sample size per control plan
- Mark sampled units to avoid re-inspection

**Step 3: Inspection Execution**
- Follow inspection checklist for product
- Measure all critical characteristics
- Perform functional tests if required
- Record all data on inspection form

**Step 4: Analysis**
- Compare to specification limits
- Plot on control charts
- Identify trends or patterns
- Calculate process capability (Cpk) weekly

**Step 5: Feedback**
- Provide immediate feedback to Production
- Highlight any concerns or trends
- Document in AI Ops Assistant
- Escalate out-of-spec conditions immediately

### 3.3 Final Inspection and Testing

**Frequency**: 100% of assemblies before packaging

**Procedure:**

**Step 1: Pre-Inspection**
- Verify all routing sheet signatures complete
- Check for in-process inspection stamps
- Confirm work order matches unit
- Verify traceability information

**Step 2: Visual Inspection**
- Check for completeness of assembly
- Look for damage or defects
- Verify all fasteners installed and torqued
- Check cleanliness

**Step 3: Dimensional Verification**
- Measure critical final dimensions
- Check for proper alignment
- Verify clearances and gaps
- Document measurements

**Step 4: Functional Testing**
- Perform all required functional tests
- Follow test procedure exactly
- Use calibrated test equipment
- Record all test data

**Step 5: Final Disposition**
- **Pass**: Apply QC passed label, release for packaging
- **Fail**: Red tag, document issue, segregate unit
- Log all dispositions in AI Ops Assistant

**Step 6: Documentation**
- Complete Final Inspection Report
- Attach to work order
- Update quality database
- Archive records per retention policy

---

## 4. Non-Conformance Management

### 4.1 Non-Conformance Identification

**Sources:**
- Inspection findings
- Production reports
- Customer complaints
- Supplier issues
- Audit findings

### 4.2 Non-Conformance Processing

**Step 1: Documentation**
- Complete Non-Conformance Report (NCR)
- Assign unique NCR number
- Log in AI Ops Assistant with full details
- Photograph defect if applicable
- Attach supporting documentation

**Step 2: Containment**
- Physically segregate non-conforming product
- Apply red tag with NCR number
- Prevent further processing or shipment
- Assess scope (how many units potentially affected)
- Implement temporary containment action

**Step 3: Root Cause Analysis**
- Initiate 5-Whys analysis in AI Ops Assistant
- Create Ishikawa (fishbone) diagram
- Identify all contributing factors
- Determine root cause category (Man, Machine, Material, Method, Measurement, Environment)
- Document findings thoroughly

**Step 4: Corrective Action**
- Develop corrective action plan
- Assign responsibility and due date
- Address root cause, not just symptoms
- Consider preventive actions for similar issues
- Obtain approval from Quality Manager

**Step 5: Disposition Decision**

| Disposition | Criteria | Approval Required |
|-------------|----------|-------------------|
| **Rework** | Can be brought to specification | Supervisor |
| **Use-As-Is** | Deviation is acceptable | Customer approval |
| **Scrap** | Cannot be economically repaired | Quality Manager |
| **Return to Supplier** | Supplier responsibility | Purchasing Manager |

**Step 6: Verification**
- Re-inspect after rework
- Verify corrective action effective
- Monitor for recurrence (30 days)
- Document verification results

**Step 7: Closure**
- Obtain approval signatures
- Update AI Ops Assistant status to "Closed"
- File NCR in quality records
- Share learnings in quality meeting

---

## 5. Root Cause Analysis Process

### 5.1 When to Conduct RCA

**Mandatory RCA Triggers:**
- Customer complaints
- Recurring defects (3+ occurrences)
- Critical safety issues
- Significant financial impact (>$5,000)
- Product recall or field failure

### 5.2 5-Whys Methodology

**Procedure:**
1. Clearly define the problem statement
2. Ask "Why did this happen?" - record the answer
3. Ask "Why?" about the previous answer - record the answer
4. Repeat until reaching the root cause (typically 5 iterations)
5. Final answer becomes the root cause
6. Document entire chain in AI Ops Assistant

**Example:**
- **Problem**: Gearbox vibration detected
- **Why 1**: Why was there vibration? → Bearings were worn
- **Why 2**: Why were bearings worn? → Lubrication schedule was missed
- **Why 3**: Why was schedule missed? → Maintenance crew understaffed
- **Why 4**: Why understaffed? → Two technicians on extended leave
- **Why 5**: Why no coverage plan? → No cross-training program exists
- **Root Cause**: Lack of workforce planning and cross-training program

### 5.3 Ishikawa (Fishbone) Diagram

**Categories (6M):**
- **Man**: Human factors, training, experience
- **Machine**: Equipment condition, capability, maintenance
- **Material**: Raw materials, components, supplies
- **Method**: Procedures, work instructions, processes
- **Measurement**: Inspection methods, calibration, gauges
- **Environment**: Temperature, humidity, cleanliness, lighting

**Procedure:**
1. Define the problem (fishbone head)
2. Draw main categories as bones
3. Brainstorm causes under each category
4. Identify most likely root causes
5. Generate diagram using AI Ops Assistant
6. Use for team discussion and analysis

---

## 6. Calibration Management

### 6.1 Equipment Requiring Calibration
- Micrometers, calipers, height gauges
- Torque wrenches
- Pressure gauges
- Temperature measuring devices
- Test equipment
- Scales and balances

### 6.2 Calibration Procedure

**Step 1: Scheduling**
- Maintain calibration database
- Schedule calibrations per manufacturer recommendations
- Typical frequency: Annually or per usage (torque wrenches)
- Send reminder 30 days before due date

**Step 2: Pre-Calibration Check**
- Inspect for damage or wear
- Clean equipment
- Document current readings (as-found)

**Step 3: Calibration Execution**
- Use certified calibration standards (traceable to NIST)
- Follow calibration procedure
- Record as-found and as-left values
- Make adjustments if necessary
- Document all readings

**Step 4: Labeling**
- Apply calibration label with:
  - Calibration date
  - Due date
  - Unique equipment ID
  - Technician initials

**Step 5: Documentation**
- Complete calibration certificate
- File in calibration records
- Update calibration database
- Log in AI Ops Assistant for tracking

**Step 6: Out-of-Tolerance Action**
- If equipment found out-of-tolerance:
  - Quarantine immediately (red tag)
  - Investigate impact on previous measurements
  - Notify Quality Manager
  - Initiate NCR if product affected
  - Determine need for product recall/rework

---

## 7. Customer Complaint Management

### 7.1 Complaint Receipt

**Step 1: Documentation**
- Log in AI Ops Assistant immediately upon receipt
- Assign unique complaint number
- Record customer information
- Document complaint details
- Obtain samples if available

**Step 2: Acknowledgment**
- Acknowledge receipt to customer within 24 hours
- Provide complaint number
- Commit to investigation timeline
- Assign owner (typically Quality Engineer)

### 7.2 Investigation

**Step 1: Verification**
- Reproduce the issue if possible
- Inspect returned product
- Review manufacturing records for affected lot
- Interview personnel involved

**Step 2: Root Cause Analysis**
- Conduct 5-Whys analysis
- Create fishbone diagram
- Identify all contributing factors
- Determine root cause

**Step 3: Containment**
- Check inventory for potentially affected product
- Quarantine if necessary
- Assess need for customer notification
- Implement temporary fix

### 7.3 Response and Closure

**Step 1: Corrective Action**
- Develop comprehensive corrective action plan
- Implement process improvements
- Update work instructions if needed
- Verify effectiveness

**Step 2: Customer Response**
- Prepare formal response within 5 business days
- Include root cause and corrective action
- Provide evidence of implementation
- Offer resolution (replacement, credit, etc.)

**Step 3: Follow-up**
- Monitor for recurrence (90 days)
- Conduct effectiveness review
- Update AI Ops Assistant status
- Close complaint with approval

---

## 8. Statistical Process Control (SPC)

### 8.1 Control Chart Implementation

**When to Use:**
- High-volume production
- Critical characteristics
- Process capability concerns

**Types of Charts:**
- **X-bar and R chart**: Variable data (measurements)
- **P chart**: Attribute data (pass/fail, defect rate)
- **C chart**: Count data (defects per unit)

### 8.2 Control Limits

**Calculation:**
- Upper Control Limit (UCL) = Mean + 3σ
- Lower Control Limit (LCL) = Mean - 3σ
- Recalculate limits quarterly or after process changes

### 8.3 Out-of-Control Signals

**Action Required When:**
- Any point outside control limits
- 7 consecutive points on one side of centerline
- 7 consecutive points trending up or down
- 14 points alternating up and down

**Response:**
- Stop production immediately
- Investigate cause
- Document in AI Ops Assistant
- Implement corrective action
- Verify process stability before resuming

---

## 9. Supplier Quality Management

### 9.1 Supplier Approval Process
- Evaluate supplier capability
- Request quality certifications (ISO 9001, etc.)
- Conduct on-site audit if critical supplier
- Establish quality agreement
- Set acceptance criteria (AQL levels)

### 9.2 Supplier Performance Monitoring

**Monthly Scorecard Metrics:**
- On-time delivery rate
- Defect rate (PPM - parts per million)
- Lot acceptance rate
- Corrective action responsiveness
- Overall rating (A/B/C/D)

### 9.3 Supplier Corrective Action

**When Required:**
- Defect rate exceeds threshold (>1000 PPM)
- Critical defect found
- Repeated issues (3+ occurrences)
- Customer complaint traced to supplier

**Process:**
1. Issue Supplier Corrective Action Request (SCAR)
2. Supplier responds within 5 business days
3. Review and approve corrective action plan
4. Monitor effectiveness (60 days)
5. Update supplier scorecard
6. Close SCAR or escalate if ineffective

---

## 10. Quality Audits

### 10.1 Internal Quality Audits

**Frequency**: Monthly rotating schedule

**Process:**
1. Develop audit checklist
2. Notify area 1 week in advance
3. Conduct audit (typically 2-4 hours)
4. Document findings
5. Issue audit report within 3 days
6. Track corrective actions
7. Verify closure

**Audit Areas:**
- Production process compliance
- Documentation accuracy
- Calibration status
- 5S adherence
- Training records
- Quality records retention

### 10.2 External Audits

**Types:**
- Customer audits
- ISO 9001 certification audits
- Regulatory compliance audits

**Preparation:**
- Review audit scope and schedule
- Prepare required documents
- Conduct pre-audit internal review
- Assign escorts and subject matter experts
- Ensure facility readiness

**During Audit:**
- Provide requested documents promptly
- Answer questions honestly and accurately
- Take notes of findings
- Do not make commitments without authorization

**After Audit:**
- Review findings
- Develop corrective action plan
- Implement actions per timeline
- Provide evidence to auditor
- Close findings

---

## 11. Training and Competency

### 11.1 Required Training

**New QC Inspector:**
- Quality system overview (4 hours)
- Inspection techniques (16 hours)
- Measurement equipment use (8 hours)
- AI Ops Assistant system (2 hours)
- Product-specific training (varies)
- On-the-job training with mentor (80 hours)

### 11.2 Ongoing Training
- Annual calibration refresher
- Updates to inspection criteria
- New product training
- Root cause analysis techniques
- Customer-specific requirements

### 11.3 Competency Verification
- Annual written assessment
- Practical measurement test
- Observation during inspection
- Inter-rater reliability study (compare inspector results)

---

## 12. Performance Metrics

### 12.1 Daily Metrics
- Inspection quantities (incoming, in-process, final)
- Non-conformances identified
- Customer complaints received
- Calibration due dates approaching

### 12.2 Weekly Metrics
- First-pass yield
- Defect rate by category
- NCR closure rate
- Supplier rejection rate

### 12.3 Monthly Metrics
- Overall defect rate (PPM)
- Cost of quality (scrap, rework, warranty)
- Customer complaint rate
- On-time corrective action closure
- Process capability indices (Cpk)
- Audit findings and closure

---

## 13. AI Operations Assistant Integration

### 13.1 Issue Logging
- All quality issues entered into system
- Use standardized categories
- Attach photos and supporting documents
- Assign priority level

### 13.2 Root Cause Analysis
- Conduct 5-Whys directly in system
- Generate fishbone diagrams
- Link related issues
- Track corrective actions to completion

### 13.3 Reporting and Analytics
- Generate weekly quality reports
- Identify recurring issues
- Track trends and patterns
- Calculate mean time to resolution
- Highlight improvement opportunities

---

## 14. Key Forms and Documents

- Incoming Inspection Report
- In-Process Inspection Checklist
- Final Inspection Report
- Non-Conformance Report (NCR)
- Supplier Corrective Action Request (SCAR)
- Customer Complaint Log
- Calibration Certificate
- Control Chart
- Audit Checklist and Report

---

## Revision History

| Version | Date | Author | Description of Changes |
|---------|------|--------|------------------------|
| 1.0 | 2025-11-22 | AI Ops Team | Initial release |

---

## Approval Signatures

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Quality Manager | | | |
| Production Manager | | | |
| Plant Manager | | | |

---

**Document Owner**: Quality Manager
**Next Review Date**: 2026-02-22
**Distribution**: All Quality Personnel
