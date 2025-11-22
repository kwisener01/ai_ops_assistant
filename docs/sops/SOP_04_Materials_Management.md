# Standard Operating Procedure: Materials Management
## Transmission Assembly Plant

---

## SOP Information

| Field | Details |
|-------|---------|
| **SOP ID** | MAT-001 |
| **Department** | Materials Management / Supply Chain |
| **Effective Date** | 2025-11-22 |
| **Version** | 1.0 |
| **Owner** | Materials Manager |
| **Review Frequency** | Quarterly |

---

## 1. Purpose and Scope

### 1.1 Purpose
This SOP establishes procedures for materials management to ensure timely availability of materials, optimize inventory levels, and maintain accurate records.

### 1.2 Scope
Applies to all personnel involved in purchasing, receiving, storing, issuing, and tracking materials and components.

---

## 2. Responsibilities

### 2.1 Materials Manager
- Strategic sourcing and supplier management
- Inventory level optimization
- Cost management and budgeting
- Performance metrics and reporting
- Team leadership and development

### 2.2 Buyer/Purchasing Agent
- Purchase order creation and management
- Supplier communication
- Price negotiation
- Expediting and delivery coordination
- Contract management

### 2.3 Receiving Clerk
- Material receipt and inspection
- Documentation verification
- System transactions
- Quality notification
- Damage claim processing

### 2.4 Warehouse/Inventory Coordinator
- Material storage and organization
- Inventory accuracy
- Cycle counting
- Material issuing
- 5S maintenance

### 2.5 Material Handler
- Material movement and delivery
- Kitting and staging
- Forklift operation
- Replenishment to production
- Packaging and shipping

---

## 3. Purchasing Procedures

### 3.1 Purchase Requisition

**Triggers:**
- Inventory reaches reorder point
- Production schedule requirements
- Special project needs
- Capital equipment requests

**Procedure:**

**Step 1: Requisition Creation**
- Requester completes purchase requisition form
- Specify: part number, description, quantity, due date, account code
- Attach specifications or drawings if applicable
- Obtain supervisor approval

**Step 2: Requisition Review**
- Buyer reviews for completeness
- Verify budget availability
- Check if item already on order
- Consolidate with other requirements if possible
- Approve or return for clarification

### 3.2 Supplier Selection

**Criteria:**
- Price competitiveness
- Quality history
- Delivery reliability
- Financial stability
- Technical capability
- Location (lead time considerations)

**Process:**
- Maintain approved supplier list
- Request quotes from 3 suppliers (orders >$5,000)
- Evaluate total cost of ownership (not just price)
- Consider payment terms and freight
- Select supplier and document decision

### 3.3 Purchase Order (PO) Creation

**Procedure:**

**Step 1: PO Development**
- Create PO in purchasing system
- Assign unique PO number
- Include all required information:
  - Supplier name and address
  - Bill-to and ship-to addresses
  - Part number, description, quantity
  - Unit price and total price
  - Required delivery date
  - Payment terms
  - Shipping method
  - Quality requirements
  - Special instructions

**Step 2: PO Approval**

| PO Value | Approval Required |
|----------|-------------------|
| <$1,000 | Buyer |
| $1,000 - $10,000 | Materials Manager |
| $10,000 - $50,000 | Plant Manager |
| >$50,000 | Director of Operations |

**Step 3: PO Transmission**
- Send PO to supplier (email, EDI, portal)
- Request acknowledgment
- Confirm delivery date
- File copy in purchasing records
- Communicate critical dates to Production

### 3.4 Expediting and Follow-up

**Routine Follow-up:**
- Contact supplier 1 week before due date
- Confirm on-time delivery
- Address any issues proactively

**Expediting (Late Deliveries):**
- Daily contact with supplier
- Escalate to supplier management
- Arrange premium freight if necessary
- Document in AI Ops Assistant
- Notify Production of delays
- Consider alternate sources

### 3.5 Purchase Order Changes

**Change Types:**
- Quantity change
- Date change
- Price change
- Cancellation

**Procedure:**
1. Contact supplier to negotiate change
2. Obtain supplier agreement
3. Issue formal PO revision with revision number
4. Document reason for change
5. Update system and communicate to stakeholders

---

## 4. Receiving Procedures

### 4.1 Material Receipt

**Procedure:**

**Step 1: Carrier Arrival**
- Greet driver and obtain bill of lading (BOL)
- Review BOL for shipper and PO number
- Verify destination is correct

**Step 2: Physical Inspection**
- Count packages/pallets
- Check for obvious shipping damage
- Note on BOL if damage observed
- Photograph damage for claim
- Accept or reject delivery

**Step 3: Documentation Check**
- Match BOL to open purchase orders
- Verify quantities match PO
- Check packing slip for part numbers
- Confirm supplier name and address

**Step 4: Unloading**
- Use appropriate material handling equipment
- Place materials in receiving inspection area
- Segregate by supplier and PO
- Handle with care to prevent damage

**Step 5: System Transaction**
- Create goods receipt in system
- Enter PO number
- Record quantity received
- Record receiving date and time
- Print receiving tag and attach to material
- Forward paperwork to Quality for inspection

**Step 6: Notification**
- Notify Quality Control for inspection
- Alert Production if critical material
- Inform Buyer of receipt
- Email expeditor if partial shipment

### 4.2 Discrepancy Resolution

**Types:**
- Quantity short or over
- Wrong part received
- Damage discovered
- Missing documentation

**Procedure:**
1. Document discrepancy with photos
2. Notify Buyer immediately
3. Hold material in receiving area
4. Contact supplier for resolution
5. Issue debit memo if applicable
6. Update system with actual quantity
7. Log issue in AI Ops Assistant

### 4.3 Inspection Coordination

**After Receipt:**
- Place material in "Inspection Hold" area
- Provide Quality with paperwork
- Quality performs incoming inspection (see QC SOP)
- Material dispositioned as Accept, Reject, or Conditional
- Accepted material moved to warehouse
- Rejected material segregated, tagged, and buyer notified

---

## 5. Inventory Management

### 5.1 Inventory Storage

**Storage Principles:**
- First-In, First-Out (FIFO) rotation
- Like items stored together
- Clear labeling of all locations
- Aisles kept clear for safety
- Height limits observed
- Special storage for hazardous materials (MSDS available)

**Storage Locations:**
- **Primary**: Warehouse racking (bulk storage)
- **Secondary**: Production floor point-of-use
- **Tertiary**: Quarantine area (inspection, hold, reject)

**Location System:**
- Alphanumeric location codes (e.g., A-12-03 = Aisle A, Bay 12, Level 3)
- All locations labeled clearly
- System updated with location for each part
- Periodic location audits

### 5.2 Material Issuing

**Procedure:**

**Step 1: Issue Request**
- Production provides work order number
- Specifies part numbers and quantities needed
- Indicates delivery location
- Confirms required delivery date/time

**Step 2: Picking**
- Locate material using inventory system
- Verify part number matches request
- Count quantity to be issued
- Check material condition
- Use FIFO rotation (oldest stock first)

**Step 3: Documentation**
- Create issue transaction in system
- Record work order number
- Deduct from inventory balance
- Print issue ticket
- Attach to material being issued

**Step 4: Delivery**
- Deliver to production location
- Obtain signature from Production representative
- Place in designated staging area
- Dispose of packaging materials properly

**Step 5: Shortage Handling**
- If insufficient stock, issue available quantity
- Notify Production and Buyer of shortage
- Create emergency purchase requisition if needed
- Log in AI Ops Assistant for tracking

### 5.3 Inventory Accuracy

**Target**: 95% inventory accuracy

**Measurement**: Cycle counting program (see Section 5.4)

**Root Causes of Inaccuracy:**
- Incorrect issue transactions
- Receiving errors
- Scrap not reported
- Returns not processed
- Unauthorized withdrawals
- Location errors

**Corrective Actions:**
- Training reinforcement
- Process improvements
- System enhancements
- Physical security measures

### 5.4 Cycle Counting

**Cycle Count Program:**
- All parts counted at least once per year
- High-value or critical parts counted monthly
- Medium-value parts counted quarterly
- Low-value parts counted annually

**ABC Classification:**
- **A Items** (20% of items, 80% of value): Monthly count
- **B Items** (30% of items, 15% of value): Quarterly count
- **C Items** (50% of items, 5% of value): Annual count

**Cycle Count Procedure:**

**Step 1: Count Assignment**
- System generates count list
- Assigns locations and parts to counter
- Prints count sheet (part number, location, expected qty)

**Step 2: Physical Count**
- Go to location
- Count all material at that location
- Do not reference expected quantity
- Record actual count on sheet
- Note any condition issues (damage, obsolete, etc.)

**Step 3: Count Entry**
- Enter count into system
- System compares to book quantity
- Variance calculated automatically

**Step 4: Variance Investigation**

| Variance | Action |
|----------|--------|
| 0 (perfect match) | Accept count, no action |
| ±1-5 units or <5% | Adjust inventory, document reason |
| >5 units or >5% | Recount immediately |
| >10% or high value | Supervisor recount, investigate root cause |

**Step 5: Adjustment**
- Supervisor approves adjustment
- System updated with actual quantity
- Variance logged for metrics
- If significant, log in AI Ops Assistant for root cause analysis

**Step 6: Root Cause Analysis (if applicable)**
- Conduct 5-Whys for significant variances
- Identify process breakdown
- Implement corrective action
- Update procedures if needed

### 5.5 Physical Inventory (Annual)

**Procedure:**
- Schedule during slow production period (e.g., year-end shutdown)
- All material movement frozen during count
- Count entire warehouse in one day
- All personnel participate
- Two-person count teams
- Supervisor spot-checks 10% of counts
- Reconcile variances before resuming operations

---

## 6. Inventory Optimization

### 6.1 Inventory Levels

**Terminology:**
- **Minimum (Min)**: Reorder point, safety stock level
- **Maximum (Max)**: Maximum inventory before overstock
- **Reorder Quantity (ROQ)**: Standard order quantity
- **Lead Time**: Time from PO to receipt
- **Safety Stock**: Buffer for variability

**Min Calculation:**
```
Min = (Average daily usage × Lead time in days) + Safety stock
```

**Max Calculation:**
```
Max = Min + Reorder quantity
```

**Example:**
- Average daily usage: 10 units
- Lead time: 14 days
- Safety stock: 20 units (2 days)
- Min = (10 × 14) + 20 = 160 units (reorder point)
- Reorder quantity: 200 units
- Max = 160 + 200 = 360 units

### 6.2 Economic Order Quantity (EOQ)

**EOQ Formula:**
```
EOQ = √(2 × Annual demand × Order cost / Holding cost per unit)
```

**Application:**
- Balances ordering costs vs. holding costs
- Determines optimal order quantity
- Reviewed annually for high-value items

### 6.3 Slow-Moving and Obsolete Inventory

**Identification:**
- No usage in past 12 months = obsolete candidate
- Low usage (<3 transactions/year) = slow-moving

**Disposition:**
1. Review with Engineering for future use
2. Offer to other facilities
3. Return to supplier (if possible)
4. Sell as surplus
5. Scrap (last resort)
6. Write off and remove from inventory

**Prevention:**
- Regular review of inventory turns
- Coordinate with Engineering on design changes
- Order min quantities for low-volume parts
- Supplier consignment programs

---

## 7. Supplier Management

### 7.1 Supplier Performance Metrics

**Monthly Scorecard:**

| Metric | Weight | Target | Measurement |
|--------|--------|--------|-------------|
| On-Time Delivery | 40% | 95% | On-time receipts / Total receipts |
| Quality (Reject Rate) | 40% | <1% | Rejected lots / Total lots |
| Responsiveness | 10% | 24 hrs | Avg. time to respond to inquiry |
| Lead Time | 10% | Per agreement | Actual vs. quoted lead time |

**Overall Rating:**
- A: >90% (preferred supplier)
- B: 80-90% (acceptable)
- C: 70-80% (needs improvement)
- D: <70% (unacceptable, source alternative)

### 7.2 Supplier Corrective Action

**Triggers:**
- Quality rejection
- Late delivery (>3 days)
- Repeated issues
- Score drops to C or D

**Process:**
1. Issue Supplier Corrective Action Request (SCAR)
2. Supplier responds within 5 business days
3. Root cause and corrective action plan required
4. Review and accept or negotiate
5. Monitor effectiveness for 90 days
6. Update scorecard
7. Close SCAR or escalate

### 7.3 Supplier Development

**Activities:**
- Quarterly business reviews
- On-site visits
- Joint improvement projects
- Volume commitments for better pricing
- Technology sharing
- Long-term agreements

---

## 8. Shipping and Logistics

### 8.1 Outbound Shipping

**Procedure:**

**Step 1: Pick and Pack**
- Receive customer order
- Pick from finished goods inventory
- Verify part numbers and quantities
- Package per customer requirements
- Apply labels and shipping documents

**Step 2: Documentation**
- Create bill of lading (BOL)
- Include packing list
- Generate commercial invoice if export
- Certificate of conformance if required
- Country of origin documentation

**Step 3: Carrier Coordination**
- Schedule pickup with carrier
- Confirm freight terms (FOB, CIF, etc.)
- Obtain tracking number
- Provide to customer

**Step 4: System Update**
- Create shipment transaction
- Deduct from finished goods inventory
- Update order status
- Archive documents

### 8.2 Returns Processing

**Incoming Returns:**
- Obtain Return Material Authorization (RMA) number
- Inspect upon receipt
- Document condition with photos
- Forward to Quality for disposition
- Process credit or replacement

**Outgoing Returns (to Supplier):**
- Contact supplier for RMA
- Package and label per supplier instructions
- Include documentation (NCR, photos, etc.)
- Ship and track
- Follow up on credit or replacement

---

## 9. Hazardous Materials Management

### 9.1 Hazmat Identification
- Flammable liquids (oils, solvents, cleaners)
- Compressed gases
- Corrosives (acids, caustics)
- Toxic materials

### 9.2 Safety Data Sheets (SDS)
- Obtain SDS from supplier before first use
- Maintain SDS binder/electronic database
- Accessible to all employees
- Updated when formulations change

### 9.3 Storage Requirements
- Segregate incompatible materials
- Store in approved cabinets/rooms
- Proper ventilation
- Spill containment
- Bonding and grounding for flammables
- Clear labeling with hazard symbols

### 9.4 Spill Response
- Minor spills (<1 gallon): Use spill kit, clean up, report to supervisor
- Major spills: Evacuate area, call emergency response team, notify environmental manager
- Dispose of contaminated materials per regulations

---

## 10. Performance Metrics

### 10.1 Daily Metrics
- Purchase orders issued
- Receipts processed
- Materials issued to production
- Inventory transactions

### 10.2 Weekly Metrics
- On-time delivery rate (suppliers)
- Receiving accuracy
- Material shortage incidents
- Cycle count completion rate

### 10.3 Monthly Metrics
- Inventory accuracy percentage
- Inventory turnover ratio
- Days of inventory on hand
- Supplier scorecard results
- Expedite costs
- Freight costs
- Obsolete inventory value

### 10.4 Key Performance Indicators (KPIs)

| KPI | Target | Calculation |
|-----|--------|-------------|
| Inventory Accuracy | 95% | (Accurate counts / Total counts) × 100 |
| Inventory Turns | 6-12/year | Cost of goods sold / Avg. inventory value |
| On-Time Delivery | 95% | On-time receipts / Total receipts |
| Fill Rate | 98% | Complete orders / Total orders |
| Carrying Cost | <25% of inv. value | Storage + obsolescence + capital cost |

---

## 11. AI Operations Assistant Integration

### 11.1 Issue Tracking
- Log material shortages
- Document supplier quality issues
- Track delivery delays
- Record root causes of inventory variances

### 11.2 Analytics
- Identify recurring shortage patterns
- Analyze supplier performance trends
- Optimize reorder points and quantities
- Forecast material needs

### 11.3 Reporting
- Generate inventory reports
- Supplier performance dashboards
- Cost analysis and savings opportunities

---

## 12. Training Requirements

### 12.1 New Hire Training
- Safety orientation (4 hours)
- Forklift certification (if applicable, 8 hours)
- Inventory system training (8 hours)
- Receiving procedures (4 hours)
- Hazmat awareness (2 hours)
- AI Ops Assistant (2 hours)

### 12.2 Ongoing Training
- Annual forklift recertification
- Hazmat refresher (annually)
- System updates and enhancements
- Continuous improvement techniques

---

## 13. Key Forms and Documents

- Purchase Requisition
- Purchase Order
- Goods Receipt
- Inventory Adjustment Form
- Cycle Count Sheet
- Material Issue Ticket
- Supplier Corrective Action Request (SCAR)
- Return Material Authorization (RMA)
- Bill of Lading (BOL)

---

## Revision History

| Version | Date | Author | Description of Changes |
|---------|------|--------|------------------------|
| 1.0 | 2025-11-22 | AI Ops Team | Initial release |

---

## Approval Signatures

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Materials Manager | | | |
| Quality Manager | | | |
| Plant Manager | | | |

---

**Document Owner**: Materials Manager
**Next Review Date**: 2026-02-22
**Distribution**: All Materials Management Personnel
