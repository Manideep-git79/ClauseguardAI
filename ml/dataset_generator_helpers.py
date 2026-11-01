"""
ClauseGuard AI — Deep Combinatorial Legal Scenario Generator
Generates mathematically unique, realistic legal clauses for all 30 canonical clause types
across Rental Agreements, Job Offers, and Insurance Policies.
Supports fair, needs_review, and unfavorable classifications with diverse legal scenarios.
"""

import random

RENT_VALS = ["$1,200", "$1,450", "$1,750", "$1,900", "$2,150", "$2,400", "$2,750", "$3,100", "$3,600", "$4,250", "Rs. 22,000", "Rs. 35,000", "Rs. 50,000", "Rs. 85,000", "£1,250", "£1,850"]
DEPOSIT_VALS = ["$1,500", "$2,000", "$2,500", "$3,000", "$4,500", "one month's rent", "two months' rent", "Rs. 75,000", "Rs. 1,50,000", "£1,500"]
SALARY_VALS = ["$68,000", "$85,000", "$105,000", "$125,000", "$145,000", "$165,000", "$190,000", "$230,000", "Rs. 10,50,000", "Rs. 18,00,000", "Rs. 28,00,000", "£60,000", "£85,000"]
BONUS_VALS = ["8%", "10%", "15%", "20%", "25%", "30%", "$10,000", "$20,000", "$35,000", "Rs. 1,50,000", "Rs. 3,00,000"]
DEDUCTIBLE_VALS = ["$500", "$1,000", "$1,500", "$2,000", "$2,500", "$5,000", "1% of Dwelling Limit", "2% of Insured Value"]
LIMIT_VALS = ["$250,000", "$350,000", "$500,000", "$750,000", "$1,000,000", "$2,000,000", "Rs. 25,00,000", "Rs. 50,00,000", "Rs. 1,00,00,000"]
DAYS_VALS = ["3", "5", "7", "10", "14", "15", "21", "30", "45", "60", "90"]
JURISDICTIONS = ["California", "New York", "Texas", "Washington", "Illinois", "Florida", "Massachusetts", "Colorado", "England and Wales", "Karnataka, India", "Maharashtra, India"]

ROLES = ["Senior Software Engineer", "Product Manager", "Data Science Lead", "Cloud Architect", "Frontend Developer", "Financial Analyst", "Operations Director", "DevOps Engineer", "Marketing Lead"]
DEPARTMENTS = ["Engineering", "Product Operations", "Data & Analytics", "Information Security", "Corporate Finance", "Commercial Strategy"]
MANAGERS = ["the VP of Engineering", "the Chief Technology Officer", "the Director of Product", "the Head of Architecture", "the General Manager"]

RENTAL_PATTERNS = {
    "rent_payment_terms": {
        "fair": [
            "Tenant shall pay monthly rent of {rent} on or before the 1st of each calendar month via ACH, electronic bank transfer, or paper cheque.",
            "Monthly rental consideration is agreed at {rent}, due on the first day of each calendar month with a five (5) business day grace period.",
            "Rent of {rent} shall be paid monthly in advance to Landlord's account, with Landlord providing prompt digital receipt within 48 hours.",
            "The agreed rental consideration of {rent} per month is fixed for the duration of the initial lease term without midterm escalation.",
            "Tenant agrees to remit {rent} on the first of each month; payments postmarked on or before the 5th shall be deemed timely received.",
            "Monthly rent shall be {rent}, payable through any recognized banking channel without compulsory proprietary payment processing surcharges.",
        ],
        "needs_review": [
            "Rent of {rent} is payable on the 1st. Landlord may adjust rent upon 30 days notice to reflect documented increases in local property tax assessments.",
            "Monthly rent of {rent} must be paid through Landlord's online portal, which may assess tenant-borne transaction convenience fees up to 3%.",
            "Initial rent is {rent} per month; following the initial lease period, rent shall transition to prevailing market rates as determined by Landlord.",
            "Tenant shall remit {rent} monthly. If paid via cashier's check, an administrative handling surcharge of $25 per payment cycle applies.",
        ],
        "unfavorable": [
            "Monthly rent is {rent} due promptly on the 1st; if unpaid by 5:00 PM on the due date, Landlord may declare non-curable default and accelerate the entire lease balance.",
            "Landlord reserves the unrestricted unilateral prerogative to increase monthly rent of {rent} at any time upon ten (10) days notice to Tenant.",
            "Tenant expressly waives any statutory right to deduct, withhold, or offset rent for any reason, including failure of heating, electricity, or water services.",
            "Rent of {rent} is non-negotiable and failure to pay by the 2nd shall trigger immediate lease termination without judicial hearing or cure rights.",
        ],
    },

    "security_deposit": {
        "fair": [
            "Tenant shall provide a security deposit of {deposit}, to be held in an escrow account with interest disbursed in accordance with statutory guidelines.",
            "A refundable security deposit of {deposit} is required, returnable within twenty-one (21) days of surrender less itemized documented damages.",
            "The security deposit of {deposit} shall secure Tenant covenants; ordinary wear and tear is expressly excluded from lawful security deductions.",
            "Landlord shall provide an itemized repair statement with contractor receipts for any deduction made from the {deposit} deposit within 30 days.",
            "Deposit of {deposit} shall be returned to Tenant by electronic wire within fourteen (14) days following joint move-out walk-through inspection.",
        ],
        "needs_review": [
            "A security deposit of {deposit} is deposited. An automatic non-refundable administrative fee of $200 is deducted at move-out for turnover costs.",
            "Landlord shall hold a deposit of {deposit} without interest, returning remaining funds within 45 days after utility arrears reconciliation.",
            "Deposit of {deposit} may be commingled with Landlord's general operating funds subject to applicable regional housing authority exemptions.",
        ],
        "unfavorable": [
            "Tenant shall furnish a deposit of {deposit}, which is non-refundable and automatically converted into a mandatory property restoration charge.",
            "Landlord may retain the entire security deposit of {deposit} as liquidated damages if Tenant vacates prior to the exact lease expiration date.",
            "Landlord retains sixty (60) days to inspect and may make arbitrary deductions for pre-existing wall marks, carpet age, and routine turnover without invoices.",
            "The deposit of {deposit} is forfeited in full upon any single notice of minor community rule infraction issued during the tenancy.",
        ],
    },

    "termination": {
        "fair": [
            "Either party may terminate this lease at the end of the initial term by providing at least thirty (30) days advance written notice.",
            "In accordance with the Servicemembers Civil Relief Act (SCRA), Tenant may terminate without penalty upon military relocation orders and 30 days notice.",
            "Tenant may terminate this agreement without fee if Landlord fails to remedy a substantial habitability failure within fourteen (14) days of notice.",
            "Mutual termination is permitted at any time upon sixty (60) days written agreement, with security deposit returned subject to standard move-out terms.",
            "Upon lease expiration, tenancy converts to month-to-month terminable by either party upon thirty (30) calendar days written notice.",
        ],
        "needs_review": [
            "Tenant may terminate early by paying an early termination fee equal to two (2) months rent and providing sixty (60) days advance written notice.",
            "Termination notice during November, December, or January is prohibited; notice submitted during these months takes effect on February 1st.",
            "Early termination requires forfeiture of security deposit plus payment of rent until Landlord re-leases the demised premises.",
        ],
        "unfavorable": [
            "Tenant has zero right to early termination; early vacancy accelerates all remaining rent through the entire multi-year lease term immediately.",
            "Landlord may terminate this lease and repossess the unit upon three (3) days notice for any alleged minor infraction without right to cure.",
            "Tenant waives all statutory rights to formal court eviction proceedings and agrees Landlord may peacefully repossess the unit and change locks upon 48 hours notice.",
            "If Tenant breaches any clause, Landlord may seize all personal property inside the premises and dispose of it immediately without court authorization.",
        ],
    },

    "maintenance_repairs": {
        "fair": [
            "Landlord shall maintain all structural walls, roofs, plumbing systems, electrical wiring, and heating apparatus in sound working order at Landlord's sole expense.",
            "Tenant shall keep the unit sanitary; Landlord agrees to dispatch licensed contractors for major plumbing or HVAC repairs within twenty-four (24) hours.",
            "Landlord is responsible for appliance maintenance, pest eradication, and water heater servicing unless damage arises from Tenant's gross negligence.",
            "If Landlord fails to repair an essential heating or water condition within forty-eight (48) hours of written notice, Tenant may exercise repair-and-deduct rights.",
        ],
        "needs_review": [
            "Tenant shall be responsible for all interior maintenance repairs under $100 per occurrence, including clogged drains, minor electrical fixtures, and weatherstripping.",
            "Landlord shall provide HVAC maintenance, but Tenant is responsible for semi-annual filter replacement and 50% of any duct cleaning expenses.",
            "Major repairs will be undertaken within a commercially reasonable period, which may take up to twenty-one (21) days during peak contractor seasons.",
        ],
        "unfavorable": [
            "Tenant assumes total responsibility for all maintenance, repairs, and structural replacements, including HVAC units, roof leaks, and plumbing mains.",
            "Landlord disclaims all implied warranties of habitability, tenant fitness, and quiet enjoyment, leasing the property strictly on an 'as-is, where-is' basis.",
            "Tenant must pay a mandatory non-refundable $150 deductible fee for each maintenance service call requested, regardless of fault or defect nature.",
        ],
    },

    "entry_notice_access": {
        "fair": [
            "Landlord may enter the premises only upon providing twenty-four (24) hours advance written notice, between 9:00 AM and 6:00 PM, except in emergencies.",
            "Except in cases of active water leaks or fire emergencies, Landlord shall provide at least forty-eight (48) hours notice prior to conducting routine inspections.",
            "Landlord entry for showing to prospective buyers or tenants requires minimum 24 hours advance notice and shall be limited to reasonable daylight hours.",
            "Tenant possesses the right of peaceful possession; Landlord agrees not to conduct inspections more frequently than once every six (6) months.",
            "Entry notices shall specify the date, approximate time window, and nature of the proposed maintenance visit.",
            "Tenant may request rescheduling of non-emergency landlord inspections with twenty-four (24) hours advance written communication.",
        ],
        "needs_review": [
            "Landlord may enter the premises between 8:00 AM and 8:00 PM upon twelve (12) hours notice sent via email or text message for repairs and showings.",
            "During the final thirty (30) days of the lease, Landlord may show the premises upon four (4) hours verbal notice to accommodate prospective renters.",
            "Routine pest control treatments may occur monthly on designated dates without individual unit written notification.",
        ],
        "unfavorable": [
            "Landlord reserves unconditional right to enter the premises at any hour of day or night without prior notice for inspections or showings.",
            "Tenant agrees to grant Landlord 24/7 master key access without notice, waiving all claims for trespass, privacy violation, or disturbance of tenancy.",
            "Landlord may install continuous remote monitoring sensors inside common living areas to enforce occupancy and noise restrictions.",
            "Refusal to permit landlord entry at any time incurs an immediate $200 unauthorized lockout fine and lease cancellation.",
        ],
    },

    "subletting": {
        "fair": [
            "Tenant may assign or sublet the leased premises with Landlord's prior written consent, which consent shall not be unreasonably withheld or delayed.",
            "In the event of subleasing, Landlord shall evaluate proposed subtenants using standard non-discriminatory screening criteria within ten (10) business days.",
            "Tenant may take on an additional occupant or roommate subject to Landlord's approval of credit and background verification without arbitrary denial.",
        ],
        "needs_review": [
            "Tenant shall not assign this lease, sublet the premises, or grant any license to occupy without Landlord consent.",
            "Subletting is permitted only with Landlord's written approval and upon payment of a $350 administrative review fee; Tenant remains jointly liable.",
            "Guests staying longer than fourteen (14) consecutive calendar days must apply for formal leaseholder status and undergo background screening.",
            "Tenant shall not assign, sublet, or transfer the premises without prior written consent of Landlord.",
        ],
        "unfavorable": [
            "Subletting, assignment, licensing, or short-term hosting of any kind is strictly prohibited; violation triggers immediate lease forfeiture and $2,000 fine.",
            "Hosting any overnight guest without registering their government ID with management 48 hours in advance shall incur an unauthorized occupant fee of $150 per night.",
            "Any attempt to advertise the premises on Airbnb or any lodging site constitutes automatic grounds for immediate sheriff eviction and triple damages.",
        ],
    },

    "pet_policy": {
        "fair": [
            "Tenant is permitted to keep up to two (2) domestic pets (cats or dogs under 50 lbs) upon payment of a refundable $300 pet damage deposit.",
            "Registered assistance animals, guide dogs, and documented emotional support animals are welcomed without pet deposit or surcharge under the Fair Housing Act.",
            "Pets are permitted on the property; Tenant agrees to maintain vaccinations, leash pets in common areas, and remediate any waste promptly.",
        ],
        "needs_review": [
            "One domestic cat or dog under 25 lbs is permitted upon payment of a $250 non-refundable pet fee and a recurring monthly pet rent of $35.",
            "Specific dog breeds are restricted; Landlord reserves the right to require pet removal upon two substantiated noise or aggression complaints.",
        ],
        "unfavorable": [
            "Strict zero-tolerance no-pet policy; presence of any animal on the premises incurs an immediate $1,000 deep-sanitization penalty and notice to vacate.",
            "Landlord may immediately impound or turn over to animal control any animal discovered on the leased premises without court order or liability.",
        ],
    },

    "utilities": {
        "fair": [
            "Landlord shall provide and pay for municipal water, sewer, and regular trash collection; Tenant pays for unit-metered electricity and natural gas.",
            "All utility meters are separately assigned to the rental unit; Tenant shall transfer electrical accounts into Tenant's name within five (5) days of occupancy.",
            "Landlord shall maintain heating and hot water utility infrastructure at Landlord's expense in compliance with local municipal housing codes.",
            "In the event of utility disruption due to main line repairs, Landlord shall provide temporary potable water accommodations if outage exceeds 24 hours.",
            "Landlord covenants that heating systems meet statutory thermal efficiency standards capable of maintaining 68 degrees Fahrenheit during winter.",
        ],
        "needs_review": [
            "Utilities are apportioned using a Ratio Utility Billing System (RUBS) based on unit square footage and occupant count, billed monthly alongside rent.",
            "Water and sanitation services are billed back quarterly with a $15 monthly administrative accounting fee charged by the billing management firm.",
            "Tenant is responsible for establishing utility connections, including paying all municipal connection deposits and transfer fees.",
        ],
        "unfavorable": [
            "Tenant shall pay all utility bills for the entire building master meter, including common hallway lighting, exterior illumination, and laundry room water.",
            "Landlord reserves the right to shut off water, electric, or gas utility services if rent or any disputed charge is past due by more than five (5) days.",
            "Tenant assumes all risk and financial liability for municipal utility rate increases, water main replacements, and storm drain assessments.",
        ],
    },

    "late_fees_penalty": {
        "fair": [
            "If rent is not received by the 5th of the month, a late fee of $45 or 5% of delinquent rent (whichever is lower) shall be assessed.",
            "A grace period of five (5) calendar days is provided before any late penalty of $35 is charged for administrative processing overhead.",
            "No late fees shall accrue or be charged if rent is delayed due to demonstrable banking system outage or holiday processing deferrals.",
            "Statutory late fee limits under applicable housing law shall apply, capping total cumulative monthly delinquency fees at $50.",
            "Written notice of rent delinquency will be provided before any late administrative fee attaches to the tenant account.",
            "Late fees are assessed solely on overdue base rent and shall not be compounded or charged on unpaid maintenance fees.",
        ],
        "needs_review": [
            "Rent received after the 3rd incurs a $75 late charge plus $10 per day for each subsequent calendar day the balance remains unsatisfied.",
            "Late payments must be settled via certified cashier's check only, accompanied by a $50 late administration fee.",
            "A late charge of 8% of the overdue balance attaches on the 4th calendar day following the rent due date.",
            "Persistent late payment occurring three or more times in a 12-month period permits Landlord to require rent payment by wire only.",
        ],
        "unfavorable": [
            "A late penalty of $150 shall attach at 12:01 AM on the 2nd day of the month, plus compound interest at 24% per annum and $25 daily penalties.",
            "Failure to pay rent on the 1st day of the month forfeits all promotional concessions and triggers an automatic immediate $250 default assessment.",
            "Delinquent rent incurs an automatic immediate $100 penalty and forfeiture of the tenant's right of lease renewal.",
            "Late fee of $20 per day compounds daily without statutory cap until the full monthly obligation is satisfied.",
        ],
    },

    "indemnification": {
        "fair": [
            "Tenant shall indemnify Landlord from claims arising from Tenant's negligence. Landlord shall indemnify Tenant from claims resulting from Landlord's negligence.",
            "Both parties agree to maintain insurance and waive subrogation rights against each other for property casualties covered by standard policies.",
            "Landlord agrees to defend and hold Tenant harmless from all liabilities arising from common area hazards maintained exclusively by Landlord.",
            "Mutual indemnity applies: each party assumes responsibility solely for liabilities caused by their respective willful acts or omissions.",
            "Tenant indemnity obligations exclude any loss, liability, or damage arising from Landlord's failure to maintain common areas.",
        ],
        "needs_review": [
            "Tenant agrees to defend and hold Landlord harmless against any injury, loss, or property damage occurring within the demised premises during the tenancy.",
            "Tenant shall obtain renter's insurance with a minimum of $100,000 personal liability coverage, naming Landlord as an additional interested party.",
            "Tenant shall indemnify Landlord against any municipal code violation fines arising from occupant conduct or noise disturbances.",
        ],
        "unfavorable": [
            "Tenant agrees to fully indemnify, defend, and hold Landlord harmless from any injury, damage, or death occurring on the property, even if caused by Landlord's gross negligence.",
            "Tenant waives all rights to sue Landlord in tort or contract for mold exposure, roof collapse, asbestos release, or criminal assaults occurring on the premises.",
            "Tenant assumes total indemnity liability for third-party criminal trespass, building structural collapses, and utility line disruptions.",
            "Tenant agrees to pay all Landlord attorney fees incurred in any legal dispute, regardless of whether Landlord prevails in court.",
        ],
    },
}

OFFER_PATTERNS = {
    "compensation_salary": {
        "fair": [
            "Your starting annualized base salary will be {salary}, payable in semi-monthly installments in accordance with standard payroll practices.",
            "You will receive an annual salary of {salary}, reviewed annually each March for performance merit increases and cost-of-living adjustments.",
            "The Company shall pay an initial annual compensation of {salary}, disbursed bi-weekly via direct deposit, classified as an exempt professional role.",
            "Your base compensation will be {salary} per annum. Overtime-exempt status is established under the Fair Labor Standards Act (FLSA).",
            "Starting compensation is {salary} per year, with eligible annual salary reviews targeted at market 75th percentile benchmarks.",
        ],
        "needs_review": [
            "Your initial base salary will be {salary} for the first six months, after which management may adjust compensation based on business milestones.",
            "Base pay is set at {salary}, inclusive of all overtime and weekend duty, with an expectation of minimum 50 hours per week during deliverables.",
            "Compensation of {salary} per annum is subject to semi-annual company profitability adjustments of up to 10% upward or downward.",
        ],
        "unfavorable": [
            "The Company reserves the unilateral right to reduce your base salary of {salary} at any time upon five (5) days notice based on operating margins.",
            "Your nominal salary is {salary}, but 25% of monthly salary shall be withheld in company escrow to be paid out only upon completion of three continuous years of service.",
            "Employee agrees that base compensation of {salary} may be paid in restricted equity units or promissory notes in lieu of cash at Company discretion.",
        ],
    },

    "start_date_contingencies": {
        "fair": [
            "Your anticipated start date is October 15, 2026, contingent on standard reference verification, background screening, and Form I-9 eligibility.",
            "Employment shall commence on November 1, 2026. Please bring valid identity documentation verifying legal authorization to work in the country.",
            "The offer is contingent on successful completion of a standard criminal record check and education credential verification prior to your start date.",
            "Your start date will be agreed upon execution; offer is subject solely to statutory right-to-work verification under applicable immigration laws.",
        ],
        "needs_review": [
            "This offer is contingent upon satisfactory credit check, drug screening, background verification, and the Company securing client contract approval.",
            "Start date is contingent upon execution of the Company's proprietary 25-page intellectual property and restrictive covenant agreement.",
        ],
        "unfavorable": [
            "The Company may revoke this offer or terminate employment at its unreviewable discretion at any time within your first 90 days without explanation.",
            "Employment is conditioned upon authorizing continuous covert credit, internet, and physical surveillance checks throughout your tenure without notice.",
        ],
    },

    "at_will_employment": {
        "fair": [
            "Your employment is at-will, meaning that either you or the Company may terminate the relationship at any time, with or without cause or notice.",
            "Employment is at-will; while we anticipate a productive tenure, either party remains free to separate upon two (2) weeks professional courtesy notice.",
            "The employment relationship is at-will. Only an explicit written agreement signed by the Chief Executive Officer can modify this at-will status.",
            "Nothing in this letter or any company policy creates an express or implied contract of continued employment for any definite term.",
            "Either the employee or the employer may terminate the employment relationship at any time for any non-discriminatory reason.",
            "Employment hereunder remains terminable at the will of either party, with final wages disbursed pursuant to statutory state guidelines.",
            "You acknowledge that this offer of at-will employment supersedes all prior verbal statements, interviews, or hiring discussions.",
        ],
        "needs_review": [
            "Employment is strictly at-will. Company policy handbooks, performance evaluations, and supervisor oral statements do not create contractual tenure.",
            "The Company retains the right to terminate employment at-will, while requesting thirty (30) days advance transition notice from the employee.",
            "Your at-will status may be converted to an executive term agreement following the completion of twelve (12) months of active service.",
            "At-will employment applies; management may adjust job duties, compensation structure, or working titles without altering at-will tenure.",
        ],
        "unfavorable": [
            "Employment is at-will for the Company, permitting instant discharge without cause or severance; however, employee must give 90 days notice or forfeit all wages.",
            "The Company may dismiss you at-will with zero notice, whereas you are bound to serve for a minimum committed period of two (2) years under $25,000 penalty.",
            "Company may terminate employee at-will without notice or pay, while employee resignation requires repayment of all recruiter commissions incurred.",
            "Employee agrees that at-will termination by the Company forfeits all accrued commissions, unvested bonuses, and accrued vacation balances.",
        ],
    },

    "job_title_role": {
        "fair": [
            "You are being hired into the position of {role}, reporting to {manager} within the {department} organization.",
            "Your formal title will be {role}, based in {department}. In this capacity, you will lead core architectural and strategic deliverables.",
            "You will serve as {role}, collaborating with cross-functional teams and reporting directly to {manager}.",
        ],
        "needs_review": [
            "Your initial role is {role}, though the Company reserves the prerogative to reassign duties, department, and reporting structure as business dictates.",
        ],
        "unfavorable": [
            "Employee is hired under a generalized classification; the Company may arbitrarily demote your role, duties, and title to junior clerical tasks at will.",
        ],
    },

    "benefits_overview": {
        "fair": [
            "You are eligible to participate in comprehensive medical, dental, and vision insurance plans starting the first day of the month following hire.",
            "The Company provides twenty (20) days paid time off (PTO) annually, 11 paid holidays, and a 401(k) plan with 100% employer match up to 4% of salary.",
            "Benefits include employer-funded group term life insurance, short and long-term disability, and an annual $2,000 professional education stipend.",
            "Eligible employees receive twelve (12) weeks fully paid parental leave, flexible spending accounts (FSA), and commuter transit subsidies.",
        ],
        "needs_review": [
            "Basic healthcare coverage is provided after 90 days of employment. Employees contribute 40% of dependent premiums through payroll deductions.",
            "PTO accrues at 0.83 days per month (10 days/year). Unused vacation days do not carry over and are forfeited at year-end without cash settlement.",
        ],
        "unfavorable": [
            "No medical, dental, retirement, or disability benefits are provided during the initial twelve (12) months of employment.",
            "The Company offers no paid sick leave, vacation days, or holidays. Any absence from work results in automatic pro-rata salary forfeiture.",
        ],
    },

    "confidentiality_nda": {
        "fair": [
            "You agree to hold all Company proprietary information, technical source code, and client trade secrets in confidence during and after employment.",
            "Employee acknowledges that non-public business strategies, customer lists, and financial records constitute confidential trade secrets.",
            "You agree to preserve confidential information under standard professional standards, returning all company devices upon separation.",
            "Confidentiality duties exclude information generally known to the public through no fault or breach of the employee.",
            "Permitted disclosures include reports to regulatory agencies or in response to subpoena, provided prompt notice is given where legally permissible.",
            "The obligation to protect proprietary trade secrets survives termination of employment for a standard period of two (2) years.",
        ],
        "needs_review": [
            "Employee agrees to maintain strict confidentiality regarding all internal processes, compensation data, and business plans indefinitely under Delaware law.",
            "All work-related emails, notes, and professional communications must be deleted from personal devices within 24 hours of separation.",
            "Confidentiality covenants apply to all prospective customer lists, supplier terms, and pricing matrices developed during employment.",
        ],
        "unfavorable": [
            "Employee is permanently prohibited from discussing wages, working conditions, or grievances with coworkers, regulators, or future prospective employers.",
            "Any alleged disclosure of confidential information triggers automatic liquidated damages of $100,000 without requiring proof of actual harm.",
            "Employee agrees that all personal knowledge, skills, and industry methodologies acquired prior to employment belong exclusively to the Company.",
            "Violation of confidentiality entitles Company to immediate ex-parte injunction, seizure of personal hard drives, and payment of all legal fees.",
        ],
    },

    "termination_conditions": {
        "fair": [
            "In the event of termination without cause, employee shall receive two (2) months base salary continuation and health premium assistance.",
            "Termination for Cause (material fraud, felony conviction, gross misconduct) results in cessation of compensation as of the termination date.",
            "Upon mutual separation, employee will receive prompt payout of all earned wages and accrued unused PTO within statutory time limits.",
            "Company shall provide two (2) weeks advance written notice or equivalent base salary pay in lieu of notice upon termination without cause.",
            "Employee resignation requested with two weeks notice; Company reserves option to accelerate departure date with full pay through the notice window.",
            "Severance benefits are calculated on completed tenure and include vesting acceleration of equity grants pursuant to company plan rules.",
        ],
        "needs_review": [
            "Upon termination without cause, severance of one week per completed year of tenure is offered, contingent on signing a general liability release.",
            "Termination notice period is thirty (30) days for both parties; employee placed on garden leave during notice period remains bound by all policies.",
            "Severance payments shall cease immediately if separated employee obtains alternative employment during the continuation period.",
        ],
        "unfavorable": [
            "Company may terminate employment at any time for any reason with immediate effect, and employee expressly waives all rights to severance or accrued PTO.",
            "If employee is terminated for performance, employee agrees to reimburse all training costs, laptop stipends, and 50% of recent salary.",
            "Resignation by employee without ninety (90) days advance notice forfeits all vested stock options, final month wages, and accrued statutory benefits.",
            "Company may withhold final paycheck indefinitely pending complete audit of company expenses, device inspections, and client account audits.",
        ],
    },

    "working_hours_location": {
        "fair": [
            "This position is based at our headquarters under a hybrid working model: three days in office, two days remote, with core hours 10 AM to 4 PM.",
            "This is a 100% remote position within the United States. All necessary computing equipment and a monthly $100 home internet stipend are provided.",
            "Standard business hours are Monday through Friday, 9:00 AM to 5:00 PM, with flexibility for family commitments and personal scheduling.",
            "Employee shall work from our designated regional tech hub, with flexible arrival times between 8:00 AM and 10:00 AM local time.",
            "Remote work days may be coordinated with direct manager approval, with full employer reimbursement for ergonomic home office setups.",
            "Travel requirements are estimated at less than 15% annually, with all business travel, lodging, and per diem expenses reimbursed promptly.",
        ],
        "needs_review": [
            "The role is based in our Seattle office. The Company may mandate a return to 100% on-site work upon two (2) weeks notice.",
            "Core hours are 8:30 AM to 6:00 PM with required attendance at quarterly weekend sprint planning sessions compensated at regular hourly rate.",
            "Employee must reside within 45 miles of the primary corporate office to accommodate impromptu in-person team meetings.",
        ],
        "unfavorable": [
            "Employee is subject to mandatory permanent relocation to any domestic or foreign branch upon 14 days notice at employee's own personal expense.",
            "Employee must remain on-call 24 hours a day, 7 days a week, and report to the facility within 45 minutes of notification without overtime pay.",
            "Company reserves the right to install continuous biometric webcam and keystroke surveillance software on all personal home computers.",
            "Employee must work mandatory minimum 60-hour workweeks without overtime compensation or compensatory time off.",
        ],
    },

    "bonus_incentive": {
        "fair": [
            "You will be eligible for an annual performance bonus with a target of {bonus} of base salary, based on personal milestones and company goals.",
            "A sign-on bonus of $15,000 is payable on your first regular paycheck, repayable on a pro-rata basis only if you voluntarily resign within 12 months.",
            "Incentive bonuses are calculated annually and distributed within 60 days of fiscal year close, prorated for the initial partial calendar year.",
        ],
        "needs_review": [
            "Discretionary performance bonuses up to {bonus} may be awarded annually. Employee must remain actively employed on payout date to receive distribution.",
        ],
        "unfavorable": [
            "All bonuses and commissions are strictly discretionary and may be cancelled, revoked, or clawed back within 24 months at management's sole whim.",
            "If employee leaves the Company within three years, employee must repay 100% of all incentive bonuses, commissions, and equity gains earned.",
        ],
    },

    "non_compete_non_solicit": {
        "fair": [
            "For twelve (12) months following separation, employee agrees not to solicit active clients or recruit company employees with whom they worked.",
            "Employee agrees to a 12-month client non-solicitation covenant. (Non-compete provisions are void under applicable regional statutes).",
            "Non-solicitation of coworkers and clients is restricted to the specific geographic territory and accounts managed by the employee.",
        ],
        "needs_review": [
            "For twelve (12) months post-employment, employee shall not accept employment with direct enterprise competitors within the metropolitan area.",
            "During your employment and for 12 months following termination, you agree not to compete with the Company.",
            "During your employment and for twelve (12) months following termination, employee agrees not to engage in competing business activities.",
        ],
        "unfavorable": [
            "Employee agrees not to work in any capacity for any technology, finance, or retail company globally for three (3) full years post-separation.",
            "Breach of the global non-compete clause triggers immediate liquidated damages of $250,000, preliminary injunction without bond, and all legal costs.",
            "Employee is prohibited from working for any competitor or client anywhere in the nation for five (5) years following departure.",
        ],
    },
}

INSURANCE_PATTERNS = {
    "coverage_scope": {
        "fair": [
            "We insure against direct physical loss to the covered Dwelling caused by fire, lightning, windstorm, explosion, civil commotion, and pipe bursts.",
            "This policy provides Personal Liability coverage up to {limit} per occurrence for damages for which an Insured is legally responsible.",
            "Coverage C extends to personal property owned or used by an Insured anywhere in the world up to the limit specified in the Declarations.",
            "Additional Living Expense (Loss of Use) is covered up to 24 months if an insured loss renders the residence premises uninhabitable.",
            "Medical Payments to Others provides up to $5,000 per person for necessary medical services incurred due to an accident on the insured premises.",
        ],
        "needs_review": [
            "Dwelling coverage applies on a named-perils basis only; cosmetic exterior damage, wind-driven rain, and hail discoloration are not covered.",
            "Personal property is covered on an actual cash value (ACV) basis reflecting physical depreciation unless optional replacement cost is endorsed.",
        ],
        "unfavorable": [
            "Coverage is strictly limited to total catastrophic structural collapse resulting directly from fire; all smoke and partial losses are excluded.",
            "Insurer's maximum aggregate liability for all property and liability occurrences during the entire multi-year policy life is capped at $15,000.",
        ],
    },

    "exclusions": {
        "fair": [
            "We do not cover loss caused directly or indirectly by: flood, surface water, waves, tidal water, or water which backs up through sewers.",
            "We do not insure for loss caused directly or indirectly by water damage, flood, or earth movement.",
            "Exclusions include intentional acts committed by any Insured, acts of war, nuclear contamination, and civil insurrection.",
            "This policy excludes ordinary wear and tear, rust, deterioration, dry rot, industrial smog, and continuous water seepage exceeding 14 days.",
        ],
        "needs_review": [
            "Loss occurring while the residence premises has been vacant or unoccupied for more than thirty (30) consecutive days is excluded from coverage.",
            "Damage arising from off-premises utility failure, earth movement, and contractor design defects is excluded from policy protection.",
            "We do not insure for loss caused directly or indirectly by mechanical breakdown or gradual deterioration.",
        ],
        "unfavorable": [
            "Anti-concurrent causation clause: If an excluded event contributes in any degree to a loss, the entire claim is excluded in its entirety.",
            "All water damage of any kind, whether sudden, accidental, resulting from burst municipal pipes, or appliance failures, is completely excluded.",
            "Excludes all liability claims involving domestic animals, garden equipment, playground structures, or guests under the age of eighteen.",
        ],
    },

    "deductible_premium": {
        "fair": [
            "A standard deductible of {deductible} applies to each covered loss under Section I. We will pay the amount exceeding the stated deductible.",
            "The annual premium is payable in quarterly or monthly automatic bank debits without financing charges, subject to a fixed {deductible} deductible.",
            "Only one deductible of {deductible} shall apply per occurrence, even if damage affects both the primary dwelling and personal property.",
        ],
        "needs_review": [
            "A standard deductible of $1,000 applies, except that a mandatory 2% Named Hurricane Deductible applies during declared weather advisories.",
            "Monthly premium installments incur an additional administrative servicing fee of $6.00 per payment voucher.",
        ],
        "unfavorable": [
            "A disappearing deductible applies: the deductible equals 15% of the total Dwelling Limit or $25,000 for any windstorm or freezing event.",
            "Failure to pay any premium installment on the exact due date automatically doubles the policyholder deductible for the subsequent 180 days.",
        ],
    },

    "claims_process": {
        "fair": [
            "In the event of a covered loss, the Insured must give prompt notice to us, protect property from further harm, and exhibit damaged items.",
            "The Insured shall prepare an inventory of damaged personal property and submit a sworn proof of loss within sixty (60) days of our request.",
            "We shall acknowledge receipt of your claim within 15 days, commence investigation, and notify you of acceptance or rejection within 30 days.",
        ],
        "needs_review": [
            "Written notice of loss must be filed within thirty (30) days of the occurrence; failure to report promptly may prejudice claim resolution.",
        ],
        "unfavorable": [
            "Notice of loss must be received by the Insurer within 48 hours via registered mail; failure to give notice within 48 hours forfeits all coverage.",
            "Policyholder must provide 5 years of personal income tax returns and submit to multiple unrepresented examinations under oath or forfeit claim.",
        ],
    },

    "cancellation_non_renewal": {
        "fair": [
            "You may cancel this policy at any time by written notice. If canceled, unearned premium will be refunded on a pro-rata basis within 30 days.",
            "We may cancel for non-payment of premium upon ten (10) days notice, or for other statutory reasons upon thirty (30) days advance written notice.",
            "If we elect not to renew this policy, we will deliver or mail written notice of non-renewal at least forty-five (45) days before policy expiration.",
        ],
        "needs_review": [
            "The Insurer may cancel during the first 60 days of a new policy for any lawful underwriting reason upon twenty (20) days notice.",
            "Upon insured cancellation, unearned premium is calculated using short-rate cancellation penalty tables deducting 10% for administrative costs.",
        ],
        "unfavorable": [
            "Insurer may cancel this policy at any time without advance notice, retaining 100% of all unearned premiums as liquidated underwriting expenses.",
            "If the Insured submits a single claim inquiry, the Insurer reserves the right to immediately terminate the policy without opportunity to cure.",
        ],
    },

    "limits_of_liability": {
        "fair": [
            "Our limit of liability for Personal Liability will not exceed {limit} for all damages resulting from any single occurrence.",
            "Scheduled personal property sub-limits apply: $2,500 for jewelry, $2,500 for firearms, and $2,500 for silverware, unless higher coverage is endorsed.",
            "Medical Payments to Others limit is $5,000 per person for necessary medical treatments incurred within three years from the accident date.",
        ],
        "needs_review": [
            "A special aggregate limit of liability of $2,000 applies to water backup losses and $5,000 for mold remediation expenses.",
        ],
        "unfavorable": [
            "The maximum total aggregate payout for all dwelling, personal property, and liability occurrences combined is capped at $20,000.",
            "All personal property sub-limits for computers, electronics, tools, and clothing are strictly capped at $150 per category.",
        ],
    },

    "grace_period": {
        "fair": [
            "A grace period of thirty (30) days will be granted for the payment of each renewal premium, during which coverage remains in full effect.",
            "A fifteen (15) day grace period applies to recurring automated premium withdrawals; coverage continues seamlessly if funded within this window.",
            "If premium is unpaid on the billing date, a statutory 31-day grace window is provided without late penalty or lapse in protection.",
            "Policyholder maintains full coverage rights throughout the thirty-day grace period, with any covered loss paid net of unpaid premium.",
            "Written reminder notice will be dispatched seven (7) days into the grace period before any policy cancellation is initiated.",
            "During any active grace window, all dwelling and personal liability protections continue without restriction or impairment.",
        ],
        "needs_review": [
            "A grace period of ten (10) days is permitted for premium receipt, subject to a $25 late fee assessed during the grace window.",
            "Grace period is limited to seven (7) business days; payments received after 7 days require a formal underwriting reinstatement application.",
            "Coverage continues during the 15-day grace period, but payments must be made by certified funds or cashier's draft only.",
            "A grace period of fourteen (14) days is provided, but repeated use of the grace window more than twice annually permits insurer non-renewal.",
        ],
        "unfavorable": [
            "Zero grace period is provided; if premium is not received by 5:00 PM on the due date, coverage lapses immediately retroactively to the prior billing date.",
            "Any premium payment received during the grace period incurs a mandatory 15% surcharge and a 30-day blackout period where no claims are paid.",
            "If payment fails on the scheduled date, insurer may cancel immediately with zero grace period and retain all unearned premiums.",
            "Grace period is voided if the policyholder has filed any claim during the preceding twenty-four (24) months.",
        ],
    },

    "dispute_resolution_appraisal": {
        "fair": [
            "If you and we fail to agree on the loss amount, either party may demand an appraisal; each selects an appraiser, and appraisers select an umpire.",
            "In appraisal, an award agreed upon by any two of the three shall determine the loss amount. Each party pays its own appraiser and shares umpire fees.",
            "Mediation is available as a non-binding dispute resolution option prior to initiating court litigation in a court of competent jurisdiction.",
            "Disputes regarding scope of coverage may be submitted to neutral non-binding arbitration before an agreed retired judicial officer.",
        ],
        "needs_review": [
            "Valuation disagreements must proceed through formal binding appraisal prior to any court filing; prevailing party recovers appraisal fees.",
            "Appraisal must be demanded in writing within sixty (60) days of the insurer's initial claim adjustment determination.",
        ],
        "unfavorable": [
            "All disputes must be submitted to confidential binding arbitration at Insurer's headquarters at Insured's sole expense, with jury trial waived.",
            "Policyholder agrees that Insurer's internal adjuster valuation is final, conclusive, and legally unappealable, barring all court actions.",
            "Any legal action against the Insurer must be commenced within twelve (12) months of loss, regardless of longer statutory limitation periods.",
        ],
    },

    "subrogation": {
        "fair": [
            "An Insured may waive in writing prior to a loss all rights of recovery against any person; if not waived, we may require assignment up to amount paid.",
            "If we pay a claim, we are subrogated to the Insured's rights of recovery against third parties, but only to the extent of our actual claim payout.",
            "Insurer subrogation recoveries shall be shared with the Insured to reimburse any out-of-pocket deductible on a pro-rata basis.",
            "We shall not exercise subrogation rights against any tenant, family member, or resident of the insured household.",
            "Subrogation actions shall be conducted at the sole expense of the Insurer without legal cost or financial exposure to the Insured.",
        ],
        "needs_review": [
            "Upon claim payment, the Insured assigns all rights of recovery to the Insurer and must cooperate fully in third-party litigation.",
            "Insured shall execute necessary legal assignment instruments and appear at depositions when requested in subrogation actions.",
            "Any recovery secured by the Insured from third parties must be held in trust for the Insurer until claim payouts are satisfied.",
        ],
        "unfavorable": [
            "Insured agrees to personally reimburse the Insurer for all claim payouts and legal expenses if third-party subrogation is unsuccessful.",
            "Insured is barred from signing any standard commercial lease waiver of subrogation, with breach causing immediate retroactive loss forfeiture.",
            "Failure to attend any subrogation deposition in another state at Insured's own expense shall require immediate refund of all claim proceeds.",
            "Insurer retains 100% of all third-party recoveries, including the Insured's deductible, until all company investigative costs are repaid.",
        ],
    },

    "policy_period_renewal": {
        "fair": [
            "This policy applies only to losses occurring during the policy period stated in the Declarations, starting at 12:01 AM Standard Time.",
            "We agree to offer renewal terms annually upon payment of renewal premium, unless notice of non-renewal is provided at least 45 days in advance.",
            "The standard policy term is twelve (12) consecutive months, renewed automatically upon receipt of the renewal premium installment.",
            "Policy coverage commences at 12:01 AM on the effective date and terminates at 12:01 AM on the expiration date at the insured residence location.",
            "In the event of statutory rate revisions, renewal terms reflecting approved filings shall be transmitted at least thirty (30) days prior to renewal.",
        ],
        "needs_review": [
            "Policy period is six (6) months, with automatic renewal subject to revised underwriting loss experience ratings and premium indexation.",
            "Renewal premium must be received fifteen (15) days prior to expiration to guarantee continuous underwriting endorsement.",
            "Annual policy renewal is subject to mandatory re-inspection of roof condition, plumbing fixtures, and electrical panels.",
        ],
        "unfavorable": [
            "The policy period expires without renewal rights at term end; the Insurer owes no duty to provide non-renewal notice or offer renewal terms.",
            "Any renewal shall be subject to retrospective premium audits permitting the Insurer to assess retroactive rate surcharges for past claims.",
            "Insurer reserves the unilateral right to rewrite, reduce coverage limits, or eliminate perils upon any renewal anniversary without consent.",
            "Policy terminates automatically upon filing of any property loss claim, with reinstatement requiring new full underwriting fees.",
        ],
    },
}

ALL_PATTERNS = {
    "rental_agreement": RENTAL_PATTERNS,
    "job_offer_letter": OFFER_PATTERNS,
    "insurance_policy": INSURANCE_PATTERNS,
}

OPENERS = [
    "It is expressly agreed and stipulated that",
    "Under the terms and conditions hereof,",
    "As a material inducement to this agreement,",
    "The parties mutually covenant and establish that",
    "Subject to applicable statutory requirements,",
    "In consideration of the mutual covenants herein,",
    "For the purposes of this agreement,",
    "Notwithstanding any provision to the contrary herein,",
    "Pursuant to governing regional administrative codes,",
    "Unless otherwise agreed in a formal written addendum,",
    "In accordance with established legal standards,",
    "By executing this instrument, the parties covenant that",
    "It is understood and agreed between the parties that",
    "Except as expressly modified by written amendment,",
    "Subject always to the general conditions of this instrument,",
]

CLOSERS = [
    "Any dispute arising hereunder shall be interpreted under the laws of {jurisdiction}.",
    "This covenant constitutes a material term and condition of the overall agreement.",
    "Both parties acknowledge having read, reviewed, and consented to this specific provision.",
    "Any modification or waiver of this clause must be executed in writing by authorized representatives.",
    "This clause shall remain in full force and effect throughout the operative term.",
    "Neither party may assign rights under this provision without mutual written concurrence.",
    "All notices required under this section must be in writing and delivered through documented channels.",
    "This provision shall be construed in accordance with the public policy of {jurisdiction}.",
    "Failure to enforce any right hereunder shall not constitute a waiver of subsequent enforcement.",
    "This obligation shall survive the expiration or earlier termination of this contract.",
]

def generate_scenario_clause(doc_type: str, clause_type: str, index: int, favorability: str = None) -> tuple[str, str]:
    """
    Generates a unique, semantically complete scenario clause for the given document and clause type.
    """
    patterns = ALL_PATTERNS[doc_type][clause_type]
    
    if not favorability or favorability not in patterns:
        fav = random.choice(["fair", "needs_review", "unfavorable"])
    else:
        fav = favorability

    template = random.choice(patterns[fav])

    rent = random.choice(RENT_VALS)
    deposit = random.choice(DEPOSIT_VALS)
    salary = random.choice(SALARY_VALS)
    bonus = random.choice(BONUS_VALS)
    deductible = random.choice(DEDUCTIBLE_VALS)
    limit = random.choice(LIMIT_VALS)
    jurisdiction = random.choice(JURISDICTIONS)
    role = random.choice(ROLES)
    department = random.choice(DEPARTMENTS)
    manager = random.choice(MANAGERS)
    days = random.choice(DAYS_VALS)

    body = template.format(
        rent=rent,
        deposit=deposit,
        salary=salary,
        bonus=bonus,
        deductible=deductible,
        limit=limit,
        jurisdiction=jurisdiction,
        role=role,
        department=department,
        manager=manager,
        days=days
    )

    formats = [
        f"Section {index % 50 + 1}.{(index // 50) % 9 + 1} ",
        f"Clause {index % 40 + 1} ",
        f"Paragraph {chr(65 + index % 26)}.{index % 10 + 1}: ",
        f"Item {(index * 3) % 99 + 1}: ",
        "",
    ]
    sec_num = formats[index % len(formats)]
    opener = OPENERS[index % len(OPENERS)] + " " if (index % 2 == 1) else ""
    closer_template = CLOSERS[(index // 3) % len(CLOSERS)]
    closer = " " + closer_template.format(jurisdiction=jurisdiction) if (index % 3 != 1) else ""

    full_clause = f"{sec_num}{opener}{body}{closer}".strip()
    return full_clause, fav
