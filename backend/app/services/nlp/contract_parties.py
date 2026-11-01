import re
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

def extract_contract_parties(raw_text: str, document_type: Optional[str] = None) -> Dict[str, Any]:
    """
    High-precision contractual entity, party, and signatory extractor.
    Identifies:
    1. Agreement Title & Type
    2. Effective Date / Execution Date
    3. Governing Law / Jurisdiction
    4. Primary Contracting Parties (Name, Role, Entity Type, Address/Location)
    5. Authorized Signatories (Name, Title, Entity, Date)
    6. Contract Specific Identifiers (e.g. Premises for Lease, Policyholder for Insurance, Candidate for Job Offer)
    """
    if not raw_text or not raw_text.strip():
        return {
            "agreement_title": None,
            "effective_date": None,
            "governing_law": None,
            "parties": [],
            "signatories": [],
            "summary": "No contract text available."
        }

    parties: List[Dict[str, Any]] = []
    signatories: List[Dict[str, Any]] = []
    effective_date: Optional[str] = None
    agreement_title: Optional[str] = None
    governing_law: Optional[str] = None
    specific_details: Dict[str, Any] = {}

    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]

    # 1. Agreement Title Extraction
    if lines:
        for l in lines[:6]:
            if any(term in l.upper() for term in [
                "AGREEMENT", "OFFER LETTER", "LEASE", "POLICY", "CONTRACT", "MEMORANDUM", "TERMS"
            ]):
                agreement_title = l
                break
        if not agreement_title:
            agreement_title = lines[0]

    # 2. Effective Date / Execution Date Extraction
    m_eff = re.search(
        r'(?:as of|effective as of|dated as of|effective date[:\s]+|dated[:\s]+)\s*([A-Za-z]+\s+\d{1,2},\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        raw_text,
        re.IGNORECASE
    )
    if m_eff:
        effective_date = m_eff.group(1).strip()
    else:
        m_date = re.search(r'(?:^|\n)Date:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})', raw_text)
        if m_date:
            effective_date = m_date.group(1).strip()

    # 3. Governing Law / Jurisdiction
    m_gov = re.search(
        r'(?:laws of the State of|laws of|governed by the laws of)\s+([A-Za-z\s]+?)(?:\s*,\s*without|\.|\;)',
        raw_text,
        re.IGNORECASE
    )
    if m_gov:
        governing_law = m_gov.group(1).strip()

    # 4. Contracting Parties Extraction

    # Pattern A: Standard Corporate "by and between Party A ... ("Role") ... and Party B ... ("Role")"
    m_between = re.search(
        r'by and between\s+(.+?)\s+and\s+(.+?)(?:\.|\n\n|;\s*WHEREAS|\s+WHEREAS|\s*\(collectively)',
        raw_text,
        re.IGNORECASE | re.DOTALL
    )
    if m_between:
        part1 = m_between.group(1).strip()
        part2 = m_between.group(2).strip()

        def parse_single_party(p_str: str, default_role: str = "Party") -> Dict[str, Any]:
            role_m = re.search(r'\(\s*(?:the\s+)?["\']([A-Za-z0-9_\s]+?)["\']\s*\)', p_str)
            role = role_m.group(1).strip() if role_m else default_role

            clean_str = re.sub(r'\(\s*(?:the\s+)?["\'][A-Za-z0-9_\s]+?["\']\s*\)', '', p_str).strip()
            # If leading with role label like "Landlord, John" or "Tenant, Emily"
            clean_str = re.sub(r'^(?:Landlord|Tenant|Lessor|Lessee|Customer|Provider|Buyer|Seller)[,\s]+', '', clean_str, flags=re.IGNORECASE).strip()

            ent_type_m = re.search(
                r'(?:,\s*)?a[n]?\s+([A-Za-z\s]+?(?:corporation|limited liability company|LLC|company|partnership|trust))',
                clean_str,
                re.IGNORECASE
            )
            if ent_type_m:
                ent_type = ent_type_m.group(1).strip()
            else:
                agent_m = re.search(r'managing agent for\s+([^,\n]+)', clean_str, re.IGNORECASE)
                ent_type = agent_m.group(1).strip() if agent_m else "Entity"

            addr_m = re.search(
                r'with (?:its )?(?:principal )?(?:place of business|office|residence) at\s+([^,\n]+(?:,\s*[^,\n]+){1,3})',
                clean_str,
                re.IGNORECASE
            )
            address = addr_m.group(1).strip() if addr_m else None

            name_part = clean_str.split(',')[0].strip()
            name_sub = re.split(
                r'\s+a[n]?\s+(?:Delaware|California|New York|Texas|foreign|\w+\s+corporation|\w+\s+LLC)',
                name_part,
                flags=re.IGNORECASE
            )[0].strip()
            name = re.sub(r'^(?:the\s+)', '', name_sub, flags=re.IGNORECASE).strip()

            return {
                "name": name,
                "role": role,
                "entity_type": ent_type,
                "address": address
            }

        p1_info = parse_single_party(part1, "Customer / Party A")
        p2_info = parse_single_party(part2, "Provider / Party B")
        if p1_info["name"] and p2_info["name"] and len(p1_info["name"]) > 1:
            parties.append(p1_info)
            parties.append(p2_info)

    # Pattern B: Lease "entered into between Landlord, [Name], and Tenant, [Name]"
    if not parties:
        m_lease = re.search(
            r'(?:entered into|executed|by and)?\s*between\s+Landlord[,\s]+([A-Za-z\.\s\-]+?)(?:\s*\([^)]+\))?[,\s]+(?:managing agent for\s+([^,\n]+?))?[,\s]+and\s+Tenant[,\s]+([A-Za-z\.\s\-]+?)(?:\s*\([^)]+\))?(?:[,\s]+for|\.|\;|\n)',
            raw_text,
            re.IGNORECASE
        )
        if m_lease:
            landlord_name = m_lease.group(1).strip()
            landlord_ent = m_lease.group(2).strip() if m_lease.group(2) else "Property Owner"
            tenant_name = m_lease.group(3).strip()
            parties.append({
                "name": landlord_name,
                "role": "Landlord",
                "entity_type": landlord_ent,
                "address": None
            })
            parties.append({
                "name": tenant_name,
                "role": "Tenant",
                "entity_type": "Individual / Lessee",
                "address": None
            })

    # Pattern C: Job Offer Letter (Company & Candidate)
    if not parties:
        m_cand = re.search(r'(?:Dear|Candidate:)\s*([A-Za-z\.\s\-]+?)(?:,|\n)', raw_text)
        m_comp = re.search(
            r'(?:^|\n)([A-Za-z0-9\.\s\-]+?(?:Inc\.|LLC|Corp\.|Technologies|Solutions|Company))\s*(?:\([^)]+\))?\s*(?:is pleased to extend|extends|offers|delighted to extend)',
            raw_text
        )
        if not m_comp:
            m_comp = re.search(r'(?:^|\n)([A-Za-z0-9\.\s\-]+?)\s+\((?:the\s+)?["\']Company["\']\)', raw_text)

        if m_cand or m_comp:
            if m_comp:
                parties.append({
                    "name": m_comp.group(1).strip(),
                    "role": "Employer / Company",
                    "entity_type": "Corporate Entity",
                    "address": None
                })
            if m_cand:
                cand_name = m_cand.group(1).strip()
                if cand_name.lower() not in ["applicant", "candidate", "sir", "madam"]:
                    parties.append({
                        "name": cand_name,
                        "role": "Employee / Candidate",
                        "entity_type": "Individual",
                        "address": None
                    })

    # Pattern D: Insurance Policy (Named Insured & Policy Number)
    if not parties:
        m_insured = re.search(r'Named Insured:\s*([^\n]+)', raw_text, re.IGNORECASE)
        m_pol_num = re.search(r'Policy Number:\s*([^\n]+)', raw_text, re.IGNORECASE)
        m_premise = re.search(r'Residence Premises:\s*([^\n]+)', raw_text, re.IGNORECASE)
        if m_insured:
            parties.append({
                "name": m_insured.group(1).strip(),
                "role": "Named Insured / Policyholder",
                "entity_type": "Insured Person",
                "address": m_premise.group(1).strip() if m_premise else None
            })
            if m_pol_num:
                specific_details["policy_number"] = m_pol_num.group(1).strip()

    # 5. Signatories Extraction
    # Matches patterns like:
    # [COMPANY NAME]
    # By: John Smith, Chief Executive Officer
    # Date: January 15, 2026
    sig_blocks = re.findall(
        r'(?:([A-Z0-9\s,\.]{3,60})\n+)?(?:By|Signature):\s*([A-Za-z\.\s\-]+?)(?:,\s*([^\n]+))?\n+(?:(?:Date|Dated):\s*([^\n]+))?',
        raw_text,
        re.IGNORECASE
    )
    for sm in sig_blocks:
        ent_raw = sm[0].strip() if sm[0] else None
        ent = None
        if ent_raw:
            ent_lines = [l.strip() for l in ent_raw.splitlines() if l.strip()]
            cand_ent = ent_lines[-1] if ent_lines else None
            if cand_ent and not any(bad in cand_ent.lower() for bad in ["effective date", "witness whereof", "date:", "duly authorized"]):
                ent = cand_ent

        name = sm[1].strip()
        title = sm[2].strip() if sm[2] else None
        dt = sm[3].strip() if sm[3] else None
        if name.lower() not in ["the", "an", "a", "date", "name", "signature"] and len(name) > 2:
            signatories.append({
                "name": name,
                "title": title or "Authorized Signatory",
                "entity": ent,
                "date": dt
            })

    # Additional Lease Property Premises Check
    m_property = re.search(
        r'(?:premises located at|property located at|dwelling unit designated as)\s+([^,\n]+(?:,\s*[^,\n]+){1,3})',
        raw_text,
        re.IGNORECASE
    )
    if m_property:
        specific_details["premises"] = m_property.group(1).strip()

    # Additional Job Offer Role Check
    m_job_role = re.search(
        r'(?:position of|role of)\s+([A-Za-z\s]+?)(?:\.|\,|\;|\n)',
        raw_text,
        re.IGNORECASE
    )
    if m_job_role:
        specific_details["job_role"] = m_job_role.group(1).strip()

    # 6. Concise Summary Formulation
    if parties:
        p_strs = [f"{p['name']} ({p['role']})" for p in parties]
        summary_str = f"Agreement between { ' and '.join(p_strs) }"
        if signatories:
            s_strs = [f"{s['name']} ({s.get('title', 'Signatory')})" for s in signatories]
            summary_str += f", executed by { ', '.join(s_strs) }"
        if effective_date:
            summary_str += f" on {effective_date}."
        else:
            summary_str += "."
    else:
        summary_str = f"{agreement_title or 'Legal Agreement'} without explicit named parties."

    return {
        "agreement_title": agreement_title,
        "effective_date": effective_date,
        "governing_law": governing_law,
        "parties": parties,
        "signatories": signatories,
        "details": specific_details,
        "summary": summary_str
    }
