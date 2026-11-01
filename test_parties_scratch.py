import re
from pathlib import Path

def extract_contract_parties(raw_text: str, document_type: str = None) -> dict:
    parties = []
    signatories = []
    effective_date = None
    agreement_title = None
    governing_law = None

    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    if lines:
        first_few = lines[:5]
        for l in first_few:
            if any(t in l.upper() for t in ["AGREEMENT", "OFFER LETTER", "LEASE", "POLICY", "CONTRACT"]):
                agreement_title = l
                break
        if not agreement_title:
            agreement_title = lines[0]

    # Effective Date
    m_eff = re.search(r'(?:as of|effective as of|dated as of|effective date[:\s]+)\s*([A-Za-z]+\s+\d{1,2},\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4})', raw_text, re.IGNORECASE)
    if m_eff:
        effective_date = m_eff.group(1).strip()
    else:
        m_date = re.search(r'Date:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})', raw_text)
        if m_date:
            effective_date = m_date.group(1).strip()

    # Governing Law
    m_gov = re.search(r'(?:laws of the State of|laws of|governed by the laws of)\s+([A-Za-z\s]+?)(?:\s*,\s*without|\.|\;)', raw_text, re.IGNORECASE)
    if m_gov:
        governing_law = m_gov.group(1).strip()

    # Pattern A: Standard Corporate "by and between Party A ... ("Role") ... and Party B ... ("Role")"
    m_between = re.search(r'by and between\s+(.+?)\s+and\s+(.+?)(?:\.|\n\n|;\s*WHEREAS)', raw_text, re.IGNORECASE | re.DOTALL)
    if m_between:
        part1 = m_between.group(1).strip()
        part2 = m_between.group(2).strip()

        def parse_single_party(p_str: str) -> dict:
            # Look for role in parentheses quotes: ("Customer") or (the "Provider")
            role_m = re.search(r'\(\s*(?:the\s+)?["\']([A-Za-z0-9_\s]+?)["\']\s*\)', p_str)
            role = role_m.group(1).strip() if role_m else "Party"
            
            # Entity name is usually before the first comma, or before ("Role") or before "a Delaware"
            clean_str = re.sub(r'\(\s*(?:the\s+)?["\'][A-Za-z0-9_\s]+?["\']\s*\)', '', p_str).strip()
            
            # Check entity type / jurisdiction: e.g. "a Delaware corporation"
            ent_type_m = re.search(r'(?:,\s*)?a[n]?\s+([A-Za-z\s]+?(?:corporation|limited liability company|LLC|company|partnership|trust))', clean_str, re.IGNORECASE)
            ent_type = ent_type_m.group(1).strip() if ent_type_m else "Entity"

            # Check address
            addr_m = re.search(r'with (?:its )?(?:principal )?(?:place of business|office|residence) at\s+([^,\n]+(?:,\s*[^,\n]+){1,3})', clean_str, re.IGNORECASE)
            address = addr_m.group(1).strip() if addr_m else None

            # Name is the leading corporate name before the comma or entity type description
            name_part = clean_str.split(',')[0].strip()
            # If name_part still contains "a [state] [entity]", split on " a "
            name_sub = re.split(r'\s+a[n]?\s+(?:Delaware|California|New York|Texas|foreign|\w+\s+corporation|\w+\s+LLC)', name_part, flags=re.IGNORECASE)[0].strip()
            name = re.sub(r'^(?:the\s+)', '', name_sub, flags=re.IGNORECASE).strip()

            return {
                "name": name,
                "role": role,
                "entity_type": ent_type,
                "address": address
            }

        p1_info = parse_single_party(part1)
        p2_info = parse_single_party(part2)
        if p1_info["name"] and p2_info["name"]:
            parties.append(p1_info)
            parties.append(p2_info)

    # Pattern B: Lease "entered into between Landlord, [Name], and Tenant, [Name]"
    if not parties:
        m_lease = re.search(
            r'entered into between\s+Landlord[,\s]+([A-Za-z\.\s\-]+?)[,\s]+and\s+Tenant[,\s]+([A-Za-z\.\s\-]+?)(?:[,\s]+for|\.|\;)',
            raw_text,
            re.IGNORECASE
        )
        if m_lease:
            parties.append({
                "name": m_lease.group(1).strip(),
                "role": "Landlord",
                "entity_type": "Individual / Owner"
            })
            parties.append({
                "name": m_lease.group(2).strip(),
                "role": "Tenant",
                "entity_type": "Individual / Lessee"
            })

    # Pattern C: Job Offer "Company" and "Dear [Name]"
    if not parties:
        m_cand = re.search(r'Dear\s+([A-Za-z\.\s\-]+?),', raw_text)
        m_comp = re.search(r'(?:^|\n)([A-Za-z0-9\.\s\-]+?(?:Inc\.|LLC|Corp\.|Technologies|Solutions|Company))\s*(?:\([^)]+\))?\s*(?:is pleased to extend|extends|offers)', raw_text)
        if not m_comp:
            m_comp = re.search(r'(?:^|\n)([A-Za-z0-9\.\s\-]+?)\s+\((?:the\s+)?["\']Company["\']\)', raw_text)
        if m_cand or m_comp:
            if m_comp:
                parties.append({
                    "name": m_comp.group(1).strip(),
                    "role": "Employer / Company",
                    "entity_type": "Corporation"
                })
            if m_cand:
                parties.append({
                    "name": m_cand.group(1).strip(),
                    "role": "Employee / Candidate",
                    "entity_type": "Individual"
                })

    # Signature Block Extraction
    # Search for blocks like:
    # [COMPANY NAME]
    # By: [Name], [Title]
    # Date: [Date]
    sig_blocks = re.findall(
        r'(?:([A-Z0-9\s,\.]{3,60})\n+)?(?:By|Signature):\s*([A-Za-z\.\s\-]+?)(?:,\s*([^\n]+))?\n+(?:(?:Date|Dated):\s*([^\n]+))?',
        raw_text,
        re.IGNORECASE
    )
    for sm in sig_blocks:
        ent_raw = sm[0].strip() if sm[0] else None
        if ent_raw:
            # Take only the last non-empty line of entity block
            ent_lines = [l.strip() for l in ent_raw.splitlines() if l.strip()]
            ent = ent_lines[-1] if ent_lines else None
            # Filter out non-entity phrases like "Effective Date"
            if ent and any(bad in ent.lower() for bad in ["effective date", "witness whereof"]):
                ent = None
        else:
            ent = None

        name = sm[1].strip()
        title = sm[2].strip() if sm[2] else None
        dt = sm[3].strip() if sm[3] else None
        if name.lower() not in ["the", "an", "a", "date", "name", "signature"] and len(name) > 2:
            signatories.append({
                "name": name,
                "title": title or "Authorized Representative",
                "entity": ent,
                "date": dt
            })

    return {
        "agreement_title": agreement_title,
        "effective_date": effective_date,
        "governing_law": governing_law,
        "parties": parties,
        "signatories": signatories,
    }

if __name__ == "__main__":
    docs = [
        Path("sample_documents/synthetic_contracts/contract_a_ordinary_agreement.txt"),
        Path("sample_documents/sample_residential_lease.txt"),
        Path("sample_documents/sample_tech_offer_letter.txt"),
    ]
    for doc_path in docs:
        if doc_path.exists():
            txt = doc_path.read_text(encoding="utf-8")
            res = extract_contract_parties(txt)
            print(f"=== {doc_path.name} ===")
            print("Title:", res["agreement_title"])
            print("Effective Date:", res["effective_date"])
            print("Governing Law:", res["governing_law"])
            print("Parties:", res["parties"])
            print("Signatories:", res["signatories"])
            print()
