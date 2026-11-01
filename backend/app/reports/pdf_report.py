import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)

DISCLAIMER_TEXT = (
    "ClauseGuard AI provides automated, non-expert analysis for informational purposes only. "
    "It is not legal advice. Consult a qualified attorney for decisions with legal consequences."
)

def generate_pdf_report(document, clauses, missing_clauses, deadlines, pii_findings, integrity_findings=None) -> bytes:
    """
    Generates a professional, print-ready PDF analysis report for a document using ReportLab.
    Includes overall score, risk bands, executive summary clause counts, per-clause findings,
    missing clauses, deadlines, PII summary, excluded block transparency, and the mandatory
    permanent non-legal-advice disclaimer.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#000000"),
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#666666"),
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#111827"),
        spaceBefore=14,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1F2937"),
    )
    body_bold = ParagraphStyle(
        "ReportBodyBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1F2937"),
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#B91C1C"),
    )

    elements = []

    elements.append(Paragraph("ClauseGuard AI — Risk Analysis Report", title_style))
    elements.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')} | Document ID: {document.id}",
            subtitle_style,
        )
    )
    elements.append(Spacer(1, 10))

    disclaimer_table = Table(
        [[Paragraph(f"<b>LEGAL NOTICE:</b> {DISCLAIMER_TEXT}", disclaimer_style)]],
        colWidths=[540],
    )
    disclaimer_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEF2F2")),
            ("BORDER", (0, 0), (-1, -1), 1, colors.HexColor("#FCA5A5")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ])
    )
    elements.append(disclaimer_table)
    elements.append(Spacer(1, 12))

    # Partition clauses into substantive vs excluded (non-substantive)
    substantive_clauses = [
        c for c in clauses
        if getattr(c, "is_substantive", True) and getattr(c, "content_type", "SUBSTANTIVE_CLAUSE") == "SUBSTANTIVE_CLAUSE"
    ]
    excluded_clauses = [
        c for c in clauses
        if not (getattr(c, "is_substantive", True) and getattr(c, "content_type", "SUBSTANTIVE_CLAUSE") == "SUBSTANTIVE_CLAUSE")
    ]

    unfavorable_count = sum(1 for c in substantive_clauses if (getattr(c, "favorability_label", "") or "").lower() == "unfavorable")
    review_count = sum(
        1 for c in substantive_clauses
        if (getattr(c, "favorability_label", "") or "").lower() != "unfavorable"
        and ((getattr(c, "favorability_label", "") or "").lower() == "needs_review"
             or getattr(c, "clause_type", "") == "UNCERTAIN"
             or getattr(c, "human_review", False))
    )
    fair_count = max(0, len(substantive_clauses) - unfavorable_count - review_count)

    elements.append(Paragraph("1. Document Overview & Executive Summary", section_heading))
    
    band_colors = {
        "low": colors.HexColor("#15803D"),
        "medium": colors.HexColor("#B45309"),
        "high": colors.HexColor("#C2410C"),
        "critical": colors.HexColor("#B91C1C"),
    }
    score_color = band_colors.get((document.risk_band or "").lower(), colors.HexColor("#15803D"))

    score_text = f"<b>{document.overall_risk_score} / 100</b> ({document.risk_band.upper() if document.risk_band else 'N/A'})"
    
    overview_data = [
        [Paragraph("<b>Filename:</b>", body_style), Paragraph(document.filename, body_style),
         Paragraph("<b>Overall Risk Score:</b>", body_style), Paragraph(score_text, ParagraphStyle("Score", parent=body_style, textColor=score_color))],
        [Paragraph("<b>Document Type:</b>", body_style), Paragraph(document.document_type.replace('_', ' ').title(), body_style),
         Paragraph("<b>Pages:</b>", body_style), Paragraph(str(document.page_count or 1), body_style)],
        [Paragraph("<b>Model Version:</b>", body_style), Paragraph(document.model_version or "N/A", body_style),
         Paragraph("<b>Analyzed At:</b>", body_style), Paragraph(document.analyzed_at.strftime('%Y-%m-%d %H:%M') if document.analyzed_at else "N/A", body_style)],
    ]

    overview_table = Table(overview_data, colWidths=[100, 170, 110, 160])
    overview_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F9FAFB")),
            ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#F9FAFB")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    elements.append(overview_table)
    elements.append(Spacer(1, 8))

    # Executive Clause Breakdown Table (Matching UI counts exactly)
    breakdown_data = [
        [
            Paragraph("<b>Substantive Clauses</b>", body_bold),
            Paragraph("<b>Confirmed Hazards</b>", body_bold),
            Paragraph("<b>Needs Review</b>", body_bold),
            Paragraph("<b>Fair / Standard</b>", body_bold),
            Paragraph("<b>Excluded Metadata</b>", body_bold),
        ],
        [
            Paragraph(f"<b>{len(substantive_clauses)}</b>", body_style),
            Paragraph(f"<font color='#B91C1C'><b>{unfavorable_count}</b></font>", body_style),
            Paragraph(f"<font color='#B45309'><b>{review_count}</b></font>", body_style),
            Paragraph(f"<font color='#15803D'><b>{fair_count}</b></font>", body_style),
            Paragraph(f"<b>{len(excluded_clauses)}</b>", body_style),
        ],
    ]
    breakdown_table = Table(breakdown_data, colWidths=[108, 108, 108, 108, 108])
    breakdown_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F4F6")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ])
    )
    elements.append(breakdown_table)
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("2. Privacy & PII Redaction Summary", section_heading))
    pii_counts = {}
    for p in pii_findings:
        pii_counts[p.entity_type] = pii_counts.get(p.entity_type, 0) + 1

    if pii_counts:
        pii_rows = [[Paragraph("<b>Redacted Entity Type</b>", body_bold), Paragraph("<b>Count Detected</b>", body_bold)]]
        for ent, count in sorted(pii_counts.items()):
            pii_rows.append([Paragraph(ent.replace('_', ' ').title(), body_style), Paragraph(str(count), body_style)])
        pii_table = Table(pii_rows, colWidths=[270, 270])
        pii_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F4F6")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ])
        )
        elements.append(pii_table)
    else:
        elements.append(Paragraph("No sensitive personal identification items detected for redaction.", body_style))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("3. Expected but Missing Clauses", section_heading))
    if missing_clauses:
        m_rows = [[Paragraph("<b>Clause Type</b>", body_bold), Paragraph("<b>Severity</b>", body_bold), Paragraph("<b>Standard Protection Note</b>", body_bold)]]
        for m in missing_clauses:
            sev_color = colors.HexColor("#B91C1C") if m.severity == "high" else (colors.HexColor("#B45309") if m.severity == "medium" else colors.HexColor("#15803D"))
            m_rows.append([
                Paragraph(m.clause_type.replace('_', ' ').title(), body_style),
                Paragraph(f"<b>{m.severity.upper()}</b>", ParagraphStyle("Sev", parent=body_style, textColor=sev_color)),
                Paragraph(m.checklist_note or "Expected standard provision.", body_style),
            ])
        m_table = Table(m_rows, colWidths=[140, 70, 330])
        m_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F4F6")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ])
        )
        elements.append(m_table)
    else:
        elements.append(Paragraph("All standard expected clauses for this document type were identified.", body_style))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("4. Deadlines & Date-Bound Obligations", section_heading))
    if deadlines:
        dl_rows = [[
            Paragraph("<b>Obligation & Type</b>", body_bold),
            Paragraph("<b>Action, Trigger & Context</b>", body_bold),
            Paragraph("<b>Timeline / Due Date</b>", body_bold),
            Paragraph("<b>Status & Review</b>", body_bold),
        ]]
        for dl in deadlines:
            is_review = getattr(dl, "status", "") == "NEEDS_REVIEW" or dl.confidence == "needs_review"
            conf_text = "NEEDS REVIEW" if is_review else "CONFIDENT"
            conf_color = colors.HexColor("#B45309") if is_review else colors.HexColor("#15803D")

            # Column 1: Type, Obligation, Party, Page
            col1_lines = []
            title_text = getattr(dl, "obligation_title", None) or (dl.deadline_type.replace('_', ' ').title() if dl.deadline_type else "Obligation")
            col1_lines.append(f"<b>{title_text}</b>")
            if getattr(dl, "responsible_party", None):
                col1_lines.append(f"Party: <b>{dl.responsible_party}</b>")
            if getattr(dl, "page_number", None):
                col1_lines.append(f"Page: <b>{dl.page_number}</b>")
            
            # Column 2: Action, Trigger, Evidence
            col2_lines = []
            if getattr(dl, "required_action", None):
                col2_lines.append(f"Action: <b>{dl.required_action}</b>")
            if getattr(dl, "trigger_event", None) and getattr(dl, "trigger_event", "") != "explicit_calendar_date":
                col2_lines.append(f"Trigger: <i>{dl.trigger_event}</i>")
            if getattr(dl, "conditions", None):
                col2_lines.append(f"Condition: {dl.conditions}")
            raw_snippet = dl.raw_text[:140] + "..." if len(dl.raw_text) > 140 else dl.raw_text
            col2_lines.append(f"<font size='8' color='#4B5563'>\"{raw_snippet}\"</font>")

            # Column 3: Timeline / Due Date
            col3_lines = []
            if dl.parsed_date:
                col3_lines.append(f"<b>{dl.parsed_date.strftime('%Y-%m-%d')}</b>")
                if getattr(dl, "status", "") == "RESOLVED_CALCULATED":
                    col3_lines.append("<font size='7' color='#0284C7'>[Calculated Date]</font>")
                else:
                    col3_lines.append("<font size='7' color='#15803D'>[Explicit Calendar]</font>")
            elif getattr(dl, "duration_count", None) and getattr(dl, "duration_unit", None):
                unit_clean = dl.duration_unit.replace('_', ' ')
                col3_lines.append(f"<b>{dl.duration_count} {unit_clean}</b>")
                col3_lines.append("<font size='7' color='#D97706'>[Unresolved Trigger]</font>")
            elif dl.relative_days:
                col3_lines.append(f"<b>{dl.relative_days} days</b>")
                col3_lines.append("<font size='7' color='#D97706'>[Relative Duration]</font>")
            else:
                col3_lines.append("<i>Unresolved</i>")

            if getattr(dl, "recurrence", None):
                col3_lines.append(f"<font size='7' color='#4B5563'>Recurrence: {dl.recurrence}</font>")

            # Column 4: Status and Review Reason
            col4_lines = [f"<b>{conf_text}</b>"]
            if getattr(dl, "review_reason", None):
                col4_lines.append(f"<font size='7' color='#B45309'>{dl.review_reason[:65]}</font>")

            dl_rows.append([
                Paragraph("<br/>".join(col1_lines), body_style),
                Paragraph("<br/>".join(col2_lines), body_style),
                Paragraph("<br/>".join(col3_lines), body_style),
                Paragraph("<br/>".join(col4_lines), ParagraphStyle("Conf", parent=body_style, textColor=conf_color)),
            ])

        dl_table = Table(dl_rows, colWidths=[120, 230, 110, 80])
        dl_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F4F6")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ])
        )
        elements.append(dl_table)
    else:
        elements.append(Paragraph("No time-sensitive deadlines or relative notice obligations detected.", body_style))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("5. Substantive Clause-by-Clause Risk Breakdown", section_heading))
    if substantive_clauses:
        for c in substantive_clauses:
            fav_raw = (c.favorability_label or "fair").lower()
            
            # Authoritative coloring and labeling:
            # - Fair / Neutral: Green (#15803D)
            # - Needs Review / Uncertain: Amber (#B45309)
            # - Unfavorable (Confirmed Hazard): Red (#B91C1C) or Orange (#C2410C)
            # Neutral clauses are NEVER styled critical-red!
            if fav_raw == "unfavorable":
                if (c.risk_score or 0.0) >= 60.0:
                    fav_label = "UNFAVORABLE (HIGH RISK)"
                    fav_color = colors.HexColor("#B91C1C")
                else:
                    fav_label = "UNFAVORABLE (MODERATE RISK)"
                    fav_color = colors.HexColor("#C2410C")
            elif fav_raw == "needs_review" or getattr(c, "clause_type", "") == "UNCERTAIN" or getattr(c, "human_review", False):
                fav_label = "NEEDS REVIEW (UNCERTAIN)"
                fav_color = colors.HexColor("#B45309")
            elif fav_raw in ("fair", "neutral"):
                fav_label = "FAIR / STANDARD"
                fav_color = colors.HexColor("#15803D")
            else:
                fav_label = "STANDARD / UNCLASSIFIED"
                fav_color = colors.HexColor("#4B5563")

            c_type_display = (c.clause_type or "Unclassified").replace("_", " ").title()
            page_info = f" | Page: <b>{c.page_number}</b>" if getattr(c, "page_number", None) else ""
            c_header = (
                f"<b>Clause {c.clause_index + 1}: {c_type_display}</b>{page_info} "
                f"| Risk Score: <b>{c.risk_score}</b> | Favorability: <font color='{fav_color.hexval()}'><b>{fav_label}</b></font>"
            )
            if getattr(c, "human_review", False) and fav_raw != "unfavorable":
                c_header += " <font color='#B45309'>[FLAGGED FOR HUMAN REVIEW]</font>"

            clause_block = [
                Paragraph(c_header, body_bold),
                Spacer(1, 2),
            ]
            if getattr(c, "human_review", False) and getattr(c, "filter_reason", None):
                clause_block.append(
                    Paragraph(f"<font color='#B45309'><b>Review Note:</b> {c.filter_reason}</font>", body_style)
                )
                clause_block.append(Spacer(1, 2))
            clause_block.extend([
                Paragraph(c.redacted_text, body_style),
                Spacer(1, 6),
                HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E5E7EB"), spaceBefore=2, spaceAfter=6),
            ])
            elements.append(KeepTogether(clause_block))
    else:
        elements.append(Paragraph("No substantive legal clauses identified in this document.", body_style))
    elements.append(Spacer(1, 10))

    # Excluded Structural Blocks (Headers, Footers, Metadata)
    if excluded_clauses:
        elements.append(Paragraph("6. Excluded Structural Content (0.0 Risk)", section_heading))
        elements.append(
            Paragraph(
                "The following blocks were excluded from substantive legal risk scoring under ClauseGuard AI "
                "eligibility rules and assigned strictly 0.0 risk.",
                body_style
            )
        )
        elements.append(Spacer(1, 4))
        
        ex_rows = [[
            Paragraph("<b>Block #</b>", body_bold),
            Paragraph("<b>Content Classification</b>", body_bold),
            Paragraph("<b>Filter Rationale</b>", body_bold),
            Paragraph("<b>Assigned Risk</b>", body_bold),
        ]]
        for ec in excluded_clauses:
            ex_rows.append([
                Paragraph(f"Block {ec.clause_index + 1}", body_style),
                Paragraph((ec.content_type or "NON_SUBSTANTIVE").replace('_', ' ').title(), body_style),
                Paragraph(ec.filter_reason or "Non-substantive structural block", body_style),
                Paragraph("<b>0.0 (Excluded)</b>", ParagraphStyle("Zero", parent=body_style, textColor=colors.HexColor("#15803D"))),
            ])
        ex_table = Table(ex_rows, colWidths=[70, 160, 230, 80])
        ex_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F4F6")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ])
        )
        elements.append(ex_table)
        elements.append(Spacer(1, 10))

    # 7. Document Integrity & Safety Signals
    elements.append(Paragraph("7. Document Integrity & Safety Signals", section_heading))
    if integrity_findings:
        elements.append(
            Paragraph(
                "<font color='#B45309'><b>Advisory:</b></font> The system detected suspicious formatting, "
                "visually obscured text, or instruction anomalies. These signals do not alter legal risk scoring "
                "but are documented below for manual auditor review.",
                body_style
            )
        )
        elements.append(Spacer(1, 4))
        int_rows = [[
            Paragraph("<b>Category</b>", body_bold),
            Paragraph("<b>Severity</b>", body_bold),
            Paragraph("<b>Page</b>", body_bold),
            Paragraph("<b>Explanation & Evidence</b>", body_bold),
        ]]
        for inf in integrity_findings:
            sev = (inf.severity if hasattr(inf, "severity") else inf.get("severity", "medium")).upper()
            sev_color = "#B91C1C" if sev in ("CRITICAL", "HIGH") else "#B45309"
            cat = (inf.category if hasattr(inf, "category") else inf.get("category", "")).replace("_", " ").title()
            pg = str(inf.page_number if hasattr(inf, "page_number") else inf.get("page_number", "-") or "-")
            expl = inf.explanation if hasattr(inf, "explanation") else inf.get("explanation", "")
            snip = inf.evidence_snippet if hasattr(inf, "evidence_snippet") else inf.get("evidence_snippet", "")
            detail_p = f"{expl}<br/><font color='#6B7280'>Snippet: {snip}</font>" if snip else expl

            int_rows.append([
                Paragraph(cat, body_style),
                Paragraph(f"<font color='{sev_color}'><b>{sev}</b></font>", body_style),
                Paragraph(pg, body_style),
                Paragraph(detail_p, body_style),
            ])
        int_table = Table(int_rows, colWidths=[120, 60, 40, 320])
        int_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#FEF3C7")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ])
        )
        elements.append(int_table)
    else:
        elements.append(
            Paragraph(
                "<font color='#15803D'><b>Verified Clean:</b></font> No hidden text, abnormal styling anomalies, "
                "off-page coordinates, or prompt injection directives were detected in this document.",
                body_style
            )
        )
    elements.append(Spacer(1, 10))

    elements.append(disclaimer_table)

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
