from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def build_evaluation_report_pdf(report_payload):
    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4, topMargin=32, bottomMargin=32)
    styles = getSampleStyleSheet()

    story = [
        Paragraph("Ethical AI Response Evaluation Report", styles["Title"]),
        Spacer(1, 12),
    ]

    summary_rows = [
        ["Prompt", report_payload.get("prompt", "")],
        ["AI Response", report_payload.get("response", "")],
        ["Ethical Score", str(report_payload.get("ethical_score", 0))],
        ["Risk Classification", report_payload.get("behavior", "")],
        ["Confidence Score", str(report_payload.get("confidence_score", 0))],
        ["Timestamp", report_payload.get("timestamp", "")],
    ]
    summary_table = Table(summary_rows, colWidths=[140, 360])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef6ff")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c7d9ef")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEADING", (0, 0), (-1, -1), 12),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Detected Issues", styles["Heading2"]))
    issues = report_payload.get("issue_summary", []) or ["No detected issues."]
    for issue in issues:
        story.append(Paragraph(f"- {issue}", styles["BodyText"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Explanations", styles["Heading2"]))
    explanations = report_payload.get("structured_explanations", [])
    if explanations:
        for explanation in explanations:
            text = (
                f"<b>Issue:</b> {explanation.get('issue', '')}<br/>"
                f"<b>Matched Phrase:</b> {explanation.get('matched_phrase', '')}<br/>"
                f"<b>Reason:</b> {explanation.get('reason', '')}"
            )
            story.append(Paragraph(text, styles["BodyText"]))
            story.append(Spacer(1, 8))
    else:
        story.append(Paragraph("No structured explanations available.", styles["BodyText"]))

    document.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
