from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors





def create_pdf_report(metrics_df, macro_avg, micro_avg, output_path):
    """
    Create a PDF report for precision, recall, and F1-score metrics (including stdev & CI).
    """
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Add title
    title = Paragraph("Classification Metrics Report", styles["Title"])
    story.append(title)
    story.append(Spacer(1, 12))

    # Convert DataFrame (metrics_df) into a table suitable for PDF
    # metrics_df columns might include:
    # [class, precision, recall, f1-score, support,
    #  precision_std, recall_std, f1_std,
    #  precision_CI_lower, precision_CI_upper, ... etc.]

    # Build table data
    table_header = list(metrics_df.columns)
    table_data = [table_header] + metrics_df.values.tolist()

    class_metrics_table = Table(
        table_data,
        colWidths=[65] * len(table_header),
        rowHeights=20
    )
    class_metrics_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(class_metrics_table)
    story.append(Spacer(1, 12))

    # Add macro and micro averages in a simple table
    macro_micro_data = [
        ["Metric", "Precision", "Recall", "F1-Score", "Support"],
        ["Macro Avg", macro_avg["precision"], macro_avg["recall"], macro_avg["f1-score"], macro_avg["support"]],
        ["Micro Avg", micro_avg["precision"], micro_avg["recall"], micro_avg["f1-score"], micro_avg["support"]],
    ]
    macro_micro_table = Table(macro_micro_data, colWidths=[80, 80, 80, 80, 80])
    macro_micro_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(macro_micro_table)

    # Build the PDF
    doc.build(story)

