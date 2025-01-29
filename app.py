import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd
import random
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import io
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as ReportLabImage,
    Table,
    TableStyle,
    HRFlowable,
    PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
from markdown2 import markdown
from bs4 import BeautifulSoup
import datetime

# ---------------------------------------------------------------------
# Enneagram Type Map ("One" => "Type One", etc.)
# ---------------------------------------------------------------------
eg_map = {
    "one": "Type One",
    "two": "Type Two",
    "three": "Type Three",
    "four": "Type Four",
    "five": "Type Five",
    "six": "Type Six",
    "seven": "Type Seven",
    "eight": "Type Eight",
    "nine": "Type Nine",
}

def parse_eg_type(type_str: str) -> str:
    """
    Convert a string like "Nine" or "Two" into "Type Nine"/"Type Two".
    If invalid, return "" so we skip it.
    """
    if not type_str:
        return ""
    lower = type_str.strip().lower()
    if lower in eg_map:
        return eg_map[lower]
    return ""

# ---------------------------------------------------------------------
# Updated Prompts
# ---------------------------------------------------------------------
initial_context = """
You are an expert organizational psychologist specializing in team dynamics using the Enneagram framework.

We have nine Enneagram types, referred to as Type One through Type Nine.
In your writing:
- Always spell out the type number (Type One, Type Two, etc.).
- Avoid references to other frameworks (MBTI, DISC, TypeFinder).
- Round all percentages to the nearest whole number.

Below are the team details:
Team Size: {TEAM_SIZE}

Team Members and their Enneagram Types:
{TEAM_MEMBERS_LIST}

Your goal is to create a team personality report that includes:

1. Introduction & Team Dynamics (combined)
   - Briefly introduce the Enneagram (Type One ... Type Nine).
   - Provide a breakdown (table or list) of each type (Type One ... Type Nine), with count & percentage.
   - Under a subheading "Types on the Team," list each present type with a short description (1–2 bullet points) plus count & %.
   - Under a subheading "Types Not on the Team," do the same for absent types (count=0).
   - Include subheadings "Dominant Types" (most common) and "Less Represented Types" (absent/scarce). Explain how each affects communication, decision-making, or team perspective.
   - End with a short "Summary" subheading (2–3 sentences).

2. Team Insights
   - Strengths (at least four, each in bold, followed by a paragraph)
   - Potential Blind Spots (similarly bolded)
   - Communication
   - Teamwork
   - Conflict

3. Next Steps
   - Provide actionable recommendations or next steps for team leaders in bullet form.

Use a clear heading hierarchy in Markdown:
- `##` for each main section (1,2,3).
- `###` for subheadings (e.g., "Types on the Team," "Dominant Types," etc.).
- Blank lines between paragraphs, bullet points, and headings.

Maintain a professional, neutral tone.
"""

prompts = {
    "Intro_Dynamics": """
{INITIAL_CONTEXT}

**Your Role:**

Write **Section 1: Introduction & Team Dynamics**.

## Section 1: Introduction & Team Dynamics

- Briefly introduce the Enneagram system (Type One … Type Nine).
- Provide a breakdown of each type (1–9) with count and percentage (even if count=0).
- `### Types on the Team`: List types present with short bullet points, count, and %.
- `### Types Not on the Team`: List absent types with the same format (count=0, 0%).
- `### Dominant Types`: Most common types, how they shape communication/decisions.
- `### Less Represented Types`: Discuss missing or rare types, the impact on the team.
- `### Summary`: 2–3 sentences wrapping up the distribution insights.

Required length: ~600 words total.

**Begin your section below:**
""",
    "Team Insights": """
{INITIAL_CONTEXT}

**Report So Far:**

{REPORT_SO_FAR}

**Your Role:**

Write **Section 2: Team Insights**.

## Section 2: Team Insights

Use these subheadings (`###` or `####` as needed):

1. **Strengths**  
   - At least four strengths; each strength in **bold** on one line, followed by a paragraph.

2. **Potential Blind Spots**  
   - At least four possible challenges; each in **bold** + paragraph.

3. **Communication**  
   - 1–2 paragraphs describing how these types communicate.

4. **Teamwork**  
   - 1–2 paragraphs on collaboration, delegation, synergy.

5. **Conflict**  
   - 1–2 paragraphs on friction points and resolution ideas.

Required length: ~700 words.

**Continue the report below:**
""",
    "NextSteps": """
{INITIAL_CONTEXT}

**Report So Far:**

{REPORT_SO_FAR}

**Your Role:**

Write **Section 3: Next Steps**.

## Section 3: Next Steps

- Provide actionable recommendations for team leaders.
- Use subheadings (`###`) for each recommendation area.
- Offer bullet points or numbered lists, with blank lines between items.
- No concluding paragraph: end after the last bullet.

Required length: ~400 words.

**Conclude the report below:**
"""
}

# ---------------------------------------------------------------------
# The Streamlit App
# ---------------------------------------------------------------------
st.title('Enneagram Team Report Generator (CSV-based)')

# -- Cover Page Details
st.subheader("Cover Page Details")
logo_path = "truity_logo.png"  # For your cover page
company_name = st.text_input("Company Name (for cover page)", "Example Corp")
team_name = st.text_input("Team Name (for cover page)", "Marketing Team")
today_str = datetime.date.today().strftime("%B %d, %Y")
custom_date = st.text_input("Date (for cover page)", today_str)

# CSV upload
st.subheader("Upload CSV with columns: User Name, EG Type")
uploaded_csv = st.file_uploader("Upload CSV", type=["csv"])

if st.button("Generate Report from CSV"):
    if not uploaded_csv:
        st.error("Please upload a valid CSV file first.")
    else:
        with st.spinner("Processing CSV..."):
            df = pd.read_csv(uploaded_csv)
            # We only care about 'User Name' and 'EG Type'
            # We'll skip rows missing or blank 'EG Type'
            valid_rows = []
            for i, row in df.iterrows():
                name_val = row.get("User Name", "")
                eg_val = row.get("EG Type", "")
                # Convert them to strings, strip
                name_str = str(name_val).strip()
                eg_str = str(eg_val).strip()
                # parse Enneagram type
                parsed = parse_eg_type(eg_str)
                if name_str and parsed:
                    valid_rows.append((name_str, parsed))
            
            if not valid_rows:
                st.error("No valid Enneagram types found in CSV. Nothing to report.")
            else:
                # Team size, members list
                team_size = len(valid_rows)
                team_members_list = ""
                for i, (nm, t) in enumerate(valid_rows):
                    team_members_list += f"{i+1}. {nm}: {t}\n"

                # Count distribution
                from collections import Counter
                type_counts = Counter([r[1] for r in valid_rows])
                total_members = len(valid_rows)
                type_percentages = {
                    k: round((v / total_members) * 100)
                    for k, v in type_counts.items()
                }

                # Create bar chart
                sns.set_style('whitegrid')
                plt.rcParams.update({'font.family': 'serif'})
                plt.figure(figsize=(10, 6))
                sorted_types = sorted(type_counts.keys())
                sorted_counts = [type_counts[t] for t in sorted_types]
                sns.barplot(
                    x=sorted_types,
                    y=sorted_counts,
                    palette='viridis'
                )
                plt.title('Enneagram Type Distribution', fontsize=16)
                plt.xlabel('Enneagram Types', fontsize=14)
                plt.ylabel('Number of Team Members', fontsize=14)
                plt.xticks(rotation=45)
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                type_distribution_plot = buf.getvalue()
                plt.close()

                # Prepare LLM
                from langchain_community.chat_models import ChatOpenAI
                chat_model = ChatOpenAI(
                    openai_api_key=st.secrets['API_KEY'],
                    model_name='gpt-4o-2024-08-06',
                    temperature=0.2
                )

                # Format initial context
                from langchain.prompts import PromptTemplate
                initial_context_template = PromptTemplate.from_template(initial_context)
                formatted_initial_context = initial_context_template.format(
                    TEAM_SIZE=str(team_size),
                    TEAM_MEMBERS_LIST=team_members_list
                )

                # Generate sections
                section_order = ["Intro_Dynamics", "Team Insights", "NextSteps"]
                report_sections = {}
                report_so_far = ""

                from langchain.chains import LLMChain
                for section_name in section_order:
                    prompt_template = PromptTemplate.from_template(prompts[section_name])
                    prompt_vars = {
                        "INITIAL_CONTEXT": formatted_initial_context.strip(),
                        "REPORT_SO_FAR": report_so_far.strip()
                    }
                    chain = LLMChain(prompt=prompt_template, llm=chat_model)
                    section_text = chain.run(**prompt_vars)
                    report_sections[section_name] = section_text.strip()
                    report_so_far += f"\n\n{section_text.strip()}"

                # Display in Streamlit
                for s in section_order:
                    st.markdown(report_sections[s])
                    if s == "Intro_Dynamics":
                        st.header("Enneagram Type Distribution Plot")
                        st.image(type_distribution_plot, use_column_width=True)

                # -----------------------------
                # Cover Page + PDF Generation
                # -----------------------------
                def build_cover_page(logo_path, company_name, team_name, date_str):
                    cover_elems = []
                    styles = getSampleStyleSheet()

                    cover_title_style = ParagraphStyle(
                        'CoverTitle',
                        parent=styles['Title'],
                        fontName='Times-Bold',
                        fontSize=24,
                        leading=28,
                        alignment=TA_CENTER,
                        spaceAfter=20
                    )
                    cover_text_style = ParagraphStyle(
                        'CoverText',
                        parent=styles['Normal'],
                        fontName='Times-Roman',
                        fontSize=14,
                        alignment=TA_CENTER,
                        spaceAfter=8
                    )

                    # Some vertical space
                    cover_elems.append(Spacer(1, 80))

                    try:
                        logo = ReportLabImage(logo_path, width=140, height=52)
                        cover_elems.append(logo)
                    except:
                        pass

                    cover_elems.append(Spacer(1, 50))

                    # Title
                    title_para = Paragraph("Enneagram For The Workplace<br/>Team Report", cover_title_style)
                    cover_elems.append(title_para)

                    cover_elems.append(Spacer(1, 50))

                    # Horizontal line
                    sep = HRFlowable(width="70%", color=colors.darkgoldenrod)
                    cover_elems.append(sep)
                    cover_elems.append(Spacer(1, 20))

                    # Company, Team, Date
                    comp_para = Paragraph(company_name, cover_text_style)
                    cover_elems.append(comp_para)
                    tm_para = Paragraph(team_name, cover_text_style)
                    cover_elems.append(tm_para)
                    dt_para = Paragraph(date_str, cover_text_style)
                    cover_elems.append(dt_para)

                    cover_elems.append(Spacer(1, 60))
                    cover_elems.append(PageBreak())
                    return cover_elems

                def convert_markdown_to_pdf_with_cover(report_dict, distribution_plot,
                                                       logo_path, company_name, team_name, date_str):
                    pdf_buffer = io.BytesIO()
                    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
                    elements = []

                    # Cover page
                    cover_elems = build_cover_page(logo_path, company_name, team_name, date_str)
                    elements.extend(cover_elems)

                    # Normal styling
                    styles = getSampleStyleSheet()
                    styleH1 = ParagraphStyle(
                        'Heading1Custom',
                        parent=styles['Heading1'],
                        fontName='Times-Bold',
                        fontSize=18,
                        leading=22,
                        spaceAfter=10,
                    )
                    styleH2 = ParagraphStyle(
                        'Heading2Custom',
                        parent=styles['Heading2'],
                        fontName='Times-Bold',
                        fontSize=16,
                        leading=20,
                        spaceAfter=8,
                    )
                    styleH3 = ParagraphStyle(
                        'Heading3Custom',
                        parent=styles['Heading3'],
                        fontName='Times-Bold',
                        fontSize=14,
                        leading=18,
                        spaceAfter=6,
                    )
                    styleH4 = ParagraphStyle(
                        'Heading4Custom',
                        parent=styles['Heading4'],
                        fontName='Times-Bold',
                        fontSize=12,
                        leading=16,
                        spaceAfter=4,
                    )
                    styleN = ParagraphStyle(
                        'Normal',
                        parent=styles['Normal'],
                        fontName='Times-Roman',
                        fontSize=12,
                        leading=14,
                    )
                    styleList = ParagraphStyle(
                        'List',
                        parent=styles['Normal'],
                        fontName='Times-Roman',
                        fontSize=12,
                        leading=14,
                        leftIndent=20,
                    )

                    def process_md(md_text):
                        html = markdown(md_text, extras=['tables'])
                        soup = BeautifulSoup(html, 'html.parser')
                        for elem in soup.contents:
                            if isinstance(elem, str):
                                continue

                            if elem.name == 'h1':
                                elements.append(Paragraph(elem.text, styleH1))
                                elements.append(Spacer(1, 12))
                            elif elem.name == 'h2':
                                elements.append(Paragraph(elem.text, styleH2))
                                elements.append(Spacer(1, 12))
                            elif elem.name == 'h3':
                                elements.append(Paragraph(elem.text, styleH3))
                                elements.append(Spacer(1, 12))
                            elif elem.name == 'h4':
                                elements.append(Paragraph(elem.text, styleH4))
                                elements.append(Spacer(1, 12))
                            elif elem.name == 'p':
                                elements.append(Paragraph(elem.decode_contents(), styleN))
                                elements.append(Spacer(1, 12))
                            elif elem.name == 'ul':
                                for li in elem.find_all('li', recursive=False):
                                    elements.append(Paragraph('• ' + li.text, styleList))
                                    elements.append(Spacer(1, 6))
                            elif elem.name == 'table':
                                table_data = []
                                thead = elem.find('thead')
                                if thead:
                                    header_row = []
                                    for th in thead.find_all('th'):
                                        header_row.append(th.get_text(strip=True))
                                    if header_row:
                                        table_data.append(header_row)
                                tbody = elem.find('tbody')
                                if tbody:
                                    rows = tbody.find_all('tr')
                                else:
                                    rows = elem.find_all('tr')
                                for row in rows:
                                    cols = row.find_all(['td', 'th'])
                                    table_row = [c.get_text(strip=True) for c in cols]
                                    table_data.append(table_row)
                                if table_data:
                                    t = Table(table_data, hAlign='LEFT')
                                    t.setStyle(TableStyle([
                                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                        ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
                                        ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
                                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                                    ]))
                                    elements.append(t)
                                    elements.append(Spacer(1, 12))
                            else:
                                elements.append(Paragraph(elem.get_text(strip=True), styleN))
                                elements.append(Spacer(1, 12))

                    # Process each section
                    for s in section_order:
                        process_md(report_dict[s])
                        if s == "Intro_Dynamics":
                            elements.append(Spacer(1, 12))
                            img_buf = io.BytesIO(distribution_plot)
                            img = ReportLabImage(img_buf, width=400, height=240)
                            elements.append(img)
                            elements.append(Spacer(1, 12))

                    doc.build(elements)
                    pdf_buffer.seek(0)
                    return pdf_buffer

                pdf_data = convert_markdown_to_pdf_with_cover(
                    report_sections,
                    type_distribution_plot,
                    logo_path,
                    company_name,
                    team_name,
                    custom_date
                )

                st.download_button(
                    "Download Report as PDF",
                    data=pdf_data,
                    file_name="team_enneagram_report.pdf",
                    mime="application/pdf"
                )
