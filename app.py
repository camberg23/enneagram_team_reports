import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
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
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.enums import TA_LEFT
from reportlab.lib import colors
from markdown2 import markdown
from bs4 import BeautifulSoup

# --------------------------------------------------------------------
# Updated Context & Prompts for Enneagram with Spelled-Out Type Names
# --------------------------------------------------------------------
#
# We respect the guidance:
#   - Always spell out type numbers (Type One, Type Two, etc.).
#   - Use headings correctly, just as we did in the DISC updates.
#   - Generate a team report with the same four sections.
#   - Round percentages, no referencing other frameworks (MBTI, TypeFinder).
#

initial_context = """
You are an expert organizational psychologist specializing in team dynamics and personality assessments using the Enneagram framework.

In your writing:
- Always **spell out** the Enneagram type numbers (e.g., Type One, Type Two, Type Nine).
- Do **not** write "Type 1" or "Type 2"; instead, write "Type One," "Type Two," etc.
- Offer specific but concise explanations about how these types shape team dynamics.

**Team Size:** {TEAM_SIZE}

**Team Members and their Enneagram Types (numeric form):**

{TEAM_MEMBERS_LIST}

You will create a comprehensive team personality report based on the Enneagram types present in this team. There are nine Enneagram types total, typically referred to as Type One through Type Nine.

The report consists of four sections:

1. **Team Profile**
2. **Type Distribution**
3. **Team Insights**
4. **Actions and Next Steps**

**Formatting Requirements:**

- Use clear headings and subheadings in Markdown format (##, ###, etc.).
- Use bullet points and tables where appropriate.
- Round all percentages to the nearest whole number.
- Do not mention any other frameworks (MBTI, TypeFinder, DISC, etc.).
- The total team size and type distribution are given; use them directly.
- Maintain a professional, neutral tone.
"""

prompts = {
    "Team Profile": """
{INITIAL_CONTEXT}

**Your Role:**

Write the **Team Profile** section of the report (Section 1).

## Section 1: Team Profile

- Briefly introduce the concept of the Enneagram as a framework describing nine distinct personality types (Type One through Type Nine).
- Summarize **each Enneagram type present** on this team, spelling out the type number (e.g., Type Three, Type Five).
- Include core motivations, general behavioral tendencies, and how these shape foundational team dynamics.
- Required length: Approximately 500 words.

**Begin your section below:**
""",
    "Type Distribution": """
{INITIAL_CONTEXT}

**Report So Far:**

{REPORT_SO_FAR}

**Your Role:**

Write the **Type Distribution** section of the report (Section 2).

## Section 2: Type Distribution

- Present a breakdown (list or table) of how many team members fall into each spelled-out Enneagram type (Type One, Type Two, etc.), along with the percentage of the total team.
- Discuss what it means to have certain Enneagram types more dominant, and how less represented types add diversity.
- Include any immediate implications for communication, decision-making, and problem-solving.
- Required length: Approximately 500 words.

**Continue the report by adding your section below:**
""",
    "Team Insights": """
{INITIAL_CONTEXT}

**Report So Far:**

{REPORT_SO_FAR}

**Your Role:**

Write the **Team Insights** section of the report (Section 3).

## Section 3: Team Insights

Create these subheadings (use `###` or `####` as appropriate):

1. **Strengths**  
   - At least four strengths from the dominant Enneagram types.
   - Each strength in **bold** on one line, followed by a paragraph explanation.

2. **Potential Blind Spots**  
   - At least four potential challenges or areas of improvement.
   - Each blind spot in **bold** on one line, followed by a paragraph explanation.

3. **Communication**  
   - Summarize how this particular mix of Enneagram types typically communicates.

4. **Teamwork**  
   - Overview of how these types collaborate, delegate, and resolve tasks as a team.

5. **Conflict**  
   - Possible sources of conflict based on the Enneagram makeup, plus suggestions for resolution.

- Required length: ~700 words total.

**Continue the report by adding your section below:**
""",
    "Actions and Next Steps": """
{INITIAL_CONTEXT}

**Report So Far:**

{REPORT_SO_FAR}

**Your Role:**

Write the **Actions and Next Steps** section of the report (Section 4).

## Section 4: Actions and Next Steps

- Provide actionable recommendations for team leaders, leveraging the spelled-out Enneagram types (Type One, Type Two, etc.).
- Use subheadings (`###`) for each major recommendation area.
- Offer a brief justification linking each recommendation to the Enneagram types.
- Present recommendations as bullet points or numbered lists, with blank lines between.
- End output immediately after the last bullet (no concluding paragraph).
- Required length: ~400 words.

**Conclude the report by adding your section below:**
"""
}

# -------------------------------
# Enneagram Types (1-9)
# -------------------------------
enneagram_types = [str(i) for i in range(1, 10)]

# -------------------------------
# Callback Function
# -------------------------------
def randomize_types_callback():
    randomized_types = [random.choice(enneagram_types) for _ in range(int(st.session_state['team_size']))]
    for i in range(int(st.session_state['team_size'])):
        key = f'enn_{i}'
        st.session_state[key] = randomized_types[i]

# -------------------------------
# Streamlit App Layout
# -------------------------------
st.title('Enneagram Team Report Generator')

if 'team_size' not in st.session_state:
    st.session_state['team_size'] = 5

team_size = st.number_input(
    'Enter the number of team members (up to 30)',
    min_value=1, max_value=30, value=5, key='team_size'
)

st.button('Randomize Types', on_click=randomize_types_callback)

st.subheader('Enter Enneagram types for each team member')
for i in range(int(team_size)):
    if f'enn_{i}' not in st.session_state:
        st.session_state[f'enn_{i}'] = 'Select Enneagram Type'

team_enneagram_types = []
for i in range(int(team_size)):
    e_type = st.selectbox(
        f'Team Member {i+1}',
        options=['Select Enneagram Type'] + enneagram_types,
        key=f'enn_{i}'
    )
    if e_type != 'Select Enneagram Type':
        team_enneagram_types.append(e_type)
    else:
        team_enneagram_types.append(None)

if st.button('Generate Report'):
    if None in team_enneagram_types:
        st.error('Please select Enneagram types for all team members.')
    else:
        with st.spinner('Generating report, please wait...'):
            # Build a string listing team members and their typed Enneagram
            # We do not rename numeric references here but let the LLM
            # output spelled-out type names in the text itself.
            team_members_list = "\n".join([
                f"{i+1}. Team Member {i+1}: Enneagram {e_type}"
                for i, e_type in enumerate(team_enneagram_types)
            ])

            # Compute counts & percentages
            type_counts = Counter(team_enneagram_types)
            total_members = len(team_enneagram_types)
            type_percentages = {
                t: round((c / total_members) * 100)
                for t, c in type_counts.items()
            }

            # Generate bar plot for distribution
            sns.set_style('whitegrid')
            plt.rcParams.update({'font.family': 'serif'})

            plt.figure(figsize=(10, 6))
            sns.barplot(
                x=list(type_counts.keys()),
                y=list(type_counts.values()),
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

            # Initialize LLM
            chat_model = ChatOpenAI(
                openai_api_key=st.secrets['API_KEY'],
                model_name='gpt-4o-2024-08-06',
                temperature=0.2
            )

            # Prepare initial context
            initial_context_template = PromptTemplate.from_template(initial_context)
            formatted_initial_context = initial_context_template.format(
                TEAM_SIZE=str(team_size),
                TEAM_MEMBERS_LIST=team_members_list
            )

            section_order = [
                "Team Profile",
                "Type Distribution",
                "Team Insights",
                "Actions and Next Steps"
            ]

            report_sections = {}
            report_so_far = ""

            # Generate each section
            for section_name in section_order:
                prompt_template = PromptTemplate.from_template(prompts[section_name])
                prompt_variables = {
                    "INITIAL_CONTEXT": formatted_initial_context.strip(),
                    "REPORT_SO_FAR": report_so_far.strip()
                }
                chat_chain = LLMChain(prompt=prompt_template, llm=chat_model)
                section_text = chat_chain.run(**prompt_variables)
                report_sections[section_name] = section_text.strip()
                report_so_far += f"\n\n{section_text.strip()}"

            # Display the final report
            for section_name in section_order:
                st.markdown(report_sections[section_name])
                if section_name == "Type Distribution":
                    st.header("Enneagram Type Distribution Plot")
                    st.image(type_distribution_plot, use_column_width=True)

            # ------------------------------------------------
            # PDF Generation (with improved heading styling)
            # ------------------------------------------------
            def convert_markdown_to_pdf(report_sections_dict, distribution_plot):
                pdf_buffer = io.BytesIO()
                doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
                elements = []
                styles = getSampleStyleSheet()

                # Define separate heading styles
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

                def process_markdown(text):
                    # Convert Markdown to HTML
                    html = markdown(text, extras=['tables'])
                    soup = BeautifulSoup(html, 'html.parser')

                    for elem in soup.contents:
                        if isinstance(elem, str):
                            continue

                        # Handle tables
                        if elem.name == 'table':
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
                                table_row = [col.get_text(strip=True) for col in cols]
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

                        elif elem.name == 'h1':
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

                        else:
                            elements.append(Paragraph(elem.get_text(strip=True), styleN))
                            elements.append(Spacer(1, 12))

                # Build PDF content from each report section
                for sec in section_order:
                    process_markdown(report_sections_dict[sec])
                    if sec == "Type Distribution":
                        # Insert distribution plot below type distribution text
                        elements.append(Spacer(1, 12))
                        img_buffer = io.BytesIO(distribution_plot)
                        img = ReportLabImage(img_buffer, width=400, height=240)
                        elements.append(img)
                        elements.append(Spacer(1, 12))

                doc.build(elements)
                pdf_buffer.seek(0)
                return pdf_buffer

            # Generate PDF
            pdf_data = convert_markdown_to_pdf(report_sections, type_distribution_plot)
            st.download_button(
                label="Download Report as PDF",
                data=pdf_data,
                file_name="team_enneagram_report.pdf",
                mime="application/pdf"
            )
