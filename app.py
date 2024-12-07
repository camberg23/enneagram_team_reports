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

# -------------------------------
# Initial Context and Prompts (Enneagram)
# -------------------------------

# We only have Enneagram types (1 through 9). We will generate a team report with similar structure to the original,
# but without references to TypeFinder or letter-based preferences.
# The sections will be simplified, focusing on:
# 1. Team Profile
# 2. Type Distribution
# 3. Team Insights
# 4. Actions and Next Steps
#
# The content should be adapted to Enneagram. We have no preference dichotomies here.

initial_context = """
You are an expert organizational psychologist specializing in team dynamics and personality assessments using the Enneagram framework.

**Team Size:** {TEAM_SIZE}

**Team Members and their Enneagram Types:**

{TEAM_MEMBERS_LIST}

You will create a comprehensive team personality report based on the Enneagram types present in this team. The Enneagram types are 1 through 9. Each has distinct motivations, fears, and core drivers that shape behavior.

The report consists of four sections:

1. **Team Profile**
2. **Type Distribution**
3. **Team Insights**
4. **Actions and Next Steps**

**Formatting Requirements:**

- Use clear headings and subheadings.
- Write in Markdown format.
- Use bullet points and tables where appropriate.
- Offer specific, actionable insights.
- Base all insights on the provided Enneagram types; do not invent data.
- Round all percentages to the nearest whole number.
- Do not mention "MBTI" or "TypeFinder". Stick strictly to the Enneagram framework.
- The total team size and type distribution are provided, use them as-is without recalculation beyond percentages.

Your tone should be professional, neutral, and focused on providing value to team leaders.
"""

prompts = {
    "Team Profile": """
{INITIAL_CONTEXT}

**Your Role:**

You are responsible for writing the **Team Profile** section of the report.

**Section 1: Team Profile**

- Introduce the concept of the Enneagram as a framework describing nine distinct personality types.
- Summarize each Enneagram type present in the team, including its core motivations and general behavioral tendencies.
- Highlight how the combination of these Enneagram types shapes the team's foundational personality and interpersonal dynamics.
- Required length: Approximately 500 words.

**Begin your section below:**
""",
    "Type Distribution": """
{INITIAL_CONTEXT}

**Report So Far:**

{REPORT_SO_FAR}

**Your Role:**

You are responsible for writing the **Type Distribution** section of the report.

**Section 2: Type Distribution**

- Present a breakdown of how many team members fall into each of the Enneagram types present.
- Convert these counts into percentages of the total team.
- Discuss what it means to have certain Enneagram types more dominant and how less represented types contribute diversity.
- Highlight implications for communication, decision-making, and problem-solving based on these distributions.
- Required length: Approximately 500 words.

**Continue the report by adding your section below:**
""",
    "Team Insights": """
{INITIAL_CONTEXT}

**Report So Far:**

{REPORT_SO_FAR}

**Your Role:**

You are responsible for writing the **Team Insights** section of the report.

**Section 3: Team Insights**

- Create two subheadings: **Strengths** and **Potential Blind Spots**.
- For **Strengths**, identify at least four strengths emerging from the dominant Enneagram themes in the group. Each strength should be bolded as a single sentence followed by a paragraph explanation.
- For **Potential Blind Spots**, identify at least four potential challenges or areas of improvement based on the Enneagram composition. Each blind spot should be bolded as a single sentence followed by a paragraph explanation.
- Required length: Approximately 700 words total.

**Continue the report by adding your section below:**
""",
    "Actions and Next Steps": """
{INITIAL_CONTEXT}

**Report So Far:**

{REPORT_SO_FAR}

**Your Role:**

You are responsible for writing the **Actions and Next Steps** section of the report.

**Section 4: Actions and Next Steps**

- Provide actionable recommendations for team leaders to enhance collaboration, leveraging the Enneagram composition.
- Use subheadings for each area of action.
- Offer a brief justification for each recommendation, linking it to the types present.
- Present the recommendations as bullet points or numbered lists of specific actions.
- End output immediately after the last bullet with no concluding paragraph.
- Required length: Approximately 400 words.

**Conclude the report by adding your section below:**
"""
}

# -------------------------------
# Enneagram Types
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
            team_types_str = ', '.join(team_enneagram_types)
            team_members_list = "\n".join([
                f"{i+1}. Team Member {i+1}: Enneagram {e_type}"
                for i, e_type in enumerate(team_enneagram_types)
            ])
            
            # Compute counts and percentages
            type_counts = Counter(team_enneagram_types)
            total_members = len(team_enneagram_types)
            type_percentages = {t: round((c/total_members)*100) for t, c in type_counts.items()}
            
            # Generate a type distribution plot
            sns.set_style('whitegrid')
            plt.rcParams.update({'font.family': 'serif'})

            plt.figure(figsize=(10, 6))
            sns.barplot(x=list(type_counts.keys()), y=list(type_counts.values()), palette='viridis')
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

            # Initialize the LLM
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

            report_sections = {}
            report_so_far = ""
            section_order = [
                "Team Profile",
                "Type Distribution",
                "Team Insights",
                "Actions and Next Steps"
            ]

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

            final_report = "\n\n".join([report_sections[s] for s in section_order])

            # Display the report
            for section_name in section_order:
                st.markdown(report_sections[section_name])
                if section_name == "Type Distribution":
                    st.header("Enneagram Type Distribution Plot")
                    st.image(type_distribution_plot, use_column_width=True)

            # PDF Generation
            def convert_markdown_to_pdf(report_sections_dict, distribution_plot):
                pdf_buffer = io.BytesIO()
                doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
                elements = []
                styles = getSampleStyleSheet()

                styleN = ParagraphStyle(
                    'Normal',
                    parent=styles['Normal'],
                    fontName='Times-Roman',
                    fontSize=12,
                    leading=14,
                )
                styleH = ParagraphStyle(
                    'Heading',
                    parent=styles['Heading1'],
                    fontName='Times-Bold',
                    fontSize=18,
                    leading=22,
                    spaceAfter=10,
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
                    html = markdown(text, extras=['tables'])
                    soup = BeautifulSoup(html, 'html.parser')
                    for elem in soup.contents:
                        if isinstance(elem, str):
                            continue
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
                        elif elem.name in ['h1', 'h2', 'h3']:
                            elements.append(Paragraph(elem.text, styleH))
                            elements.append(Spacer(1, 12))
                        elif elem.name == 'p':
                            elements.append(Paragraph(elem.decode_contents(), styleN))
                            elements.append(Spacer(1, 12))
                        elif elem.name == 'ul':
                            for li in elem.find_all('li', recursive=False):
                                elements.append(Paragraph('• ' + li.text, styleList))
                                elements.append(Spacer(1, 12))
                        else:
                            elements.append(Paragraph(elem.get_text(strip=True), styleN))
                            elements.append(Spacer(1, 12))

                # Add sections
                for sec in section_order:
                    process_markdown(report_sections_dict[sec])
                    if sec == "Type Distribution":
                        elements.append(Spacer(1, 12))
                        img_buffer = io.BytesIO(distribution_plot)
                        img = ReportLabImage(img_buffer, width=400, height=240)
                        elements.append(img)
                        elements.append(Spacer(1, 12))

                doc.build(elements)
                pdf_buffer.seek(0)
                return pdf_buffer

            pdf_data = convert_markdown_to_pdf(report_sections, type_distribution_plot)

            st.download_button(
                label="Download Report as PDF",
                data=pdf_data,
                file_name="team_enneagram_report.pdf",
                mime="application/pdf"
            )
