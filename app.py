import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate 
from langchain.chains import LLMChain

# Define the initial context shared across all LLM calls
initial_context = """
You are an expert organizational psychologist specializing in team dynamics and personality assessments based on the Myers-Briggs Type Indicator (MBTI) framework.

**Team Size:** {TEAM_SIZE}

**Team Members and their MBTI Types:**

{TEAM_MEMBERS_LIST}

Your task is to contribute to a comprehensive team personality report. The report consists of five sections, and you will be responsible for generating one of them. Please ensure that your section aligns seamlessly with the previous sections and maintains a consistent tone and style throughout the report.

The sections are:

1. **Team Profile**

2. **Type Distribution**

3. **Team Insights**

4. **Type Preference Breakdown**

5. **Actions and Next Steps**

**Formatting Requirements:**

- Use clear headings and subheadings for your section.
- Write in Markdown format for easy readability.
- Use bullet points and tables where appropriate.
- Ensure the content is specific to the provided team MBTI types and offers unique insights.
- Avoid redundancy with previous sections!
"""

# Define prompts for each section, emphasizing the use of provided MBTI types
prompts = {
    "Team Profile": """
{INITIAL_CONTEXT}

**Your Role:**

You are responsible for writing the **Team Profile** section of the report.

**Section 1: Team Profile**

- Determine the overall "team type" by identifying the most common traits among the team members based on their MBTI types provided above.
- Provide an overview of this team type, including key characteristics and how it influences team dynamics.
- **Only use the MBTI types provided; do not introduce new types.**
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

- Present a percentage breakdown of each MBTI type within the team (only include MBTI types present among the team members).
- Ensure calculations are accurate based on the team size and MBTI types provided.
- Explain what the distribution suggests about the team's diversity and similarity in thinking and working styles.
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

- **Strengths:** Highlight the collective strengths of the team based on the prevalent personality traits present in the team. Required length: Approximately 350 words.
- **Blind Spots:** Identify potential blind spots or challenges the team may face due to less represented traits within the team. Required length: Approximately 350 words.

**Continue the report by adding your section below:**
""",
    "Type Preference Breakdown": """
{INITIAL_CONTEXT}

**Report So Far:**

{REPORT_SO_FAR}

**Your Role:**

You are responsible for writing the **Type Preference Breakdown** section of the report.

**Section 4: Type Preference Breakdown**

- For each MBTI dimension (Extraversion vs. Introversion, Sensing vs. Intuition, Thinking vs. Feeling, Judging vs. Perceiving), calculate the team's percentage distribution based on the MBTI types of the team members provided.
- Make a separate table for each dimension, followed by a 200-word discussion of the implications of this specific distribution for workplace dynamics.
- Explain what these percentages mean for team communication, decision-making, and problem-solving.

**Continue the report by adding your section below:**
""",
    "Actions and Next Steps": """
{INITIAL_CONTEXT}

**Report So Far:**

{REPORT_SO_FAR}

**Your Role:**

You are responsible for writing the **Actions and Next Steps** section of the report.

**Section 5: Actions and Next Steps**

- Provide actionable recommendations for team leaders to enhance collaboration and performance, based on the analysis in the previous sections and the specific MBTI types present in the team.
- Required length: Approximately 450 words.

**Conclude the report by adding your section below:**
"""
}

st.title('Team Personality Report Generator')

# Input for team size
team_size = st.number_input('Enter the number of team members (up to 30)', min_value=1, max_value=30, value=5)

# Initialize list to store MBTI types
team_mbti_types = []

# Input for MBTI types of each team member
st.header('Enter MBTI types for each team member')
mbti_options = ['INTJ', 'INTP', 'ENTJ', 'ENTP',
                'INFJ', 'INFP', 'ENFJ', 'ENFP',
                'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ',
                'ISTP', 'ISFP', 'ESTP', 'ESFP']

for i in range(int(team_size)):
    mbti_type = st.selectbox(f'Team Member {i+1}', options=['Select MBTI Type'] + mbti_options, key=f'mbti_{i}')
    if mbti_type != 'Select MBTI Type':
        team_mbti_types.append(mbti_type)
    else:
        team_mbti_types.append(None)  # Ensure the list has the same length as team_size

# Submit button
if st.button('Generate Report'):
    if None in team_mbti_types:
        st.error('Please select MBTI types for all team members.')
    else:
        with st.spinner('Generating report, please wait...'):
            # Prepare the team types as a string
            team_types_str = ', '.join(team_mbti_types)
            # Prepare the team members list
            team_members_list = "\n".join([f"{i+1}. Team Member {i+1}: {mbti_type}" for i, mbti_type in enumerate(team_mbti_types)])
            # Initialize the LLM
            chat_model = ChatOpenAI(openai_api_key=st.secrets['API_KEY'], model_name='gpt-4-1106-preview', temperature=0.2)
            
            # Prepare the initial context
            initial_context_template = PromptTemplate.from_template(initial_context)
            formatted_initial_context = initial_context_template.format(
                TEAM_SIZE=str(team_size),
                TEAM_MEMBERS_LIST=team_members_list,
                TEAM_TYPES=team_types_str
            )
            
            # Initialize variables to store the report
            report_sections = []
            report_so_far = ""
            # Iterate over each section
            for section_name in ["Team Profile", "Type Distribution", "Team Insights", "Type Preference Breakdown", "Actions and Next Steps"]:
                # Prepare the prompt
                prompt_template = PromptTemplate.from_template(prompts[section_name])
                # Prepare the variables for the prompt
                prompt_variables = {
                    "INITIAL_CONTEXT": formatted_initial_context.strip(),
                    "REPORT_SO_FAR": report_so_far.strip()
                }
                # Create the chain
                chat_chain = LLMChain(prompt=prompt_template, llm=chat_model)
                # Generate the section
                section_text = chat_chain.run(**prompt_variables)
                # Append the section to the report
                report_sections.append(section_text.strip())
                # Update the report so far
                report_so_far += f"\n\n{section_text.strip()}"
            
            # Combine all sections into the final report
            final_report = "\n\n".join(report_sections)
            
            # Display the report using markdown
            st.markdown(final_report)
            
            # Download button for the report
            st.download_button(
                label="Download Report",
                data=final_report,
                file_name="team_personality_report.md",
                mime="text/markdown"
            )
