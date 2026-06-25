from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm

MODEL = LiteLlm(model="groq/llama-3.3-70b-versatile")

#jobserach_agent
jobsearch_agent = LlmAgent(
    name='jobsearch_agent',
    model=MODEL,
    instruction="""You are a job search specialist. When given a job title, provide:
        Your job:
        - Research the job title deeply 
        - Find 10 interesting and surprising facts about the job
        - Find real world examples and stories about the job
        - Find statistics and data about the job
        - Think about what job seekers would find most interesting
        Output everything as detailed research notes.
        Be thorough — the job seeker depends on your research.""",
        output_key="job_research",
        )

#job_analysis_agent
job_analysis_agent = LlmAgent(
    name="job_analysis_agent",
    model=MODEL,
    instruction="""You are a job analysis expert.
        Use this research to create a job analysis:
        {job_research}

        Create this exact structure:

        JOB OVERVIEW:
        - Summary of the job title
        - Key responsibilities and duties

        REQUIRED SKILLS AND QUALIFICATIONS:
        - List of essential skills and qualifications
        - Any certifications or degrees required

        SALARY AND BENEFITS:
        - Average salary range
        - Common benefits offered

        JOB OUTLOOK:
        - Employment growth projections
        - Industry trends and demand

        TIPS FOR SUCCESS:
        - Advice for job seekers in this field
        - Common challenges and how to overcome them

        Keep it informative and useful for job seekers.""",
        output_key="job_analysis",
        )

#Resume Tailoring Agent
resume_tailoring_agent = LlmAgent(
    name="resume_tailoring_agent",
    model=MODEL,
    instruction="""You are a resume tailoring expert. Given a job description and a candidate's resume, modify the resume to better match the job requirements.
        - Highlight relevant skills and experiences
        - Adjust the summary to align with the job's key responsibilities
        - Suggest improvements to the work experience section
        - Ensure the resume uses keywords from the job description

        Keep it professional and effective for job applications.""",
        output_key="tailored_resume",
    )
root_agent = SequentialAgent(
    name="JobSearch_Agent", 
    sub_agents=[jobsearch_agent, job_analysis_agent, resume_tailoring_agent],
)