import os
import uuid
import csv
from crewai.tools import BaseTool
from crewai import Agent, Task, Crew, Process, LLM, BaseTool
from crewai_tools import DirectoryReadTool, FileReadTool, SerperDevTool

# Initialize LLM
google_llm = LLM(
    model="gemini/gemini-2.0-flash-exp",
    temperature=0.7
)

SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Define custom tools
class JobScraperTool(BaseTool):
    name = "Job Scraper"
    description = "Scrapes job listings from LinkedIn and Indian job portals"

    def _run(self, topic: str, location: str) -> list:
        search_tool = SerperDevTool(api_key=SERPER_API_KEY)
        results = search_tool.run(f"{topic} jobs in {location} site:linkedin.com OR naukri.com OR indeed.co.in")
        return results or [{"title": "Sample Job Title", "company": "Tech Corp", "link": "https://example.com/job/123"}]

class ResumeModifierTool(BaseTool):
    name = "Resume Modifier"
    description = "Modifies resume based on job description"

    def _run(self, resume: str, job_desc: str) -> str:
        prompt = f"Modify this resume:\n{resume}\nTo better match this job description:\n{job_desc}"
        return google_llm.invoke(prompt)

# Create agents
job_researcher = Agent(
    role="Senior Job Researcher",
    goal="Find relevant job postings in specified location",
    backstory="Expert in scraping job portals and analyzing market trends",
    llm=google_llm,
    tools=[JobScraperTool()],
    verbose=True
)

resume_editor = Agent(
    role="Professional Resume Editor",
    goal="Tailor resumes to match job descriptions perfectly",
    backstory="Experienced HR professional with decade of resume optimization experience",
    llm=google_llm,
    tools=[ResumeModifierTool()],
    verbose=True
)

file_manager = Agent(
    role="File Organization Specialist",
    goal="Manage resume versions and tracking data",
    backstory="Meticulous organizer with perfect track record in data management",
    llm=google_llm,
    tools=[],  # Replace DirectoryReadTool if needed
    verbose=True
)

# Define tasks
research_task = Task(
    description="Search for {topic} jobs in {location}, India...",
    agent=job_researcher,
    expected_output="List of job listings with company names, links, and descriptions"
)

modify_resume_task = Task(
    description="Read the original resume and modify it for each job listing...",
    agent=resume_editor,
    expected_output="Modified resume text for each job application",
    context=[research_task]
)

file_management_task = Task(
    description="Create unique folder for each application with ID...",
    agent=file_manager,
    expected_output="Structured folder hierarchy and updated CSV file",
    context=[modify_resume_task]
)

# Create and run crew
crew = Crew(
    agents=[job_researcher, resume_editor, file_manager],
    tasks=[research_task, modify_resume_task, file_management_task],
    process=Process.sequential,
    verbose=2
)

def process_jobs(topic: str, location: str, resume_path: str):
    if not os.path.exists('applications'):
        os.makedirs('applications')
    with open('applications/jobs.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Job Title', 'Company', 'Job Link', 'Folder ID'])
    result = crew.kickoff(inputs={'topic': topic, 'location': location, 'resume_path': resume_path})

# Example usage
process_jobs(
    topic="Agentic AI",
    location="Gurugram",
    resume_path="res.txt"
)
