import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from langchain_openai import ChatOpenAI

# Load environment variables for API keys
load_dotenv()
# --- 1. Define Tools ---
# Initialize the tools the sharks will use for their research
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()
llm = ChatOpenAI(
    model="gemini/gemini-2.5-flash",  # Note the provider prefix
    temperature=0.5,
)

# --- 2. Define The Agents (The Cast) ---

# The Pitcher
entrepreneur = Agent(
    role="Entrepreneur",
    goal="Develop a groundbreaking business idea and create a perfect pitch to secure investment.",
    backstory="You are a visionary entrepreneur with a knack for identifying market gaps. You are resilient and excel at refining your ideas based on expert feedback.",
    allow_delegation=False,
    llm=llm,
    verbose=True

)

# The Sharks
tech_shark = Agent(
    role="Technical Shark",
    goal="Analyze the technical feasibility, scalability, and innovation of a business idea.",
    backstory="You are a seasoned CTO and tech investor, like a mix of Elon Musk and a Silicon Valley venture capitalist. You dissect technology stacks and product roadmaps for breakfast.",
    tools=[search_tool, scrape_tool],
    verbose=True,
    llm=llm
)

marketing_shark = Agent(
    role="Marketing Shark",
    goal="Evaluate the market viability, branding, and customer acquisition strategy of a business idea.",
    backstory="You are a marketing guru and branding genius, reminiscent of the minds behind Nike's and Apple's greatest campaigns. You can spot a winning brand from a mile away.",
    tools=[search_tool, scrape_tool],
    verbose=True,
    llm=llm
)

finance_shark = Agent(
    role="Financial Shark",
    goal="Scrutinize the business model, financial projections, and overall profitability of an idea.",
    backstory="You are a shrewd investor and numbers wizard, like a blend of Warren Buffett and the sharpest minds on Wall Street. You live and breathe spreadsheets, and no financial detail escapes your notice.",
    tools=[search_tool, scrape_tool],
    verbose=True,
    llm=llm
)


# --- 3. The Main Loop (The Pitch) ---

# The initial business idea that starts the conversation
initial_idea = "A subscription-based service for personalized, 3D-printed vitamins tailored to an individual's DNA and lifestyle data."
pitch_context = f"Initial Idea: {initial_idea}"
turn_count = 1

print(f"🚀 Starting Shark Tank with the idea: {initial_idea}\n")

while True:
    print(f"\n----- Turn {turn_count}: Refining the Pitch -----")

    # --- Dynamically Create Tasks for the Current Turn ---
    
    pitch_creation_task = Task(
        description=f"Based on the following context, create or refine a business pitch. The pitch should include: The Problem, The Solution, Business Model, and Target Market.\n\nContext:\n{pitch_context}",
        expected_output="A comprehensive and persuasive business pitch document.",
        agent=entrepreneur,
        async_execution=False,
    )

    # Shark review tasks can run in parallel
    tech_review_task = Task(
        description="From a technical standpoint, analyze the provided business pitch. Assess its feasibility, scalability, and any potential technological hurdles or innovations. Use your tools to research competing technologies.",
        expected_output="A bullet-point list of technical strengths, weaknesses, and suggestions for improvement.",
        agent=tech_shark,
        async_execution=False,
        
    )

    market_review_task = Task(
        description="From a marketing perspective, analyze the business pitch. Evaluate the target market, branding, and customer acquisition strategy. Use your tools to research market size and competitors.",
        expected_output="A bullet-point list of marketing strengths, weaknesses, and strategic recommendations.",
        agent=marketing_shark,
        async_execution=False
        
    )

    finance_review_task = Task(
        description="From a financial perspective, analyze the business pitch. Scrutinize the business model, potential revenue streams, and overall profitability. What are the financial risks?",
        expected_output="A bullet-point list of financial strengths, weaknesses, and questions for the entrepreneur.",
        agent=finance_shark,
        async_execution=False
    )

    # This task consolidates feedback and makes the Entrepreneur refine the pitch
    pitch_refinement_task = Task(
        description="You have received feedback from the sharks. Your task is to integrate their suggestions to create a new, improved version of your business pitch. Address each of their points clearly.",
        expected_output="A revised and improved business pitch document that directly addresses the sharks' feedback.",
        agent=entrepreneur,
        context=[tech_review_task, market_review_task, finance_review_task],
        async_execution=False,
        output_file="final_pitch.md"
    )

    # The final decision task for the turn
    decision_task = Task(
        description="""Analyze your refined pitch and the feedback that led to it. Is the pitch now robust, compelling, and addresses all major concerns? Is it ready for investment?
        Respond with a single word: INVESTMENT_READY if the pitch is complete, or NEEDS_REVISION if it still requires work.""",
        expected_output="A single word: INVESTMENT_READY or NEEDS_REVISION.",
        agent=entrepreneur,
        context=[pitch_refinement_task],
        async_execution=False
    )

    # --- Assemble and Run the Crew for This Turn ---
    
    shark_tank_crew = Crew(
        agents=[entrepreneur, tech_shark, marketing_shark, finance_shark],
        tasks=[
            pitch_creation_task,
            tech_review_task,
            market_review_task,
            finance_review_task,
            pitch_refinement_task,
            decision_task
        ],
        process=Process.sequential,
        verbose=True,
        chat_llm=llm
    )
    
    # The result of the kickoff is the output of the LAST task (the decision task)
    decision = shark_tank_crew.kickoff()
    print(f"\nEntrepreneur's Decision: {decision}")
    
    # --- Check the Decision and Update Context ---
    if "INVESTMENT_READY" in decision.raw.upper():
        print("\n🎉 ----- The Sharks are IN! The pitch is finalized! ----- 🎉")
        final_pitch = pitch_refinement_task.output
        break
    
    # Update context for the next turn with the refined pitch
    pitch_context = f"This is Turn {turn_count}. Here is the latest pitch to be improved:\n" + pitch_refinement_task.output.raw_output
    turn_count += 1

print("\n\n---(Final Investment Pitch) ---")
print(final_pitch)