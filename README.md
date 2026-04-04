# 1. Clone the repository
git clone https://github.com/<YOUR_USERNAME>/personal-agent.git
cd personal-agent

# 2. Create and activate a Python virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install the tools you know the agent needs 
# (e.g., langgraph, langchain, openai)
pip install langgraph langchain-openai python-dotenv

# 4. Run the main application or test files
python app.py
