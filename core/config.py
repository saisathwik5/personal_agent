import os
from dotenv import load_dotenv

# Load the main .env from the parent `agents` folder
root_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env")
load_dotenv(dotenv_path=root_env, override=True)

class AgentConfig:
    LLM_MODEL = os.getenv("PERSONAL_AGENT_MODEL", "gpt-4o-mini")
    MAX_ITERATIONS = 10  # Fallback to prevent infinite loops

config = AgentConfig()
