# 🤖 Advanced AI Personal Agent

An intelligent, stateful personal agent built using LangGraph and OpenAI. This agent is designed to act as a digital representation of my professional profile, capable of answering background questions, coordinating meetings, and executing software engineering tasks.

## 🌟 What Problem It Solves
Building reliable autonomous agents requires more than just API calls—it requires state management, reliable tool execution, and clear orchestration. Traditional CLI chatbots lack the ability to retain conversational history effectively and securely execute local commands. This project solves that by using LangGraph to build a cyclical, stateful graph where the agent uses specialized tools and can pause for human approval before taking high-risk actions.

## 🛠️ Topics & Technology Stack
*   **Framework**: LangGraph, LangChain
*   **LLM Model**: OpenAI GPT-4o
*   **Interface**: Gradio Chat Interface
*   **Environment**: Python 3.10+
*   **Key Concepts**: Agentic Orchestration, Tool Calling, Stateful Graphs

## 🏗️ Architecture Diagram

```
User → Gradio UI → App Engine (app.py)
                        │
          ┌─────────────▼──────────────────────────────┐
          │              Agentic System                 │
          │                                             │
          │      ┌───────────────────────────────┐     │
          │      │    LangGraph Orchestrator     │     │
          │      │         │           │         │     │
          │      │      GPT-4o     Tool Node     │     │
          │      │                /    |    \    │     │
          │      │        RAG Tool  Scheduler  Bash   │     │
          │      │                    Executor        │     │
          │      │      ┌─────────────────────┐  │     │
          │      │      │   Checkpointer DB   │  │     │
          │      │      └─────────────────────┘  │     │
          │      └───────────────────────────────┘     │
          └─────────────────────────────────────────────┘
                                    │
                              Local Filesystem
```

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/personal-agent.git
   cd personal-agent
   ```
2. **Create a virtual environment and activate it:**
   ```bash
   python -m venv .venv
   # Windows
   .\.venv\Scripts\Activate.ps1
   # macOS/Linux
   source .venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install gradio langgraph langchain-openai python-dotenv typing-extensions
   ```
4. **Set up environment variables:**
   Create a `.env` file in the root directory and add your OpenAI API key:
   ```text
   OPENAI_API_KEY=sk-your-secret-key
   ```
5. **Run the agent:**
   ```bash
   python app.py
   ```
   *Navigate to `http://localhost:7861` to interact with your agent!*
