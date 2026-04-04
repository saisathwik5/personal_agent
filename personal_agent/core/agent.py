from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
import uuid
import sys
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

# Ensure we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import config
from tools import get_all_tools

# ==========================================
# 1. State Definitions
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    success_criteria: str
    feedback: Optional[str]
    success_met: bool
    needs_user_input: bool

class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Critical feedback on the assistant's previous response.")
    success_met: bool = Field(description="True if the user's implicit/explicit request is fully answered.")
    needs_user_input: bool = Field(
        description="True if the request is ambiguous, impossible, or requires explicit confirmation."
    )

# ==========================================
# 2. Graph Construction
# ==========================================
class SaiSathwikAgent:
    
    def __init__(self):
        self.worker_llm = ChatOpenAI(model=config.LLM_MODEL)
        self.evaluator_llm = ChatOpenAI(model=config.LLM_MODEL).with_structured_output(EvaluatorOutput)
        self.tools = get_all_tools()
        self.worker_with_tools = self.worker_llm.bind_tools(self.tools)
        
        # Build the DAG
        self._build_graph()

    def _worker_node(self, state: AgentState) -> Dict[str, Any]:
        """The primary actor. Invokes tools or drafts a response."""
        system_prompt = f"""You are the advanced personal and engineering assistant for Sai Sathwik.
You represent Sai Sathwik to recruiters, and assist Sai himself with coding/engineering tasks.
You have access to tools to read his resume, log inquiries, and execute local engineering workflows.

If the user is asking a question about Sai, use the `get_resume` tool.
If the user wants to contact Sai, use the `record_inquiry` tool.
If the user (likely Sai himself) asks you to perform a file operation or run code, use the appropriate engineering tools.

SUCCESS CRITERIA FOR THIS TASK:
{state.get('success_criteria', 'Help the user concisely and accurately.')}
"""
        if state.get("feedback"):
            system_prompt += f"\nEVALUATOR FEEDBACK (Fix these issues):\n{state['feedback']}"

        # Ensure system message is injected properly
        msgs = state.get("messages", [])
        if not msgs or not isinstance(msgs[0], SystemMessage):
            msgs = [SystemMessage(content=system_prompt)] + msgs

        response = self.worker_with_tools.invoke(msgs)
        return {"messages": [response]}

    def _evaluator_node(self, state: AgentState) -> Dict[str, Any]:
        """Critiques the worker's output before releasing it to the user."""
        messages = state.get("messages", [])
        last_response = messages[-1].content if messages else "No response generated."

        eval_sys_prompt = """You are a strict QA evaluator. Your job is to review the draft response of an AI Assistant acting on behalf of Sai Sathwik.
Decide if the response is complete, accurate, and professional. 
If the Assistant used a tool successfully and answered the query, mark success_met=True.
If the response implies hallucination or failed to answer the prompt, provide critical feedback and mark success_met=False.
If the Assistant is asking a clarifying question to the user, mark needs_user_input=True and success_met=True (as asking the question is the success criteria)."""

        eval_user_prompt = f"""
Original Request / Conversation: 
{self._format_history(messages[:-1])}

Draft Response to Evaluate:
{last_response}

Success Criteria to Verify against:
{state.get('success_criteria', 'Satisfy the user query.')}
"""
        eval_msgs = [SystemMessage(content=eval_sys_prompt), HumanMessage(content=eval_user_prompt)]
        eval_result = self.evaluator_llm.invoke(eval_msgs)

        # Update state based on evaluation
        return {
            "feedback": eval_result.feedback,
            "success_met": eval_result.success_met,
            "needs_user_input": eval_result.needs_user_input,
            "messages": [AIMessage(content=f"[Internal Evaluator]: {eval_result.feedback}")] if not eval_result.success_met else []
        }

    def _route_worker(self, state: AgentState) -> str:
        """Route from Worker either to Tools or to Evaluator"""
        last_msg = state.get("messages", [])[-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return "evaluator"

    def _route_evaluator(self, state: AgentState) -> str:
        """Route from Evaluator back to Worker (retry) or END"""
        if state.get("success_met") or state.get("needs_user_input"):
            return END
        return "worker"

    def _build_graph(self):
        builder = StateGraph(AgentState)
        
        builder.add_node("worker", self._worker_node)
        builder.add_node("tools", ToolNode(self.tools))
        builder.add_node("evaluator", self._evaluator_node)

        builder.add_edge(START, "worker")
        builder.add_conditional_edges("worker", self._route_worker, {"tools": "tools", "evaluator": "evaluator"})
        builder.add_edge("tools", "worker")
        builder.add_conditional_edges("evaluator", self._route_evaluator, {"worker": "worker", END: END})

        self.graph = builder.compile()

    def _format_history(self, messages: List[BaseMessage]) -> str:
        out = []
        for m in messages:
            if isinstance(m, HumanMessage):
                out.append(f"User: {m.content}")
            elif isinstance(m, AIMessage):
                out.append(f"Assistant: {m.content or '[Tool Use]'}")
        return "\n".join(out)

    def invoke(self, user_input: str, thread_id: str = str(uuid.uuid4())):
        """Main entry point for chatting with the agent."""
        state = {
            "messages": [HumanMessage(content=user_input)],
            "success_criteria": "Resolve the user's prompt acting as Sai Sathwik's highly capable assistant.",
            "success_met": False,
            "needs_user_input": False,
            "feedback": None
        }
        
        # We invoke the graph. For a production app, we would use a checkpointer 
        # (like langgraph.checkpoint.memory.MemorySaver) to persist state across `invoke` calls.
        # But for this V1, we simply run the DAG end-to-end for the current interaction.
        final_state = self.graph.invoke(state, {"configurable": {"thread_id": thread_id}})
        
        # Extract the last REAL message (prior to Evaluator feedback if it failed)
        # We backtrack through messages to find the last substantive AIMessage generated before END
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage) and not msg.content.startswith("[Internal Evaluator]"):
                return msg.content
        return "An error occurred retrieving the final response."
