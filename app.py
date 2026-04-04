import gradio as gr
import sys
import os

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.agent import SaiSathwikAgent

class PersonalAgentApp:
    def __init__(self):
        print("Initializing the Sai Sathwik Advanced Personal Agent...")
        try:
            self.agent = SaiSathwikAgent()
        except Exception as e:
            print(f"Failed to initialize Agent: {e}")
            self.agent = None

    def chat_interface(self, message, history):
        if not self.agent:
            return "Agent initialization failed. Please check your `.env` for OPENAI_API_KEY."
        
        # The agent graph handles the logic. We pass the new message to it.
        # Note: In a fully stateful app, we would pass the Thread ID and let the checkpointer handle history. 
        # For this Gradio demo, we pass the current input as the trigger.
        try:
            response = self.agent.invoke(message)
            return response
        except Exception as e:
            return f"Agent encountered an error during execution: {e}"

if __name__ == "__main__":
    app_instance = PersonalAgentApp()
    
    # Launch Gradio interface (similar to Week 1 app.py)
    interface = gr.ChatInterface(
        fn=app_instance.chat_interface,
        title="Sai Sathwik | Advanced AI Assistant",
        description="Ask me about Sai's background, request a meeting, or give me engineering tasks to execute.",
        theme="monochrome",
        type="messages"
    )
    
    interface.launch(server_name="0.0.0.0", server_port=7861, share=False)
