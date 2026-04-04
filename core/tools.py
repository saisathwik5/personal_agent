from langchain_core.tools import tool
import os
import subprocess
import glob

# ==========================================
# Representative Tools
# ==========================================

@tool
def get_resume() -> str:
    """Reads Sai Sathwik's resume to answer questions about his background, education, or skills."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    resume_path = os.path.join(base_dir, "..", "data", "sai_sathwik_resume.md")
    try:
        with open(resume_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading resume: {e}"

@tool
def record_inquiry(name: str, inquiry: str, contact_info: str) -> str:
    """Records an inquiry or message from a user/recruiter seeking to contact Sai."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    inquiries_path = os.path.join(base_dir, "..", "data", "inquiries.log")
    try:
        with open(inquiries_path, "a", encoding="utf-8") as f:
            f.write(f"Name: {name}\nContact: {contact_info}\nInquiry: {inquiry}\n---\n")
        return "Inquiry successfully recorded."
    except Exception as e:
        return f"Failed to record inquiry: {e}"


# ==========================================
# Engineering & Operations Tools
# ==========================================

@tool
def execute_python_code(code: str) -> str:
    """
    Executes a python snippet locally.
    WARNING: Use with caution. Do not execute destructive commands.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(base_dir, "..", "data", "temp_exec.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)
    try:
        result = subprocess.run(
            ["python", script_path], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        output = result.stdout
        if result.stderr:
            output += f"\nErrors:\n{result.stderr}"
        return output if output else "Script executed successfully with no output."
    except Exception as e:
        return f"Execution Failed: {e}"

@tool
def list_local_files(directory_path: str = ".") -> str:
    """Lists files and directories in a given path."""
    try:
        files = glob.glob(os.path.join(directory_path, "*"))
        return "\n".join(files) if files else "Directory is empty."
    except Exception as e:
        return f"Failed to list directory: {e}"

@tool
def read_local_file(file_path: str) -> str:
    """Reads the contents of a local file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Failed to read file: {e}"

def get_all_tools():
    """Returns the list of tools for the LangGraph Worker LLM bind."""
    return [
        get_resume,
        record_inquiry, 
        execute_python_code,
        list_local_files,
        read_local_file
    ]
