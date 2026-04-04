import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.agent import SaiSathwikAgent

def test_representative():
    print("Testing Representative Role...")
    agent = SaiSathwikAgent()
    print("User: What experience does Sai have with Big Data and PySpark?")
    response = agent.invoke("What experience does Sai have with Big Data and PySpark?")
    print(f"Agent:\n{response}\n")

def test_engineering():
    print("Testing Engineering Role...")
    agent = SaiSathwikAgent()
    print("User: Run a python snippet that prints out the sum of 5 and 7, using your code execution tool.")
    response = agent.invoke("Run a python snippet that prints out the sum of 5 and 7, using your code execution tool.")
    print(f"Agent:\n{response}\n")

if __name__ == "__main__":
    test_representative()
    print("="*40)
    test_engineering()
