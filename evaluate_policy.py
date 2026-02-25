import os
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# --- CONFIGURATION ---
# Replace with the model you pulled (e.g., "llama3" or "mistral")
MODEL_NAME = "qwen3-coder:30b" 
POLICY_PATH = "OPP-115/sanitized_policies/105_amazon.com.html"
# ---------------------

def run_evaluation():
    # 1. Initialize the LLM via LangChain
    # Assumes Ollama is running on the default port 11434
    llm = OllamaLLM(model=MODEL_NAME)

    # 2. Load the Policy Text
    if not os.path.exists(POLICY_PATH):
        print(f"Error: Policy file not found at {POLICY_PATH}")
        return

    with open(POLICY_PATH, 'r', encoding='utf-8') as f:
        policy_text = f.read()

    # 3. Define the Research Prompt (as per Chinni's script logic)
    template = """
    You are evaluating a privacy policy. 
    
    POLICY TEXT:
    {policy_content}
    
    TASK:
    Identify and quote exact sentences about children, teens, age restrictions, or parental guidance. 
    Put each sentence on a new line. Do not paraphrase. 
    If there are none, output ONLY the word 'NONE'.
    """

    prompt = PromptTemplate(input_variables=["policy_content"], template=template)

    # 4. Execute the Simulation
    print(f"--- Starting Evaluation of {POLICY_PATH} ---")
    chain = prompt | llm
    response = chain.invoke({"policy_content": policy_text})

    # 5. Output Results
    print("\n--- LLM EXTRACTED SENTENCES ---")
    print(response)
    print("\n--- END OF EVALUATION ---")

if __name__ == "__main__":
    run_evaluation()
