import re
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

# Ensure basic NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

class LLMJudgeOutput(BaseModel):
    is_correct: bool = Field(description="True if the prediction semantically fully matches / contains the facts of the expected output.")
    reasoning: str = Field(description="Brief reasoning for the decision.")

def normalize_text(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return " ".join(text.split())

def exact_match(prediction: str, ground_truth: str) -> bool:
    """Return True if normalized prediction contains the normalized ground truth."""
    pred_norm = normalize_text(prediction)
    gt_norm = normalize_text(ground_truth)
    return gt_norm in pred_norm

def calculate_rouge(prediction: str, ground_truth: str) -> dict:
    """Calculates Rouge-1, Rouge-2, and Rouge-L scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(ground_truth, prediction)
    return {
        "rouge1": scores['rouge1'].fmeasure,
        "rouge2": scores['rouge2'].fmeasure,
        "rougeL": scores['rougeL'].fmeasure
    }

def calculate_bleu(prediction: str, ground_truth: str) -> float:
    """Calculates BLEU score between prediction and ground truth."""
    reference = [nltk.word_tokenize(ground_truth.lower())]
    candidate = nltk.word_tokenize(prediction.lower())
    chencherry = SmoothingFunction()
    return sentence_bleu(reference, candidate, smoothing_function=chencherry.method1)

def llm_as_judge(prediction: str, ground_truth: str, prompt: str, llm_model: str = "gpt-4o-mini") -> dict:
    """Uses LLM to judge semantic correctness."""
    llm = ChatOpenAI(model=llm_model, temperature=0).with_structured_output(LLMJudgeOutput)
    
    sys_prompt = "You are an expert evaluator. Compare the Prediction to the Expected Output for the given Prompt. Decide if the Prediction is factually consistent and correct."
    user_prompt = f"Prompt: {prompt}\nExpected Output: {ground_truth}\nPrediction: {prediction}"
    
    messages = [SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)]
    try:
        response = llm.invoke(messages)
        return {"is_correct": response.is_correct, "reasoning": response.reasoning}
    except Exception as e:
        return {"is_correct": False, "reasoning": f"Error calling LLM: {str(e)}"}
