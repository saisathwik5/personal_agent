import os
import sys
import json
import uuid
import datetime
import glob

# Ensure that the root personal_agent directory is in the path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from core.agent import SaiSathwikAgent
from evals.scorers import exact_match, calculate_bleu, calculate_rouge, llm_as_judge

def run_evaluations():
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(base_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    
    test_set_path = os.path.join(base_dir, "test_set.json")
    with open(test_set_path, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    # Find the most recent run for comparison
    previous_run = None
    run_files = sorted(glob.glob(os.path.join(runs_dir, "run_*.json")))
    if run_files:
        with open(run_files[-1], "r", encoding="utf-8") as f:
            previous_run = json.load(f)

    # Initialize agent
    print("Initializing Agent for evaluation...")
    agent = SaiSathwikAgent()

    results = []
    totals = {
        "exact_match_count": 0,
        "bleu_sum": 0.0,
        "rougeL_sum": 0.0,
        "llm_judge_correct_count": 0,
        "total_cases": len(test_set)
    }

    print(f"Starting evaluation over {len(test_set)} cases...")
    
    for case in test_set:
        prompt = case["prompt"]
        ground_truth = case["expected_output"]
        
        print(f"\n[Case] {prompt}")
        
        # Invoke agent
        # We define a unique thread so there's no memory bleed between cases
        thread_id = str(uuid.uuid4())
        try:
            prediction = agent.invoke(prompt, thread_id=thread_id)
        except Exception as e:
            prediction = f"ERROR: {str(e)}"
        
        # Scoring
        em_score = exact_match(prediction, ground_truth)
        bleu_score = calculate_bleu(prediction, ground_truth)
        rouge_scores = calculate_rouge(prediction, ground_truth)
        llm_score = llm_as_judge(prediction, ground_truth, prompt)

        results.append({
            "prompt": prompt,
            "expected_output": ground_truth,
            "prediction": prediction,
            "metrics": {
                "exact_match": em_score,
                "bleu": bleu_score,
                "rouge1": rouge_scores["rouge1"],
                "rouge2": rouge_scores["rouge2"],
                "rougeL": rouge_scores["rougeL"],
                "llm_judge_is_correct": llm_score["is_correct"],
                "llm_judge_reasoning": llm_score["reasoning"]
            }
        })
        
        totals["exact_match_count"] += 1 if em_score else 0
        totals["bleu_sum"] += bleu_score
        totals["rougeL_sum"] += rouge_scores["rougeL"]
        totals["llm_judge_correct_count"] += 1 if llm_score["is_correct"] else 0

        print(f"  Prediction: {prediction}")
        print(f"  EM: {em_score} | BLEU: {bleu_score:.3f} | ROUGE-L: {rouge_scores['rougeL']:.3f} | LLM Judged Correct: {llm_score['is_correct']}")

    # Aggregates
    N = totals["total_cases"]
    aggregates = {
        "exact_match_rate": totals["exact_match_count"] / N if N else 0,
        "avg_bleu": totals["bleu_sum"] / N if N else 0,
        "avg_rougeL": totals["rougeL_sum"] / N if N else 0,
        "llm_judge_pass_rate": totals["llm_judge_correct_count"] / N if N else 0
    }

    # Compare to previous run checking for regressions
    regression_report = {}
    if previous_run and "aggregates" in previous_run:
        prev_aggs = previous_run["aggregates"]
        regression_report = {
            "exact_match_diff": aggregates["exact_match_rate"] - prev_aggs.get("exact_match_rate", 0),
            "bleu_diff": aggregates["avg_bleu"] - prev_aggs.get("avg_bleu", 0),
            "rougeL_diff": aggregates["avg_rougeL"] - prev_aggs.get("avg_rougeL", 0),
            "llm_judge_pass_rate_diff": aggregates["llm_judge_pass_rate"] - prev_aggs.get("llm_judge_pass_rate", 0)
        }
        
    print("\n========= EVALUATION SUMMARY =========")
    for k, v in aggregates.items():
        print(f"  {k}: {v:.3f}")
    if regression_report:
        print("\n========= REGRESSIONS =========")
        for k, v in regression_report.items():
            sym = "+" if v >= 0 else "-"
            print(f"  {k}: {sym}{abs(v):.3f}")
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(runs_dir, f"run_{timestamp}.json")
    
    output_data = {
        "timestamp": timestamp,
        "totals": totals,
        "aggregates": aggregates,
        "regression_report": regression_report,
        "results": results
    }
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
        
    print(f"\nEvaluation complete. Results saved to {out_file}")

if __name__ == "__main__":
    run_evaluations()
