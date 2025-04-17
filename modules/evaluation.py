from rouge_score import rouge_scorer
from typing import Dict, Any
import pandas as pd
import os
import json
import time
from datetime import datetime
from config import get_config
from utils.logger import Logger


class SummaryEvaluator:
    """
    Evaluates the quality of generated summaries using ROUGE metrics and other measures.
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = Logger(log_file=self.config["evaluation"]["log_file"])
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.results_dir = "logs/evaluation_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def evaluate_summary(self, generated_summary: str, reference_summary: str, method: str = "", prompt_strategy: str = "") -> Dict[str, Any]:
        """
        Evaluate a generated summary against a reference summary using ROUGE metrics,
        save the results, and perform analysis.
        
        Args:
            generated_summary: The summary produced by the system
            reference_summary: The reference/gold standard summary,
            prompt_strategy: The prompt strategy used to generate the summary
            
        Returns:
            Dictionary containing ROUGE scores, metrics, and analysis
        """
        start_time = time.time()
        
        # Calculate ROUGE scores
        scores = self.scorer.score(reference_summary, generated_summary)
        
        # Extract precision, recall and f1 scores
        metrics = {}
        for metric, score in scores.items():
            metrics[f"{metric}_precision"] = score.precision
            metrics[f"{metric}_recall"] = score.recall
            metrics[f"{metric}_f1"] = score.fmeasure
            
        # Add additional metrics
        metrics.update({
            "summary_length": len(generated_summary.split()),
            "reference_length": len(reference_summary.split()),
            "length_ratio": len(generated_summary.split()) / len(reference_summary.split())
        })
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = [metrics]  # Create a list with single result for consistency
        
        # Save as CSV
        df = pd.DataFrame(results)
        csv_path = os.path.join(self.results_dir, f"summary_evaluation_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        
        # Save as JSON with additional info
        json_path = os.path.join(self.results_dir, f"summary_evaluation_{timestamp}.json")
        evaluation_data = {
            "generated_summary": generated_summary,
            "reference_summary": reference_summary,
            "method": method,
            "prompt_strategy": prompt_strategy,
            "metrics": metrics
        }
        with open(json_path, 'w') as f:
            json.dump(evaluation_data, f, indent=2)
            
        # Perform analysis
        analysis = {
            "rouge_scores": {
                "rouge1_f1": metrics["rouge1_f1"],
                "rouge2_f1": metrics["rouge2_f1"],
                "rougeL_f1": metrics["rougeL_f1"]
            },
            "length_analysis": {
                "summary_length": metrics["summary_length"],
                "reference_length": metrics["reference_length"],
                "length_ratio": metrics["length_ratio"]
            }
        }
        
        # Combine metrics and analysis
        final_results = {
            "metrics": json.dumps(metrics, indent=2),
            "analysis": json.dumps(analysis, indent=2),
            "file_paths": {
                "csv": csv_path,
                "json": json_path
            }
        }
        
        end_time = time.time()
        self.logger.metrics("Summary Evaluation", start_time, end_time, final_results)
        return final_results
    


def start_evaluation():
    """
    Interactive function to evaluate a summary against a reference.
    Prompts user for both summaries, runs evaluation, and displays results.
    """

    config = get_config()
    logger = Logger(log_file=config["evaluation"]["log_file"])

    logger.info("\n=== Summary Evaluation ===")
    logger.info("This tool evaluates how well a generated summary matches a reference summary.")
    
    evaluator = SummaryEvaluator()
    
    # Get user input for generated summary
    logger.info("\nEnter the generated summary (the one to evaluate):")
    logger.info("Type your text (can be multiple lines). When finished, type 'END' on a new line:")
    generated_summary_lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        generated_summary_lines.append(line)
    generated_summary = "\n".join(generated_summary_lines)
    
    # Get user input for reference summary
    logger.info("\nEnter the reference summary (the gold standard):")
    logger.info("Type your text (can be multiple lines). When finished, type 'END' on a new line:")
    reference_summary_lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        reference_summary_lines.append(line)
    reference_summary = "\n".join(reference_summary_lines)
    
    if not generated_summary or not reference_summary:
        logger.error("Error: Both summaries must be provided.")
        return
    
    # Evaluate and display results
    logger.info("\nEvaluating summaries...")
    evaluator.evaluate_summary(generated_summary, reference_summary)

