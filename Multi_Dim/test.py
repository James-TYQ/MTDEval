import os
import torch
import pandas as pd
from tqdm import tqdm
import logging
import sys
import json
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_log.txt')
    ]
)
logger = logging.getLogger(__name__)

# model and data path
MODEL_PATH = "./results/multi_turn_ArmoRM-llama3-8b-v0.1_bsz32_lr5e-5_ss_lr1e-2_sens0.5_spec0.5_epochs1_warmup0.1_conf0.0_labeltemp2.0multi-turn-dialogue"  # replace with your actual model path
CSV_PATH = "../data/Daily-MTD/Daily-MTD-Dim.csv"  # replace with your actual test data path

# import evaluation pipeline
try:
    from inference import MTDEvalPipeline
    logger.info("Successfully imported MTDEvalPipeline")
except ImportError as e:
    logger.error(f"Failed to import MTDEvalPipeline: {e}")
    sys.exit(1)

# initialize evaluation pipeline
logger.info(f"Loading model: {MODEL_PATH}")
try:
    pipeline = MTDEvalPipeline(
        model_id=MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
    logger.info("Model loaded successfully")
    logger.info(f"Evaluation dimensions: {pipeline.dimensions}")
except Exception as e:
    logger.error(f"Failed to load model: {e}", exc_info=True)
    sys.exit(1)

def parse_conversation(conversation_str):
    conv = conversation_str.replace('""', '"').replace('\r\n', '\n').replace('\r', '\n')
    rounds = re.split(r'"role": "Human",', conv)
    dialog_A = []
    dialog_B = []
    for r in rounds[1:]:  
        match_human = re.search(r'"text":\s*"([^"]*)"', r)
        match_A = re.search(r'"A":\s*"([^"]*)"', r)
        match_B = re.search(r'"B":\s*"([^"]*)"', r)
        if match_human and match_A and match_B:
            user_msg = {"role": "user", "content": match_human.group(1)}
            a_msg = {"role": "assistant", "content": match_A.group(1)}
            b_msg = {"role": "assistant", "content": match_B.group(1)}
            dialog_A.append(user_msg)
            dialog_A.append(a_msg)
            dialog_B.append(user_msg)
            dialog_B.append(b_msg)
    if dialog_A and dialog_B:
        return dialog_A, dialog_B
    return None, None

def main():
    logger.info(f"Reading CSV file: {CSV_PATH}")
    try:
        df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
        logger.info(f"Read {len(df)} dialogs")
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}")
        return

    results = []
    detailed_results = []
    dimension_stats = {dim: {"correct": 0, "total": 0} for dim in pipeline.dimensions}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating dialogs"):
        try:
            dialog_A, dialog_B = parse_conversation(row['conversation'])
            if dialog_A is None or dialog_B is None:
                raise ValueError("Failed to parse conversation field")
        except Exception as e:
            logger.error(f"Failed to parse dialog {idx}: {e}")
            continue

        try:
            eval_result = pipeline.compare_responses(dialog_A, dialog_B)
            ground_truth = row.get("out_conclusion", "Unknown")
            
            overall_correct = eval_result["overall_winner"] == ground_truth
            results.append({
                "dialog_id": row.get("dialog_id", idx),
                "model_winner": eval_result["overall_winner"],
                "ground_truth": ground_truth,
                "correct": overall_correct,
                "A_overall_score": eval_result["A_overall_score"],
                "B_overall_score": eval_result["B_overall_score"],
            })
            
            detail_record = {
                "dialog_id": row.get("dialog_id", idx),
                "ground_truth": ground_truth,
                "overall_correct": overall_correct,
                "dimension_comparisons": eval_result["dimension_comparisons"],
                "A_dimensional_scores": eval_result["A_dimensional_scores"],
                "B_dimensional_scores": eval_result["B_dimensional_scores"],
            }
            detailed_results.append(detail_record)
            
            for dim in pipeline.dimensions:
                dim_result = eval_result["dimension_comparisons"][dim]
                dimension_stats[dim]["total"] += 1
                if dim_result["winner"] == ground_truth:
                    dimension_stats[dim]["correct"] += 1
            
            logger.info(f"Dialog {idx}: Overall judgment: {eval_result['overall_winner']}, Ground truth: {ground_truth}")
            
            dim_winners = []
            for dim in pipeline.dimensions:
                winner = eval_result["dimension_comparisons"][dim]["winner"]
                score_A = eval_result["dimension_comparisons"][dim]["A_score"]
                score_B = eval_result["dimension_comparisons"][dim]["B_score"]
                dim_winners.append(f"{dim}:{winner}({score_A:.3f} vs {score_B:.3f})")
            logger.info(f"Dimension judgment: {'; '.join(dim_winners)}")
            
        except Exception as e:
            logger.error(f"Error evaluating dialog {idx}: {e}", exc_info=True)

    if results:
        correct_count = sum(1 for r in results if r["correct"])
        overall_accuracy = correct_count / len(results)
        
        dimension_accuracies = {}
        for dim, stats in dimension_stats.items():
            if stats["total"] > 0:
                accuracy = stats["correct"] / stats["total"]
                dimension_accuracies[dim] = accuracy
            else:
                dimension_accuracies[dim] = 0.0
        
        result_df = pd.DataFrame(results)
        output_dir = os.path.join(os.path.dirname(CSV_PATH), "result")
        os.makedirs(output_dir, exist_ok=True)
        result_path = os.path.join(output_dir, "fine.csv")
        result_df.to_csv(result_path, index=False)
        
        detailed_result_path = os.path.join(output_dir, "detailed_fine_results.json")
        with open(detailed_result_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        stats_path = os.path.join(output_dir, "dimension_accuracy_stats_fine.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump({
                "overall_accuracy": overall_accuracy,
                "dimension_accuracies": dimension_accuracies,
                "dimension_stats": dimension_stats,
                "total_evaluated": len(results)
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluation completed! Evaluated {len(results)} dialogs")
        logger.info(f"Overall accuracy: {overall_accuracy:.4f} ({correct_count}/{len(results)})")
        logger.info(f"\nDimension accuracies:")
        for dim, accuracy in dimension_accuracies.items():
            logger.info(f"  {dim}: {accuracy:.4f} ({dimension_stats[dim]['correct']}/{dimension_stats[dim]['total']})")
        
        logger.info(f"\nResult files:")
        logger.info(f"  Basic results: {result_path}")
        logger.info(f"  Detailed results: {detailed_result_path}")
        logger.info(f"  Dimension statistics: {stats_path}")
        logger.info(f"{'='*60}")
        
    else:
        logger.warning("No successful evaluations")

if __name__ == "__main__":
    main()