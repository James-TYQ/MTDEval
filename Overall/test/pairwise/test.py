import os
import torch
import pandas as pd
from tqdm import tqdm
import logging
import sys
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

# Model and data path
MODEL_PATH = "./results/multi_turn_ArmoRM-llama3-8b-v0.1_bsz32_lr5e-5_ss_lr1e-2_sens0.5_spec0.5_epochs1_warmup0.1_conf0.0_labeltemp2.0multi-turn-dialogue"  # replace with your actual model path
CSV_PATH = "../data/Daily-MTD/Daily-MTD-Pair.csv"  # replace with your actual test data path
OUTPUT_PATH = "../Daily-MTD-Pair.csv"  # replace with your actual output path

# Import evaluation pipeline
try:
    from inference import SaMerPipeline
    logger.info("Successfully imported SaMerPipeline")
except ImportError as e:
    logger.error(f"Failed to import SaMerPipeline: {e}")
    sys.exit(1)

# Initialize evaluation pipeline
logger.info(f"Loading model: {MODEL_PATH}")
try:
    pipeline = SaMerPipeline(
        model_id=MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}", exc_info=True)
    sys.exit(1)

def parse_conversation(conversation_str):
    """
    Load the conversation from the string and return the dialog_A and dialog_B
    Return dialog_A, dialog_B (both are list[dict], format as the pipeline needs)
    """

    # Preprocess
    conv = conversation_str.replace('""', '"').replace('\r\n', '\n').replace('\r', '\n')

    rounds = re.split(r'"role": "Human",', conv)
    dialog_A = []
    dialog_B = []
    for r in rounds[1:]:  # The first one is empty
        # Extract Human content
        match_human = re.search(r'"text":\s*"([^"]*)"', r)
        # Extract A/B content
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
    # If there is content, return
    if dialog_A and dialog_B:
        return dialog_A, dialog_B
    return None, None

def main():
    logger.info(f"Reading CSV file: {CSV_PATH}")
    try:
        df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
        logger.info(f"Read {len(df)} conversations")
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}")
        return

    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating conversations"):
        try:
            dialog_A, dialog_B = parse_conversation(row['conversation'])
            if dialog_A is None or dialog_B is None:
                raise ValueError("Failed to parse conversation field")
        except Exception as e:
            logger.error(f"Failed to parse conversation {idx}: {e}")
            continue

        try:
            eval_result = pipeline.compare_responses(dialog_A, dialog_B)
            ground_truth = row.get('conclusion', 'Unknown')
            results.append({
                "dialog_id": row.get('dialog_id', idx),
                "A_score": eval_result.get("A_score"),
                "B_score": eval_result.get("B_score"),
                "model_winner": eval_result.get("winner"),
                "ground_truth": ground_truth,
                "correct": eval_result.get("winner") == ground_truth,
                "A_Overall": eval_result.get("A_Overall"),
                "B_Overall": eval_result.get("B_Overall"),
            })
            logger.info(f"Conversation {idx}: Model judgment: {eval_result['winner']}, True label: {ground_truth}")
        except Exception as e:
            logger.error(f"Error evaluating conversation {idx}: {e}", exc_info=True)

    if results:
        correct_count = sum(1 for r in results if r["correct"])
        accuracy = correct_count / len(results)
        result_df = pd.DataFrame(results)

        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        
        result_df.to_csv(OUTPUT_PATH, index=False)
        logger.info(f"\nEvaluation completed! Total {len(results)} conversations")
        logger.info(f"Accuracy: {accuracy:.4f} ({correct_count}/{len(results)})")
        logger.info(f"Detailed results saved to: {OUTPUT_PATH}")
    else:
        logger.warning("No successful evaluation")

if __name__ == "__main__":
    main()