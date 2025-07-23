import os
import re
import torch
import pandas as pd
from tqdm import tqdm
import logging
import sys
from scipy.stats import pearsonr, spearmanr, kendalltau

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
CSV_PATH = "../data/Daily-MTD/Daily-MTD-Single.csv"  # replace with your actual test data path
OUT_PATH = "../Daily-MTD-Single.csv"  # replace with your actual output path

# Import evaluation pipeline
try:
    from inference import MTDEvalPipeline
    logger.info("Successfully imported MTDEvalPipeline")
except ImportError as e:
    logger.error(f"Failed to import MTDEvalPipeline: {e}")
    sys.exit(1)

# Initialize evaluation pipeline
logger.info(f"Loading model: {MODEL_PATH}")
try:
    pipeline = MTDEvalPipeline(
        model_id=MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}", exc_info=True)
    sys.exit(1)

def parse_conversation(conv_str):
    """Parse a single conversation string into a list of messages"""
    conv = conv_str.replace('""','"').replace('\r\n','\n').replace('\r','\n')
    rounds = re.split(r'"role": "Human",', conv)
    messages = []
    
    for r in rounds[1:]:  # The first one is empty
        m_h = re.search(r'"text":\s*"([^"]*)"', r)
        m_a = re.search(r'"role": "Assistant",\s*"text":\s*"([^"]*)"', r)
        
        if m_h and m_a:
            messages.append({
                "role": "user", 
                "content": m_h.group(1).strip()
            })
            messages.append({
                "role": "assistant", 
                "content": m_a.group(1).strip()
            })
    
    return messages if messages else None

def main():
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    logger.info(f"Read {len(df)} conversations")

    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="score"):
        try:
            # Parse conversation
            messages = parse_conversation(row['conversation'])
            if not messages:
                logger.warning(f"{idx} Failed to parse, skip")
                continue
                
            pipeline_output = pipeline(messages) 
            model_score = pipeline_output["overall_score"]    # Use model to score
            original_score = row.get('score', None)           # Get original score
            
            # Save results
            results.append({
                "dialog_id": row.get("dialog_id", idx),
                "model_score": model_score,
                "original_score": original_score,
            })
            
        except Exception as e:
            logger.warning(f"{idx} Failed to process: {e}")
            continue

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(OUT_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Create directory: {output_dir}")

    # Save results
    pd.DataFrame(results).to_csv(OUT_PATH, index=False)
    logger.info(f"Done! Results saved in {OUT_PATH}")
    
    # Calculate the correlation between the model score and the original score (if any)
    if len(results) >= 2 and all(r.get('original_score') is not None for r in results):
        model_scores = [r['model_score'] for r in results]
        original_scores = [r['original_score'] for r in results]
        
        pearson_corr, p_value = pearsonr(model_scores, original_scores)
        spearman_corr, s_p_value = spearmanr(model_scores, original_scores)
        kendall_corr, k_p_value = kendalltau(model_scores, original_scores)
        
        logger.info(f"Pearson: {pearson_corr:.4f} (p-value: {p_value:.4f})")
        logger.info(f"Spearman: {spearman_corr:.4f} (p-value: {s_p_value:.4f})")
        logger.info(f"Kendall-Tau: {kendall_corr:.4f} (p-value: {k_p_value:.4f})")
    elif len(results) < 2:
        logger.warning("Invalid results")

if __name__ == "__main__":
    main()
