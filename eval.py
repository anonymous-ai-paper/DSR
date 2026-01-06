import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, roc_auc_score

# --- Dynamic Path Configuration ---
# Add project root to sys.path to import config.py
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import DATA_ROOT, LLM_PATH

# Import local modules
from dataset_loader import DialectiGraphDataset
from models.judge_agent import JudgeAgent

# ================= Configuration =================
# Update this path to the specific checkpoint you want to evaluate
BEST_MODEL_PATH = os.path.join(project_root, "checkpoints/run_20260105_121136_gossipcop/best_model.pth")

DATASET_FILTER = "gossipcop" # Options: politifact, gossipcop, fakeddit, all
DEVICE = "cuda:0"            
SEED = 42                    
# =================================================

def to_device_and_dtype(data, device, dtype):
    """Recursively move tensors to device and convert precision."""
    if isinstance(data, torch.Tensor):
        if torch.is_floating_point(data):
            return data.to(device=device, dtype=dtype)
        return data.to(device=device)
    elif isinstance(data, dict):
        return {k: to_device_and_dtype(v, device, dtype) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device_and_dtype(v, device, dtype) for v in data]
    return data

def compute_metrics_bundle(labels, preds, probs):
    """Calculate standard classification metrics."""
    if len(labels) == 0: return None
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', pos_label=1, zero_division=0)
    _, _, f1_each, _ = precision_recall_fscore_support(labels, preds, average=None, labels=[0, 1], zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = 0.5
    return {
        "Acc": acc, 
        "Precision": p, 
        "Recall": r, 
        "F1": f1, 
        "Macro-F1": macro_f1, 
        "AUC": auc,
        "F1_Real": f1_each[0],
        "F1_Fake": f1_each[1]
    }

def run_evaluation():
    print(f"🚀 Starting Single-GPU Evaluation...")
    print(f"📦 Dataset: {DATASET_FILTER.upper()}")
    
    # 1. Load Dataset
    full_test_raw = DialectiGraphDataset(data_root=DATA_ROOT, split="test", dataset_filter=DATASET_FILTER)
    
    # 2. Split Logic (Sync with train.py)
    val_size = len(full_test_raw) // 2
    test_size = len(full_test_raw) - val_size
    _, test_ds = random_split(full_test_raw, [val_size, test_size], generator=torch.Generator().manual_seed(SEED))
    
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=DialectiGraphDataset.collate_fn, num_workers=0)
    print(f"📊 Evaluation samples: {len(test_ds)}")

    # 3. Initialize Model
    print(f"🧠 Loading Judge Agent from {LLM_PATH}...")
    model = JudgeAgent(llm_path=LLM_PATH, device=DEVICE)
    
    # --- Precision Alignment ---
    target_dtype = model.system_prompt_embeds.dtype
    print(f"⚙️ Target Dtype detected: {target_dtype}. Syncing modules...")
    
    model.feature_encoders.to(device=DEVICE, dtype=target_dtype)
    model.path_encoder.to(device=DEVICE, dtype=target_dtype)
    model.agg_attention.to(device=DEVICE, dtype=target_dtype)
    model.evidence_projector.to(device=DEVICE, dtype=target_dtype)
    model.internal_projector.to(device=DEVICE, dtype=target_dtype)
    model.aux_classifier.to(device=DEVICE, dtype=target_dtype)
    model.classifier.to(device=DEVICE, dtype=torch.float32) # Keep classifier FP32
    
    # 4. Load Weights
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"❌ Error: Checkpoint not found at {BEST_MODEL_PATH}")
        return
        
    print(f"💾 Loading weights from {BEST_MODEL_PATH}...")
    state_dict = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    all_labels, all_preds, all_probs, all_sources = [], [], [], []

    # 5. Inference
    print("⚡ Running inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # Move batch to device and match dtype
            news_input_ids = batch['news_input_ids'].to(DEVICE)
            news_attn_mask = batch['news_attention_mask'].to(DEVICE)
            internal_embs = batch['internal_embs'].to(device=DEVICE, dtype=target_dtype)
            
            fake_path_data = to_device_and_dtype(batch['fake_path_data'], DEVICE, target_dtype)
            real_path_data = to_device_and_dtype(batch['real_path_data'], DEVICE, target_dtype)

            logits, _, _, _ = model(
                news_input_ids, 
                news_attn_mask,
                fake_path_data, 
                real_path_data,
                internal_embs=internal_embs
            )
            
            probs = F.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            
            all_labels.extend(batch['labels'].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            for fid in batch['ids']:
                if "politifact" in fid: all_sources.append("politifact")
                elif "gossipcop" in fid: all_sources.append("gossipcop")
                else: all_sources.append("fakeddit")

    # 6. Statistics
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_sources = np.array(all_sources)
    
    results = {"Overall_Test": compute_metrics_bundle(all_labels, all_preds, all_probs)}
    for group in ["politifact", "gossipcop", "fakeddit"]:
        mask = (all_sources == group)
        if np.any(mask): 
            results[group] = compute_metrics_bundle(all_labels[mask], all_preds[mask], all_probs[mask])

    # 7. Print Report
    print("\n" + "="*25 + " FINAL TEST REPORT " + "="*25)
    row_fmt = "{:>15} | {:>6} | {:>6} | {:>6} | {:>6} | {:>7} | {:>6}"
    print(row_fmt.format("Subset", "Acc", "Pre", "Rec", "F1", "MacroF1", "AUC"))
    print("-" * 75)
    for grp, m in results.items():
        if m:
            print(row_fmt.format(grp.upper(), f"{m['Acc']:.4f}", f"{m['Precision']:.4f}", f"{m['Recall']:.4f}", f"{m['F1']:.4f}", f"{m['Macro-F1']:.4f}", f"{m['AUC']:.4f}"))
    
    # 8. Save Results
    output_file = BEST_MODEL_PATH.replace("best_model.pth", "final_test_results.json")
    with open(output_file, 'w') as f: 
        json.dump(results, f, indent=4)
    print(f"\n✅ JSON results saved to: {output_file}")

if __name__ == "__main__":
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    run_evaluation()