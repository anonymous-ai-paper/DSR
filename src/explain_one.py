import os
import torch
import torch.nn.functional as F
import json
import sys
from torch.utils.data import DataLoader

# --- Dynamic Path Configuration ---
# Add project root to sys.path to import config.py and local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import DATA_ROOT, LLM_PATH, PROJECT_ROOT
from dataset_loader import DialectiGraphDataset, get_tokenizer
from models.judge_agent import JudgeAgent

# ================= Configuration =================
DEVICE = "cuda:0"

# Path to the specific checkpoint for analysis
MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints/run_20260105_105338_politifact/best_model.pth")
# News ID to analyze
TARGET_NEWS_ID = "politifact-14840" 
# =================================================

def to_device_and_dtype(data, device, dtype):
    """Recursively move tensors to device and match model precision."""
    if isinstance(data, torch.Tensor):
        if torch.is_floating_point(data):
            return data.to(device=device, dtype=dtype)
        return data.to(device=device)
    elif isinstance(data, dict):
        return {k: to_device_and_dtype(v, device, dtype) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device_and_dtype(v, device, dtype) if isinstance(v, (torch.Tensor, dict, list)) else v for v in data]
    return data

def run_explanation():
    print(f"🔍 Analyzing Sample: {TARGET_NEWS_ID}")
    
    # 1. Locate the specific JSON file
    file_path = None
    for root, dirs, files in os.walk(DATA_ROOT):
        if f"{TARGET_NEWS_ID}.json" in files:
            file_path = os.path.join(root, f"{TARGET_NEWS_ID}.json")
            break
    
    if not file_path:
        print(f"❌ Error: Cannot find news file for ID {TARGET_NEWS_ID}")
        return

    # 2. Setup Dataset Loader
    tokenizer = get_tokenizer()
    dataset = DialectiGraphDataset(DATA_ROOT, split="test", dataset_filter="all")
    # Override file list to process only the target sample
    dataset.file_paths = [file_path] 
    loader = DataLoader(dataset, batch_size=1, collate_fn=DialectiGraphDataset.collate_fn)
    batch = next(iter(loader))

    # 3. Initialize Model
    print(f"🧠 Loading Judge Agent from {LLM_PATH}...")
    model = JudgeAgent(llm_path=LLM_PATH, device=DEVICE)
    
    # Detect target precision (BFloat16 or Float16)
    target_dtype = model.system_prompt_embeds.dtype
    print(f"⚙️ Syncing modules to {DEVICE} with {target_dtype}...")
    
    # Move sub-modules to device and align dtypes
    model.feature_encoders.to(device=DEVICE, dtype=target_dtype)
    model.path_encoder.to(device=DEVICE, dtype=target_dtype)
    model.agg_attention.to(device=DEVICE, dtype=target_dtype)
    model.evidence_projector.to(device=DEVICE, dtype=target_dtype)
    model.internal_projector.to(device=DEVICE, dtype=target_dtype)
    model.aux_classifier.to(device=DEVICE, dtype=target_dtype)
    model.classifier.to(device=DEVICE, dtype=torch.float32)

    # 4. Load Weights with Mismatch Protection
    print(f"💾 Loading Checkpoint from {MODEL_PATH}...")
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    
    model_state = model.state_dict()
    new_state_dict = {}
    skipped_layers = []
    
    for k, v in state_dict.items():
        if k in model_state:
            # Check for shape consistency (handles the 512 vs 768 projector fix)
            if v.shape != model_state[k].shape:
                skipped_layers.append(k)
                continue
            new_state_dict[k] = v
            
    if skipped_layers:
        print(f"⚠️ Warning: Skipped loading mismatch layers (Randomly initialized):")
        for k in skipped_layers:
            print(f"   - {k}")
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # 5. Execute Inference
    print(f"⚡ Running Inference...")
    with torch.no_grad():
        news_in = batch['news_input_ids'].to(DEVICE)
        news_mask = batch['news_attention_mask'].to(DEVICE)
        int_emb = batch['internal_embs'].to(device=DEVICE, dtype=target_dtype)
        
        f_data = to_device_and_dtype(batch['fake_path_data'], DEVICE, target_dtype)
        r_data = to_device_and_dtype(batch['real_path_data'], DEVICE, target_dtype)

        logits, _, _, path_weights = model(news_in, news_mask, f_data, r_data, int_emb)
        prob = F.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1).item()
        
        # Generate Natural Language Explanation
        explanation = model.generate_explanation(news_in, news_mask, f_data, r_data, int_emb)

    # 6. Print Visual Analysis Report
    print("\n" + "="*30 + " DGR ANALYSIS REPORT " + "="*30)
    print(f"News ID:      {TARGET_NEWS_ID}")
    print(f"Title:        {batch['news_titles'][0]}")
    print(f"Ground Truth: {'FAKE' if batch['labels'][0]==1 else 'REAL'}")
    print(f"Prediction:   {'FAKE' if pred==1 else 'REAL'} (Confidence: {prob[0][pred]:.4f})")
    print("-" * 80)
    
    # Helper to decode and print evidence path content
    def print_path_text(path_data, weights, label_type):
        best_idx = torch.argmax(weights).item()
        weight_val = weights[best_idx].item()
        print(f"🔥 Key {label_type} Evidence (Path {best_idx}, Weight {weight_val:.4f}):")
        
        text_indices = path_data.get('text_indices', [])
        img_indices = path_data.get('img_indices', [])
        tokens = None
        if path_data.get('text_inputs') is not None:
            tokens = path_data['text_inputs']['input_ids']
        
        found_nodes = False
        for b, p, n, flat_idx in text_indices:
            if p == best_idx:
                found_nodes = True
                if tokens is not None:
                    if isinstance(flat_idx, torch.Tensor): flat_idx = flat_idx.item()
                    node_text = tokenizer.decode(tokens[flat_idx], skip_special_tokens=True)
                    print(f"  [Text Node {n}] {node_text[:200]}...") 
        
        for b, p, n, flat_idx in img_indices:
            if p == best_idx:
                found_nodes = True
                print(f"  [Image Node {n}] Visual features detected.")
        
        if not found_nodes:
            print("  (This path contains no accessible text/image nodes)")

    if 'fake' in path_weights:
        print_path_text(batch['fake_path_data'], path_weights['fake'][0], "FAKE")
        print("-" * 40)
    if 'real' in path_weights:
        print_path_text(batch['real_path_data'], path_weights['real'][0], "REAL")
    
    print("\n" + "="*30 + " AI REASONING " + "="*30)
    clean_explanation = explanation[0]
    if "Analysis & Final Conclusion:" in clean_explanation:
        clean_explanation = clean_explanation.split("Analysis & Final Conclusion:")[-1].strip()
    print(clean_explanation)
    print("="*80)

if __name__ == "__main__":
    run_explanation()