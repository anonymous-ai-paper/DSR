import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import logging
import json
import sys
import gc
import re
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, roc_auc_score
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.utils import set_seed

# --- Dynamic Path Configuration ---
# Add project root to sys.path to import config.py
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import DATA_ROOT, LLM_PATH, CHECKPOINT_DIR

# --- 1. Environment Robustness ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Local imports
from dataset_loader import DialectiGraphDataset
from models.judge_agent import JudgeAgent 

def setup_logging(run_dir, is_main_process):
    handlers = [logging.StreamHandler()]
    if is_main_process:
        handlers.append(logging.FileHandler(os.path.join(run_dir, "train.log")))
    
    logging.basicConfig(
        level=logging.INFO if is_main_process else logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description="DialectiGraph Training")
    
    # Path parameters using config.py defaults
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--llm_path", type=str, default=LLM_PATH) 
    parser.add_argument("--output_dir", type=str, default=CHECKPOINT_DIR)
    
    # Training hyperparameters
    parser.add_argument("--dataset_name", type=str, default="all", choices=["all", "politifact", "gossipcop", "fakeddit"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1) 
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    
    # Optimizer settings
    parser.add_argument("--lr_head", type=float, default=1e-4)
    parser.add_argument("--lr_lora", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--aux_weight", type=float, default=0.3)
    
    # LoRA settings
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    # System settings
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    
    return parser.parse_args()

def compute_metrics_bundle(labels, preds, probs):
    if len(labels) == 0: return None
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', pos_label=1, zero_division=0)
    try: auc = roc_auc_score(labels, probs)
    except: auc = 0.5
    return {"Acc": acc, "Precision": p, "Recall": r, "F1": f1, "Macro-F1": macro_f1, "AUC": auc}

def evaluate(model, loader, accelerator, criterion):
    model.eval()
    all_labels, all_preds, all_probs, all_sources = [], [], [], []
    total_val_loss = 0.0
    
    for batch in tqdm(loader, disable=not accelerator.is_main_process, desc="Evaluating"):
        with torch.no_grad():
            logits, aux_fake, aux_real, _ = model(
                batch['news_input_ids'], batch['news_attention_mask'],
                batch['fake_path_data'], batch['real_path_data'],
                internal_embs=batch['internal_embs']
            )
            
            # Loss calculation
            loss_main = criterion(logits, batch['labels'])
            loss_aux = criterion(aux_fake, torch.ones_like(batch['labels'])) + \
                       criterion(aux_real, torch.zeros_like(batch['labels']))
            batch_loss = loss_main + 0.3 * loss_aux 
            
            probs = F.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            
            # Gather metrics across GPUs
            out_logits, out_labels, out_probs, out_preds, out_loss = accelerator.gather_for_metrics(
                (logits, batch['labels'], probs, preds, batch_loss)
            )
            
            total_val_loss += out_loss.mean().item()

            batch_sources = []
            for fid in batch['ids']:
                if "politifact" in fid: batch_sources.append("politifact")
                elif "gossipcop" in fid: batch_sources.append("gossipcop")
                else: batch_sources.append("fakeddit")
            
            all_labels.extend(out_labels.cpu().numpy())
            all_preds.extend(out_preds.cpu().numpy())
            all_probs.extend(out_probs.cpu().numpy())
            all_sources.extend(batch_sources * accelerator.num_processes) 

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    min_len = min(len(all_labels), len(all_sources))
    all_labels, all_preds, all_probs = all_labels[:min_len], all_preds[:min_len], all_probs[:min_len]
    all_sources = np.array(all_sources[:min_len])
    
    results = {"Current_Set": compute_metrics_bundle(all_labels, all_preds, all_probs)}
    
    # Per-dataset statistics
    for group in ["politifact", "gossipcop", "fakeddit"]:
        mask = (all_sources == group)
        if np.any(mask):
            results[group] = compute_metrics_bundle(all_labels[mask], all_preds[mask], all_probs[mask])
    
    fn_mask = (all_sources == "politifact") | (all_sources == "gossipcop")
    if np.any(fn_mask):
        results["fakenewsnet"] = compute_metrics_bundle(all_labels[fn_mask], all_preds[fn_mask], all_probs[fn_mask])
            
    return results, total_val_loss / len(loader)

def train():
    args = get_args()
    # DDP stability config
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    # Increase timeout to 2 hours for large model synchronization
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        log_with="tensorboard",
        project_dir=args.output_dir,
        kwargs_handlers=[ddp_kwargs, init_kwargs]
    )
    device = accelerator.device
    set_seed(args.seed)

    # Directory sync logic
    if accelerator.is_main_process:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(args.output_dir, f"run_{timestamp}_{args.dataset_name}")
        os.makedirs(run_dir, exist_ok=True)
        with open("last_run_dir.tmp", "w") as f: f.write(run_dir)
    
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        import time
        while not os.path.exists("last_run_dir.tmp"): time.sleep(0.1)
        with open("last_run_dir.tmp", "r") as f: run_dir = f.read().strip()
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and os.path.exists("last_run_dir.tmp"): os.remove("last_run_dir.tmp")
    
    logger = setup_logging(run_dir, accelerator.is_main_process)
    
    # --- Dataset Splitting (Val from Test) ---
    train_ds = DialectiGraphDataset(args.data_root, split="train", dataset_filter=args.dataset_name)
    full_test_raw = DialectiGraphDataset(args.data_root, split="test", dataset_filter=args.dataset_name)
    
    # 50/50 split for validation and final testing
    val_size = len(full_test_raw) // 2
    test_size = len(full_test_raw) - val_size
    val_ds, test_ds = random_split(full_test_raw, [val_size, test_size], generator=torch.Generator().manual_seed(args.seed))
    
    if accelerator.is_main_process:
        logger.info(f"📊 Dataset: {args.dataset_name.upper()} | Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=DialectiGraphDataset.collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=DialectiGraphDataset.collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=DialectiGraphDataset.collate_fn, num_workers=0)

    # Model init
    model = JudgeAgent(llm_path=args.llm_path, device=device, lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    
    # Param grouping for specialized learning rates
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora" in n]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora" not in n]
    optimizer = torch.optim.AdamW([{'params': lora_params, 'lr': args.lr_lora}, {'params': head_params, 'lr': args.lr_head}], weight_decay=args.weight_decay)
    
    total_steps = (len(train_loader) * args.epochs) // args.grad_accum_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_ratio * total_steps), num_training_steps=total_steps)
    
    model, optimizer, train_loader, val_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader, scheduler
    )
    criterion = nn.CrossEntropyLoss()
    
    # Early stopping tracked by Loss
    start_epoch, best_val_loss, patience_counter = 0, float('inf'), 0

    # Resume Logic
    if args.resume_path and os.path.exists(args.resume_path):
        logger.info(f"🔄 Resuming from {args.resume_path}")
        ckpt = torch.load(args.resume_path, map_location="cpu")
        accelerator.unwrap_model(model).load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
        if 'epoch' in ckpt: start_epoch = ckpt['epoch'] + 1
        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                best_val_loss = ckpt.get('best_val_loss', float('inf'))
            except: pass

    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=os.path.join(run_dir, "logs"))

    # Main Training Loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch}")
        
        for step, batch in enumerate(pbar):
            with accelerator.accumulate(model):
                logits, aux_fake, aux_real, path_weights = model(
                    batch['news_input_ids'], batch['news_attention_mask'],
                    batch['fake_path_data'], batch['real_path_data'],
                    internal_embs=batch['internal_embs']
                )
                loss_main = criterion(logits, batch['labels'])
                loss_aux = criterion(aux_fake, torch.ones_like(batch['labels'])) + criterion(aux_real, torch.zeros_like(batch['labels']))
                loss = loss_main + args.aux_weight * loss_aux
                
                # Forward-Backward pass
                total_loss = loss + (sum(p.sum() for p in model.parameters() if p.requires_grad) * 0.0)
                accelerator.backward(total_loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                
                # Active garbage collection to prevent memory fragmentation
                if step % 100 == 0:
                    gc.collect(); torch.cuda.empty_cache()

                epoch_loss += loss.item()
                if accelerator.is_main_process:
                    pbar.set_postfix({'loss': epoch_loss / (step + 1)})

        # Validation phase
        logger.info(f"--- Validating Epoch {epoch} ---")
        val_results, val_avg_loss = evaluate(model, val_loader, accelerator, criterion)
        
        stop_flag = torch.tensor(0, device=device)
        if accelerator.is_main_process:
            logger.info(f"\n" + "="*60)
            logger.info(f"EPOCH {epoch} REPORT (Val Loss: {val_avg_loss:.4f})")
            row_fmt = "{:>12} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6}"
            logger.info(row_fmt.format("Dataset", "Acc", "Pre", "Rec", "F1", "Mac-F1", "AUC"))
            for grp, m in val_results.items():
                if m: logger.info(row_fmt.format(grp.upper(), f"{m['Acc']:.4f}", f"{m['Precision']:.4f}", f"{m['Recall']:.4f}", f"{m['F1']:.4f}", f"{m['Macro-F1']:.4f}", f"{m['AUC']:.4f}"))
            logger.info("="*60)
            
            writer.add_scalar("Loss/Val", val_avg_loss, epoch)
            
            # Save logic for best model and checkpointing
            if val_avg_loss < best_val_loss:
                best_val_loss = val_avg_loss
                patience_counter = 0
                torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(run_dir, "best_model.pth"))
                logger.info(f"🌟 Best Model Updated (New Low Loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                logger.info(f"⚠️ Patience: {patience_counter}/{args.patience}")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'metrics': val_results
            }, os.path.join(run_dir, f"epoch_{epoch}_checkpoint.pth"))

            if patience_counter >= args.patience:
                logger.info("🛑 Early stopping triggered.")
                stop_flag = torch.tensor(1, device=device)

        stop_flag = accelerator.reduce(stop_flag, reduction="max")
        if stop_flag.item() == 1: break

    # Final Testing using best model found
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("🏁 Final Test with Best Model...")
        best_path = os.path.join(run_dir, "best_model.pth")
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=device), strict=False)
        
        final_res, _ = evaluate(model, test_loader, accelerator, criterion)
        logger.info("\n" + "!"*20 + " FINAL TEST RESULT " + "!"*20)
        row_fmt = "{:>12} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6}"
        logger.info(row_fmt.format("Dataset", "Acc", "Pre", "Rec", "F1", "Mac-F1", "AUC"))
        for grp, m in final_res.items():
            if m: logger.info(row_fmt.format(grp.upper(), f"{m['Acc']:.4f}", f"{m['Precision']:.4f}", f"{m['Recall']:.4f}", f"{m['F1']:.4f}", f"{m['Macro-F1']:.4f}", f"{m['AUC']:.4f}"))
        
        with open(os.path.join(run_dir, "final_test_metrics.json"), "w") as f:
            json.dump(final_res, f, indent=4)
        writer.close()

if __name__ == "__main__":
    train()