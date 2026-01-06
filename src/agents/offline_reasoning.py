import os
import json
import torch
import re
import argparse
import sys
import time
from glob import glob
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import DATA_ROOT, EVIDENCE_PATHS_DIR, QWEN_VL_PATH, PROJECT_ROOT

# --- Resource Optimization ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Import Prompts
try:
    from .prompts import get_agent_prompt
except ImportError:
    from prompts import get_agent_prompt

# --- Global Constraints for Titan RTX 24GB ---
MAX_IMAGES = 5         
MAX_POSTS = 40          
MAX_PIXELS = 512 * 512  
MAX_PATH_LEN = 5        
POSTS_PRE_TRUNCATE = 500  # Truncate before sorting to save CPU time

class AgentReasoner:
    def __init__(self, device="cuda"):
        print(f"[{time.strftime('%H:%M:%S')}] Loading Qwen2-VL from {QWEN_VL_PATH}...")
        # SDPA implementation for Turing architecture acceleration
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            QWEN_VL_PATH,
            torch_dtype=torch.float16, 
            attn_implementation="sdpa", 
            device_map=device,
        )
        self.processor = AutoProcessor.from_pretrained(QWEN_VL_PATH)
        self.device = device

    def build_context_content(self, data):
        """Build multimodal input context with aggressive truncation."""
        posts = data.get('posts', [])
        if len(posts) > POSTS_PRE_TRUNCATE:
            posts = posts[:POSTS_PRE_TRUNCATE]
        
        # Filter nodes by PageRank and CLIP scores
        imgs = data.get('images', [])
        selected_images = sorted(imgs, key=lambda x: x.get('clip_score', 0), reverse=True)[:MAX_IMAGES]
        selected_posts = sorted(posts, key=lambda x: x.get('pagerank_score', 0), reverse=True)[:MAX_POSTS]

        content_list = []
        self.img_idx_map = {} 

        # 1. Process Image Nodes
        for idx, img in enumerate(selected_images):
            img_path = img['path'] if isinstance(img, dict) else img
            # Construct path relative to project root
            abs_path = os.path.join(PROJECT_ROOT, img_path)
            
            if os.path.exists(abs_path):
                content_list.append({
                    "type": "image", "image": abs_path, "max_pixels": MAX_PIXELS
                })
                self.img_idx_map[idx] = img_path
                
        # 2. Process Text Nodes
        text_context = "【Context Information】\n"
        for idx, img in enumerate(selected_images):
            if idx in self.img_idx_map:
                pr = img.get('pagerank_score', 0) if isinstance(img, dict) else 0
                clip = img.get('clip_score', 0) if isinstance(img, dict) else 0
                text_context += f"[Node: image_{idx}] (PR: {pr:.4f}, CLIP: {clip:.2f})\n"
        
        text_context += f"\nTitle: {data.get('title','')}\nContent: {data.get('text','')[:1000]}\n\nSocial Nodes:\n"
        for post in selected_posts:
            pid = post['id']
            pr = post.get('pagerank_score', 0)
            trust = post.get('user_profile', {}).get('history_trust_score', 0.5)
            text_context += f"- [Node: post_{pid}] (PR: {pr:.3f}, Trust: {trust:.2f}) \"{post.get('text', '')[:100]}\"\n"
            
        # Hard truncate context to avoid VRAM overflow
        if len(text_context) > 12000:
            text_context = text_context[:12000] + "..."

        content_list.append({"type": "text", "text": text_context})
        return content_list

    def reconstruct_paths(self, llm_output, data):
        """Trace leaf nodes back to News root using parent_id metadata."""
        news_id = data['id']
        full_paths = []
        post_lookup = {str(p['id']): p for p in data.get('posts', [])}
        
        # Parse mixed format LLM output
        items = []
        if isinstance(llm_output, list):
            for i in llm_output:
                if isinstance(i, list) and i: items.append(str(i[-1]))
                else: items.append(str(i))
        
        for raw_node in items:
            path_chain = [news_id]
            node_id = raw_node.strip()
            
            if "image_" in node_id:
                try:
                    idx = int(re.search(r'image_(\d+)', node_id).group(1))
                    if idx in self.img_idx_map:
                        path_chain.append(self.img_idx_map[idx])
                        full_paths.append(path_chain)
                except: pass
            elif "post_" in node_id:
                curr_id = node_id.replace("post_", "").strip()
                temp_chain = []
                safety_limit = 0
                # Trace back to source
                while curr_id and curr_id != "SOURCE" and safety_limit < 15:
                    if curr_id in temp_chain: break 
                    post = post_lookup.get(curr_id)
                    if not post: break
                    temp_chain.append(curr_id)
                    next_id = str(post.get('parent_id', 'SOURCE'))
                    if next_id == curr_id: break 
                    curr_id = next_id
                    safety_limit += 1
                
                temp_chain = temp_chain[::-1] 
                if (1 + len(temp_chain)) > MAX_PATH_LEN:
                    path_chain.extend(temp_chain[-(MAX_PATH_LEN-1):])
                else:
                    path_chain.extend(temp_chain)
                if len(path_chain) > 1: full_paths.append(path_chain)
        return full_paths

    def run_inference(self, data, agent_type):
        """Execute single inference pass."""
        try:
            content = self.build_context_content(data)
            content.append({"type": "text", "text": get_agent_prompt(agent_type)})
            messages = [{"role": "user", "content": content}]
            
            prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(text=[prompt_text], images=image_inputs, videos=video_inputs,
                                    padding=True, return_tensors="pt").to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=400, do_sample=False)
            
            out_ids = [ids[len(in_ids):] for in_ids, ids in zip(inputs.input_ids, generated_ids)]
            output_text = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0]
            
            # Extract JSON from LLM output
            json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
            if json_match:
                llm_out = json.loads(json_match.group()).get('paths', [])
                return self.reconstruct_paths(llm_out, data)
            return []
        except: return []

    def process_shard(self, shard_id, num_shards):
        """Distributed reasoning with checkpoint scanning and task skipping."""
        os.makedirs(EVIDENCE_PATHS_DIR, exist_ok=True)
        datasets = ["fakenewsnet", "fakeddit"]
        splits = ["train", "val", "test"]
        
        for ds in datasets:
            for sp in splits:
                input_dir = os.path.join(DATA_ROOT, ds, sp)
                save_path = os.path.join(EVIDENCE_PATHS_DIR, f"{ds}_{sp}_shard_{shard_id}.json")
                
                # Check for already processed samples in output dir
                global_processed_ids = set()
                existing_shards = glob(os.path.join(EVIDENCE_PATHS_DIR, f"{ds}_{sp}_shard_*.json"))
                
                results = {}
                if os.path.exists(save_path):
                    try:
                        with open(save_path, 'r') as f: results = json.load(f)
                    except: results = {}

                for es in existing_shards:
                    try:
                        with open(es, 'r') as f:
                            shard_data = json.load(f)
                            for fid in shard_data.keys():
                                global_processed_ids.add(fid)
                    except: continue
                
                # Filter files to run
                all_files = sorted(glob(os.path.join(input_dir, "*.json")))
                my_files = [f for i, f in enumerate(all_files) if i % num_shards == shard_id]
                
                files_to_run = []
                for fpath in my_files:
                    fname = os.path.basename(fpath).replace('.json','')
                    if fname in global_processed_ids: continue
                    
                    # Double check if source file already has evidence injected
                    try:
                        with open(fpath, 'r', encoding='utf-8') as f:
                            source_data = json.load(f)
                        
                        ev = source_data.get('evidence_subset', {})
                        if ev.get('fake_agent_paths') or ev.get('real_agent_paths'):
                            continue
                            
                        files_to_run.append((fname, source_data))
                    except: continue
                
                if not files_to_run: 
                    print(f"✅ [Shard {shard_id}] {ds}/{sp}: No pending tasks.")
                    continue
                
                print(f"🚀 [Shard {shard_id}] {ds}/{sp}: {len(files_to_run)} tasks remaining.", flush=True)
                pbar = tqdm(files_to_run, desc=f"GPU{shard_id}-{ds[:4]}", position=shard_id)
                
                for i, (fid, data) in enumerate(pbar):
                    try:
                        fake = self.run_inference(data, "fake")
                        real = self.run_inference(data, "real")
                        results[fid] = {"fake_agent_paths": fake, "real_agent_paths": real}
                        
                        torch.cuda.empty_cache()
                        # Backup save every 20 samples
                        if (i + 1) % 20 == 0:
                            with open(save_path, 'w') as f: json.dump(results, f, indent=2)
                    except Exception: continue
                
                with open(save_path, 'w') as f: json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_id", type=int, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    args = parser.parse_args()
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    reasoner = AgentReasoner(device="cuda")
    reasoner.process_shard(args.shard_id, args.num_shards)