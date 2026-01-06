import json
import os
import torch
import math
import random
import numpy as np
import sys
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer, CLIPProcessor

# --- Dynamic Path Configuration ---
# Add project root to sys.path to ensure 'config' can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import ROBERTA_PATH, CLIP_PATH, CACHE_PATH, PROJECT_ROOT

# --- Global Configurations ---
MAX_PATHS = 5          
MAX_PATH_LEN = 5       
MAX_TEXT_LEN = 512     
USER_FEAT_DIM = 4      

# Global singletons to prevent reloading models in collate_fn
_GLOBAL_TOKENIZER = None
_GLOBAL_CLIP_PROCESSOR = None

def get_tokenizer():
    global _GLOBAL_TOKENIZER
    if _GLOBAL_TOKENIZER is None:
        _GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(ROBERTA_PATH)
    return _GLOBAL_TOKENIZER

def get_clip_processor():
    global _GLOBAL_CLIP_PROCESSOR
    if _GLOBAL_CLIP_PROCESSOR is None:
        _GLOBAL_CLIP_PROCESSOR = CLIPProcessor.from_pretrained(CLIP_PATH)
    return _GLOBAL_CLIP_PROCESSOR

class DialectiGraphDataset(Dataset):
    def __init__(self, data_root, split="train", limit=None, dataset_filter="all"):
        """
        Args:
            data_root: Root directory for unified_json.
            split: 'train', 'val', or 'test'.
            dataset_filter: 'all', 'politifact', 'gossipcop', 'fakeddit'.
        """
        self.split = split
        self.data_dir_fakenewsnet = os.path.join(data_root, "fakenewsnet", split)
        self.data_dir_fakeddit = os.path.join(data_root, "fakeddit", split)
        
        # 1. Scan for all available files
        files_politifact = []
        files_gossipcop = []
        files_fakeddit = []

        if os.path.exists(self.data_dir_fakenewsnet):
            for f in os.listdir(self.data_dir_fakenewsnet):
                if not f.endswith('.json'): continue
                full_path = os.path.join(self.data_dir_fakenewsnet, f)
                if "politifact" in f: files_politifact.append(full_path)
                elif "gossipcop" in f: files_gossipcop.append(full_path)

        if os.path.exists(self.data_dir_fakeddit):
            for f in os.listdir(self.data_dir_fakeddit):
                if f.endswith('.json'):
                    files_fakeddit.append(os.path.join(self.data_dir_fakeddit, f))

        # 2. Apply dataset filtering
        self.file_paths = []
        if dataset_filter == "politifact":
            self.file_paths = files_politifact
        elif dataset_filter == "gossipcop":
            self.file_paths = files_gossipcop
        elif dataset_filter == "fakeddit":
            self.file_paths = files_fakeddit
        else: # "all" mode with balancing
            files_fakenewsnet = files_politifact + files_gossipcop
            # Balance small FakeNewsNet against large Fakeddit during training
            if split == "train" and len(files_fakenewsnet) > 0 and len(files_fakeddit) > 0:
                ratio = len(files_fakeddit) // len(files_fakenewsnet)
                oversample_factor = min(max(ratio // 2, 2), 10) 
                print(f"⚖️ Balancing: Oversampling FakeNewsNet by {oversample_factor}x")
                files_fakenewsnet = files_fakenewsnet * oversample_factor
            self.file_paths = files_fakenewsnet + files_fakeddit

        if split == "train": random.shuffle(self.file_paths)
        if limit: self.file_paths = self.file_paths[:limit]
            
        print(f"[{split}] Dataset initialized with {len(self.file_paths)} samples.")

        # 3. Load Internal Knowledge Cache
        if os.path.exists(CACHE_PATH):
            try:
                self.internal_cache = torch.load(CACHE_PATH, map_location="cpu", mmap=True)
                print("Cache loaded with mmap.")
            except:
                self.internal_cache = torch.load(CACHE_PATH, map_location="cpu")
        else:
            print("⚠️ Warning: Cache not found at", CACHE_PATH)
            self.internal_cache = {}

        # Warm up processors
        get_tokenizer()
        get_clip_processor()

    def __len__(self):
        return len(self.file_paths)

    def _load_image(self, path):
        """Safely load image, return black placeholder on failure."""
        abs_path = os.path.join(PROJECT_ROOT, path)
        try:
            image = Image.open(abs_path).convert("RGB")
            return image
        except:
            return Image.new('RGB', (224, 224), color='black')

    def _normalize_user_features(self, profile):
        """Map user metadata to 4-dim feature vector."""
        if not profile: return [0.0, 0.0, 0.0, 0.5]
        fol = profile.get('followers_count') or 0
        fol_norm = math.log1p(float(fol)) / 10.0 
        fri = profile.get('friends_count') or 0
        fri_norm = math.log1p(float(fri)) / 10.0
        ver = 1.0 if profile.get('verified') else 0.0
        trust = float(profile.get('history_trust_score', 0.5))
        return [fol_norm, fri_norm, ver, trust]

    def _get_node_content(self, node_id_str, raw_data, lookup_table):
        """Extract node content using IDs with prefix handling."""
        node_type = "unknown"
        content = None
        user_feat = [0, 0, 0, 0.5]
        
        # 1. Clean potential prefixes (e.g., 'post:123' -> '123')
        clean_id = node_id_str
        if node_id_str.startswith("news:"):
            node_type = "text"; clean_id = node_id_str[5:]
        elif node_id_str.startswith("image:"):
            node_type = "image"; clean_id = node_id_str[6:]
        elif node_id_str.startswith("post:"):
            node_type = "text"; clean_id = node_id_str[5:]
        elif node_id_str.startswith("user:"):
            node_type = "user"; clean_id = node_id_str[5:]
        
        clean_id = clean_id.strip()

        # 2. Logic to match content
        if clean_id == str(raw_data['id']): # It's the News root
            node_type = "text"
            content = raw_data.get('title', "")
        elif clean_id in lookup_table: # It's a social post
            node_type = "text"
            post = lookup_table[clean_id]
            raw_text = post.get('text', "")
            content = raw_text[:500] if raw_text else ""
            user_feat = self._normalize_user_features(post.get('user_profile'))
        elif "/" in clean_id or clean_id.lower().endswith(('.jpg', '.png', '.jpeg')): # Image path
            node_type = "image"
            content = self._load_image(clean_id)
        else: # Handle unknown/user nodes
            if node_id_str.startswith("user:"):
                node_type = "user"
                content = "[User Node]"
            else:
                node_type = "text"
                content = "" # Pad with empty string

        return {"type": node_type, "content": content, "user_feat": user_feat}

    def _process_path_chain(self, chain_ids, raw_data, lookup_table):
        processed_nodes = []
        for node_id in chain_ids:
            node_data = self._get_node_content(node_id, raw_data, lookup_table)
            processed_nodes.append(node_data)
        return processed_nodes

    def __getitem__(self, idx):
        try:
            fpath = self.file_paths[idx]
            news_id = os.path.basename(fpath).replace('.json', '')
            
            with open(fpath, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                
            post_lookup = {str(p['id']): p for p in raw_data.get('posts', [])}
            
            news_title = raw_data.get('title', "") or ""
            news_text = (raw_data.get('text', "") or "")[:2000]
            label = 1 if raw_data.get('label') == 'fake' else 0
            
            # Evidence routing: prefer high-quality Gemini evidence if available
            if 'gemini_evidence' in raw_data and raw_data['gemini_evidence']:
                ev_data = raw_data['gemini_evidence']
            else:
                ev_data = raw_data.get('evidence_subset', {})
            
            fake_chains = ev_data.get('fake_agent_paths', [])[:MAX_PATHS]
            real_chains = ev_data.get('real_agent_paths', [])[:MAX_PATHS]
            
            fake_nodes_data = [self._process_path_chain(p, raw_data, post_lookup)[:MAX_PATH_LEN] for p in fake_chains]
            real_nodes_data = [self._process_path_chain(p, raw_data, post_lookup)[:MAX_PATH_LEN] for p in real_chains]
            
            # ID Normalization for internal knowledge lookup
            internal_emb = self.internal_cache.get(news_id)
            if internal_emb is None:
                norm_id = news_id.replace("-", "").lower()
                internal_emb = self.internal_cache.get(norm_id)
            if internal_emb is None:
                json_id = str(raw_data.get('id', '')).replace("-", "").lower()
                internal_emb = self.internal_cache.get(json_id)
            if internal_emb is None:
                internal_emb = torch.zeros(3584)
            
            source = "fakeddit"
            if "politifact" in fpath: source = "politifact"
            elif "gossipcop" in fpath: source = "gossipcop"
            
            return {
                "id": news_id, "source": source,
                "title_text": news_title, "text_content": news_text,
                "label": label,
                "fake_paths": fake_nodes_data, "real_paths": real_nodes_data,
                "internal_emb": internal_emb
            }
        except Exception as e:
            print(f"Error loading index {idx} ({fpath}): {e}")
            return self.__getitem__((idx + 1) % len(self))

    @staticmethod
    def collate_fn(batch):
        tokenizer = get_tokenizer()
        clip_processor = get_clip_processor()

        ids = [x['id'] for x in batch]
        sources = [x['source'] for x in batch]
        labels = torch.tensor([x['label'] for x in batch], dtype=torch.long)
        internal_embs = torch.stack([torch.as_tensor(x['internal_emb']) for x in batch])
        
        titles = [x['title_text'] for x in batch]
        news_enc = tokenizer(titles, padding=True, truncation=True, max_length=MAX_TEXT_LEN, return_tensors="pt")
        
        def prepare_path_tensor_inputs(paths_batch_list):
            batch_size = len(paths_batch_list)
            flat_texts, flat_images = [], []
            mask = torch.zeros(batch_size, MAX_PATHS, MAX_PATH_LEN, dtype=torch.float)
            user_tensor = torch.zeros(batch_size, MAX_PATHS, MAX_PATH_LEN, USER_FEAT_DIM)
            txt_ptr, img_ptr = 0, 0
            text_indices, img_indices = [], []
            
            for b, paths in enumerate(paths_batch_list):
                for p, nodes in enumerate(paths):
                    if p >= MAX_PATHS: break
                    for n, node in enumerate(nodes):
                        if n >= MAX_PATH_LEN: break
                        mask[b, p, n] = 1.0
                        user_tensor[b, p, n] = torch.tensor(node['user_feat'])
                        if node['type'] == 'image':
                            flat_images.append(node['content'])
                            img_indices.append((b, p, n, img_ptr)); img_ptr += 1
                        else:
                            txt_content = str(node['content'])
                            flat_texts.append(txt_content)
                            text_indices.append((b, p, n, txt_ptr)); txt_ptr += 1
                            
            text_inputs = tokenizer(flat_texts, padding=True, truncation=True, max_length=64, return_tensors="pt") if flat_texts else None
            image_inputs = clip_processor(images=flat_images, return_tensors="pt") if flat_images else None
            return {"text_inputs": text_inputs, "image_inputs": image_inputs, "user_tensor": user_tensor, "mask": mask, "text_indices": text_indices, "img_indices": img_indices}

        fake_batch_data = prepare_path_tensor_inputs([x['fake_paths'] for x in batch])
        real_batch_data = prepare_path_tensor_inputs([x['real_paths'] for x in batch])
        
        return {
            "ids": ids, "sources": sources, "labels": labels,
            "news_input_ids": news_enc['input_ids'], "news_attention_mask": news_enc['attention_mask'],
            "internal_embs": internal_embs,
            "fake_path_data": fake_batch_data, "real_path_data": real_batch_data,
            "news_titles": titles, "news_texts": [x['text_content'] for x in batch] 
        }