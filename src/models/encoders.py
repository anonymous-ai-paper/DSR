import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, CLIPVisionModel
import os
import sys
from config import CLIP_PATH, ROBERTA_PATH

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

EMBED_DIM = 768  # Target dimension alignment (RoBERTa default)

class FeatureEncoders(nn.Module):
    def __init__(self, freeze_backbones=True, device="cuda"):
        super().__init__()
        self.device = device
        
        # 1. Text Encoder (RoBERTa)
        print(f"Loading Text Encoder from: {ROBERTA_PATH}...")
        self.text_encoder = AutoModel.from_pretrained(ROBERTA_PATH)
        
        # 2. Image Encoder (CLIP)
        print(f"Loading Image Encoder from: {CLIP_PATH}...")
        self.image_encoder = CLIPVisionModel.from_pretrained(CLIP_PATH)
        
        # [Crucial Fix] CLIP ViT-Base pooler_output is 768-dim.
        self.image_projector = nn.Linear(768, EMBED_DIM)
        
        # 3. User Encoder (Metadata MLP)
        self.user_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, EMBED_DIM)
        )
        
        # Freeze backbones to save VRAM
        if freeze_backbones:
            print("Freezing Backbone Parameters (RoBERTa & CLIP)...")
            for p in self.text_encoder.parameters(): p.requires_grad = False
            for p in self.image_encoder.parameters(): p.requires_grad = False

    def forward_news(self, input_ids, attention_mask):
        """
        Encode the news title/content to serve as the Query for the Attention mechanism.
        Output shape: [Batch, 768]
        """
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token as the sentence representation
        return outputs.last_hidden_state[:, 0, :]

    def forward_nodes(self, path_data_dict):
        """
        Core logic: Reconstruct 4D node tensors from the heterogeneous dictionary batch.
        Input: path_data_dict (from collate_fn)
        Output: node_embeddings [Batch, Max_Paths, Max_Nodes, 768]
        """
        # Unpack data from batch dictionary
        text_inputs = path_data_dict['text_inputs']
        image_inputs = path_data_dict['image_inputs']
        user_tensor = path_data_dict['user_tensor'].to(self.device)
        mask = path_data_dict['mask'].to(self.device)
        
        text_indices = path_data_dict['text_indices'] # List of (b, p, n, flat_idx)
        img_indices = path_data_dict['img_indices']   # List of (b, p, n, flat_idx)
        
        # 1. Initialize base embeddings using User Metadata
        # Shape transition: [B, MP, MN, 4] -> [B, MP, MN, 768]
        node_embeddings = self.user_encoder(user_tensor)
        
        # 2. Inject text features (RoBERTa)
        if text_inputs is not None and len(text_indices) > 0:
            ti_ids = text_inputs['input_ids'].to(self.device)
            ti_mask = text_inputs['attention_mask'].to(self.device)
            
            # Forward pass through RoBERTa (Flattened) -> [N_txt, 768]
            txt_out = self.text_encoder(input_ids=ti_ids, attention_mask=ti_mask)
            txt_embeds = txt_out.last_hidden_state[:, 0, :] # Get [CLS] tokens
            
            # Scatter calculated text vectors into the 4D node_embeddings container
            b_idx, p_idx, n_idx, flat_idx = zip(*text_indices)
            node_embeddings[b_idx, p_idx, n_idx] += txt_embeds

        # 3. Inject image features (CLIP)
        if image_inputs is not None and len(img_indices) > 0:
            pix_vals = image_inputs['pixel_values'].to(self.device)
            
            # Forward pass through CLIP Vision (Flattened) -> [N_img, 768]
            img_out = self.image_encoder(pixel_values=pix_vals)
            img_pool = img_out.pooler_output
            
            # Project 768 -> 768 (Alignment layer)
            img_embeds = self.image_projector(img_pool) 
            
            # Scatter visual vectors into the 4D container
            b_idx, p_idx, n_idx, flat_idx = zip(*img_indices)
            node_embeddings[b_idx, p_idx, n_idx] += img_embeds
            
        return node_embeddings

class PathEncoder(nn.Module):
    """
    Compresses a sequence of node embeddings [Seq_Len, Dim] into a single path vector [Dim].
    Uses GRU followed by an Attention Pooling layer.
    """
    def __init__(self, input_dim=768, hidden_dim=768):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.attn_linear = nn.Linear(hidden_dim, 1)
        
    def forward(self, paths_tensor, mask=None):
        """
        Args:
            paths_tensor: [Batch * Num_Paths, Seq_Len, Dim]
            mask: [Batch * Num_Paths, Seq_Len]
        Returns:
            context_vectors: [Batch * Num_Paths, Dim]
        """
        # GRU Forward pass
        output, _ = self.gru(paths_tensor)
        
        # Calculate attention scores for pooling
        scores = self.attn_linear(output)
        
        if mask is not None:
            # Mask padding nodes with the minimum possible value for the current dtype
            fill_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask.unsqueeze(-1) == 0, fill_value)
            
        # Softmax to get weights [B*NP, L, 1]
        weights = F.softmax(scores, dim=1)
        
        # Perform weighted sum to get the final path representation
        return torch.sum(output * weights, dim=1)