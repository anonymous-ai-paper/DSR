import torch
import torch.nn as nn
import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType

# --- Dynamic Path Configuration ---
# Add project root to sys.path to import config.py and local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import LLM_PATH
from src.models.encoders import FeatureEncoders, PathEncoder
from src.models.prompt import JUDGE_SYSTEM_PROMPT

class JudgeAgent(nn.Module):
    def __init__(self, llm_path=LLM_PATH, device="cuda", lora_r=16, lora_alpha=32, lora_dropout=0.05):
        super().__init__()
        self.device = device
        
        # --- 1. Base Encoders (Text/Image/User) ---
        self.feature_encoders = FeatureEncoders(freeze_backbones=True, device=device)
        
        # --- 2. Path Encoder (Sequence to Vector) ---
        self.path_encoder = PathEncoder(input_dim=768, hidden_dim=768)
        
        # --- 3. Aggregation Layer (News Query -> Path Keys) ---
        self.agg_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        
        # --- 4. Load 4-bit Quantized LLM ---
        print(f"Loading LLM from {llm_path} in 4-bit...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16, # Recommended for TITAN RTX
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            quantization_config=bnb_config,
            device_map={"": device}, # Pin model to specific GPU
            trust_remote_code=True
        )

        # [Critical Fix] Disable cache for gradient checkpointing compatibility
        self.llm.config.use_cache = False  
        
        # Enable gradient checkpointing with non-reentrant mode to resolve DDP conflicts
        self.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
        
        # --- 5. LoRA Configuration ---
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        self.llm = get_peft_model(self.llm, peft_config)
        self.llm.enable_input_require_grads() # Required for custom inputs_embeds
        
        # --- 6. Dimension Projectors ---
        self.llm_dim = self.llm.config.hidden_size
        
        # Project 768-dim evidence to LLM latent space
        self.evidence_projector = nn.Sequential(
            nn.Linear(768, self.llm_dim),
            nn.LayerNorm(self.llm_dim),
            nn.GELU()
        )
        
        # Project LLM internal knowledge hidden states
        self.internal_projector = nn.Sequential(
            nn.Linear(self.llm_dim, self.llm_dim),
            nn.LayerNorm(self.llm_dim),
            nn.GELU()
        )
        
        # --- 7. Classification Head ---
        self.classifier = nn.Sequential(
            nn.Linear(self.llm_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 2)
        )
        self.aux_classifier = nn.Linear(768, 2)
        
        # --- 8. Pre-compute System Prompt Embeddings ---
        prompt_ids = self.llm_tokenizer(JUDGE_SYSTEM_PROMPT, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            # Registered as buffer for DDP synchronization
            self.register_buffer("system_prompt_embeds", self.llm.get_input_embeddings()(prompt_ids))

    def encode_paths_batch(self, path_data):
        """Encode a batch of heterogeneous graph paths into vectors."""
        node_embeds = self.feature_encoders.forward_nodes(path_data)
        B, MP, MN, D = node_embeds.shape
        flat_nodes = node_embeds.view(B * MP, MN, D)
        mask = path_data['mask'].to(self.device).view(B * MP, MN)
        path_vecs = self.path_encoder(flat_nodes, mask)
        return path_vecs.view(B, MP, D)

    def forward(self, news_input_ids, news_attn_mask, fake_path_data, real_path_data, internal_embs):
        """
        Forward pass with Dialectical reasoning.
        Combines Internal Knowledge + Fake Evidence + Real Evidence.
        """
        batch_size = news_input_ids.shape[0]
        target_dtype = self.system_prompt_embeds.dtype
        
        # 1. Feature Encoding
        news_feat = self.feature_encoders.forward_news(news_input_ids, news_attn_mask)
        fake_path_vecs = self.encode_paths_batch(fake_path_data)
        real_path_vecs = self.encode_paths_batch(real_path_data)
        
        # 2. Aggregation via Multi-head Attention
        query = news_feat.unsqueeze(1)
        e_fake, fake_attn_weights = self.agg_attention(query, fake_path_vecs, fake_path_vecs)
        e_real, real_attn_weights = self.agg_attention(query, real_path_vecs, real_path_vecs)
        e_fake = e_fake.squeeze(1)
        e_real = e_real.squeeze(1)
        
        # 3. Auxiliary Classification (Evidence Supervision)
        aux_logits_fake = self.aux_classifier(e_fake)
        aux_logits_real = self.aux_classifier(e_real)
        
        # 4. LLM Input Construction (Project & Align)
        proj_fake = self.evidence_projector(e_fake).to(dtype=target_dtype)
        proj_real = self.evidence_projector(e_real).to(dtype=target_dtype)
        proj_internal = self.internal_projector(internal_embs.to(self.device, dtype=target_dtype))
        
        # Concatenate: [System Prompt, Internal, Fake, Real]
        embeds_list = [
            self.system_prompt_embeds.expand(batch_size, -1, -1),
            proj_internal.unsqueeze(1),
            proj_fake.unsqueeze(1),
            proj_real.unsqueeze(1)
        ]
        inputs_embeds = torch.cat(embeds_list, dim=1)
        
        # 5. LLM Reasoning Pass
        outputs = self.llm(
            inputs_embeds=inputs_embeds, 
            output_hidden_states=True,
            use_cache=False
        )
        
        # 6. Final Logits Extraction
        # Cast back to float32 before classifier for stability
        last_hidden = outputs.hidden_states[-1][:, -1, :].to(dtype=torch.float32)
        logits = self.classifier(last_hidden)
        
        path_weights = {
            "fake": fake_attn_weights.squeeze(1).detach().cpu(),
            "real": real_attn_weights.squeeze(1).detach().cpu()
        }
        
        return logits, aux_logits_fake, aux_logits_real, path_weights

    def generate_explanation(self, news_input_ids, news_attn_mask, fake_path_data, real_path_data, internal_embs):
        """Inference mode: Generate natural language explanation for the judgment."""
        self.eval()
        batch_size = news_input_ids.shape[0]
        target_dtype = self.system_prompt_embeds.dtype
        
        with torch.no_grad():
            news_feat = self.feature_encoders.forward_news(news_input_ids, news_attn_mask)
            fake_path_vecs = self.encode_paths_batch(fake_path_data)
            real_path_vecs = self.encode_paths_batch(real_path_data)
            
            query = news_feat.unsqueeze(1)
            e_fake, _ = self.agg_attention(query, fake_path_vecs, fake_path_vecs)
            e_real, _ = self.agg_attention(query, real_path_vecs, real_path_vecs)
            
            proj_fake = self.evidence_projector(e_fake.squeeze(1)).to(dtype=target_dtype)
            proj_real = self.evidence_projector(e_real.squeeze(1)).to(dtype=target_dtype)
            proj_internal = self.internal_projector(internal_embs.to(self.device, dtype=target_dtype))

            embeds_list = [
                self.system_prompt_embeds.expand(batch_size, -1, -1),
                proj_internal.unsqueeze(1),
                proj_fake.unsqueeze(1),
                proj_real.unsqueeze(1)
            ]
            inputs_embeds = torch.cat(embeds_list, dim=1)
            
            # Append trigger text to guide the reasoning output
            expl_trigger_text = "\n\nAnalysis & Final Conclusion:"
            expl_trigger = self.llm_tokenizer(expl_trigger_text, return_tensors="pt").input_ids.to(self.device)
            expl_embeds = self.llm.get_input_embeddings()(expl_trigger).expand(batch_size, -1, -1)
            gen_inputs = torch.cat([inputs_embeds, expl_embeds], dim=1)
            
            generated_ids = self.llm.generate(
                inputs_embeds=gen_inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.llm_tokenizer.eos_token_id
            )
            
            return self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)