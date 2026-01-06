import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT = os.path.join(PROJECT_ROOT, "data/processed/unified_json")
EVIDENCE_PATHS_DIR = os.path.join(PROJECT_ROOT, "data/processed/evidence_paths")
CACHE_PATH = os.path.join(PROJECT_ROOT, "data/processed/internal_knowledge_generated.pt")

PRETRAINED_DIR = os.path.join(PROJECT_ROOT, "pretrained_models")
LLM_PATH = os.path.join(PRETRAINED_DIR, "Qwen2.5-7B-Instruct")
ROBERTA_PATH = os.path.join(PRETRAINED_DIR, "roberta-base")
CLIP_PATH = os.path.join(PRETRAINED_DIR, "clip-vit-base-patch32")

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOG_DIR = os.path.join(PROJECT_ROOT, "logger")

GEMINI_API_KEY = "YOUR_API_KEY"