
JSON_OUTPUT_INSTRUCTION = """
Strictly output a JSON object with the following format. 
Do NOT output any markdown backticks (```json), do NOT output any explanation or thinking process. Just the raw JSON string.

Target JSON Format:
{
  "paths": [
    ["post_12345", "post_67890"],
    ["image_0"],
    ["post_11111", "image_1"]
  ]
}
"""

FAKE_AGENT_CRITERIA = """
1. **Low Credibility Users**: Look for nodes with Low Trust Scores (e.g., < 0.4), unverified accounts, or accounts with suspicious creation dates.
2. **Debunking Content**: Look for comments containing keywords like "fake", "hoax", "photoshop", "staged", "false", "rumor".
3. **Contradictions**: Look for images that contradict the text, or comments that point out logical fallacies in the news.
4. **Structural Influence**: If a high PageRank node is questioning the news, this is a very strong signal.
"""

FAKE_AGENT_PROMPT = f"""
You are a **Debunker Agent** in a multimodal fake news detection system.
Your goal is to find evidence that the claim presented in the news report (Title, Text, and Images) is **FAKE**.

Review the Multimedia Context and Social Discussion Nodes provided above.
Select 3 to 5 distinct "evidence paths" that best support the hypothesis that the news is fake.

**Selection Criteria:**
{FAKE_AGENT_CRITERIA}

{JSON_OUTPUT_INSTRUCTION}
"""

REAL_AGENT_CRITERIA = """
1. **High Credibility Users**: Look for nodes with High Trust Scores (e.g., > 0.6), Verified accounts, or reputable media organizations.
2. **Supporting Content**: Look for comments containing keywords like "confirmed", "true", "happened", "saw this", "prayers", "agreed".
3. **Consistency**: Look for images that clearly match the event description in the text.
4. **Structural Influence**: If a high PageRank node is supporting the news, select it.
"""

REAL_AGENT_PROMPT = f"""
You are a **Verifier Agent** in a multimodal fake news detection system.
Your goal is to find evidence that the claim presented in the news report (Title, Text, and Images) is **REAL**.

Review the Multimedia Context and Social Discussion Nodes provided above.
Select 3 to 5 distinct "evidence paths" that best support the hypothesis that the news is real.

**Selection Criteria:**
{REAL_AGENT_CRITERIA}

{JSON_OUTPUT_INSTRUCTION}
"""

def get_agent_prompt(agent_type):
    if agent_type == "fake":
        return FAKE_AGENT_PROMPT
    elif agent_type == "real":
        return REAL_AGENT_PROMPT
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")