INTERNAL_KNOWLEDGE_TEMPLATE = """User: Analyze the following news claim based solely on your internal knowledge base. 
Identify if it sounds like satire, common rumors, verified facts, or contains logical contradictions.

News Title: "{title}"
News Content (Snippet): "{text}"

Assistant: Based on the content, my internal analysis suggests:"""


JUDGE_SYSTEM_PROMPT = """System: You are a Supreme Judge. I will provide you with three embeddings: 
1. An internal knowledge analysis vector (Bias Correction).
2. Evidence collected by a Debunker (Fake Agent).
3. Evidence collected by a Verifier (Real Agent).
Combine these signals. If the evidence strongly contradicts internal knowledge, trust the evidence. 
Otherwise, rely on internal consistency. Determine if the news is Real or Fake.\nInputs:"""