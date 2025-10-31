# ========================================
# Step 1: Gemini 生成追问 (Follow-up Question)
# ========================================
FOLLOWUP_SYSTEM = """You are a careful follow-up question generator.
Given a conversation history and the last Q&A turn, generate ONE concise, helpful follow-up question that:
- is consistent with the conversation history
- stays on-topic and advances understanding
- demonstrates contextual understanding and relevance
- avoids simple yes/no questions unless actionable
- does not repeat previous questions
- asks for exactly one specific thing

Return ONLY valid JSON format: {"followup": "<your_question>"}
No extra text or explanations."""

FOLLOWUP_USER_TMPL = """Conversation History (oldest → newest):
{history_str}

Latest Turn:
Question: {q}
Initial Answer: {a}
Corrected Answer: {c}

Task: Based on the conversation history and the latest Q&A, generate a natural follow-up question that advances the discussion.
Return format: {{"followup": "<your_question>"}}"""

# ========================================
# Step 2: Llama 生成回答 (Answer)
# ========================================
LLAMA_SYSTEM = """You are a helpful assistant. Provide detailed, comprehensive answers that:
- length should no more than 500 characters
- Maintain consistency with the conversation history
- Give practical and helpful information
"""

# ========================================
# Step 3: Gemini 生成修正 (Correction)
# ========================================
CORRECTION_SYSTEM = """You are an expert answer corrector and improver.
Given a conversation history, a question, and an initial answer from a model, your task is to:
1. Identify any inaccuracies, incompleteness, or areas for improvement
2. Generate a CORRECTED and IMPROVED answer that is:
   - More accurate and factually correct
   - More complete and comprehensive
   - Better aligned with user expectations
   - More helpful and actionable
   - length should no more than 500 characters

The corrected answer should maintain the helpful tone while addressing any flaws in the original answer.

Return ONLY valid JSON format: {"correction": "<improved_answer>"}
No extra text or explanations."""

CORRECTION_USER_TMPL = """Conversation History (oldest → newest):
{history_str}

Current Question: {question}
Initial Answer from Model: {answer}

Task: Analyze the initial answer and provide a corrected, improved version that addresses any issues and better serves the user's needs.
Make your correction length should no more than 500 characters.
Return format: {{"correction": "<improved_answer>"}}"""
