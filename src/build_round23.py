import os, json, time, random
from pathlib import Path
from typing import List, Dict
from tenacity import retry, wait_exponential_jitter, stop_after_attempt
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

import google.generativeai as genai
from prompts import (
    FOLLOWUP_SYSTEM, FOLLOWUP_USER_TMPL,
    LLAMA_SYSTEM,
    CORRECTION_SYSTEM, CORRECTION_USER_TMPL
)

# ====== 加载 .env 文件 ======
load_dotenv()

# ====== 配置参数 ======
NUM_ROUNDS = 2  # 🎯 要生成几轮对话（从第2轮开始算）
INP = Path("data/round1/round1_history_seed.jsonl")

# ====== Gemini 初始化 (用于生成 followup 和 correction) ======
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_KEY:
    raise RuntimeError("❌ Please set GEMINI_API_KEY in .env file")
genai.configure(api_key=GEMINI_KEY)
GEMINI_MODEL = "gemini-2.5-flash-lite"

# ====== Together AI 初始化 (用于生成 answer) ======
TOGETHER_KEY = os.environ.get("TOGETHER_API_KEY")
if not TOGETHER_KEY:
    raise RuntimeError("❌ Please set TOGETHER_API_KEY in .env file")
together_client = OpenAI(
    api_key=TOGETHER_KEY,
    base_url="https://api.together.xyz/v1"
)
LLAMA_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"


# 这里把历史保留 question、answer 和 correction
def _rounds_to_str(rounds: List[dict]) -> str:
    """将 rounds 列表转为对话历史字符串（包含 question, answer, correction）"""
    lines = []
    for r in rounds:
        q = (r.get("question") or "").strip()
        a = (r.get("answer") or "").strip()
        c = (r.get("correction") or "").strip()
        if q:
            lines.append(f"User: {q}")
        if a:
            lines.append(f"Assistant: {a}")
        if c:
            lines.append(f"Assistant: {c}")
    return "\n".join(lines[-16:])  # 保留最近16条消息（约8轮对话）



def _rounds_to_messages(rounds: List[dict]) -> List[Dict]:
    """将 rounds 转为 OpenAI 格式的消息列表（包含 question, answer, correction）"""
    messages = []
    for r in rounds:
        q = (r.get("question") or "").strip()
        a = (r.get("answer") or "").strip()
        c = (r.get("correction") or "").strip()
        if q:
            messages.append({"role": "user", "content": q})
        if a:
            messages.append({"role": "assistant", "content": a})
        if c:
            messages.append({"role": "assistant", "content": c})
    return messages[-16:]  # 保留最近16条消息



@retry(wait=wait_exponential_jitter(initial=1, max=12), stop=stop_after_attempt(6))
def generate_followup(history_str: str, q: str, a: str, c: str) -> str:
    """用 Gemini 生成追问"""
    user_msg = FOLLOWUP_USER_TMPL.format(history_str=history_str, q=q, a=a, c=c)
    
    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=FOLLOWUP_SYSTEM)
    resp = model.generate_content(
        contents=[{"role": "user", "parts": [user_msg]}],
        generation_config={
            "temperature": 0.7,
            "top_p": 0.9,
            "max_output_tokens": 150,
            "response_mime_type": "application/json",
        },
        safety_settings=None,
    )

    txt = (resp.text or "").strip()
    if txt.startswith("```"):
        txt = txt.strip("`")
        if "\n" in txt:
            txt = txt.split("\n", 1)[1].rsplit("\n", 1)[0].strip()

    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        if txt.lower().startswith("followup"):
            data = {"followup": txt.split(":", 1)[-1].strip()}
        else:
            raise

    followup = (data.get("followup") or "").strip()
    if not followup:
        raise ValueError("Empty followup")
    return followup

@retry(wait=wait_exponential_jitter(initial=1, max=12), stop=stop_after_attempt(6))
def generate_answer(history_messages: List[Dict], question: str) -> str:
    """用 Llama 基于 history 生成回答"""
    # 使用 prompts.py 中定义的系统提示
    system_msg = {"role": "system", "content": LLAMA_SYSTEM}
    messages = [system_msg] + history_messages + [{"role": "user", "content": question}]
    
    resp = together_client.chat.completions.create(
        model=LLAMA_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=300,  # 约500字符（1字符≈1.5-2tokens）
    )
    
    answer = resp.choices[0].message.content.strip()
    if not answer:
        raise ValueError("Empty answer")
    return answer

@retry(wait=wait_exponential_jitter(initial=1, max=12), stop=stop_after_attempt(6))
def generate_correction(history_str: str, question: str, answer: str) -> str:
    """用 Gemini 生成修正后的回答"""
    user_msg = CORRECTION_USER_TMPL.format(
        history_str=history_str,
        question=question,
        answer=answer
    )
    
    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=CORRECTION_SYSTEM)
    resp = model.generate_content(
        contents=[{"role": "user", "parts": [user_msg]}],
        generation_config={
            "temperature": 0.6,
            "top_p": 0.9,
            "max_output_tokens": 300,  # 增加到300，允许详细的修正回答（2-4句话）
            "response_mime_type": "application/json",
        },
        safety_settings=None,
    )

    txt = (resp.text or "").strip()
    if txt.startswith("```"):
        txt = txt.strip("`")
        if "\n" in txt:
            txt = txt.split("\n", 1)[1].rsplit("\n", 1)[0].strip()

    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        # 兜底：直接返回原答案
        return answer

    correction = (data.get("correction") or "").strip()
    if not correction:
        return answer  # 兜底
    return correction

def main():
    """迭代生成多轮对话"""
    # 读取初始数据
    with INP.open("r", encoding="utf-8") as f:
        initial_data = [json.loads(line) for line in f]
    
    print(f"📚 Loaded {len(initial_data)} samples from {INP}")
    print(f"🎯 Will generate {NUM_ROUNDS} additional rounds\n")
    
    # 迭代生成每一轮
    for round_num in range(2, 2 + NUM_ROUNDS):
        print(f"\n{'='*60}")
        print(f"🔄 Generating Round {round_num}")
        print(f"{'='*60}\n")
        
        # 输出路径
        out_dir = Path(f"data/round{round_num}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_history = out_dir / f"round{round_num}_history.jsonl"
        out_single = out_dir / f"round{round_num}_single_turn.jsonl"
        
        new_data = []
        
        with out_history.open("w", encoding="utf-8") as f_hist, \
             out_single.open("w", encoding="utf-8") as f_single:
            
            for ex in tqdm(initial_data, desc=f"Round {round_num}"):
                rid = ex.get("id", "")
                rounds = ex.get("rounds", []) or []
                
                if not rounds:
                    continue
                
                # 获取最后一轮的 q, a, c
                last_round = rounds[-1]
                last_q = last_round.get("question", "") or ""
                last_a = last_round.get("answer", "") or ""
                last_c = last_round.get("correction", "") or ""
                
                # 构建历史上下文
                hist_str = _rounds_to_str(rounds[:-1]) if len(rounds) > 1 else ""
                hist_msgs = _rounds_to_messages(rounds)
                
                try:
                    # Step 1: Gemini 生成追问
                    followup_q = generate_followup(hist_str, last_q, last_a, last_c)
                    
                    # Step 2: Llama 生成回答
                    followup_a = generate_answer(hist_msgs, followup_q)
                    
                    # Step 3: Gemini 生成修正
                    full_hist_str = _rounds_to_str(rounds)  # 包含所有历史
                    followup_c = generate_correction(full_hist_str, followup_q, followup_a)
                    
                except Exception as e:
                    print(f"⚠️  Error for {rid}: {type(e).__name__}: {str(e)[:100]}")
                    print(f"   Using fallback responses...")
                    followup_q = "Could you provide more details?"
                    followup_a = "I need more context to answer properly."
                    followup_c = "Let me know what specific aspect you'd like me to clarify."
                
                # 构建新一轮
                new_round = {
                    "question": followup_q,
                    "answer": followup_a,
                    "correction": followup_c
                }
                
                # 追加到 rounds
                updated_rounds = rounds + [new_round]
                
                # 保存完整 history
                f_hist.write(json.dumps({
                    "id": rid,
                    "rounds": updated_rounds
                }, ensure_ascii=False) + "\n")
                
                # 保存单轮数据
                f_single.write(json.dumps({
                    "id": rid,
                    **new_round
                }, ensure_ascii=False) + "\n")
                
                # 更新数据供下一轮使用
                new_data.append({"id": rid, "rounds": updated_rounds})
                
                # 温和速率
                time.sleep(random.uniform(0.1, 0.3))
        
        print(f"\n✅ Round {round_num} completed!")
        print(f"   📄 History: {out_history}")
        print(f"   📄 Single turn: {out_single}")
        
        # 更新 initial_data 供下一轮迭代
        initial_data = new_data
    
    print(f"\n{'='*60}")
    print(f"🎉 All {NUM_ROUNDS} rounds generated successfully!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
