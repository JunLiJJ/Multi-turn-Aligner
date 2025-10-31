# Multi-turn Dialogue Dataset Generator

自动生成多轮对话数据集，使用 Gemini + Llama 的混合方案。

## 🎯 目标

从 aligner-20K 单轮数据集生成 3 轮标准的问答修正数据，最终形状：`[question, answer, correction].shape = (3, 3, 3)`

## 📊 数据流程

```
Round 1 (种子数据)
  ↓
Round 2 (生成第2轮)
  ↓
Round 3 (生成第3轮)
```

### 每轮生成流程

1. **Gemini** → 生成追问 (followup question)
2. **Llama 3.1 8B** → 基于 history 生成回答 (answer)
3. **Gemini** → 根据 history + Q&A 生成修正 (correction)

## 🚀 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 设置 API Keys

```bash
export GEMINI_API_KEY="your-gemini-api-key"
export TOGETHER_API_KEY="your-together-api-key"
```

### 3. 准备第1轮数据

```bash
python src/prepare_round1.py
```

生成：
- `data/round1/round1_single_turn.jsonl` - 单轮三元组
- `data/round1/round1_history_seed.jsonl` - 格式化为 rounds 列表

### 4. 生成第2、3轮

```bash
python src/build_round23.py
```

在脚本开头修改 `NUM_ROUNDS` 参数：
```python
NUM_ROUNDS = 2  # 生成2轮（round2 和 round3）
```

生成：
- `data/round2/round2_history.jsonl` - 完整的2轮对话历史
- `data/round2/round2_single_turn.jsonl` - 只有第2轮的 Q&A&C
- `data/round3/round3_history.jsonl` - 完整的3轮对话历史
- `data/round3/round3_single_turn.jsonl` - 只有第3轮的 Q&A&C

## 📁 数据格式

### History 文件格式

```json
{
  "id": "r1-000001",
  "rounds": [
    {
      "question": "高速公路开多快？",
      "answer": "随便开",
      "correction": "55-70英里/小时，具体看路标"
    },
    {
      "question": "雨天要减速多少？",
      "answer": "稍微慢点",
      "correction": "雨天建议降低10-15英里/小时"
    }
  ]
}
```

### Single Turn 文件格式

```json
{
  "id": "r1-000001",
  "question": "雨天要减速多少？",
  "answer": "稍微慢点",
  "correction": "雨天建议降低10-15英里/小时"
}
```

## ⚙️ 配置

### 模型选择

**build_round23.py**:
- `GEMINI_MODEL = "gemini-2.5-flash-lite"` - 最便宜的 Gemini 模型
- `LLAMA_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"` - Together AI 上的 Llama

### 轮数控制

**build_round23.py**:
```python
NUM_ROUNDS = 2  # 要生成几轮（2 表示生成 round2 和 round3）
```

### 采样数量

**prepare_round1.py**:
```python
N = 200  # 取200条数据，设为 None 则取全部
```

## 💰 成本优化

- 使用最便宜的 Gemini 模型
- 限制上下文长度（最近16条消息）
- 限制输出 token 数量
- 自动重试机制（避免浪费）
- 温和的 API 调用速率

## 📦 项目结构

```
.
├── data/
│   ├── round1/
│   │   ├── round1_history_seed.jsonl
│   │   └── round1_single_turn.jsonl
│   ├── round2/
│   │   ├── round2_history.jsonl
│   │   └── round2_single_turn.jsonl
│   └── round3/
│       ├── round3_history.jsonl
│       └── round3_single_turn.jsonl
├── src/
│   ├── prepare_round1.py     # 准备第1轮数据
│   ├── build_round23.py       # 生成第2、3轮
│   └── prompts.py             # Prompt 模板
└── requirements.txt
```

## 🔧 故障排除

### SSL 证书错误

```bash
pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

### API 限流

如果遇到 429 错误，脚本会自动重试。可以调整 sleep 时间：

```python
time.sleep(random.uniform(0.1, 0.3))  # 调大这个值
```

## 📝 注意事项

1. **成本控制**：生成 200 条 × 2 轮数据，预计成本约 $1-2
2. **时间估算**：每轮约需 10-15 分钟（200 条数据）
3. **API Keys**：确保两个 API key 都有额度
4. **数据质量**：Gemini 负责 question 和 correction（质量高），Llama 负责 answer（故意不完美）

## 🎉 最终输出

运行完成后，你会得到：
- **3轮完整对话历史**
- **每轮单独的 Q&A&C 数据**
- **总共 200 × 3 = 600 条问答对**

形状符合要求：`[question, answer, correction].shape = (3, 3, 3)`

