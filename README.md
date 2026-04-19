# Agent

一个基于 ReAct（Reasoning + Acting）框架的 AI 智能体，支持任务规划、工具调用和记忆管理。

## 项目结构

```
agent/
├── core.py        # 主循环：ReAct 推理-行动循环
├── planner.py     # 任务规划与分解
├── executor.py    # 工具调用执行器
├── memory.py      # 智能体记忆管理
└── __init__.py
```

## 工作原理

1. **Planner** 将用户目标分解为有序子任务
2. **AgentCore** 运行 ReAct 循环：调用 LLM → 执行工具 → 获取结果 → 继续推理
3. **Executor** 负责分发工具调用并收集结果
4. **AgentMemory** 在对话轮次间维护上下文状态

## 快速开始

```bash
pip install -r requirements.txt
python main.py
```

## 依赖

- Python 3.10+
- [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python)
- Rich（终端美化输出）
