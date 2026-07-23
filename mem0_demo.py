"""
Mem0 最小 Demo
运行前设置：$env:MEM0_API_KEY="m0-xxx..."
注册：https://app.mem0.ai（免费 tier 足够）
"""
import os
from mem0 import MemoryClient

client = MemoryClient(api_key=os.environ["MEM0_API_KEY"])
USER_ID = "drugforge-demo"

# ── 1. 添加记忆 ──────────────────────────────────────────────────────────────
print("=== 1. add() ===")
msgs = [
    {"role": "user",      "content": "我上次做DPP4靶点时，更倾向于优化口服生物利用度，logP在2-3之间的分子表现最好"},
    {"role": "assistant", "content": "好的，记住你对DPP4靶点先导化合物的优化偏好：优先口服生物利用度，logP 2-3"},
]
result = client.add(msgs, user_id=USER_ID)
print(result)

# ── 2. 语义搜索 ──────────────────────────────────────────────────────────────
print("\n=== 2. search() ===")
hits = client.search("用户对先导化合物优化有什么偏好？", user_id=USER_ID)
for h in hits:
    print(f"  [{h['score']:.2f}] {h['memory']}")

# ── 3. 获取全部 ──────────────────────────────────────────────────────────────
print("\n=== 3. get_all() ===")
all_mem = client.get_all(user_id=USER_ID)
print(f"  共 {len(all_mem)} 条记忆")
for m in all_mem:
    print(f"  - {m['memory']}")
