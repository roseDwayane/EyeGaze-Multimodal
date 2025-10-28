"""
驗證生成的 complete_metadata.json 檔案
檢查以下項目：
1. Pair 18 是否被排除
2. 各個 class 的數量分布
3. Pair 的範圍是否正確 (12-40, 排除18)
4. 資料完整性檢查
"""

import json
from collections import Counter

# 讀取生成的 JSON 檔案
with open("./complete_metadata.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("=== 資料驗證報告 ===\n")

# 1. 檢查 Pair 18
pairs = [item["pair"] for item in data]
has_pair_18 = 18 in pairs
print(f"1. Pair 18 是否被排除: {'[X] 失敗 (發現Pair 18)' if has_pair_18 else '[O] 成功 (已排除Pair 18)'}")

# 2. Pair 範圍檢查
unique_pairs = sorted(set(pairs))
print(f"\n2. Pair 範圍:")
print(f"   - 最小 Pair: {min(pairs)}")
print(f"   - 最大 Pair: {max(pairs)}")
print(f"   - Pair 總數: {len(unique_pairs)}")
print(f"   - Pair 列表: {unique_pairs}")
expected_pairs = [p for p in range(12, 41) if p != 18]
missing_pairs = set(expected_pairs) - set(unique_pairs)
extra_pairs = set(unique_pairs) - set(expected_pairs)
if missing_pairs:
    print(f"   - [!] 缺少的 Pair: {sorted(missing_pairs)}")
if extra_pairs:
    print(f"   - [!] 額外的 Pair: {sorted(extra_pairs)}")
if not missing_pairs and not extra_pairs:
    print(f"   - [O] Pair 範圍正確")

# 3. Class 分布
classes = [item["class"] for item in data]
class_counter = Counter(classes)
print(f"\n3. Class 分布:")
for class_name, count in sorted(class_counter.items()):
    print(f"   - {class_name}: {count} 筆")

# 4. 資料完整性檢查
print(f"\n4. 資料完整性:")
print(f"   - 總資料筆數: {len(data)}")

# 檢查是否有空值
empty_formal = sum(1 for item in data if not item.get("formal_sen"))
empty_lively = sum(1 for item in data if not item.get("lively_sen"))
print(f"   - 空的 formal_sen: {empty_formal} 筆")
print(f"   - 空的 lively_sen: {empty_lively} 筆")

# 5. 每個 Pair 的資料數量
print(f"\n5. 每個 Pair 的資料筆數:")
pair_counter = Counter(pairs)
for pair, count in sorted(pair_counter.items())[:5]:  # 只顯示前5個
    print(f"   - Pair {pair}: {count} 筆")
print(f"   ... (共 {len(unique_pairs)} 個 Pair)")

# 檢查每個 pair 的資料數量是否一致
pair_counts = list(pair_counter.values())
if len(set(pair_counts)) == 1:
    print(f"   - [O] 所有 Pair 的資料數量一致 (每個 {pair_counts[0]} 筆)")
else:
    print(f"   - [!] Pair 的資料數量不一致")
    print(f"     最少: {min(pair_counts)} 筆, 最多: {max(pair_counts)} 筆")

# 6. 範例資料展示
print(f"\n6. 第一筆資料範例:")
example = data[0]
print(f"   Pair: {example['pair']}")
print(f"   Player1: {example['player1']}")
print(f"   Player2: {example['player2']}")
print(f"   Class: {example['class']}")
print(f"   Formal_sen: {example['formal_sen'][:80]}...")
print(f"   Lively_sen: {example['lively_sen'][:80]}...")

print("\n=== 驗證完成 ===")
