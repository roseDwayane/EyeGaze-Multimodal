"""
整理所有資料成為一個JSON檔案，包含以下欄位：
1. "pair": 記錄這筆資料是第幾個pair (總共12~40，18剔除)
2. "player1": 紀錄第一位玩家的檔名 (不含副檔名)
3. "player2": 紀錄第二位玩家的檔名 (不含副檔名)
4. "class": 依據檔名給予互動類別(Single, Competition, Cooperation)
5. "formal_sen": 從正式描述JSON讀取
6. "lively_sen": 從生動描述JSON讀取

原始資料路徑說明：
- Image 原始資料: G:\共用雲端硬碟\CNElab_林佳誼_Gaze\B.GazeImage\01.data\bgOn_heatmapOn_trajOn
  使用方式: player1 + ".jpg" 或 player2 + ".jpg"
  例如: "Pair-12-A-Single-EYE_trial01_player.jpg"

- EEG 原始資料: G:\共用雲端硬碟\CNElab_林佳誼_Gaze\B.GazeImage\01.data\EEGseg
  使用方式: player1 + ".csv" 或 player2 + ".csv"
  例如: "Pair-12-A-Single-EYE_trial01_player.csv"

- Sentence 資料: G:\共用雲端硬碟\CNElab_林佳誼_Gaze\B.GazeImage\01.data\image_text_descriptions
  已整合至 formal_sen 和 lively_sen 欄位中
"""

import json
import os

# 設定資料路徑
FORMAL_JSON_PATH = "../raw/Sentence/formal_description_json.json"
LIVELY_JSON_PATH = "../raw/Sentence/lively_description_json.json"
OUTPUT_PATH = "./complete_metadata.json"

# 原始資料路徑（供參考）
IMAGE_DATA_PATH = r"G:\共用雲端硬碟\CNElab_林佳誼_Gaze\B.GazeImage\01.data\bgOn_heatmapOn_trajOn"
EEG_DATA_PATH = r"G:\共用雲端硬碟\CNElab_林佳誼_Gaze\B.GazeImage\01.data\EEGseg"
SENTENCE_DATA_PATH = r"G:\共用雲端硬碟\CNElab_林佳誼_Gaze\B.GazeImage\01.data\image_text_descriptions"

# 讀取 formal 和 lively 描述
print("讀取 formal 和 lively 描述...")
with open(FORMAL_JSON_PATH, "r", encoding="utf-8") as f:
    formal_data = json.load(f)

with open(LIVELY_JSON_PATH, "r", encoding="utf-8") as f:
    lively_data = json.load(f)

# 建立索引字典，用於快速查找
print("建立索引...")
lively_dict = {}
for item in lively_data:
    key = (item["pair"], item["image1"], item["image2"])
    lively_dict[key] = item["class"]

# 從檔名提取類別
def get_class_from_filename(image_name):
    """從檔名提取互動類別"""
    if "A-Single" in image_name or "B-Single" in image_name:
        return "Single"
    elif "Comp" in image_name:
        return "Competition"
    elif "Coop" in image_name:
        return "Cooperation"
    return "Unknown"

# 生成完整的資料結構
print("生成完整資料結構...")
complete_data = []
excluded_count = 0

for item in formal_data:
    pair = item["pair"]

    # 排除 Pair 18
    if pair == 18:
        excluded_count += 1
        continue

    image1 = item["image1"]
    image2 = item["image2"]
    formal_sen = item["class"]

    # 從 lively_dict 獲取對應的 lively 句子
    key = (pair, image1, image2)
    lively_sen = lively_dict.get(key, "")

    # 從檔名提取類別
    class_type = get_class_from_filename(image1)

    # 移除副檔名 (.jpg)
    player1 = image1.replace(".jpg", "")
    player2 = image2.replace(".jpg", "")

    # 組合新的資料結構
    complete_data.append({
        "pair": pair,
        "player1": player1,
        "player2": player2,
        "class": class_type,
        "formal_sen": formal_sen,
        "lively_sen": lively_sen
    })

print(f"已排除 Pair 18 的資料: {excluded_count} 筆")

# 儲存為JSON檔案
print(f"儲存資料到 {OUTPUT_PATH}...")
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(complete_data, f, ensure_ascii=False, indent=4)

print(f"完成！總共生成 {len(complete_data)} 筆資料。")
print(f"資料已儲存至: {OUTPUT_PATH}")

# 顯示前3筆資料作為預覽
print("\n前3筆資料預覽：")
for i, item in enumerate(complete_data[:3], 1):
    print(f"\n=== 資料 {i} ===")
    print(f"Pair: {item['pair']}")
    print(f"Player1: {item['player1']}")
    print(f"Player2: {item['player2']}")
    print(f"Class: {item['class']}")
    print(f"Formal: {item['formal_sen'][:100]}...")
    print(f"Lively: {item['lively_sen'][:100]}...")


# 提供輔助函數，方便獲取原始資料的完整路徑
def get_image_path(player_name, base_path=IMAGE_DATA_PATH):
    """
    從player名稱獲取image檔案的完整路徑

    Args:
        player_name: player1 或 player2 的檔名 (不含副檔名)
        base_path: 圖像資料的基礎路徑

    Returns:
        完整的圖像檔案路徑
    """
    return os.path.join(base_path, f"{player_name}.jpg")


def get_eeg_path(player_name, base_path=EEG_DATA_PATH):
    """
    從player名稱獲取EEG檔案的完整路徑

    Args:
        player_name: player1 或 player2 的檔名 (不含副檔名)
        base_path: EEG資料的基礎路徑

    Returns:
        完整的EEG檔案路徑
    """
    return os.path.join(base_path, f"{player_name}.csv")


# 使用範例
print("\n\n=== 使用範例 ===")
example_item = complete_data[0]
print(f"Player1 Image路徑: {get_image_path(example_item['player1'])}")
print(f"Player2 Image路徑: {get_image_path(example_item['player2'])}")
print(f"Player1 EEG路徑: {get_eeg_path(example_item['player1'])}")
print(f"Player2 EEG路徑: {get_eeg_path(example_item['player2'])}")
