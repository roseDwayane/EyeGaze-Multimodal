# Dual EEG Transformer Training Guide

## 概述

这是一个用于双人EEG信号融合的Transformer模型训练流程，用于分类 Single / Competition / Cooperation 三种模式。

## 架构特点

### 1. Temporal Convolution Frontend
- 对每位玩家的EEG信号进行时序卷积降采样
- 从 (C, T) 降采样到 (T̃, d)，其中 T̃ = T / (stride^num_layers)
- 混合通道信息并投影到 d_model 维度

### 2. IBS (Inter-Brain Synchrony) Token
- **跨脑同步先验**：计算双人EEG的跨脑特征
- 计算多个频段（theta, alpha, beta, gamma）的：
  - PLV (Phase Locking Value) - 相位锁定值
  - Power Correlation - 功率相关性
  - Phase Difference - 相位差
- 投影成一个 token，插入到两个玩家的序列中

### 3. Token Sequence
每个玩家的序列结构：
```
[CLS, IBS, H(1), H(2), ..., H(T̃)]
```
- CLS: 可学习的分类 token
- IBS: 跨脑同步 token（两个玩家共享）
- H(i): 时序卷积后的嵌入

### 4. Siamese Transformer Encoder
- 共享权重的 Transformer Encoder
- 分别处理两个玩家的序列
- 得到 Z₁, Z₂ ∈ ℝ^{(T̃+2)×d}

### 5. Cross-Brain Attention
- 双向交叉注意力：Z₁ ↔ Z₂
- 允许两个玩家的表征互相关注
- 得到 Z₁', Z₂'

### 6. Symmetric Fusion
- 提取 CLS token: cls₁, cls₂
- 对称融合算子：f(z₁, z₂) = f(z₂, z₁)
- 操作：add, multiply, abs_diff, concat
- 生成 f_pair

### 7. Classification
- 拼接特征：[f_pair, mp₁', mp₂']
  - f_pair: 对称融合的 CLS
  - mp₁', mp₂': 交叉注意后的均值池化
- 通过分类头得到 logits

## 损失函数

### 主损失：Cross-Entropy Loss
```
L_ce = CrossEntropy(logits, labels)
```

### 可选损失（可以后续开启）

#### 1. 对称性损失 (Symmetry Loss)
```
L_sym = ||cls₁ - cls₂||²
```
鼓励两个玩家的 CLS 表征相似（适用于合作场景）

#### 2. IBS 对齐损失 (IBS Alignment Loss)
```
L_ibs = InfoNCE(t_IBS, {cls₁, cls₂})
```
使用 InfoNCE 鼓励 IBS token 与同窗的 CLS token 对齐

### 总损失
```
L = L_ce + λ_sym·L_sym + λ_ibs·L_ibs
```

**建议**：初期训练只用 L_ce，等基线跑稳后再开启 L_sym 和 L_ibs

## 数据格式

### 输入数据
- EEG CSV 文件：每个文件包含一个玩家的EEG信号
- 格式：(Channels, Timepoints) 或 (Timepoints, Channels)
- 采样率：250 Hz（可配置）

### Metadata JSON
```json
{
  "pair": 12,
  "player1": "Pair-12-A-Single-EYE_trial01_player",
  "player2": "Pair-12-A-Single-EYE_trial01_observer",
  "class": "Single"
}
```

## 训练流程

### 1. 安装依赖
```bash
pip install -r requirements.txt
pip install scipy  # For EEG filtering
```

### 2. 配置文件
编辑 `Experiments/configs/dual_eeg_transformer.yaml`：
- 调整 `data.eeg_base_path` 指向你的EEG数据目录
- 调整 `model.in_channels` 匹配你的EEG通道数
- 调整 `data.window_size` 和 `data.stride` 来控制窗口大小和重叠

### 3. 开始训练
```bash
# 基础训练
python Experiments/scripts/train_art.py --config Experiments/configs/dual_eeg_transformer.yaml

# 使用不同配置
python Experiments/scripts/train_art.py --config path/to/your/config.yaml
```

### 4. 监控训练
训练开始后，查看 Wandb URL：
```
wandb: 🚀 View run at https://wandb.ai/...
```

### 5. 输出结构
```
Experiments/outputs/dual_eeg_transformer/
├── best_model.pt                 # 最佳模型（基于F1 score）
├── checkpoint-epoch-5.pt         # 定期保存的checkpoint
├── checkpoint-epoch-10.pt
└── ...
```

## 超参数调优

### 模型大小
```yaml
model:
  d_model: 256        # 增大提升容量，但增加计算量
  num_layers: 6       # 更深的网络
  num_heads: 8        # 多头注意力数量
  d_ff: 1024          # FFN维度
```

### 时序卷积
```yaml
model:
  conv_kernel_size: 25   # 卷积核大小（时间窗口）
  conv_stride: 4         # 降采样率
  conv_layers: 2         # 卷积层数
```

### 数据窗口
```yaml
data:
  window_size: 1000      # 4秒 @ 250Hz
  stride: 500            # 2秒重叠
```

### 训练参数
```yaml
training:
  learning_rate: 1.0e-4  # 学习率
  per_device_train_batch_size: 16  # Batch size
  num_train_epochs: 50   # 训练轮数
  dropout: 0.1           # Dropout率
```

## 实验建议

### 阶段1：Baseline（仅 L_ce）
1. 使用默认配置训练
2. 确保模型收敛
3. 在验证集上观察F1 score

### 阶段2：添加可选损失
编辑配置：
```yaml
training:
  use_sym_loss: true
  use_ibs_loss: true
  lambda_sym: 0.1
  lambda_ibs: 0.1
```

### 阶段3：超参数搜索
- 尝试不同的 d_model: [128, 256, 512]
- 尝试不同的 num_layers: [4, 6, 8]
- 尝试不同的 learning_rate: [5e-5, 1e-4, 2e-4]

## 故障排除

### CUDA Out of Memory
```yaml
training:
  per_device_train_batch_size: 8  # 减小 batch size
```

### 训练不收敛
- 检查数据预处理是否正确
- 尝试降低学习率
- 增加 warmup steps

### EEG 文件读取错误
- 确认 CSV 格式正确
- 确认文件路径匹配 metadata 中的名称
- 检查 EEG 通道数是否匹配配置

### 数据窗口太少
- 增加 `data.stride`（减少重叠）
- 检查 EEG 文件长度是否足够

## 代码位置

- **模型架构**: `Models/backbones/dual_eeg_transformer.py`
- **数据加载器**: `Data/processed/dual_eeg_dataset.py`
- **训练脚本**: `Experiments/scripts/train_art.py`
- **配置文件**: `Experiments/configs/dual_eeg_transformer.yaml`

## 技术细节

### IBS Token 计算
当前实现使用简化的频谱分析：
- FFT 计算相位
- 功率 = 幅度平方
- PLV, 功率相关, 相位差

**改进方向**：
- 使用 MNE-Python 进行专业的EEG分析
- 使用 Wavelet Transform 提取多尺度特征
- 添加更多连接性指标（Coherence, Granger Causality等）

### 对称性设计
模型使用对称算子确保：
```
f(player1, player2) = f(player2, player2)
```
这对于 Competition 和 Cooperation 模式很重要，因为两个玩家的角色是对等的。

### 共享 vs 独立 Encoder
当前使用**共享**Encoder（Siamese），好处：
- 参数效率高
- 强制两个玩家用相同方式编码
- 更好的泛化

**可选**：使用独立Encoder
```python
self.encoder1 = TransformerEncoder(...)
self.encoder2 = TransformerEncoder(...)
```

## 参考文献

相关的跨脑同步和脑机接口研究：
- Hyperscanning and EEG-based connectivity analysis
- Phase Locking Value (PLV) for neural synchrony
- Transformer for EEG signal processing

---

**祝训练顺利！🚀**

如有问题，请查看：
- 配置文件是否正确
- EEG数据路径是否正确
- 查看训练日志和 Wandb
