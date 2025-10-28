import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor
from datasets import load_dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
import json


# ==================== Gaze-Guided Attention 機制 ====================

class GazeWeightComputer:
    """計算 Gaze 權重、遮罩和 bias"""
    
    @staticmethod
    def compute_gaze_weights(heatmap, eeg_scalar=0.5, top_p=0.7, alpha=1.0, 
                            patch_size=16, img_size=224):
        """
        從 gaze heatmap 計算 token 權重和遮罩
        
        Args:
            heatmap: (B, 1, H, W) gaze heatmap
            eeg_scalar: EEG 標量 [0,1] 或 (B,)
            top_p: 保留前 p% 的 tokens
            alpha: attention bias 的縮放係數
            patch_size: patch 大小
            img_size: 圖片大小
            
        Returns:
            soft_weights: (B, num_patches+1) 歸一化權重（含CLS）
            binary_mask: (B, num_patches+1) 二值遮罩（含CLS）
            attn_bias: (B, num_patches+1) attention bias
        """
        B = heatmap.shape[0]
        num_patches_side = img_size // patch_size
        num_patches = num_patches_side ** 2
        device = heatmap.device
        
        # (A-1) 平均池化到 patch grid
        pooled_heatmap = F.adaptive_avg_pool2d(
            heatmap, 
            (num_patches_side, num_patches_side)
        )  # (B, 1, H//P, W//P)
        
        # 展平為 token 序列
        patch_weights = pooled_heatmap.flatten(2).squeeze(1)  # (B, num_patches)
        
        # (A-2) 計算溫度：eeg_scalar 高 → tau 小 → 更尖銳
        if isinstance(eeg_scalar, (int, float)):
            tau = 1.5 - eeg_scalar * 1.2
            tau = torch.tensor(tau, device=device)
        else:
            eeg_scalar = eeg_scalar.to(device)
            tau = 1.5 - eeg_scalar * 1.2
        
        tau = torch.clamp(tau, 0.3, 1.5)
        
        # 溫度縮放的 softmax
        if tau.dim() == 0:
            tau = tau.unsqueeze(0)
        tau = tau.view(B, 1)
        
        patch_weights_scaled = patch_weights / tau
        soft_weights_patches = F.softmax(patch_weights_scaled, dim=-1)  # (B, num_patches)
        
        # 為 CLS token 添加權重（始終保留）
        cls_weight = torch.ones(B, 1, device=device)
        soft_weights = torch.cat([cls_weight, soft_weights_patches], dim=1)  # (B, num_patches+1)
        
        # (A-3) Binary Mask: Top-p 選擇
        binary_mask = torch.zeros_like(soft_weights)
        binary_mask[:, 0] = 1.0  # CLS token 始終保留
        
        for b in range(B):
            sorted_weights, sorted_indices = torch.sort(
                soft_weights_patches[b], 
                descending=True
            )
            cumsum = torch.cumsum(sorted_weights, dim=0)
            
            # 找到累積和超過 top_p 的位置
            cutoff_idx = torch.searchsorted(cumsum, top_p).item()
            cutoff_idx = max(1, min(cutoff_idx + 1, num_patches))
            
            # 選擇的 token 索引（+1 因為 CLS 在第 0 位）
            selected_indices = sorted_indices[:cutoff_idx] + 1
            binary_mask[b, selected_indices] = 1.0
        
        # (A-4) Attention Bias
        eps = 1e-8
        attn_bias = alpha * torch.log(soft_weights + eps)  # (B, num_patches+1)
        
        return soft_weights, binary_mask, attn_bias


class GazeGuidedAttention(nn.Module):
    """Gaze-guided self-attention layer"""
    
    def __init__(self, hidden_size, num_heads, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size
        
        # Q, K, V 投影
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_bias=None):
        """
        Args:
            x: (B, N, D) token embeddings
            attn_bias: (B, N) attention bias for each token
        """
        B, N, D = x.shape
        
        # QKV projection and reshape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # (B) Scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, num_heads, N, N)
        
        # Apply attention bias: broadcast bias to all heads and queries
        if attn_bias is not None:
            # attn_bias: (B, N) -> (B, 1, 1, N) for broadcasting to (B, num_heads, N, N)
            bias_expanded = attn_bias.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            attn_scores = attn_scores + bias_expanded
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = (attn_weights @ v).transpose(1, 2).reshape(B, N, D)
        
        # Output projection
        output = self.proj(attn_output)
        output = self.proj_dropout(output)
        
        return output


class GazeGuidedTransformerBlock(nn.Module):
    """Transformer block with gaze-guided attention"""
    
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.attention = GazeGuidedAttention(hidden_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        
        # MLP
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, attn_bias=None):
        # Attention with residual (Pre-LN)
        x = x + self.attention(self.norm1(x), attn_bias)
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


# ==================== Token Pruning 工具函數 ====================

def prune_tokens(x, mask):
    """
    (C) 根據 mask 對 tokens 進行 pruning
    
    Args:
        x: (B, N, D) tokens
        mask: (B, N) binary mask
    
    Returns:
        pruned_x: List of tensors, 每個 (N_kept, D)
        kept_indices: List of tensors, 每個保留的索引
    """
    B = x.shape[0]
    pruned_x = []
    kept_indices = []
    
    for b in range(B):
        indices = torch.nonzero(mask[b], as_tuple=False).squeeze(-1)
        pruned_x.append(x[b, indices, :])
        kept_indices.append(indices)
    
    return pruned_x, kept_indices


def restore_tokens(pruned_list, kept_indices_list, original_shape, device):
    """
    將 pruned tokens 還原到原始位置
    
    Args:
        pruned_list: List of (N_kept, D) tensors
        kept_indices_list: List of kept indices
        original_shape: (B, N, D)
        device: torch device
    
    Returns:
        restored_x: (B, N, D)
    """
    B, N, D = original_shape
    restored_x = torch.zeros(B, N, D, device=device)
    
    for b in range(B):
        restored_x[b, kept_indices_list[b], :] = pruned_list[b]
    
    return restored_x


def pad_pruned_tokens(pruned_list):
    """
    將不同長度的 pruned tokens padding 到相同長度
    
    Args:
        pruned_list: List of (N_kept, D) tensors
    
    Returns:
        padded_tokens: (B, max_N, D)
        padding_mask: (B, max_N) 1 表示有效，0 表示 padding
    """
    B = len(pruned_list)
    D = pruned_list[0].shape[-1]
    max_len = max(t.shape[0] for t in pruned_list)
    device = pruned_list[0].device
    
    padded_tokens = torch.zeros(B, max_len, D, device=device)
    padding_mask = torch.zeros(B, max_len, device=device)
    
    for b in range(B):
        length = pruned_list[b].shape[0]
        padded_tokens[b, :length] = pruned_list[b]
        padding_mask[b, :length] = 1.0
    
    return padded_tokens, padding_mask


# ==================== 主模型 ====================

class GazeGuidedViT(nn.Module):
    """ViT with Gaze-guided attention"""
    
    def __init__(self, base_model, patch_size=16, img_size=224, 
                 apply_gaze_from_layer=0, use_pruning=True):
        super().__init__()
        self.config = base_model.config
        self.patch_size = patch_size
        self.img_size = img_size
        self.apply_gaze_from_layer = apply_gaze_from_layer
        self.use_pruning_in_forward = use_pruning
        
        # 保留 embeddings
        self.embeddings = base_model.vit.embeddings
        
        # 建立 gaze-guided encoder blocks
        hidden_size = self.config.hidden_size
        num_heads = self.config.num_attention_heads
        num_layers = self.config.num_hidden_layers
        dropout = self.config.hidden_dropout_prob
        
        self.encoder_blocks = nn.ModuleList([
            GazeGuidedTransformerBlock(
                hidden_size, num_heads, 
                mlp_ratio=4.0, dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # 複製原始權重
        self._copy_weights_from_base_model(base_model)
        
        self.layernorm = base_model.vit.layernorm
        self.classifier = base_model.classifier
        
        print(f"✓ Gaze-guided ViT 初始化完成")
        print(f"  - 層數: {num_layers}")
        print(f"  - 從第 {apply_gaze_from_layer} 層開始應用 gaze bias")
        print(f"  - Token pruning: {'啟用' if use_pruning else '停用'}")
    
    def _copy_weights_from_base_model(self, base_model):
        """從原始 ViT 複製權重"""
        with torch.no_grad():
            for i, block in enumerate(self.encoder_blocks):
                orig_block = base_model.vit.encoder.layer[i]
                
                # 複製 attention QKV
                orig_attn = orig_block.attention.attention
                block.attention.qkv.weight.data = torch.cat([
                    orig_attn.query.weight.data,
                    orig_attn.key.weight.data,
                    orig_attn.value.weight.data
                ], dim=0)
                block.attention.qkv.bias.data = torch.cat([
                    orig_attn.query.bias.data,
                    orig_attn.key.bias.data,
                    orig_attn.value.bias.data
                ], dim=0)
                
                # 複製 attention projection
                block.attention.proj.weight.data = orig_block.attention.output.dense.weight.data.clone()
                block.attention.proj.bias.data = orig_block.attention.output.dense.bias.data.clone()
                
                # 複製 LayerNorm
                block.norm1.weight.data = orig_block.layernorm_before.weight.data.clone()
                block.norm1.bias.data = orig_block.layernorm_before.bias.data.clone()
                block.norm2.weight.data = orig_block.layernorm_after.weight.data.clone()
                block.norm2.bias.data = orig_block.layernorm_after.bias.data.clone()
                
                # 複製 MLP
                block.mlp[0].weight.data = orig_block.intermediate.dense.weight.data.clone()
                block.mlp[0].bias.data = orig_block.intermediate.dense.bias.data.clone()
                block.mlp[3].weight.data = orig_block.output.dense.weight.data.clone()
                block.mlp[3].bias.data = orig_block.output.dense.bias.data.clone()
    
    def forward(self, pixel_values, heatmap, eeg_scalar=0.5, top_p=0.7, 
                alpha=1.0, labels=None):
        """
        Forward pass
        
        Args:
            pixel_values: (B, C, H, W)
            heatmap: (B, 1, H, W)
            eeg_scalar: float or (B,)
            top_p: float
            alpha: float
            labels: (B,) 可選
        """
        device = pixel_values.device
        
        # 計算 gaze 權重、mask 和 bias
        soft_weights, binary_mask, attn_bias = GazeWeightComputer.compute_gaze_weights(
            heatmap, eeg_scalar, top_p, alpha, 
            self.patch_size, self.img_size
        )
        
        # Embeddings
        x = self.embeddings(pixel_values)  # (B, 1+num_patches, D)
        B, N, D = x.shape
        
        # 通過所有 encoder blocks
        for i, block in enumerate(self.encoder_blocks):
            # 決定是否應用 gaze bias
            current_bias = attn_bias if i >= self.apply_gaze_from_layer else None
            
            # 簡化版：不在中間做 pruning，只用 bias 引導
            # 完整 pruning 需要更複雜的實作
            x = block(x, current_bias)
        
        # LayerNorm
        x = self.layernorm(x)
        
        # Pooling
        if self.use_pruning_in_forward:
            # 使用 mask 加權平均
            masked_x = x * binary_mask.unsqueeze(-1)
            pooled_output = masked_x.sum(dim=1) / binary_mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            # 使用 CLS token
            pooled_output = x[:, 0]
        
        # 分類
        logits = self.classifier(pooled_output)
        
        # 損失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        # 統計保留的 token 數量
        num_kept_tokens = binary_mask.sum(dim=1).float().mean().item()
        
        return {
            'loss': loss,
            'logits': logits,
            'soft_weights': soft_weights,
            'binary_mask': binary_mask,
            'attn_bias': attn_bias,
            'num_kept_tokens': num_kept_tokens
        }


# ==================== 資料處理 ====================

def prepare_dataset(dataset, processor, image_dir, label2id, combine_mode='concat'):
    """
    準備資料集：讀取 2 張背景圖 + 2 張 heatmap
    
    資料格式：
    {
        'bg_image1': '背景圖1',
        'bg_image2': '背景圖2',
        'heatmap1': 'heatmap1',
        'heatmap2': 'heatmap2',
        'class': '類別',
        'eeg_scalar': 0.5  # 可選
    }
    """
    def preprocess_function(examples):
        batch_size = len(examples['bg_image1'])
        
        # 讀取背景圖
        bg_images1 = [Image.open(os.path.join(image_dir, img)).convert('RGB') 
                      for img in examples['bg_image1']]
        bg_images2 = [Image.open(os.path.join(image_dir, img)).convert('RGB') 
                      for img in examples['bg_image2']]
        
        # 讀取 heatmap
        heatmaps1 = [Image.open(os.path.join(image_dir, img)).convert('L') 
                     for img in examples['heatmap1']]
        heatmaps2 = [Image.open(os.path.join(image_dir, img)).convert('L') 
                     for img in examples['heatmap2']]
        
        # 處理背景圖
        inputs1 = processor(images=bg_images1, return_tensors="pt")
        inputs2 = processor(images=bg_images2, return_tensors="pt")
        
        pixel_values1 = inputs1['pixel_values']
        pixel_values2 = inputs2['pixel_values']
        
        # 組合背景圖
        if combine_mode == 'concat':
            combined_pixels = torch.cat([pixel_values1, pixel_values2], dim=1)
        elif combine_mode == 'multiply':
            combined_pixels = pixel_values1 * pixel_values2
        elif combine_mode == 'subtract':
            combined_pixels = pixel_values1 - pixel_values2
        else:
            raise ValueError(f"不支援的 combine_mode: {combine_mode}")
        
        # 處理 heatmap
        heatmap_tensors1 = torch.stack([
            torch.from_numpy(np.array(hm).astype(np.float32) / 255.0) 
            for hm in heatmaps1
        ]).unsqueeze(1)
        
        heatmap_tensors2 = torch.stack([
            torch.from_numpy(np.array(hm).astype(np.float32) / 255.0) 
            for hm in heatmaps2
        ]).unsqueeze(1)
        
        # 組合 heatmap（平均）
        combined_heatmap = (heatmap_tensors1 + heatmap_tensors2) / 2.0
        
        # Resize heatmap
        target_size = (pixel_values1.shape[-2], pixel_values1.shape[-1])
        combined_heatmap = F.interpolate(
            combined_heatmap, 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        # 標籤
        labels = [label2id[label] for label in examples['class']]
        
        # EEG scalar
        eeg_scalars = examples.get('eeg_scalar', [0.5] * batch_size)
        
        return {
            'pixel_values': combined_pixels,
            'heatmap': combined_heatmap,
            'labels': labels,
            'eeg_scalar': eeg_scalars
        }
    
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        batch_size=8,
        writer_batch_size=8,  # Add this
        remove_columns=dataset.column_names
    )
    
    processed_dataset.set_format(
        type='torch', 
        columns=['pixel_values', 'heatmap', 'labels', 'eeg_scalar']
    )
    
    return processed_dataset


# ==================== 訓練器 ====================

class GazeGuidedViTClassifier:
    """Gaze-Guided ViT 分類器"""
    
    def __init__(self, num_labels, model_name='google/vit-base-patch16-224', 
                 combine_mode='concat', use_gaze=True, 
                 apply_gaze_from_layer=0, use_pruning=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.combine_mode = combine_mode
        self.use_gaze = use_gaze
        
        print(f"\n{'='*60}")
        print(f"初始化 Gaze-Guided ViT Classifier")
        print(f"{'='*60}")
        print(f"裝置: {self.device}")
        print(f"圖片組合模式: {combine_mode}")
        print(f"使用 Gaze 引導: {use_gaze}")
        
        # 載入基礎模型
        base_model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        
        # 修改第一層（如果是 concat 模式）
        if combine_mode == 'concat':
            self._modify_first_layer(base_model)
        
        # 建立 Gaze-Guided 模型
        if use_gaze:
            self.model = GazeGuidedViT(
                base_model, 
                patch_size=16, 
                img_size=224,
                apply_gaze_from_layer=apply_gaze_from_layer,
                use_pruning=use_pruning
            )
        else:
            self.model = base_model
            print("✓ 使用標準 ViT（不含 Gaze 引導）")
        
        self.model.to(self.device)
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        
        print(f"✓ 模型載入完成，分類數: {num_labels}")
        print(f"{'='*60}\n")
    
    def _modify_first_layer(self, base_model):
        """修改第一層以支援 6 channels"""
        original_conv = base_model.vit.embeddings.patch_embeddings.projection
        new_conv = nn.Conv2d(
            in_channels=6,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = original_conv.weight.clone()
            new_conv.weight[:, 3:, :, :] = original_conv.weight.clone()
            new_conv.weight.data = new_conv.weight.data / 2.0
            if new_conv.bias is not None and original_conv.bias is not None:
                new_conv.bias.data = original_conv.bias.data.clone()
        
        base_model.vit.embeddings.patch_embeddings.projection = new_conv
        print("✓ 已修改第一層支援 6 channels (concat 模式)")
    
    def train(self, train_dataset, val_dataset=None, epochs=10, 
              batch_size=16, learning_rate=2e-5, 
              top_p=0.7, alpha=1.0,
              save_path='gaze_vit_model.pth'):
        """訓練模型"""
        
        print(f"\n{'='*60}")
        print(f"開始訓練")
        print(f"{'='*60}")
        print(f"訓練樣本數: {len(train_dataset)}")
        if val_dataset:
            print(f"驗證樣本數: {len(val_dataset)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Top-p: {top_p}")
        print(f"Alpha: {alpha}")
        print(f"{'='*60}\n")
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # ===== 訓練階段 =====
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            train_kept_tokens = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for batch_idx, batch in enumerate(pbar):
                pixel_values = batch['pixel_values'].to(self.device)
                heatmap = batch['heatmap'].to(self.device)
                labels = batch['labels'].to(self.device)
                eeg_scalar = batch['eeg_scalar'].to(self.device) if torch.is_tensor(batch['eeg_scalar']) else batch['eeg_scalar'][0]
                
                # 前向傳播
                if self.use_gaze:
                    outputs = self.model(
                        pixel_values=pixel_values,
                        heatmap=heatmap,
                        eeg_scalar=eeg_scalar,
                        top_p=top_p,
                        alpha=alpha,
                        labels=labels
                    )
                    loss = outputs['loss']
                    logits = outputs['logits']
                    train_kept_tokens += outputs['num_kept_tokens']
                else:
                    outputs = self.model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss
                    logits = outputs.logits
                
                # 反向傳播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 統計
                predictions = logits.argmax(dim=-1)
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)
                train_loss += loss.item()
                
                # 更新進度條
                postfix = {
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100*train_correct/train_total:.2f}%'
                }
                if self.use_gaze:
                    avg_tokens = train_kept_tokens / (batch_idx + 1)
                    postfix['tokens'] = f'{avg_tokens:.1f}'
                
                pbar.set_postfix(postfix)
            
            avg_train_loss = train_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total
            
            print(f'\nEpoch {epoch+1}/{epochs}')
            print(f'  Train Loss: {avg_train_loss:.4f}')
            print(f'  Train Acc:  {train_acc:.2f}%')
            if self.use_gaze:
                avg_tokens = train_kept_tokens / len(train_loader)
                print(f'  Avg Tokens: {avg_tokens:.1f}')
            
            # ===== 驗證階段 =====
            if val_dataset:
                val_acc = self.evaluate(val_loader, top_p, alpha)
                print(f'  Val Acc:    {val_acc:.2f}%')
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(self.model.state_dict(), save_path)
                    print(f'  ✓ 最佳模型已儲存')
            
            print()
        
        if not val_dataset:
            torch.save(self.model.state_dict(), save_path)
            print(f'✓ 訓練完成！模型已儲存到 {save_path}')
        else:
            print(f'✓ 訓練完成！最佳驗證準確率: {best_val_acc:.2f}%')
    
    def evaluate(self, dataloader, top_p=0.7, alpha=1.0):
        """評估模型"""
        self.model.eval()
        correct = 0
        total = 0
        total_kept_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating'):
                pixel_values = batch['pixel_values'].to(self.device)
                heatmap = batch['heatmap'].to(self.device)
                labels = batch['labels'].to(self.device)
                eeg_scalar = batch['eeg_scalar'].to(self.device) if torch.is_tensor(batch['eeg_scalar']) else batch['eeg_scalar'][0]
                
                if self.use_gaze:
                    outputs = self.model(
                        pixel_values=pixel_values,
                        heatmap=heatmap,
                        eeg_scalar=eeg_scalar,
                        top_p=top_p,
                        alpha=alpha
                    )
                    predictions = outputs['logits'].argmax(dim=-1)
                    total_kept_tokens += outputs['num_kept_tokens']
                else:
                    outputs = self.model(pixel_values=pixel_values)
                    predictions = outputs.logits.argmax(dim=-1)
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        accuracy = 100 * correct / total
        
        if self.use_gaze:
            avg_tokens = total_kept_tokens / len(dataloader)
            print(f'  平均保留 tokens: {avg_tokens:.1f}')
        
        return accuracy
    
    def predict(self, bg_image1_path, bg_image2_path, 
                heatmap1_path, heatmap2_path, 
                eeg_scalar=0.5, top_p=0.7, alpha=1.0):
        """
        預測單筆資料
        
        Args:
            bg_image1_path: 背景圖 1 路徑
            bg_image2_path: 背景圖 2 路徑
            heatmap1_path: Heatmap 1 路徑
            heatmap2_path: Heatmap 2 路徑
            eeg_scalar: EEG 標量
            top_p: Token 保留比例
            alpha: Attention bias 強度
        
        Returns:
            predicted_class: 預測的類別 ID
            confidence: 信心分數
            outputs: 完整的模型輸出（包含 gaze 資訊）
        """
        self.model.eval()
        
        # 讀取背景圖
        bg1 = Image.open(bg_image1_path).convert('RGB')
        bg2 = Image.open(bg_image2_path).convert('RGB')
        
        # 讀取 heatmap
        hm1 = Image.open(heatmap1_path).convert('L')
        hm2 = Image.open(heatmap2_path).convert('L')
        
        # 處理背景圖
        inputs1 = self.processor(images=bg1, return_tensors="pt")
        inputs2 = self.processor(images=bg2, return_tensors="pt")
        
        pixel_values1 = inputs1['pixel_values']
        pixel_values2 = inputs2['pixel_values']
        
        # 組合背景圖
        if self.combine_mode == 'concat':
            combined_pixels = torch.cat([pixel_values1, pixel_values2], dim=1)
        elif self.combine_mode == 'multiply':
            combined_pixels = pixel_values1 * pixel_values2
        elif self.combine_mode == 'subtract':
            combined_pixels = pixel_values1 - pixel_values2
        
        # 處理 heatmap
        hm1_tensor = torch.from_numpy(np.array(hm1).astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        hm2_tensor = torch.from_numpy(np.array(hm2).astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        combined_heatmap = (hm1_tensor + hm2_tensor) / 2.0
        
        # Resize heatmap
        target_size = (combined_pixels.shape[-2], combined_pixels.shape[-1])
        combined_heatmap = F.interpolate(
            combined_heatmap,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )
        
        # 移到 device
        combined_pixels = combined_pixels.to(self.device)
        combined_heatmap = combined_heatmap.to(self.device)
        
        # 預測
        with torch.no_grad():
            if self.use_gaze:
                outputs = self.model(
                    pixel_values=combined_pixels,
                    heatmap=combined_heatmap,
                    eeg_scalar=eeg_scalar,
                    top_p=top_p,
                    alpha=alpha
                )
                logits = outputs['logits']
            else:
                outputs = self.model(pixel_values=combined_pixels)
                logits = outputs.logits
            
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = logits.argmax(dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return predicted_class, confidence, outputs


# ==================== 主程式 ====================

if __name__ == "__main__":
    # ===== 1. 設定參數 =====
    IMAGE_DIR = "./"                          # 圖片資料夾
    TRAIN_JSON = "./train_gaze.json"          # 訓練資料
    VAL_JSON = "./val_gaze.json"              # 驗證資料（可選）
    
    # 模型配置
    COMBINE_MODE = 'subtract'                 # 'concat', 'multiply', 'subtract'
    USE_GAZE = True                           # 是否使用 Gaze 引導
    APPLY_GAZE_FROM_LAYER = 0                 # 從第幾層開始應用 gaze
    USE_PRUNING = True                        # 是否使用 token pruning
    
    # 訓練超參數
    EPOCHS = 10
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    TOP_P = 0.7                               # 保留 70% tokens
    ALPHA = 1.0                               # Attention bias 強度
    
    SAVE_PATH = f'gaze_vit_{COMBINE_MODE}_p{TOP_P}.pth'
    
    # ===== 2. 載入資料 =====
    print("載入資料集...")
    train_ds = load_dataset("json", data_files=TRAIN_JSON, split="train")
    
    # 如果有驗證集
    val_ds = None
    if os.path.exists(VAL_JSON):
        val_ds = load_dataset("json", data_files=VAL_JSON, split="train")
        print(f"✓ 載入驗證集: {len(val_ds)} 筆")
    
    print(f"✓ 載入訓練集: {len(train_ds)} 筆")
    
    # ===== 3. 建立類別映射 =====
    unique_classes = set(train_ds['class'])
    label2id = {label: idx for idx, label in enumerate(sorted(unique_classes))}
    id2label = {idx: label for label, idx in label2id.items()}
    
    print(f"\n類別映射:")
    for label, idx in sorted(label2id.items(), key=lambda x: x[1]):
        print(f"  {idx}: {label}")
    print(f"總類別數: {len(label2id)}\n")
    
    # ===== 4. 初始化模型 =====
    classifier = GazeGuidedViTClassifier(
        num_labels=len(label2id),
        combine_mode=COMBINE_MODE,
        use_gaze=USE_GAZE,
        apply_gaze_from_layer=APPLY_GAZE_FROM_LAYER,
        use_pruning=USE_PRUNING
    )
    
    # ===== 5. 預處理資料集 =====
    print("預處理訓練資料...")
    train_dataset = prepare_dataset(
        train_ds, 
        classifier.processor, 
        IMAGE_DIR, 
        label2id,
        combine_mode=COMBINE_MODE
    )
    print(f"✓ 訓練集預處理完成: {len(train_dataset)} 筆")
    
    val_dataset = None
    if val_ds is not None:
        print("預處理驗證資料...")
        val_dataset = prepare_dataset(
            val_ds, 
            classifier.processor, 
            IMAGE_DIR, 
            label2id,
            combine_mode=COMBINE_MODE
        )
        print(f"✓ 驗證集預處理完成: {len(val_dataset)} 筆")
    
    # ===== 6. 訓練模型 =====
    classifier.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        top_p=TOP_P,
        alpha=ALPHA,
        save_path=SAVE_PATH
    )
    
    # ===== 7. 儲存配置 =====
    config = {
        'label2id': label2id,
        'id2label': id2label,
        'combine_mode': COMBINE_MODE,
        'use_gaze': USE_GAZE,
        'apply_gaze_from_layer': APPLY_GAZE_FROM_LAYER,
        'use_pruning': USE_PRUNING,
        'top_p': TOP_P,
        'alpha': ALPHA,
        'model_name': 'google/vit-base-patch16-224',
        'num_labels': len(label2id)
    }
    
    config_path = f'gaze_model_config_{COMBINE_MODE}.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 模型配置已儲存到 {config_path}")
    
    # ===== 8. 預測範例 =====
    if len(train_ds) > 0:
        print("\n" + "="*60)
        print("預測範例")
        print("="*60)
        
        sample = train_ds[0]
        bg1_path = os.path.join(IMAGE_DIR, sample['bg_image1'])
        bg2_path = os.path.join(IMAGE_DIR, sample['bg_image2'])
        hm1_path = os.path.join(IMAGE_DIR, sample['heatmap1'])
        hm2_path = os.path.join(IMAGE_DIR, sample['heatmap2'])
        
        if all(os.path.exists(p) for p in [bg1_path, bg2_path, hm1_path, hm2_path]):
            predicted_class, confidence, outputs = classifier.predict(
                bg1_path, bg2_path, hm1_path, hm2_path,
                eeg_scalar=sample.get('eeg_scalar', 0.5),
                top_p=TOP_P,
                alpha=ALPHA
            )
            
            predicted_label = id2label[predicted_class]
            actual_label = sample['class']
            
            print(f"背景圖 1:     {sample['bg_image1']}")
            print(f"背景圖 2:     {sample['bg_image2']}")
            print(f"Heatmap 1:    {sample['heatmap1']}")
            print(f"Heatmap 2:    {sample['heatmap2']}")
            print(f"實際類別:     {actual_label}")
            print(f"預測類別:     {predicted_label}")
            print(f"信心分數:     {confidence:.4f}")
            
            if USE_GAZE:
                print(f"保留 tokens:  {outputs['num_kept_tokens']:.1f}")
                print(f"Bias 範圍:    [{outputs['attn_bias'].min():.2f}, {outputs['attn_bias'].max():.2f}]")
        else:
            print("找不到範例圖片檔案")
    
    print("\n" + "="*60)
    print("所有流程完成！")
    print("="*60)
    
    # ===== 9. 實驗建議 =====
    print("\n💡 實驗建議：")
    print("1. 對比實驗：")
    print("   - Baseline: USE_GAZE=False")
    print("   - Gaze (no pruning): USE_GAZE=True, USE_PRUNING=False")
    print("   - Gaze + Pruning: USE_GAZE=True, USE_PRUNING=True")
    print()
    print("2. 參數調整：")
    print("   - TOP_P: [0.3, 0.5, 0.7, 0.9] 測試不同剪枝程度")
    print("   - ALPHA: [0.5, 1.0, 2.0, 5.0] 測試 bias 強度")
    print("   - APPLY_GAZE_FROM_LAYER: [0, 6] 測試應用層數")
    print()
    print("3. 視覺化：")
    print("   - 可視化 binary_mask 查看保留的 patches")
    print("   - 繪製不同 EEG 值下的權重分佈")
    print("   - 分析 attention bias 的影響")