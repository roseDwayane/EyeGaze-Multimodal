import math
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.utils import ModelOutput
from .hf_config import ArtifactRemovalTransformerConfig

class ExpandConv1x1(nn.Module):
    """Expand channels with a 1x1 convolution and permute to (B, T, C)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        No docstring provided.
        """
        return self.conv(x).permute(0, 2, 1)

class PositionalEmbedding(nn.Module):
    """
    1. Summary:
           Represents a module that adds positional information to input sequences.

        2. Description:
           WHY: Sequence models, especially those based on attention mechanisms like Transformers, are inherently permutation-invariant and do not inherently understand the order of elements in a sequence. `PositionalEmbedding` injects information about the position of each element, enabling the model to learn relationships based on sequence order.
           WHEN: Use this class as a crucial component in sequence models, before feeding the sequence data into attention layers or other sequence-aware modules.  It is particularly important when the absolute or relative position of tokens in a sequence is significant for the task.
           WHERE: This class typically resides within the embedding layer of a sequence model. It is instantiated once during model initialization and then used repeatedly during the forward pass to add positional embeddings to input sequences.
           HOW: The class offers two modes: 'sinusoidal' (fixed positional encoding) and 'learned' (trainable positional embeddings). In 'sinusoidal' mode, it generates a fixed positional encoding matrix using sine and cosine functions of different frequencies. In 'learned' mode, it uses an `nn.Embedding` layer to learn positional embeddings during training.

        3. Parameters:
           max_len (int): The maximum sequence length that the positional embedding will support. This determines the size of the positional encoding matrix or the embedding table.
           d_model (int): The dimensionality of the input embeddings. This also determines the dimensionality of the positional embeddings.
           mode (str, optional): Specifies the type of positional embedding to use.
                - 'sinusoidal': Uses fixed sinusoidal positional encodings.
                - 'learned': Uses a learned embedding layer for positional embeddings.
                Defaults to 'sinusoidal'.

        4. Attributes:
           mode (str): Stores the chosen positional embedding mode ('sinusoidal' or 'learned').
           d_model (int): Stores the embedding dimensionality.
           pos_embed (nn.Embedding, optional):  The learned embedding layer, present only when `mode` is 'learned'.
           pe (torch.Tensor, optional): The pre-computed positional encoding matrix, present only when `mode` is 'sinusoidal'. It's a buffer, meaning it's not a parameter of the module.

        5. Example:
           ```python
           import torch
           import torch.nn as nn
           import math

           # Example with sinusoidal embeddings
           pos_embed_sin = PositionalEmbedding(max_len=500, d_model=512, mode='sinusoidal')
           input_tensor = torch.randn(1, 128, 512)  # Batch size 1, sequence length 128, embedding dim 512
           output_tensor_sin = pos_embed_sin(input_tensor)
           print(f"Output shape with sinusoidal embeddings: {output_tensor_sin.shape}")

           # Example with learned embeddings
           pos_embed_learned = PositionalEmbedding(max_len=500, d_model=512, mode='learned')
           input_tensor = torch.randn(1, 128, 512)
           output_tensor_learned = pos_embed_learned(input_tensor)
           print(f"Output shape with learned embeddings: {output_tensor_learned.shape}")
           ```
    """

    def __init__(self, max_len: int, d_model: int, mode: str='sinusoidal') -> None:
        super().__init__()
        if mode not in {'sinusoidal', 'learned'}:
            raise ValueError(f'Unsupported pos_mode: {mode}')
        self.mode = mode
        self.d_model = d_model
        if self.mode == 'learned':
            self.pos_embed = nn.Embedding(max_len, d_model)
        else:
            pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        No docstring provided.
        """
        T = x.size(1)
        if self.mode == 'learned':
            pos = torch.arange(T, device=x.device).unsqueeze(0)
            pos_emb = self.pos_embed(pos)
        else:
            pos_emb = self.pe[:, :T, :]
        return x + pos_emb

class MultiHeadAttention(nn.Module):
    """
    1. Summary:
           Represents a multi-head scaled dot-product attention mechanism.

        2. Description:
           WHY: This class implements multi-head attention, a key component of the Transformer architecture. Multi-head attention allows the model to attend to different parts of the input sequence in parallel, capturing different relationships and dependencies. This enhances the model's ability to understand complex sequences.
           WHEN: Use this class within a Transformer layer to implement the attention mechanism. It's used to compute attention weights and generate a context vector that incorporates information from different parts of the input sequence.
           WHERE: This class typically resides within a Transformer encoder or decoder layer. It takes queries, keys, and values as input and returns a context vector representing the attended information.
           HOW: The class projects the queries, keys, and values into multiple heads using linear layers. It then computes scaled dot-product attention for each head in parallel. The outputs from all heads are concatenated and projected back to the original dimension.

        3. Parameters:
           d_model (int): The input and output dimension of the module. This is the dimension of the input queries, keys, and values, and the dimension of the output context vector.
           num_heads (int): The number of attention heads. The `d_model` must be divisible by `num_heads`.
           dropout (float, optional): The dropout probability. This is applied to the attention weights for regularization. Defaults to 0.0.

        4. Attributes:
           d_k (int): The dimension of the queries, keys, and values for each head. This is equal to `d_model // num_heads`.
           num_heads (int): The number of attention heads.
           d_model (int): The input and output dimension of the module.
           q_proj (nn.Linear): The linear projection layer for the queries.
           k_proj (nn.Linear): The linear projection layer for the keys.
           v_proj (nn.Linear): The linear projection layer for the values.
           out_proj (nn.Linear): The linear projection layer for the output context vector.
           dropout (nn.Dropout): The dropout layer.

        5. Example:
           ```python
           import torch
           import torch.nn as nn

           # Example usage
           mha = MultiHeadAttention(d_model=512, num_heads=8, dropout=0.1)
           q = torch.randn(32, 128, 512)  # Batch size 32, sequence length 128, feature dimension 512
           k = torch.randn(32, 128, 512)
           v = torch.randn(32, 128, 512)
           attn_mask = torch.rand(32, 8, 128, 128) > 0.5 # Example attention mask
           output_tensor = mha(q, k, v, attn_mask)
           print(f"Output shape: {output_tensor.shape}")  # Expected: torch.Size([32, 128, 512])
           ```
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float=0.0) -> None:
        super().__init__()
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        1. Summary:
               Performs scaled dot-product attention, including linear projections, attention masking, and dropout.

            2. Description:
               WHY: This function implements the core scaled dot-product attention mechanism, a fundamental component of Transformer models. It computes attention weights based on the similarity between queries and keys, scales the weights, applies an optional mask, and then uses these weights to combine the values. This allows the model to selectively focus on different parts of the input sequence.
               WHEN: Use this function within a multi-head attention module to compute attention weights and generate a context vector that incorporates information from different parts of the input sequence.
               WHERE: This function is called within the forward pass of a multi-head attention module. It receives queries, keys, and values as input, computes the attention weights, and returns the context vector.
               HOW: The function first projects the queries, keys, and values using linear layers (`self.q_proj`, `self.k_proj`, `self.v_proj`) and reshapes them to have a separate head dimension. It then calculates the dot product between the queries and keys, scales the result by the square root of the key dimension (`self.d_k`), applies an optional attention mask, and computes the softmax to obtain the attention weights. Dropout is applied to the attention weights for regularization. Finally, the attention weights are used to combine the values, and the result is projected back to the original dimension using `self.out_proj`.

            3. Args:
               q (torch.Tensor): The query tensor. Shape: (B, T, d_model), where B is the batch size, T is the sequence length, and d_model is the embedding dimension.
               k (torch.Tensor): The key tensor. Shape: (B, T, d_model).
               v (torch.Tensor): The value tensor. Shape: (B, T, d_model).
               attn_mask (Optional[torch.Tensor], optional): An optional attention mask. Shape: (B, num_heads, T, T). If provided, it masks the attention scores, typically to prevent attending to padding tokens or future tokens. Defaults to None.

            4. Returns:
               torch.Tensor: The context vector, which is the weighted sum of the values based on the attention weights. Shape: (B, T, d_model).
        """
        B, T, _ = q.shape
        q = self.q_proj(q).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(k).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(v).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1000000000.0)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.out_proj(context)

class FeedForward(nn.Module):
    """
    1. Summary:
           Represents a feedforward network module with two linear layers, ReLU activation, and dropout.

        2. Description:
           WHY: This module implements a standard feedforward network (FFN) block, commonly used in Transformer architectures and other neural networks. The FFN expands the input dimension, applies a non-linear activation (ReLU), and then projects it back to the original dimension. Dropout is used to prevent overfitting.
           WHEN: Use this class as a key building block in neural networks where non-linear transformations and feature expansion are required. It's particularly effective after attention layers in Transformers.
           WHERE: This class typically resides within a larger module, such as a Transformer layer, where it's applied to the output of the attention mechanism.
           HOW: The module consists of two linear layers (`nn.Linear`) with a ReLU activation function in between, and dropout layers applied after each linear transformation. The forward pass applies the first linear transformation, ReLU activation, dropout, the second linear transformation, and then dropout again.

        3. Parameters:
           d_model (int): The input and output dimension of the module. This is the dimension of the input tensor and the dimension of the output tensor.
           d_ff (int): The dimension of the hidden layer in the feedforward network. This determines the size of the intermediate representation after the first linear transformation.
           dropout (float, optional): The dropout probability. This determines the probability of dropping out neurons during training to prevent overfitting. Defaults to 0.0.

        4. Attributes:
           linear1 (nn.Linear): The first linear layer that expands the input dimension to `d_ff`.
           dropout (nn.Dropout): The dropout layer.
           linear2 (nn.Linear): The second linear layer that projects the hidden layer back to the original dimension `d_model`.

        5. Example:
           ```python
           import torch
           import torch.nn as nn

           # Example usage
           ffn = FeedForward(d_model=512, d_ff=2048, dropout=0.1)
           input_tensor = torch.randn(32, 128, 512)  # Batch size 32, sequence length 128, feature dimension 512
           output_tensor = ffn(input_tensor)
           print(f"Output shape: {output_tensor.shape}")  # Expected: torch.Size([32, 128, 512])
           ```
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float=0.0) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        1. Summary:
               Applies a feedforward network consisting of two linear layers with ReLU activation and dropout.

            2. Description:
               WHY: This function implements a standard feedforward network (FFN) block, commonly used in Transformer architectures and other neural networks. The FFN expands the input dimension, applies a non-linear activation (ReLU), and then projects it back to the original dimension. Dropout is used to prevent overfitting.
               WHEN: Use this function as a key building block in neural networks where non-linear transformations and feature expansion are required. It's particularly effective after attention layers in Transformers.
               WHERE: This function typically resides within a larger module, such as a Transformer layer, where it's applied to the output of the attention mechanism.
               HOW: The function first applies a linear transformation (`self.linear1`) to the input tensor, followed by a ReLU activation function. Dropout (`self.dropout`) is then applied to the output. This result is then passed through another linear transformation (`self.linear2`), and finally, dropout is applied again.

            3. Args:
               x (torch.Tensor): The input tensor to the feedforward network. The expected shape depends on the context where this FFN is used.

            4. Returns:
               torch.Tensor: The output tensor after processing by the two linear layers, ReLU activation, and dropout. The output shape typically matches the input shape.
        """
        return self.dropout(self.linear2(self.dropout(F.relu(self.linear1(x)))))

class TransformerEncoderBlock(nn.Module):
    """
    No docstring provided.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float=0.0, attn_dropout: float=0.0) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout=attn_dropout)
        self.drop1 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model, eps=1e-05)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
        self.drop2 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-05)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        No docstring provided.
        """
        h = self.mha(x, x, x, attn_mask=attn_mask)
        x = self.ln1(x + self.drop1(h))
        h = self.ffn(x)
        x = self.ln2(x + self.drop2(h))
        return x

class TransformerEncoder(nn.Module):
    """
    No docstring provided.
    """

    def __init__(self, d_model: int, num_layers: int, num_heads: int, d_ff: int, dropout: float=0.0, attn_dropout: float=0.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout, attn_dropout=attn_dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model, eps=1e-05)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        1. Summary:
               Processes the input tensor through a stack of encoder layers and applies layer normalization.

            2. Description:
               WHY: This function implements the forward pass of an encoder module, typically part of a Transformer model. It refines the input tensor by passing it through multiple encoder layers, each performing self-attention and feedforward transformations. Layer normalization ensures stable output distributions.
               WHEN: Use this function during the encoding phase of a sequence-to-sequence model or in standalone encoder models. It takes an input sequence and transforms it into a contextualized representation.
               WHERE: This function resides within an encoder module. It is called once to encode the input sequence before feeding it into a decoder or other downstream modules.
               HOW: The function iterates over a list of encoder layers (`self.layers`), passing the input tensor `x` through each layer. An optional attention mask (`attn_mask`) is used to control the self-attention mechanism within the layers, typically masking padding tokens. Finally, the output is normalized using `self.norm`.

            3. Args:
               x (torch.Tensor): The input tensor to the encoder module. Represents the input sequence to be encoded. Shape: (B, T, d_model), where B is the batch size, T is the sequence length, and d_model is the embedding dimension.
               attn_mask (Optional[torch.Tensor], optional): A mask for the self-attention mechanism within the encoder layers. This mask prevents the encoder from attending to padding tokens or other irrelevant parts of the sequence. Shape: (B, num_heads, T, T) or (T, T). Defaults to None.

            4. Returns:
               torch.Tensor: The output tensor after processing by the encoder layers and normalization. Represents the encoded representation of the input sequence. Shape: (B, T, d_model).
        """
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        return self.norm(x)

class TransformerDecoderBlock(nn.Module):
    """
    1. Summary:
           Represents a single decoder block in a Transformer decoder, consisting of self-attention, cross-attention, a feedforward network, layer normalization, and dropout.

        2. Description:
           WHY: This class implements a standard decoder block used in Transformer models. It combines self-attention to attend to the input sequence, cross-attention to attend to the encoder's memory, and a feedforward network to transform the representations. Layer normalization and dropout are used to improve training stability and prevent overfitting.
           WHEN: Use this class to build a Transformer decoder. It can be stacked multiple times to create a deep decoder network.
           WHERE: This class resides within a Transformer decoder module. It is instantiated multiple times to create a stack of decoder layers.
           HOW: The class consists of self-attention, cross-attention, a feedforward network, layer normalization layers, and dropout layers. The forward pass applies self-attention, adds the result to the input with dropout, normalizes the result, applies cross-attention with dropout and normalization, and finally applies the feedforward network with dropout and normalization.

        3. Parameters:
           d_model (int): The input and output dimension of the module. This is the dimension of the input queries, keys, and values, and the dimension of the output context vector.
           num_heads (int): The number of attention heads in the multi-head attention layers.
           d_ff (int): The dimension of the hidden layer in the feedforward network.
           dropout (float, optional): The dropout probability applied after the attention and feedforward network. Defaults to 0.0.
           attn_dropout (float, optional): The dropout probability applied to the attention weights in the multi-head attention layers. Defaults to 0.0.

        4. Attributes:
           ln1 (nn.LayerNorm): The layer normalization layer applied after the self-attention.
           self_mha (MultiHeadAttention): The multi-head self-attention layer.
           drop1 (nn.Dropout): The dropout layer applied after the self-attention.
           ln2 (nn.LayerNorm): The layer normalization layer applied after the cross-attention.
           cross_mha (MultiHeadAttention): The multi-head cross-attention layer.
           drop2 (nn.Dropout): The dropout layer applied after the cross-attention.
           ln3 (nn.LayerNorm): The layer normalization layer applied after the feedforward network.
           ffn (FeedForward): The feedforward network.
           drop3 (nn.Dropout): The dropout layer applied after the feedforward network.

        5. Example:
           ```python
           import torch
           import torch.nn as nn

           # Example usage
           decoder_block = TransformerDecoderBlock(d_model=512, num_heads=8, d_ff=2048, dropout=0.1, attn_dropout=0.1)
           x = torch.randn(32, 128, 512)  # Batch size 32, sequence length 128, feature dimension 512
           memory = torch.randn(32, 256, 512) # Batch size 32, encoder sequence length 256, feature dimension 512
           self_attn_mask = torch.rand(32, 8, 128, 128) > 0.5 # Example self-attention mask
           cross_attn_mask = torch.rand(32, 8, 128, 256) > 0.5 # Example cross-attention mask
           output_tensor = decoder_block(x, memory, self_attn_mask, cross_attn_mask)
           print(f"Output shape: {output_tensor.shape}")  # Expected: torch.Size([32, 128, 512])
           ```
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float=0.0, attn_dropout: float=0.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=1e-05)
        self.self_mha = MultiHeadAttention(d_model, num_heads, dropout=attn_dropout)
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-05)
        self.cross_mha = MultiHeadAttention(d_model, num_heads, dropout=attn_dropout)
        self.drop2 = nn.Dropout(dropout)
        self.ln3 = nn.LayerNorm(d_model, eps=1e-05)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, self_attn_mask: Optional[torch.Tensor]=None, cross_attn_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        1. Summary:
               Processes the input through a decoder layer consisting of self-attention, cross-attention, and a feedforward network, with layer normalization and dropout applied at each step.

            2. Description:
               WHY: This function implements the forward pass of a single decoder layer, as found in Transformer models. The decoder layer combines self-attention (to attend to the input sequence), cross-attention (to attend to the encoder's memory), and a feedforward network to transform the representations. Layer normalization and dropout are used to improve training stability and prevent overfitting.
               WHEN: Use this function as part of a stack of decoder layers in a Transformer decoder. It takes the output from the previous decoder layer (or the initial input) and the encoder's memory as input and generates the next representation.
               WHERE: This function resides within a decoder layer module. It is called during the forward pass of the decoder to process the input sequence and the encoder's memory.
               HOW: The function first applies self-attention to the input `x` using `self.self_mha`. The output of the self-attention is then added to the original input with a dropout layer applied, and the result is normalized using `self.ln1`. Next, cross-attention is applied between the normalized input and the encoder's memory `memory` using `self.cross_mha`. The output of the cross-attention is added to the normalized input with dropout, and the result is normalized using `self.ln2`. Finally, the normalized input is passed through a feedforward network `self.ffn`, and the output is added to the normalized input with dropout, and the result is normalized using `self.ln3`.

            3. Args:
               x (torch.Tensor): The input tensor to the decoder layer. Represents the output from the previous decoder layer or the initial input to the decoder. Shape: (B, T, d_model), where B is the batch size, T is the sequence length, and d_model is the embedding dimension.
               memory (torch.Tensor): The memory tensor from the encoder. This tensor contains contextual information about the input sequence that the decoder uses to generate the output sequence. Shape: (B, S, d_model), where B is the batch size, S is the encoder sequence length, and d_model is the embedding dimension.
               self_attn_mask (Optional[torch.Tensor], optional): A mask for the self-attention mechanism. This mask prevents the decoder from attending to future tokens in the sequence. Shape: (B, num_heads, T, T) or (T, T). Defaults to None.
               cross_attn_mask (Optional[torch.Tensor], optional): A mask for the cross-attention mechanism. This mask controls the attention between the decoder and the encoder memory. Shape: (B, num_heads, T, S) or (T, S). Defaults to None.

            4. Returns:
               torch.Tensor: The output tensor after processing by the decoder layer. Represents the next step in the decoded sequence. Shape: (B, T, d_model).
        """
        h = self.self_mha(x, x, x, attn_mask=self_attn_mask)
        x = self.ln1(x + self.drop1(h))
        h = self.cross_mha(x, memory, memory, attn_mask=cross_attn_mask)
        x = self.ln2(x + self.drop2(h))
        h = self.ffn(x)
        x = self.ln3(x + self.drop3(h))
        return x

class TransformerDecoder(nn.Module):
    """
    1. Summary:
           Represents a Transformer decoder consisting of a stack of decoder blocks and a final layer normalization.

        2. Description:
           WHY: This class implements a Transformer decoder, a key component in sequence-to-sequence models. The decoder takes the output from an encoder and generates an output sequence, attending to both the encoder output and the previously generated tokens. The stack of decoder blocks allows the model to learn complex dependencies in the data.
           WHEN: Use this class in sequence-to-sequence models, such as machine translation, text summarization, and speech recognition, where the goal is to generate a new sequence based on an input sequence.
           WHERE: This class resides within a Transformer model. It takes the encoder output and the previously generated tokens as input and generates the next token in the output sequence. It typically used in an autoregressive manner during inference
           HOW: The class consists of a stack of `TransformerDecoderBlock` layers and a final `LayerNorm` layer. The forward pass iterates over the decoder blocks, passing the input and encoder memory through each block. The output of the last block is then normalized.

        3. Parameters:
           d_model (int): The input and output dimension of the module. This is the dimension of the input queries, keys, and values, and the dimension of the output context vector.
           num_layers (int): The number of decoder blocks in the decoder stack.
           num_heads (int): The number of attention heads in the multi-head attention layers within each decoder block.
           d_ff (int): The dimension of the hidden layer in the feedforward network within each decoder block.
           dropout (float, optional): The dropout probability applied within each decoder block. Defaults to 0.0.
           attn_dropout (float, optional): The dropout probability applied to the attention weights within each decoder block. Defaults to 0.0.

        4. Attributes:
           layers (nn.ModuleList): A list of `TransformerDecoderBlock` layers.
           norm (nn.LayerNorm): The layer normalization layer applied to the output of the last decoder block.

        5. Example:
           ```python
           import torch
           import torch.nn as nn

           # Example usage
           decoder = TransformerDecoder(d_model=512, num_layers=6, num_heads=8, d_ff=2048, dropout=0.1, attn_dropout=0.1)
           x = torch.randn(32, 128, 512)  # Batch size 32, sequence length 128, feature dimension 512
           memory = torch.randn(32, 256, 512) # Batch size 32, encoder sequence length 256, feature dimension 512
           self_attn_mask = torch.rand(32, 8, 128, 128) > 0.5 # Example self-attention mask
           cross_attn_mask = torch.rand(32, 8, 128, 256) > 0.5 # Example cross-attention mask
           output_tensor = decoder(x, memory, self_attn_mask, cross_attn_mask)
           print(f"Output shape: {output_tensor.shape}")  # Expected: torch.Size([32, 128, 512])
           ```
    """

    def __init__(self, d_model: int, num_layers: int, num_heads: int, d_ff: int, dropout: float=0.0, attn_dropout: float=0.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TransformerDecoderBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout, attn_dropout=attn_dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, self_attn_mask: Optional[torch.Tensor]=None, cross_attn_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        1. Summary:
               Processes the input tensor through a stack of decoder layers and applies layer normalization.

            2. Description:
               WHY: This function implements the forward pass of a decoder module in a sequence-to-sequence model (e.g., a Transformer decoder). It iteratively refines the input tensor by passing it through multiple decoder layers, each of which typically consists of self-attention, cross-attention, and feed-forward networks. The final layer normalization ensures stable output distributions.
               WHEN: Use this function during the decoding phase of a sequence-to-sequence model. It takes the previous decoder output and the encoder's memory (context) as input and generates the next decoder output.
               WHERE: This function resides within a decoder module. It is called repeatedly in a loop to generate the output sequence, one element at a time (or in parallel if using techniques like autoregressive decoding).
               HOW: The function iterates over a list of decoder layers (`self.layers`), passing the input tensor `x` and the memory tensor `memory` through each layer. Optional attention masks (`self_attn_mask` and `cross_attn_mask`) are used to control the attention mechanism within the layers. Finally, the output is normalized using `self.norm`.

            3. Args:
               x (torch.Tensor): The input tensor to the decoder module. Represents the output from the previous decoder step or the initial input to the decoder.  Shape typically (batch_size, sequence_length, feature_dimension).
               memory (torch.Tensor): The memory tensor from the encoder. This tensor contains contextual information about the input sequence that the decoder uses to generate the output sequence. Shape typically (batch_size, sequence_length, feature_dimension).
               self_attn_mask (Optional[torch.Tensor], optional): A mask for the self-attention mechanism within the decoder layers. This mask prevents the decoder from attending to future tokens in the sequence. Defaults to None.
               cross_attn_mask (Optional[torch.Tensor], optional): A mask for the cross-attention mechanism within the decoder layers. This mask controls the attention between the decoder and the encoder memory. Defaults to None.

            4. Returns:
               torch.Tensor: The output tensor after processing by the decoder layers and normalization. Represents the next step in the decoded sequence. Shape typically (batch_size, sequence_length, feature_dimension).
        """
        for layer in self.layers:
            x = layer(x, memory, self_attn_mask=self_attn_mask, cross_attn_mask=cross_attn_mask)
        return self.norm(x)

class Reconstructor(nn.Module):
    """
    1. Summary:
           Represents a reconstruction module that projects, optionally applies log softmax, and optionally normalizes input tensors.

        2. Description:
           WHY: This module reconstructs or maps input features to a desired output space, often used as the final layer in autoencoders or similar architectures where the goal is to generate outputs resembling the input (or some target). The module provides options for applying log softmax for probabilistic outputs and z-score normalization for improved training.
           WHEN: Use this class at the end of a neural network when you need to reconstruct an input or generate data with a specific distribution. This is useful in tasks like anomaly detection, generative modeling, and representation learning.
           WHERE: This module sits at the end of a neural network, typically after a series of encoding or transformation layers. It takes the encoded representation and maps it back to the output space.
           HOW: The class consists of a linear projection layer (`nn.Linear`) followed by optional log softmax and z-score normalization. The linear projection transforms the input features to the desired output dimensions. The log softmax converts the outputs into log probabilities. The z-score normalization standardizes the outputs, potentially improving training stability.

        3. Parameters:
           d_model (int): The dimensionality of the input features. This determines the input size of the linear projection layer.
           out_channels (int): The number of output channels (or classes). This determines the output size of the linear projection layer.
           log_softmax (bool, optional): A flag indicating whether to apply log softmax to the output. Defaults to False.
           zscore (str | None, optional): Specifies the type of z-score normalization to apply.
                - None: No z-score normalization is applied.
                - 'batch': Z-score normalization is applied across the batch dimension.
                - 'time': Z-score normalization is applied across the time dimension (if applicable).
                Defaults to None.
           eps (float, optional): A small value added to the standard deviation during z-score normalization to prevent division by zero. Defaults to 1e-10.

        4. Attributes:
           proj (nn.Linear): The linear projection layer.
           use_log_softmax (bool): Stores whether log softmax is used.
           zscore (str | None): Stores the selected z-score mode.
           eps (float): Stores the epsilon value used in z-score normalization.

        5. Example:
           ```python
           import torch
           import torch.nn as nn

           # Example usage
           reconstructor = Reconstructor(d_model=128, out_channels=64, log_softmax=True, zscore='batch')
           input_tensor = torch.randn(16, 10, 128)  # Batch size 16, sequence length 10, feature dimension 128
           output_tensor = reconstructor(input_tensor)
           print(f"Output shape: {output_tensor.shape}")
           ```
    """

    def __init__(self, d_model: int, out_channels: int, *, log_softmax: bool=False, zscore: str | None=None, eps: float=1e-10) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, out_channels)
        self.use_log_softmax = log_softmax
        self.zscore = zscore
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        1. Summary:
               Applies a linear projection, optional log softmax, and optional z-score normalization to the input tensor.

            2. Description:
               WHY: This function performs a sequence of operations commonly used in the final layer of a neural network for classification tasks. The linear projection maps the input to the desired output dimension (number of classes). Log softmax converts the output to log probabilities, suitable for use with negative log-likelihood loss. Z-score normalization can improve training stability and convergence by standardizing the output distribution.
               WHEN: Use this function as the final layer in a model when performing classification, especially when dealing with time series data or other sequential inputs. The choice of log softmax and z-score normalization depends on the specific task and dataset characteristics.
               WHERE: This function sits at the end of a neural network, just before the loss calculation. It transforms the model's raw output into a format suitable for comparison with the target labels.
               HOW: The function first applies a linear projection using `self.proj`. Then, if `self.use_log_softmax` is True, it applies the log softmax function along the last dimension. Finally, if `self.zscore` is not None, it performs z-score normalization. The z-score can be calculated across the batch or across the time dimension, depending on the value of `self.zscore`.

            3. Args:
               x (torch.Tensor): The input tensor to be processed. The expected shape depends on the model architecture.

            4. Returns:
               torch.Tensor: The processed output tensor after linear projection, optional log softmax, and optional z-score normalization. The shape of the output tensor depends on the linear projection and the input shape.

            5. Raises:
               ValueError: If `self.zscore` is set to an unsupported value (i.e., not None, 'batch', or 'time').
        """
        y = self.proj(x)
        if self.use_log_softmax:
            y = F.log_softmax(y, dim=-1)
        if self.zscore is None:
            return y
        if self.zscore == 'batch':
            mean = y.mean(dim=0, keepdim=True)
            std = y.std(dim=0, keepdim=True)
        elif self.zscore == 'time':
            mean = y.mean(dim=1, keepdim=True)
            std = y.std(dim=1, keepdim=True)
        else:
            raise ValueError(f'Unsupported zscore mode: {self.zscore}')
        return (y - mean) / (std + self.eps)

class ArtifactRemovalTransformer(nn.Module):
    """
    No docstring provided.
    """

    def __init__(self, in_channels: int, out_channels: int, embedding_size: int=128, num_encoder_layers: int=6, num_decoder_layers: int=6, num_heads: int=8, feedforward_size: int=2048, dropout: float=0.1, max_len: int=2048, pos_mode: str='sinusoidal', recon_log_softmax: bool=False, recon_zscore: str | None=None) -> None:
        super().__init__()
        self.src_embed = nn.Sequential(ExpandConv1x1(in_channels, embedding_size), PositionalEmbedding(max_len=max_len, d_model=embedding_size, mode=pos_mode), nn.Dropout(dropout))
        self.encoder = TransformerEncoder(d_model=embedding_size, num_layers=num_encoder_layers, num_heads=num_heads, d_ff=feedforward_size, dropout=dropout, attn_dropout=dropout)
        self.tgt_embed = nn.Sequential(ExpandConv1x1(out_channels, embedding_size), PositionalEmbedding(max_len=max_len, d_model=embedding_size, mode=pos_mode), nn.Dropout(dropout))
        self.decoder = TransformerDecoder(d_model=embedding_size, num_layers=num_decoder_layers, num_heads=num_heads, d_ff=feedforward_size, dropout=dropout, attn_dropout=dropout)
        self.reconstructor = Reconstructor(d_model=embedding_size, out_channels=out_channels, log_softmax=recon_log_softmax, zscore=recon_zscore)

    def forward(self, src: torch.Tensor, tgt: Optional[torch.Tensor]=None, src_mask: Optional[torch.Tensor]=None, tgt_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        No docstring provided.
        """
        src_x = self.src_embed(src)
        enc_attn_mask = None
        if src_mask is not None:
            if src_mask.dtype != torch.bool:
                src_mask = src_mask.to(torch.bool)
            enc_attn_mask = (~src_mask).unsqueeze(1).unsqueeze(2)
        memory = self.encoder(src_x, attn_mask=enc_attn_mask)
        if tgt is None:
            tgt = src
        tgt_x = self.tgt_embed(tgt)
        dec_self_mask = None
        dec_cross_mask = enc_attn_mask
        if tgt_mask is not None:
            if tgt_mask.dtype != torch.bool:
                tgt_mask = tgt_mask.to(torch.bool)
            dec_self_mask = (~tgt_mask).unsqueeze(1)
        out = self.decoder(tgt_x, memory, dec_self_mask, dec_cross_mask)
        reconstructed = self.reconstructor(out)
        return reconstructed.permute(0, 2, 1)

@dataclass
class Seq2SeqLMOutput(ModelOutput):
    """
    No docstring provided.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None

class ArtifactRemovalTransformerForConditionalGeneration(PreTrainedModel):
    """
    A Hugging Face-compatible wrapper for the ArtifactRemovalTransformer model.
    This model is designed for a conditional generation task (sequence-to-sequence),
    like removing artifacts from EEG signals.
    """
    config_class = ArtifactRemovalTransformerConfig

    def __init__(self, config: ArtifactRemovalTransformerConfig):
        super().__init__(config)
        self.model = ArtifactRemovalTransformer(in_channels=config.in_channels, out_channels=config.out_channels, embedding_size=config.embedding_size, feedforward_size=config.feedforward_size, num_encoder_layers=config.num_encoder_layers, num_decoder_layers=config.num_decoder_layers, num_heads=config.num_heads, dropout=config.dropout, max_len=config.max_len, pos_mode=config.pos_mode, recon_log_softmax=config.recon_log_softmax, recon_zscore=config.recon_zscore)
        self.loss_fct = nn.MSELoss()
        self.eps = 1e-10

    def _zscore_loss(self, logits, labels):
        """
        No docstring provided.
        """
        logits = logits.permute(0, 2, 1)
        labels = labels.permute(0, 2, 1)
        l_mean, l_std = (logits.mean(dim=1, keepdim=True), logits.std(dim=1, keepdim=True))
        lab_mean, lab_std = (labels.mean(dim=1, keepdim=True), labels.std(dim=1, keepdim=True))
        logits_z = (logits - l_mean) / (l_std + self.eps)
        labels_z = (labels - lab_mean) / (lab_std + self.eps)
        return self.loss_fct(logits_z, labels_z)

    def forward(self, input_values: torch.Tensor, labels: Optional[torch.Tensor]=None, **kwargs) -> Seq2SeqLMOutput:
        """
        The forward pass of the model.

        Args:
            input_values (`torch.Tensor` of shape `(batch_size, num_channels, sequence_length)`):
                The input EEG signals with artifacts.
            labels (`torch.Tensor` of shape `(batch_size, num_channels, sequence_length)`, *optional*):
                The clean target EEG signals. If provided, the model calculates the loss.

        Returns:
            `Seq2SeqLMOutput` with loss and logits.
        """
        decoder_input_values = labels if labels is not None else input_values
        logits = self.model(src=input_values, tgt=decoder_input_values)
        loss = None
        if labels is not None:
            if self.config.loss_zscore:
                loss = self._zscore_loss(logits, labels)
            else:
                loss = self.loss_fct(logits, labels)
        return Seq2SeqLMOutput(loss=loss, logits=logits)