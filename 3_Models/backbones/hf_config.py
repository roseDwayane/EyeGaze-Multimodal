"""
HuggingFace configuration for Artifact Removal Transformer
"""

from transformers import PretrainedConfig


class ArtifactRemovalTransformerConfig(PretrainedConfig):
    """
    Configuration class for ArtifactRemovalTransformer model.
    """
    model_type = "artifact_removal_transformer"

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        embedding_size: int = 128,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        feedforward_size: int = 2048,
        dropout: float = 0.1,
        max_len: int = 2048,
        pos_mode: str = 'sinusoidal',
        recon_log_softmax: bool = False,
        recon_zscore: str = None,
        loss_zscore: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedding_size = embedding_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.feedforward_size = feedforward_size
        self.dropout = dropout
        self.max_len = max_len
        self.pos_mode = pos_mode
        self.recon_log_softmax = recon_log_softmax
        self.recon_zscore = recon_zscore
        self.loss_zscore = loss_zscore
