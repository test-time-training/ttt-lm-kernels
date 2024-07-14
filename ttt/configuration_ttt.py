""" TTT model configuration"""

from transformers.configuration_utils import PretrainedConfig

TTT_STANDARD_CONFIGS = {
    "1b": {
        "hidden_size": 2048,
        "intermediate_size": 5504,
        "num_hidden_layers": 24,
        "num_attention_heads": 32,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
        'conv_before_ttt': True,
    },
}


class TTTConfig(PretrainedConfig):
    model_type = "ttt"

    def __init__(
        self,
        vocab_size=50277,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        seq_modeling_block='ttt-linear',
        ttt_base_lr=1.0,
        mini_batch_size=16,
        conv_before_ttt=False,
        conv_kernel=4,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        self.initializer_range = initializer_range
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.seq_modeling_block = seq_modeling_block
        self.ttt_base_lr = ttt_base_lr
        self.mini_batch_size = mini_batch_size
        self.conv_before_ttt = conv_before_ttt
        self.conv_kernel = conv_kernel

        super().__init__(
             **kwargs,
        )
