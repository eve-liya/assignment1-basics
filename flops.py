vocab_size = 50257
context_length = 1024
num_layers = 48
d_model = 1600
num_heads = 25
d_ff = 6400

norm_weight = d_model
token_embedding_weights = vocab_size * d_model
position_embedding_weights = context_length*d_model
mhsa_weight = d_model*d_model * 4
ffn = d_model * d_ff + d_ff * d_model
block_weights = mhsa_weight + norm_weight + norm_weight + ffn
layer_weights = num_layers * block_weights
lm_head_weights = d_model * vocab_size

trainable_parameters = token_embedding_weights + position_embedding_weights + layer_weights + norm_weight + lm_head_weights

models = {
    'small': {'dmodel': 768, 'layers': 12},
    'medium': {'dmodel': 1024, 'layers': 24},
    'large': {'dmodel': 1280, 'layers': 36},
    'Xlarge': {'dmodel': 1600, 'layers': 48}
}

# Constants
context_length = 1024 # 16,384 for gpt2 long context calc
vocab_size = 50257 


def calculate_flops(dmodel, layers):

    mhsa_projection_flops = 8 * context_length * dmodel**2 * layers
    mhsa_attn_flops = 4 * context_length**2 * dmodel * layers
    ffn_flops = 16 * context_length * dmodel**2 * layers
    lm_head_flops = 2 * context_length * dmodel * vocab_size

    total_flops = lm_head_flops + mhsa_projection_flops + mhsa_attn_flops + ffn_flops

    mhsa_projection_percentage = (mhsa_projection_flops / total_flops) * 100
    mhsa_attn_percentage = (mhsa_attn_flops / total_flops) * 100
    ffn_percentage = (ffn_flops / total_flops) * 100
    lm_head_percentage = (lm_head_flops / total_flops) * 100
    
    return mhsa_projection_flops, mhsa_attn_flops, ffn_flops, lm_head_flops, total_flops,  \
            mhsa_projection_percentage, mhsa_attn_percentage,  ffn_percentage, lm_head_percentage

for model_name, config in models.items():
    dmodel = config['dmodel']
    layers = config['layers']
    
    mhsa_projection_flops, mhsa_attn_flops, ffn_flops, head_flops, total_flops, mhsa_projection_percentage, mhsa_attn_percentage, ffn_percentage, head_percentage = calculate_flops(dmodel, layers)
    
    print(f"{model_name.capitalize()} Model:")
    print(f"  MHSA Projections FLOPs: {mhsa_projection_flops / 1e9:.2f}B")
    print(f"  MHSA Attention FLOPs: {mhsa_attn_flops / 1e9:.2f}B")
    print(f"  FFN FLOPs: {ffn_flops / 1e9:.2f}B")
    print(f"  LM_head FLOPs: {head_flops / 1e9:.2f}B")
    print(f"  Total FLOPs: {total_flops / 1e12:.2f}T")
    print(f"  Total FLOPs: {total_flops}")
    print(f"  MHSA Percentage: {mhsa_projection_percentage:.2f}%")
    print(f"  MHSA Percentage: {mhsa_attn_percentage:.2f}%")
    print(f"  FFN Percentage: {ffn_percentage:.2f}%")
    print(f"  LM Head Percentage: {head_percentage:.2f}%")
    print("-" * 50)
