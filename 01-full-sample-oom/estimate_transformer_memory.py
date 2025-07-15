batch_size = 1
scale_factor = 4
sequence_length = (720 / scale_factor) * (1440 / scale_factor)
hidden_size = 64
dtype_bytes = 2
include_gradients = True
vocab_size = 180
num_layers = 6
num_heads = 6

# 1. Input embeddings
embedding_activations = batch_size * sequence_length * hidden_size * dtype_bytes

# 2. Attention computations (per layer)
# Q, K, V matrices
qkv_activations = 3 * batch_size * sequence_length * hidden_size * dtype_bytes
# Attention scores (batch_size * num_heads * seq_length * seq_length)
attention_scores = batch_size * num_heads * sequence_length * sequence_length * dtype_bytes
# Attention output
attention_output = batch_size * sequence_length * hidden_size * dtype_bytes

# 3. FFN activations (per layer)
# Typically FFN hidden dim is 4x larger than model hidden dim
ffn_activations = 4 * batch_size * sequence_length * hidden_size * dtype_bytes

# 4. Layer norm activations (2 per layer)
layer_norm_activations = 2 * batch_size * sequence_length * hidden_size * dtype_bytes

# Total activations per layer
activations_per_layer = (qkv_activations + attention_scores + attention_output + 
                        ffn_activations + layer_norm_activations)

# 5. Model parameters
# Embedding parameters
embedding_params = vocab_size * hidden_size * dtype_bytes
# Attention parameters per layer
attention_params = 4 * hidden_size * hidden_size * dtype_bytes  # Q,K,V,O matrices
# FFN parameters per layer
ffn_params = 8 * hidden_size * hidden_size * dtype_bytes  # 2 linear layers with 4x hidden
# Layer norm parameters
layer_norm_params = 4 * hidden_size * dtype_bytes

params_per_layer = attention_params + ffn_params + layer_norm_params
total_params = embedding_params + (params_per_layer * num_layers)

# 6. Calculate totals
total_activations = embedding_activations + (activations_per_layer * num_layers)

# 7. Account for gradients if needed
if include_gradients:
    total_activations *= 2  # Need to store gradients for activations
    total_params *= 2       # Need to store gradients for parameters

# 8. Additional memory for optimizer states (assuming Adam)
if include_gradients:
    optimizer_memory = total_params * 8  # Adam uses 8 bytes per parameter
else:
    optimizer_memory = 0

# Calculate totals in GB
results = {
    'activation_memory_gb': 1e-9*total_activations,
    'parameter_memory_gb': 1e-9*total_params,
    'optimizer_memory_gb': 1e-9*optimizer_memory,
    'total_memory_gb': 1e-9*(total_activations + total_params + optimizer_memory)
}

print(results)
