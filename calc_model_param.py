import math

def calculate_relative_attention_bias(relative_attention_num_buckets, num_heads):
    return relative_attention_num_buckets * num_heads

def calculate_embedding_layer_params(d_model, vocab):
    return d_model * vocab

def calculate_multihead_attention_params(d_model, d_kv, num_heads):
    # W_q, W_k, W_v, W_o, layer_norm
    return (d_kv * num_heads * d_model) * 4


def calculate_positionwise_feedforward_params(d_model, d_ff):
    # W_1, W_2
    return d_model * d_ff * 2


def calculate_layer_norm_params(d_model):
    # scale, bias
    return d_model


def calculate_encoder_params(d_model, d_ff, d_kv, num_heads):
    # multihead attention, positionwise feedforward, layer norm
    return calculate_multihead_attention_params(d_model, d_kv, num_heads) + calculate_positionwise_feedforward_params(d_model, d_ff) + 2 * calculate_layer_norm_params(d_model)


def calculate_decoder_params(d_model, d_ff, d_kv, num_heads):
    # multihead attention (self-attention), multihead attention (encoder-decoder attention), positionwise feedforward, layer norm
    return calculate_multihead_attention_params(d_model, d_kv, num_heads) * 2 + calculate_positionwise_feedforward_params(d_model, d_ff) + calculate_layer_norm_params(d_model) * 3


def calculate_transformer_params(d_model, d_ff, d_kv, num_heads):
    # encoder, decoder
    return (calculate_encoder_params(d_model, d_ff, d_kv, num_heads) + calculate_decoder_params(d_model, d_ff, d_kv, num_heads))


def calculate_model_params(d_model, d_ff, num_layers, d_kv, num_heads, tie_word_embeddings, vocab_size, relative_attention_num_buckets):
    # calculate the total number of parameters in a model with multi layers
    # transformers + embedding layer + position embedding + final layer norm
    return calculate_transformer_params(d_model, d_ff, d_kv, num_heads) * num_layers + (1 if tie_word_embeddings == 1 else 2) * calculate_embedding_layer_params(d_model, vocab_size) + calculate_relative_attention_bias(relative_attention_num_buckets, num_heads) * 2 + calculate_layer_norm_params(d_model) * 2


def calculate_decrease_dff_increase_layers(d_model, d_ff, num_layers, d_kv, num_heads, tie_word_embeddings, vocab_size, relative_attention_num_buckets, decrease_dff):
    # calculate the number of layers that can be increased when d_ff is decreased by a certain percentage
    decreased_d_ff = round(d_ff * (1 - decrease_dff))
    original_params = calculate_model_params(d_model, d_ff, d_kv, num_heads, num_layers, tie_word_embeddings, vocab_size, relative_attention_num_buckets)
    decreased_params = calculate_model_params(d_model, decreased_d_ff, d_kv, num_heads, num_layers, tie_word_embeddings, vocab_size, relative_attention_num_buckets)
    print("original transformer size: ", original_params)
    print("decreased transformer size: ", decreased_params)
    print("new transformer size: ", calculate_transformer_params(d_model, decreased_d_ff, d_kv, num_heads))
    increase_in_layer = (original_params - decreased_params) / calculate_transformer_params(d_model, decreased_d_ff, d_kv, num_heads)
    return math.floor(increase_in_layer)


if __name__ == "__main__":
    # # t5-base
    # d_model = 768
    # d_ff = 3072
    # num_heads = 12
    # num_layers = 12
    # vocab_size = 32128
    # d_kv = 64
    # relative_attention_num_buckets = 32
    # tie_word_embeddings = True
    
    # # t5-large
    # d_model = 1024
    # d_ff = 4096
    # num_heads = 16
    # num_layers = 24
    # vocab_size = 32128
    # d_kv = 64
    # relative_attention_num_buckets = 32
    # tie_word_embeddings = True
    
    # t5-3b
    d_model = 1024
    d_ff = 16384
    num_heads = 32  
    num_layers = 24
    vocab_size = 32128
    d_kv = 128
    relative_attention_num_buckets = 32
    tie_word_embeddings = True
    
    # flan-t5-base
    d_model = 768
    d_ff = 2048
    num_heads = 12
    num_layers = 12
    vocab_size = 32128
    d_kv = 64
    relative_attention_num_buckets = 32
    tie_word_embeddings = False
    
    print("Total params: ", calculate_model_params(d_model, d_ff, num_layers, d_kv, num_heads, tie_word_embeddings, vocab_size, relative_attention_num_buckets))
    # decrease_dff_perc = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    # for perc in decrease_dff_perc:
    #     increase_layers = calculate_decrease_dff_increase_layers(d_model, d_ff, num_layers, d_kv, num_heads, tie_word_embeddings, vocab_size, relative_attention_num_buckets, perc)
    #     print(f"Decrease d_ff by {perc * 100}% can increase layers by {increase_layers}")
