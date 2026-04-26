import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMatrix(nn.Module):

    def __init__(self, use_mask: bool = False) -> None:
        super().__init__()
        # Mask is [batch_size x window_size_queries x window_size_keys]
        self.use_mask = use_mask

    def forward(self, K: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """
        STUDENT MUST WRITE:

        Computes attention weights given key and query matrices.

        :param K: is [batch_size x window_size_keys x embedding_size]
        :param Q: is [batch_size x window_size_queries x embedding_size]
        :return: attention matrix [batch_size x window_size_queries x window_size_keys]
        """
        window_size_queries = Q.size(1)   # window size of queries
        window_size_keys    = K.size(1)   # window size of keys
        embedding_size_keys = K.size(2)

        embedding_size_keys = K.size(2)

        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(embedding_size_keys)

        if self.use_mask:
            window_size_queries = Q.size(1)
            window_size_keys = K.size(1)
            mask = torch.triu(torch.ones(window_size_queries, window_size_keys, device=K.device), diagonal=1).bool()
            scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
        
        return F.softmax(scores, dim = -1)


class AttentionHead(nn.Module):
    def __init__(self, input_size: int, output_size: int, is_self_attention: bool) -> None:
        super().__init__()
        self.use_mask = is_self_attention

        self.key = nn.Linear(input_size, output_size)
        self.value = nn. Linear(input_size, output_size, bias = False)
        self.query = nn. Linear(input_size, output_size, bias = False)

        self.attention = AttentionMatrix(use_mask = self.use_mask)


    def forward(self, inputs_for_keys: torch.Tensor, inputs_for_values: torch.Tensor, inputs_for_queries: torch.Tensor) -> torch.Tensor:
        """
        Runs a single attention head.

        :param inputs_for_keys:    tensor of [batch_size x KEY_WINDOW_SIZE   x input_size]
        :param inputs_for_values:  tensor of [batch_size x KEY_WINDOW_SIZE   x input_size]
        :param inputs_for_queries: tensor of [batch_size x QUERY_WINDOW_SIZE x input_size]
        :return:                   tensor of [BATCH_SIZE x QUERY_WINDOW_SIZE x output_size]
        """

        K = self.key(inputs_for_keys)
        V = self.value(inputs_for_values)
        Q = self.query(inputs_for_queries)

        attn_weights = self.attention(K, Q)

        return torch.bmm(attn_weights, V)


class MultiHeadedAttention(nn.Module):
    def __init__(self, emb_sz: int, use_mask: bool) -> None:
        super().__init__()

        head_dim = emb_sz // 3

        self.head1 = AttentionHead(emb_sz, head_dim, use_mask)
        self.head2 = AttentionHead(emb_sz, head_dim, use_mask)
        self.head3 = AttentionHead(emb_sz, head_dim, use_mask)

        self.linear = nn.Linear(3 * head_dim, emb_sz)


    def forward(self, inputs_for_keys: torch.Tensor, inputs_for_values: torch.Tensor, inputs_for_queries: torch.Tensor) -> torch.Tensor:
        """
        Runs multiheaded attention.

        Requirements:
            - 3 attention heads, each of output size emb_sz // 3
            - Concatenate the three head outputs along the last dimension
            - Pass through a final linear layer

        :param inputs_for_keys:    tensor of [batch_size x KEY_WINDOW_SIZE   x emb_sz]
        :param inputs_for_values:  tensor of [batch_size x KEY_WINDOW_SIZE   x emb_sz]
        :param inputs_for_queries: tensor of [batch_size x QUERY_WINDOW_SIZE x emb_sz]
        :return:                   tensor of [BATCH_SIZE x QUERY_WINDOW_SIZE x emb_sz]
        """

        h1 = self.head1(inputs_for_keys, inputs_for_values, inputs_for_queries)
        h2 = self.head2(inputs_for_keys, inputs_for_values, inputs_for_queries)
        h3 = self.head3(inputs_for_keys, inputs_for_values, inputs_for_queries)
        
        combined = torch.cat((h1, h2, h3), dim = -1)
        return self.linear(combined)


class TransformerBlock(nn.Module):
    def __init__(self, emb_sz: int, multiheaded: bool = False, dropout: float = 0.2) -> None:
        super().__init__()

        if multiheaded:
            self.attention = MultiHeadedAttention(emb_sz, use_mask = True)
            self.cross_attention = MultiHeadedAttention(emb_sz, use_mask = False)
        else:
            self.attention = AttentionHead(emb_sz, emb_sz, is_self_attention = True)
            self.cross_attention = AttentionHead(emb_sz, emb_sz, is_self_attention = False)

        self.norm1 = nn.LayerNorm(emb_sz)
        self.norm2 = nn.LayerNorm(emb_sz)
        self.norm3 = nn.LayerNorm(emb_sz)

        self.feed_forward = nn.Sequential(nn.Linear(emb_sz, emb_sz * 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(emb_sz * 4, emb_sz))

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, context_sequence: torch.Tensor) -> torch.Tensor:
        """
        Runs one Transformer decoder block.

        :param inputs:           tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH   x EMBEDDING_SIZE]
        :param context_sequence: tensor of shape [BATCH_SIZE x CONTEXT_SEQ_LENGTH x EMBEDDING_SIZE]
        :return:                 tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH   x EMBEDDING_SIZE]
        """

        self_attn_out = self.attention(inputs, inputs, inputs)
        x = self.norm1(inputs + self.dropout1(self_attn_out))

        cross_attn_out = self.cross_attention(context_sequence, context_sequence, x)
        x = self.norm2(x + self.dropout2(cross_attn_out))

        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_out))

        return x


def positional_encoding(length: int, depth: int) -> torch.Tensor:
    """
    Generates a sinusoidal positional encoding matrix.

    :param length: number of positions (sequence length)
    :param depth:  embedding dimension (must be even)
    :return:       torch.FloatTensor of shape [length x depth]

    Hint: use alternating sin/cos at different frequencies.
    See https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer
    """
    # pe = torch.zeros(length, depth)

    # position = torch.arange(0, length, dtype = torch.float).unsqueeze(1) # [length, 1]
    # div_term = torch.exp(torch.arange(0, depth, 2).float() * (-math.log(10000.0) / depth)) # [depth / 2]

    # pe[:, 0::2] = torch.sin(position * div_term) # even
    # pe[:, 1::2] = torch.cos(position * div_term) # odd

    # return pe
    depth = depth // 2

    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)   # [length, 1]
    depths = torch.arange(depth, dtype=torch.float32).unsqueeze(0) / depth  # [1, depth]

    angle_rates = 1 / (10000 ** depths)   # [1, depth]
    angle_rads = positions * angle_rates  # [length, depth]

    pe = torch.cat([torch.sin(angle_rads), torch.cos(angle_rads)], dim=1)
    return pe


class PositionalEncoding(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, window_size: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.embed_size = embed_size

        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout)

        # Sinusoidal positional encoding which precomputed and stored as a buffer (not trainable)
        # HINT: call positional_encoding(length=window_size, depth=embed_size)
        pos_enc = positional_encoding(length=window_size, depth=embed_size)
        self.register_buffer('pos_encoding', pos_enc[:window_size, :])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        STUDENT MUST WRITE:

        :param x: integer tensor of token ids [BATCH_SIZE x WINDOW_SIZE]
        :return:  float tensor [BATCH_SIZE x WINDOW_SIZE x EMBED_SIZE]

        Steps:
          1. Embed x with self.embedding.
          2. Scale the embeddings by sqrt(embed_size)
          3. Add self.pos_encoding, broadcasted over the batch dimension
          4. Apply dropout
        """
        x = self.embedding(x)

        x = x * math.sqrt(self.embed_size)

        x = x + self.pos_encoding.unsqueeze(0)

        return self.dropout(x)
