from transformers import AutoTokenizer, AutoConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class multiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    def scaled_dot_product_attention(self,Q,K,V,mask = None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e20)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

class feedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))

class embeddings(nn.Module):
    def __init__(self, vocabSize,hiddenSize,maxPositionEmbeddings):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocabSize, 
                                             hiddenSize)
        self.position_embeddings = nn.Embedding(maxPositionEmbeddings,
                                                hiddenSize)
        self.layer_norm = nn.LayerNorm(hiddenSize, eps=1e-12)
        self.dropout = nn.Dropout()

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0).cuda()
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class encoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = multiHeadAttention(d_model, num_heads)
        self.feed_forward = feedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class decoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = multiHeadAttention(d_model, num_heads)
        self.cross_attn = multiHeadAttention(d_model, num_heads)
        self.feed_forward = feedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_ckpt = "google-bert/bert-base-multilingual-uncased"
        self.config = AutoConfig.from_pretrained(self.model_ckpt)
        self.encoder_embedding = embeddings(105879,self.config.hidden_size,self.config.max_position_embeddings)
        self.decoder_embedding = embeddings(105879,self.config.hidden_size,self.config.max_position_embeddings)
        self.encoder_layers = nn.ModuleList([encoderBlock(self.config.hidden_size,self.config.num_attention_heads, self.config.intermediate_size, self.config.hidden_dropout_prob) for _ in range(12)])
        self.decoder_layers = nn.ModuleList([decoderBlock(self.config.hidden_size,self.config.num_attention_heads, self.config.intermediate_size, self.config.hidden_dropout_prob) for _ in range(12)])
        self.fc = nn.Linear(self.config.hidden_size, 105879)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).cuda()
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3).cuda()
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().cuda()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.encoder_embedding(src)
        tgt_embedded = self.decoder_embedding(tgt)
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
        output = self.fc(dec_output)
        return output