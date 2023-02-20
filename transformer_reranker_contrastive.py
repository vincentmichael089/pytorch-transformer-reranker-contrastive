# %%
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_metric_learning.losses import NTXentLoss
from transformers import get_cosine_schedule_with_warmup
import numpy as np
import math

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

class PositionalEncoding(pl.LightningModule):
  def __init__(self, num_hiddens, dropout, max_len=2000):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(dropout)
    self.register_buffer("P", torch.zeros((1, max_len, num_hiddens)))
    self.register_buffer("tempX", torch.arange(max_len, dtype=torch.float32, device = self.device).reshape(-1, 1) / 
      torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32, device = self.device) / num_hiddens))
    self.P[:, :, 0::2] = torch.sin(self.tempX)
    self.P[:, :, 1::2] = torch.cos(self.tempX)

  def forward(self, X):
    X = X + self.P[:, :X.shape[1], :]
    return self.dropout(X)

class TokenEmbedding(pl.LightningModule):
  def __init__(self, vocab_size: int, emb_size):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, emb_size)
    self.emb_size = emb_size

  def forward(self, tokens):
    return self.embedding(tokens.to(self.device)) * math.sqrt(self.emb_size)

class gX(pl.LightningModule):
  def __init__(
    self, input_dim, intermediate_dim, output_dim, dropout
  ):
    super().__init__()
    self.dense_1 = nn.Linear(input_dim, intermediate_dim)
    self.dense_2 = nn.Linear(intermediate_dim, output_dim)
    self.activation_fn = nn.ReLU()
    self.dropout = nn.Dropout(dropout)

  def forward(self, input):
    x = self.dense_1(input[:, 0, :] )
    x = self.dropout(x)
    x = self.activation_fn(x)
    x = self.dense_2(x)
    return x

class VanillaTransformerEncoder(pl.LightningModule):
  def  __init__(
    self,
    num_encoder_layers,
    emb_size,
    nhead,
    src_vocab_size,
    dim_ff,
    dropout
  ):
    super().__init__()
    encoder_layer = nn.TransformerEncoderLayer(
      d_model=emb_size, 
      nhead=nhead, 
      dim_feedforward=dim_ff, 
      dropout=dropout, 
      activation=nn.functional.relu, 
      layer_norm_eps= 1e-5, 
      batch_first=True, 
      norm_first=False)

    encoder_norm = nn.LayerNorm(emb_size, 1e-5)
    self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
  
  def forward(self, x, padding_mask):
    return self.encoder(src=x, src_key_padding_mask=padding_mask)

class TransformerReranker(pl.LightningModule):
  def __init__(
    self,
    num_encoder_layers,
    emb_size,
    nhead,
    vocab_size_inpt,
    vocab_size_cand,
    dim_ff,
    dropout
  ):
    super().__init__()
    self.positional_encoding = PositionalEncoding(emb_size, dropout = dropout)
    self.tok_emb_inp = TokenEmbedding(vocab_size_inpt, emb_size)
    self.tok_emb_cand = TokenEmbedding(vocab_size_cand, emb_size)
    self.encoder_inp = VanillaTransformerEncoder(num_encoder_layers, emb_size, nhead, vocab_size_inpt, dim_ff, dropout)  
    self.encoder_cand = VanillaTransformerEncoder(num_encoder_layers, emb_size, nhead, vocab_size_cand, dim_ff, dropout)  
    self.gx_inp = gX(emb_size, 256, 32, 0.1)
    self.gx_cand = gX(emb_size, 256, 32, 0.1)

  def forward(self, input, candidate, padding_mask_inp, padding_mask_cand):
    inp = self.positional_encoding(self.tok_emb_inp(input))
    cand = self.positional_encoding(self.tok_emb_cand(candidate))
    encoded_inp = self.encoder_inp(inp, padding_mask_inp)
    encoded_cand = self.encoder_cand(cand, padding_mask_cand)
    compact1 = self.gx_inp(encoded_inp)
    compact2 = self.gx_cand(encoded_cand)
    return compact1, compact2

class PlTransformer(pl.LightningModule):
  def __init__(self, model, lr):
    super().__init__()
    self.model = model
    self.lr = lr
    self.softmax = nn.Softmax(dim=-2)
    self.contrastive = NTXentLoss(temperature=0.10)
    self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    for p in self.model.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)
  
  def training_step(self, batch, batch_idx): 
    input = batch[0]
    candidate = batch[1]
    padding_mask_input = (input == PAD_IDX)
    padding_mask_candidate = (candidate == PAD_IDX)

    compact_input, compact_candidate = self.model(input, candidate, padding_mask_input, padding_mask_candidate)
    embeddings = torch.cat((compact_input[0].unsqueeze(0), compact_input))
    indices = torch.arange(0, compact_candidate.size(0)+1).cuda()
    indices[1] = indices[0]
    loss = self.contrastive(embeddings, indices)
    self.log('train_loss', loss, on_step=True, on_epoch=True)
    return loss

  def validation_step(self, batch, batch_idx):
    input = batch[0]
    candidate = batch[1]
    padding_mask_input = (input == PAD_IDX)
    padding_mask_candidate = (candidate == PAD_IDX)

    compact_input, compact_candidate = self.model(input, candidate, padding_mask_input, padding_mask_candidate)
    embeddings = torch.cat((compact_input[0].unsqueeze(0), compact_input))
    indices = torch.arange(0, compact_candidate.size(0)+1).cuda()
    indices[1] = indices[0]
    loss = self.contrastive(embeddings, indices)
    self.log('validation_loss', loss, on_step=True, on_epoch=True)
    return loss


  def predict_step(self, batch, batch_idx):
    input = batch[0]
    candidate = batch[1]
    padding_mask_input = (input == PAD_IDX)
    padding_mask_candidate = (candidate == PAD_IDX)

    compact_input, compact_candidate = self.model(input, candidate, padding_mask_input, padding_mask_candidate)
    out = self.cos(compact_input, compact_candidate)
    position = out.argmin().item()
    return input[position]


  def configure_optimizers(self):
    optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9)    
    scheduler = get_cosine_schedule_with_warmup(
      optimizer,
      num_warmup_steps=10000,
      num_training_steps= 235200 
      )

    scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1, 'monitor':"val_loss"}
    return [optimizer], [scheduler]


