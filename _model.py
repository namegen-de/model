# _model.py
#  pytorch bi-lstm
# by: mika senghaas

import torch
import torch.nn as nn

class Model(nn.Module):
  def __init__(self, 
      num_countries,
      num_chars, 
      conversions,
      embedding_size,
      hidden_size,
      ):
    super().__init__()

    self.num_countries = num_countries
    self.num_chars = num_chars
    self.conversions = conversions
    self.embedding_size = embedding_size
    self.hidden_size = hidden_size

    # embed
    self.gender_embed = nn.Embedding(2, hidden_size)
    self.country_embed = nn.Embedding(num_countries, embedding_size)
    self.char_embed = nn.Embedding(num_chars, embedding_size)

    # weight matrices
    self.i2h = nn.Linear(2 * embedding_size+hidden_size, hidden_size)
    self.i2o = nn.Linear(2 * embedding_size+hidden_size, num_chars)
    self.o2o = nn.Linear(hidden_size + num_chars, num_chars)

    self.dropout = nn.Dropout(0.1)
    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, country, char, hidden, debug=False):
    # embed country and char
    country_embed = self.country_embed(country)
    char_embed = self.char_embed(char)

    # average country embeddings for mixed countries
    if country_embed.size(0) > 1:
      country_embed = country_embed.mean(0)

    # make embedded 2d
    if char_embed.dim() < 2:
      char_embed = char_embed[None, :]
    if country_embed.dim() < 2:
      country_embed = country_embed[None, :]

    if debug:
      print(country_embed.size(), char_embed.size(), hidden.size())
    input_cat = torch.cat((country_embed, char_embed, hidden), 1)
    hidden = self.i2h(input_cat)
    output = self.i2o(input_cat)

    output_cat = torch.cat((hidden, output), 1)
    output = self.o2o(output_cat)

    output = self.dropout(output)
    output = self.softmax(output)

    return output, hidden

  def init_hidden(self, gender):
    return self.gender_embed(gender)
    # return torch.zeros(1, self.hidden_size)
