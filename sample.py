# _generate.py
#  generate new names
# by: mika senghaas

import torch
import numpy as np
from torch.nn import Softmax

softmax = Softmax(dim=1)

# custom imports
from _utils import (
    get_args,
    get_country_tensor,
    get_gender_tensor,
    get_input_tensor,
    load_run)

def sample(
    model, 
    countries, 
    gender,
    start_with, 
    max_len):
  # output string
  res = start_with

  # model in evaluation mode, not saving gradients
  with torch.no_grad():
    # initialise input and hidden tensors
    input_tensor = get_input_tensor(start_with, model.conversions['ch2i'])
    country_tensor = get_country_tensor(countries, model.conversions['a2i'])
    gender_tensor = get_gender_tensor(gender)

    # initialise hidden tensor from gender
    hidden = model.init_hidden(gender_tensor)

    if input_tensor.size(0)>1:
      for i in range(input_tensor.size(0)-1):
        _, hidden = model(country_tensor, input_tensor[i], hidden)
      input_tensor = input_tensor[-1]

    for _ in range(max_len-len(start_with)):
      output, hidden = model(country_tensor, input_tensor, hidden)

      # get top 5 most likely next letters
      topv, topi = output.topk(5)
      topv = torch.exp(topv)
      topv = np.array(topv.flatten()).astype('float64')
      topv = topv / topv.sum()
      topi = np.random.choice(topi.flatten(), p=topv)
      letter = model.conversions['i2ch'][topi.item()]

      if letter == "</S>":
        break
      else:
        res += letter
        input_tensor = get_input_tensor(letter, model.conversions['ch2i'], start_token=False)

    return res

def main():
  # load args
  args = get_args(script='sample')

  # load run
  model, meta = load_run(args.most_recent)
  print(f"Model from {meta['general']['time']}")

  idx2char = {i: c for c, i in model.conversions['ch2i'].items()}

  for i in range(args.generate):
    output = sample(
        model, 
        args.countries,
        args.gender,
        args.start_with,
        args.max_len)
    print(f"{i+1}: {output}")

if __name__ == "__main__":
  main()
