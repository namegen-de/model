# _train.py
#  train the model
# by: mika senghaas

# imports
import torch
import numpy as np
from time import time
from termcolor import cprint

from torch.optim import Adam
from torch.nn import CrossEntropyLoss, NLLLoss

# custom imports
from _utils import (
    get_args, 
    get_training_info, 
    save_run, 
    load_run,
    get_target_tensor,
    time_since)

from _data import Data
from _model import Model

# lambdas
working = lambda s: print(f"> {s}")
done = lambda : cprint("done!\n", "green")

# helper
def train(model, device, data, optim, criterion):
  # set into training model, gradients are computed
  model.train() 

  model.to(device) # put model and data to device
  model.zero_grad() # zero out gradients
  loss = 0.0 # loss for epoch

  # initialise tensor of zeros as hidden state
  country_alpha, gender, name, country_tensor, gender_tensor, input_tensor, target_tensor = data.get_random_sample()

  # move input tensors on device
  country_tensor.to(device)
  gender_tensor.to(device)
  input_tensor.to(device)

  hidden = model.init_hidden(gender_tensor)
  #print(name, country_tensor, gender_tensor, input_tensor, target_tensor)

  for i in range(input_tensor.size(0)):
    output, hidden = model(country_tensor, input_tensor[i], hidden)
    loss += criterion(output.squeeze(), target_tensor[i])

  # update weights
  loss.backward()
  optim.step()

  return country_alpha, gender, loss.item() / input_tensor.size(0)

def main():
  # load args
  args = get_args(script='train')
  print(args)

  # load data into sgd loader
  working("loading data")
  data = Data(args.train_countries)
  done()

  # initialise model
  model = Model(
      num_countries=data.num_countries,
      num_chars=data.num_chars,
      conversions=data.conversions,
      embedding_size=args.embedding_size,
      hidden_size=args.hidden_size)

  # get device
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  # initialise optimiser and loss
  optim = Adam(model.parameters(), lr=args.lr)
  criterion = NLLLoss()

  smooth_loss = 0.0 # accumulate loss in verbose interval
  losses = [] # collect all losses

  working("starting training")
  start = time()
  country_count = {ca: {"M": 0, "F": 0} for ca in data.country_alphas}
  for epoch in range(args.epochs):
    country_alpha, gender, loss = train(model, device, data, optim, criterion)

    country_count[country_alpha][gender] += 1
    smooth_loss += loss
    if (epoch+1) % args.verbose == 0:
      print(f"[{time_since(start)}] Epoch {epoch+1} "\
            f"({round(((epoch+1) / args.epochs)*100, 2)}%) "\
            f"Average Loss {smooth_loss  / args.verbose}")
      losses.append(smooth_loss / args.verbose)
      smooth_loss = 0.0
  training_time = round(time() - start, 2)
  done()

  # get training info
  meta = get_training_info(args, data, device, training_time, country_count)

  # save model
  if not args.save:
    while True:
      inp = input("> Do you want to save this model? [y/n] ").lower()
      if inp == 'y':
        save_run(model, meta)
        done()
        break
      elif inp == 'n':
        working('did not save')
        break
  else:
    working("saving model")
    done()

  # save to backend
  if not args.publish:
    while True:
      inp = input("> Do you want to publish this model? [y/n] ").lower()
      if inp == 'y':
        save_run(model, meta, publish=True)
        done()
        break
      elif inp == 'n':
        working('did not publish')
        break
  else:
    working("publish model")
    save_run(model, meta, publish=True)
    done()

if __name__ == "__main__":
  main()
