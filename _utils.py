# _utils.py
#  helper functions and classes
# by: mika senghaas

import os
import json
import torch
import pickle
import argparse
import unicodedata
import pandas as pd
from time import time
from datetime import datetime
from itertools import combinations
from string import ascii_letters, punctuation, digits, whitespace

CURR = os.path.dirname(os.path.abspath(__file__))

def get_args(script):
  assert script in ['train', 'sample'], "Can only get args for script in ['train', 'sample']"

  if script == 'train':
    parser = argparse.ArgumentParser(description="Training Parser")

    # training args
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--visualise", action="store_true")
    parser.add_argument("--verbose", type=int, default=100)

    parser.add_argument("--train-countries", nargs="+")
    parser.add_argument("--epochs", type=int, default=int(1e4))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=1e-1)
    parser.add_argument("--embedding-size", type=int, default=2**5)
    parser.add_argument("--hidden-size", type=int, default=2**7)

  elif script == 'sample':
    parser = argparse.ArgumentParser(description="Sample Parser")

    parser.add_argument("--most-recent", action='store_true')
    parser.add_argument("--countries", nargs="+", required=True) 
    parser.add_argument("--gender", type=str, default="M", required=True) 
    parser.add_argument("--generate", type=int, default=1) 
    parser.add_argument("--start-with", type=str, default="")
    parser.add_argument("--max-len", type=int, default=15)

  return parser.parse_args()

def get_training_info(model, args, data, device, training_time, country_count):
  general = {
      'time': datetime.now().strftime("%d/%m/%y, %H:%M:%S"),
      'user': os.path.expanduser("~"),
      'model': 'rnn',
      }

  hyperparams = {
      'epochs': args.epochs,
      'lr': args.lr,
      'batch_size': args.batch_size,
      'embedding_size': args.embedding_size,
      'hidden_size': args.hidden_size,
      'optimiser': 'adam',
      'loss': 'cross-entropy loss',
      'device': device
      }

  training = {
      'training_time (s)': training_time,
      'training_time (m)': training_time / 60,
      'country_count': country_count
      }

  params = {
      "num_countries": model.num_countries,
      "num_chars": model.num_chars,
      "conversions": model.conversions,
      "embedding_size": model.embedding_size,
      "hidden_size": model.hidden_size
      }

  return {
      'general': general, 
      'hyperparameters': hyperparams,
      'training': training,
      'params': params
      }

def save_json(d, path):
  with open(path, "w") as f:
    json.dump(d, f)

def load_json(path):
  with open(path, "r") as f:
    return json.loads(f.read())

def save_pkl(d, path):
  with open(path, "wb") as f:
    pickle.dump(d, f)

def load_pkl(path):
  with open(path, "rb") as f:
    return pickle.loads(f)

def save_run(model, meta, publish=False):
  if not publish:
    path = "runs/" + datetime.now().strftime("%y%m%d%H%M")
  else: 
    path = "../backend/model"

  # create path if not exists
  os.makedirs(path) if not os.path.exists(path) else None

  # save trained model
  torch.save(model.state_dict(), f"{path}/model.pt") # save model
  
  # save meta information
  save_pkl(meta, f"{path}/meta.pkl")


def load_run(most_recent=True):
  path = os.path.join(CURR, "runs")
  dirs = {i: d for i, d in enumerate(os.listdir(path))}
  if most_recent:
    idx = sorted(dirs, key=dirs.get)[-1]
    run = dirs[idx]
  else:
    for i, d in dirs.items():
      print(f"[{i+1}] {d}")
    while True:
      idx = int(input(f"Choose Run [1-{len(dirs)}]: "))
      if idx in range(1, len(dirs)+1):
        break
    run = dirs[idx-1]

  path = os.path.join(path, run)

  model = torch.load(f"{path}/model.pt")

  with open(f"{path}/meta.pkl", "r") as f:
    meta = pickle.loads(f)

  return model, meta

def get_conversions():
  path = os.path.join(CURR, "data", "meta")
  meta = pd.read_csv(path, keep_default_na=False)

  # alpha2index/ index2alpha
  a2i = {a: i for i, a in enumerate(meta["alpha2"])}
  i2a = {i: a for i, a in enumerate(meta["alpha2"])}

  # alpha2country/ country2alpha
  a2c = {a: c for a, c in zip(meta["alpha2"], meta["name"])}
  c2a = {c: a for a, c in zip(meta["alpha2"], meta["name"])}

  # country2index/ index2alpha
  c2i = {a: i for i, a in enumerate(meta["name"])}
  i2c = {i: a for i, a in enumerate(meta["name"])}

  return a2i, i2a, a2c, c2a, c2i, i2c

def get_country_tensor(countries, country2idx):
  return torch.tensor([country2idx[c] for c in countries], dtype=torch.long)

def get_gender_tensor(gender):
  return torch.tensor([0 if gender=="M" else 1], dtype=torch.long)

def get_input_tensor(line, char2idx, start_token=True):
  # integer encode char seq
  indices = [char2idx[char] for char in line]
  if start_token: indices.insert(0, 0)

  return torch.tensor(indices, dtype=torch.long)

def get_target_tensor(line, char2idx, end_token=True):
  # integer encode target tensor
  indices = [char2idx[char] for char in line]
  if end_token: indices.append(1)

  return torch.tensor(indices, dtype=torch.long)

def is_valid(company, chars):
  return len(set(company) & set(chars)) == len(set(company))

def get_chars():
  return ascii_letters + punctuation + digits + whitespace

# https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s, all_letters):
  return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def time_since(start):
  s = time() - start
  m = int(s // 60)
  s -= m * 60
  return f"{m}m {round(s)}s"
