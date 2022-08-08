# _data.py
#  pytorch dataset class
# by: mika senghaas

# imports
import os
import torch
import random
from tqdm import tqdm
from collections import Counter
from string import ascii_letters, punctuation, digits, whitespace
from names_dataset import NameDataset

from _utils import (
    load_json,
    unicode_to_ascii,
    get_chars,
    get_conversions,
    get_country_tensor, 
    get_gender_tensor,
    get_input_tensor, 
    get_target_tensor, 
    is_valid)

class Data(torch.utils.data.Dataset):
  def __init__(self, train_countries=None):
    super().__init__()
    # chars are all ascii letters, punctuation,
    # digits and simple whitespace
    self.chars = get_chars()

    # integer encode all unique chars
    ch2i = {"<S>": 0, "</S>": 1} 
    ch2i.update({char: i+2 for i, char in enumerate(self.chars)})

    i2ch = {i: char for char, i in ch2i.items()}
    
    # get number of unique chars
    self.num_chars = len(ch2i)

    # conversions
    self.conversions = {name: conv for name, conv in 
        zip(["a2i", "i2a", "a2c", "c2a", "c2i", "i2c"], get_conversions())}
    self.conversions.update({"ch2i": ch2i, "i2ch": i2ch})

    # number of countries
    self.country_alphas = list(self.conversions["a2c"].keys())
    self.countries = list(self.conversions["c2a"].keys())
    self.num_countries = len(self.conversions["c2a"])

    # load top 500 names for each country
    self.data = {country: {"M": [], "F": []} for country in self.country_alphas}
    for country in self.country_alphas:
      with open(f"data/names/{country}.txt", "r") as f:
        for line in f:
          gender, name = line.strip().split(",") # read gender and name
          # name = unicode_to_ascii(name, self.chars) # transform to ascii
          self.data[country][gender].append(name)

    # train countries
    if train_countries==None:
      self.train_countries = list(self.conversions['a2i'].keys())
    else:
      self.train_countries = train_countries

  def get_random_sample(self):
    return self.__getitem__(0)

  def __len__(self):
    c = 0
    for country in self.data:
      for gender in ["M", "F"]:
        c += len(self.data[country][gender])
    return c

  def __getitem__(self, i):
    # get a random country, gender and name
    while True:
      try: 
        country_alpha = random.choice(self.train_countries)
        gender = random.choice(["M", "F"])
        name = random.choice(self.data[country_alpha][gender])
        break
      except: 
        None

    country_tensor = get_country_tensor([country_alpha], self.conversions["a2i"])
    gender_tensor = get_gender_tensor(gender)
    input_tensor = get_input_tensor(name, self.conversions["ch2i"])
    target_tensor = get_target_tensor(name, self.conversions["ch2i"])

    return country_alpha, gender, name, country_tensor, gender_tensor, input_tensor, target_tensor
