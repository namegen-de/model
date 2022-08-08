# preproces.py
#  load names from python name database
# by: mika senghaas

# imports
import os
import pickle
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from pprint import pprint
from termcolor import cprint
from names_dataset import NameDataset

# custom imports
from _utils import (
    get_chars,
    unicode_to_ascii,
    get_conversions,
    save_json, 
    time_since) 

# lambdas
working = lambda s: print(f"> {s}")
done = lambda: cprint("Done!\n", "green")

def main():
    # make directories
    dirs = ["data", "data/names"]
    for d in dirs:
      os.makedirs(d) if not os.path.exists(d) else None

    # initialise name dataset (9s)
    start = working("Loading name dataset")
    names = NameDataset()
    done()

    working("Getting and saving country conversions")
    # get country code encodings
    meta = names.get_country_codes()
    meta = pd.DataFrame([{'name': c.name, 'alpha2': c.alpha_2, 'alpha3': c.alpha_3} 
        for c in meta])\
            .sort_values("alpha2")\
            .reset_index()\
            .drop('index', axis=1)\
            .to_csv("data/meta", index=False)

    # load back all conversions
    a2i, i2a, a2c, c2a, c2i, i2c = get_conversions()
    done()

    # extract top 500 non-ascii names for each country
    all_chars = get_chars()
    for country_alpha in tqdm(a2c.keys(), desc="> Query and Save Top 500 Names"):
      data = names.get_top_names(n=500, country_alpha2=country_alpha)

      with open(f"data/names/{country_alpha}.txt", "w") as f:
        for gender in ["M", "F"]:
          i = 0
          count = 0
          while count < 500 and i < len(data[country_alpha][gender]):
            name = data[country_alpha][gender][i]
            name = unicode_to_ascii(name, all_chars)

            if name != "":
              f.writelines(f"{gender},{name}\n")
              count += 1
            i += 1

if __name__ == "__main__":
  main()
