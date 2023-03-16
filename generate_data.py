import numpy as np
from tqdm import tqdm

# Make the list of Characters to be included
list_chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '/', '*']

# Associate probablities for the distribution 
probs = [1/14, 1/14, 1/14, 1/14, 1/14, 1/14, 1/14, 1/14, 1/14, 1/14, 1/14, 1/14, 1/14, 1/14]

def spit_garbage(len_expr = 10, num_exprs=15):
  for i in range(num_exprs):
    result = "".join([np.random.choice(list_chars, p=probs)])
    print(result)

def write_to_file(path = './'):
  pass
