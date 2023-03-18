import numpy as np
from tqdm import tqdm
import math

# Make the list of Characters to be included
list_chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '/', '*']

# Associate probablities for the distribution 
probs = [1/14, 1/14, 1/14, 1/14, 1/14, 1/14, 1/14, 1/14, 1/14, 1/14, 1/14, 1/14, 1/14, 1/14]


# assert math.isclose(sum(probs), 1.0) == True


def spit_garbage(len_expr = 10):
  """ Source : https://math.wvu.edu/~hdiamond/Math222F17/Sigurd_et_al-2004-Studia_Linguistica.pdf """
  word_len_probs = [0.03160, 0.16975, 0.21192, 0.15678, 0.10852, 0.08524, 0.07724, 
      0.05623, 0.04032, 0.02766, 0.01582, 0.00917, 0.00482, 0.00262, 0.00099, 0.00050, 
      0.00027, 0.00022, 0.00011, 0.00006, 0.00005, 0.00002, 0.00001, 0.00001, 0.00001, 0.00001, 0.00005]
  
  # assert math.isclose(sum(word_len_probs), 1.0, rel_tol=1e-4) == True

  word_lens = [i+1 for i in range(len(word_len_probs))]  
  wl = np.random.choice(word_lens, p=word_len_probs)
  result = "".join([np.random.choice(list_chars, p=probs) for i in range(wl)])
  print(result)


def write_to_file(path = './'):
  pass

for i in range(15):
  spit_garbage()
