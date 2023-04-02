## Add additional statistics here as well !!
from sympy import sympify, SympifyError
import numpy as np 

def eval_expr(file_path:str):
  with open(file_path) as fp:
    file_list = np.loadtxt(fp, dtype=str)

  test_kit = file_list[:100]
  count_crct = 0
  count_incrct = 0
  for exp in test_kit:
    try:
      sympify(exp)
      count_crct += 1
    except (SympifyError, ZeroDivisionError) as error:
      count_incrct += 1

  tots = count_crct + count_incrct
  return count_crct/tots, count_incrct/tots

print('Random Distribution Statistics')
corr, incorr = eval_expr('math.txt')
print(f'Correct Probability : {corr} Incorrect Probability : {incorr}')

print('P-GLAm Distribution Statistics')
corr, incorr = eval_expr('gen_text.txt')
print(f'Correct Probability : {corr} Incorrect Probability : {incorr}')
