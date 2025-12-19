from lmbm.generators.simplex import SimplexCombiner
from lmbm.functions import Alphabet
import numpy as np
def test_simple_encoding(id_matrix):
    n = 2
    S = np.eye(n)
    sc = SimplexCombiner(S = S)
    alphabet = Alphabet(n = n).alphabet
    s = alphabet[[0,1]]
    print(sc.encode(s))



