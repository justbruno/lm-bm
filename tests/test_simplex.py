from lmbm.generators.simplex import SimplexCombiner
from lmbm.functions import Lexicon
import numpy as np
def test_simple_encoding(id_matrix):
    n = 2
    S = np.eye(n)
    lexicon = Lexicon(n = n)
    sc = SimplexCombiner(S=S, lexicon=lexicon)
    s = lexicon.lexicon[[0,1]]
    print(sc.encode(s))



