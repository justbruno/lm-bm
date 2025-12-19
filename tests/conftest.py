import pytest
import numpy as np

@pytest.fixture
def id_matrix():
    return np.eye(2)
