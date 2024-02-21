import eigen_microstate as em
import numpy as np
from math import sqrt


def test_em_shape():
    A = np.array([[1, 2], [4, 1], [3, 0]])

    ems = em.EigenMicrostate(rescale=True)
    ems.fit(A)

    assert ems.n_microstate_ == 2
    assert ems.shape_ == (2,)
    assert ems.microstates_.shape == (2, 2)
    assert np.isclose(np.sum(ems.weights_), 1)
