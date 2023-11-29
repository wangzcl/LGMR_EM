"""Eigen Microstate Decomposition
"""
import numpy as np
import numpy.typing as npt
from scipy.linalg import svd


class eigen_microstate:
    """Eigen Microstate (EM) decomposition.

    EM decomposes a statistical ensemble into a set of eigen microstates
    and their evolution. A normalized statistical ensemble N * M matrix A
    is a time series of M samples. Each sample is a microstate or a snapshot
    of a system with N agents.

    By singular value decomposition (SVD), we have

    A = U \Sigma V^T

    Here, i-th column of N * N unitary matrix U is the i-th eigen microstate;
    each column of M * M unitary matrix V is the evolution of the corresponding
    eigen microstate; and \Sigma is a diagonal matrix of eigenvalues, whose
    diagonal elements represents the probability amplitudes of eigen microstates.

    The Frobenius norm of A is normalized to 1, so that the sum of the squares of
    the eigenvalues is 1.

    In this implementation, both ensmble matrix A, microstate matrix U, and
    evolution matrix V are transposed, as ``NumPy`` stores matrices in row-major
    order by default.

    You can pass a n-dimensional array as an ensemble (n>=2), and the first axis
    will be explained as time. The rest axes will be flattened and explained as
    states.

    Parameters
    ----------
    ens: array_like of shape (n_time, *shape)
        Ensemble matrix A. The first axis is time, and the rest axes are states
        to be flattened.

    rescale: bool, default=False
        Whether to rescale the ensemble before decomposition. If True, the
        ensemble will be rescaled by dividing the standard deviation states.
        This will make all agents have the same size of fluctuations.

    Attributes
    ----------
    ``rescale``: bool
        Whether the ensemble is rescaled before decomposition.

    ``n_time_``: int
        Number of time steps

    ``n_agent_``: int
        Number of agents

    ``n_microstate_``: int
        Number of eigen microstates (=``min(n_time_, n_agent_)``)

    ``ens_``: ndarray of shape (n_time_, n_agent_)
        Preprocessed, normalized ensemble matrix

    ``shape_``: ndarray
        Shape of a microstate (sample) in the original matrix

    ``_mask_``: ndarray of shape ``shape_``

    ``microstates_``: ndarray or masked array of shape (n_microstate, *shape)
        Eigen microstates (U) of the ensemble,
        stacked along the 0th axis,
        sorted by their weights or eigenvalues.

    ``eigvals_``: ndarray of shape (n_microstate,)
        Eigenvalues (\sigma) of the microstates, also the diagonal elements of
        matrix \Sigma explained above;
        sorted by their weights or eigenvalues.

    ``weights_``: ndarray of shape (n_microstate,)
        Weights (\sigma^2')of the microstates.

    ``evolution_``: ndarray of shape (n_microstate, n_agent)
        Evolution (:math: 'V') of the microstates,
        stacked along the 0th axis,
        sorted by their weights or eigenvalues.

    See Also
    --------
    EM : Alias of EigenMicrostate

    References
    ----------
    [1] Sun, Y.; Hu, G.; Zhang, Y.; Lu, B.; Lu, Z.; Fan, J.; Li, X.; Deng, Q.; 
    Chen, X. Eigen Microstates and Their Evolutions in Complex Systems. Commun. 
    Theor. Phys. 2021, 73 (6), 065603. https://doi.org/10.1088/1572-9494/abf127.
    """

    def __init__(self, rescale: bool = False) -> None:
        self.rescale = rescale

        self.ens_ = None
        self.shape_ = None
        self._mask_ = None
        self.n_time_ = None
        self.n_agent_ = None

        self.n_microstate_ = None
        self.microstates_ = None
        self.eigvals_ = None
        self.weights_ = None
        self.evolution_ = None

    def fit(self, ens: npt.ArrayLike) -> None:
        """Decompose the ensemble into microstates and evolution."""
        shape = ens.shape[1:]
        mask = mask_nan(ens)
        ens = np.asarray(ens[:, ~mask])  # flattened ensemble without NaN

        n_time, n_agent = ens.shape
        n_microstate = min(n_time, n_agent)

        # Rescaling and normalization
        if self.rescale:
            ens /= np.std(ens, axis=0)
        ens -= np.mean(ens, axis=0)
        ens /= np.linalg.norm(ens)

        # Begin eigen microstate decomposition
        # Inintialize microstates with NaN
        # There might still be NaN values after fitting
        # (if the ensemble has NaN values)
        microstates = np.full((n_microstate, *shape), np.nan)

        evolution, eigvals, microstates[:, ~mask] = svd(ens, full_matrices=False)
        evolution = evolution.T

        self.ens_ = ens
        self.shape_ = shape
        self._mask_ = mask
        self.n_time_, self.n_agent_ = n_time, n_agent

        self.n_microstate_ = n_microstate
        self.microstates_ = microstates
        self.eigvals_ = eigvals
        self.evolution_ = evolution
        self.weights_ = eigvals**2

    def sign_reverse(self, index: tuple[int]) -> None:
        """
        Reverse the sign of selected microstates and their evolution in place.

        May be helpful for visualization.
        """
        self.microstates_[index] *= -1
        self.evolution_[index] *= -1


EM = eigen_microstate


def mask_nan(ens: npt.ArrayLike) -> np.ndarray[bool]:
    """
    Return a mask of ``NaN`` values.

    If any sample in ``ens`` has ``NaN`` on a site, the site is masked.
    """
    if isinstance(ens, np.ma.MaskedArray):
        mask = ens.mask
    else:
        mask = np.isnan(ens)
    mask = mask.any(axis=0)
    return mask
