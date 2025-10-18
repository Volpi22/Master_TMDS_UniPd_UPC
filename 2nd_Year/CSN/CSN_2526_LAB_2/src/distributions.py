from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from scipy.special import zeta, gammaln
from scipy.optimize import minimize
import numpy as np
import math

from src.utils import load_degree_sequence, LANG_DICT


@dataclass
class DegStats:
    N: int  # number of nodes
    M: float  # sum of degrees = sum(k_i)
    Mlog: float  # sum log(k_i)
    C: float  # sum log(k_i!)
    kmax_obs: int  # max observed degree


def make_stats(degrees: list[int]) -> DegStats:
    degrees = np.asarray(degrees, dtype=np.int64)
    if (degrees < 1).any():
        raise ValueError("All degrees must be >= 1 (remove unlinked nodes first).")
    return DegStats(
        N=degrees.size,
        M=float(degrees.sum()),
        Mlog=float(np.log(degrees).sum()),
        C=float(gammaln(degrees + 1).sum()),
        kmax_obs=int(degrees.max()),
    )


class Distribution(ABC):
    init = {}
    bounds = []

    @abstractmethod
    def pmf(self, k):
        raise NotImplementedError

    @abstractmethod
    def nll(self, stats: DegStats, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_params(self):
        raise NotImplementedError

    def get_init_params(self):
        return self.init

    def get_bounds(self):
        return self.bounds

    def get_prob_for_degree_k(self, k: int):
        if k == 0:
            return 0
        if k < 0:
            raise ValueError("Degree k must be a positive integer.")

        return self.pmf(k)

    @staticmethod
    def resolve_init_and_bounds(
        stats: DegStats, init_params: dict, bounds: list[tuple]
    ):
        resolved_init = defaultdict(str)
        for param, value in init_params.items():
            if callable(value):
                resolved_init[param] = value(stats)
            else:
                resolved_init[param] = value

        resolved_bounds = []
        for low, high in bounds:
            if callable(low):
                low = low(stats)
            if callable(high):
                high = high(stats)
            resolved_bounds.append((low, high))

        return dict(resolved_init), resolved_bounds

    def fit(self, data: list[int]):
        stats = make_stats(data)

        init_params, bounds = self.resolve_init_and_bounds(
            stats, self.get_init_params(), self.get_bounds()
        )

        param_names = init_params.keys()
        x0 = [init_params[name] for name in param_names]

        def nll_wrapper(x):
            kwargs = dict(zip(param_names, x))
            return self.nll(stats, **kwargs)

        res = minimize(nll_wrapper, x0=x0, bounds=bounds, method="L-BFGS-B")

        for name, val in zip(param_names, res.x):
            if name == "kmax":
                val = int(round(val))
            setattr(self, name, val)
        setattr(self, "stats", stats)

        params = dict(zip(param_names, res.x))

        K = len(params)
        N = stats.N
        AICc = -2 * (-res.fun) + 2 * K * N / (N - K - 1)
        setattr(self, "AICc", AICc)
        return params, -res.fun, AICc, res.success, res.message

    def pmf_over_degrees(self, kmax_or_ks):
        if np.isscalar(kmax_or_ks):
            ks = np.arange(1, int(kmax_or_ks) + 1, dtype=int)
        else:
            ks = np.asarray(kmax_or_ks, dtype=int)
        pmf = np.array([self.get_prob_for_degree_k(int(k)) for k in ks], dtype=float)
        pmf[pmf <= 0] = np.nan
        return ks, pmf


class DisplacedGeometric(Distribution):
    init = {"q": 0.5}
    bounds = [(1e-9, 1 - 1e-9)]

    def pmf(self, k):
        return ((1 - self.q) ** (k - 1)) * self.q

    def nll(self, stats: DegStats, q: float = None):
        q = self.q if q is None else q
        return -((stats.M - stats.N) * np.log(1 - q) + stats.N * np.log(q))

    def get_params(self):
        return {"q": self.q}


class DisplacedPoisson(Distribution):
    init = {"lam": 2.0}
    bounds = [(1e-9, None)]

    def pmf(self, k):
        log_p = (
            (k * np.log(self.lam))
            - self.lam
            - gammaln(k + 1)
            - np.log(-np.expm1(-self.lam))
        )  # calculated in logspace to avoid underflow due to huge factorial
        return float(np.exp(log_p))

    def nll(self, stats: DegStats, lam: float = None):
        lam = self.lam if lam is None else lam
        return -(
            stats.M * np.log(lam) - stats.N * (lam + np.log(1 - np.exp(-lam))) - stats.C
        )

    def get_params(self):
        return {"λ": self.lam}


class ZetaDistribution(Distribution):
    init = {"gamma": 2.5}
    bounds = [(1 + 1e-9, None)]

    def pmf(self, k):
        normalizer = self.get_normalizer()
        return k**-self.gamma / normalizer

    def get_normalizer(self, gamma=None):
        gamma = self.gamma if gamma is None else gamma
        return float(zeta(gamma, 1.0))

    def nll(self, stats: DegStats, gamma: float = None):
        gamma = self.gamma if gamma is None else gamma
        return self._nll(stats, gamma=gamma)

    def _nll(self, stats: DegStats, **dist_params):
        normalizer = self.get_normalizer(**dist_params)
        gamma = dist_params.get("gamma", 2)
        return -(-gamma * stats.Mlog - stats.N * np.log(normalizer))

    def get_params(self):
        return {"γ1": self.gamma}


class ZetaRightTruncated(ZetaDistribution):
    init = {"gamma": 2.5, "kmax": lambda stats: stats.kmax_obs}
    bounds = [(1e-9, None), (lambda stats: stats.kmax_obs, None)]

    def get_normalizer(self, gamma=None, kmax=None):
        kmax = int(round(self.kmax)) if kmax is None else int(round(kmax))
        gamma = self.gamma if gamma is None else gamma
        return sum([k**-gamma for k in range(1, kmax + 1)])

    def nll(self, stats: DegStats, gamma: float = None, kmax: int = None):
        gamma = self.gamma if gamma is None else gamma
        kmax = self.kmax if kmax is None else kmax
        return self._nll(stats, gamma=gamma, kmax=kmax)

    def get_params(self):
        return {"γ2": self.gamma, "kmax": int(round(self.kmax))}


class ZetaFixedGammaTwo(ZetaDistribution):
    init = {}
    bounds = []

    def get_normalizer(self):
        return np.pi**2 / 6

    def fit(self, data: list[int]):
        stats = make_stats(data)
        setattr(self, "gamma", 2)
        setattr(self, "stats", stats)
        L = -self.nll(stats)
        AICc = -2 * L  # no free params -> K=0 -> AICc = -2L
        setattr(self, "AICc", AICc)
        return {}, L, AICc, True, "fixed gamma=2 (no free params -> no fitting)"

    def nll(self, stats: DegStats):
        return self._nll(stats)

    def get_params(self):
        return {}


class AltmannDistribution(Distribution):

    init = {"gamma": 2.5, "delta": 0.1}
    bounds = [(1e-9, None), (0.0, None)]

    def pmf(self, k):
        if k < 1:
            return 0.0
        Z = self.get_normalizer(self.gamma, self.delta, self.stats.N)
        return (k ** (-self.gamma)) * np.exp(-self.delta * k) / Z

    @staticmethod
    def get_normalizer(gamma, delta, Ncap):
        # Z(γ,δ;N) = sum_{k=1}^N k^{-γ} e^{-δk}
        ks = np.arange(1, Ncap + 1, dtype=np.float64)
        return np.sum(np.exp(-gamma * np.log(ks) - delta * ks))

    def nll(self, stats: DegStats, gamma: float = None, delta: float = None):
        gamma = self.gamma if gamma is None else gamma
        delta = self.delta if delta is None else delta
        Z = self.get_normalizer(gamma, delta, stats.N)
        # -L = γ * sum(log k_i) + δ * sum(k_i) + N * log Z
        return gamma * stats.Mlog + delta * stats.M + stats.N * np.log(Z)

    def get_params(self):
        return {"γ": self.gamma, "δ": self.delta}
