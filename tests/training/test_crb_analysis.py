from __future__ import annotations

import itertools
import unittest

import numpy as np

from scripts.crb.analyze import U_GRID, b_ki, crb_select, pass_ustat, ustat_min_weights


class TestUstatMinWeights(unittest.TestCase):
    def test_weights_sum_to_one(self) -> None:
        for K in (5, 32, 64):
            for k in (1, 2, 3, K // 2, K):
                self.assertAlmostEqual(float(ustat_min_weights(K, k).sum()), 1.0, places=12)

    def test_k1_uniform_and_kK_one_hot(self) -> None:
        np.testing.assert_allclose(ustat_min_weights(8, 1), np.full(8, 1 / 8))
        w = ustat_min_weights(8, 8)
        np.testing.assert_allclose(w, np.eye(8)[0])


class TestBkiBruteForce(unittest.TestCase):
    def test_matches_subset_enumeration(self) -> None:
        rng = np.random.default_rng(0)
        E = np.sort(rng.random((3, 5)), axis=1)
        ks = [1, 2, 3, 5]
        got = b_ki(E, ks)
        for n in range(3):
            for j, k in enumerate(ks):
                mins = [E[n, list(c)].min() for c in itertools.combinations(range(5), k)]
                self.assertAlmostEqual(float(got[n, j]), float(np.mean(mins)), places=12)


class TestPassUstatBruteForce(unittest.TestCase):
    def test_matches_subset_enumeration(self) -> None:
        N = 6
        s = np.array([0, 1, 3, 6])
        ks = [1, 2, 4, 6]
        got = pass_ustat(s, N, ks)
        for i, si in enumerate(s):
            outcomes = [1] * si + [0] * (N - si)
            for j, k in enumerate(ks):
                hits = [max(c) for c in itertools.combinations(outcomes, k)]
                self.assertAlmostEqual(float(got[i, j]), float(np.mean(hits)), places=12)

    def test_k1_is_success_rate(self) -> None:
        s = np.array([0, 8, 16, 32])
        np.testing.assert_allclose(pass_ustat(s, 32, [1])[:, 0], s / 32)


class TestCrbSelect(unittest.TestCase):
    def test_synthetic_surface(self) -> None:
        T = 5
        # coverage percentiles get worse with epoch, reliability percentiles get better
        cov = np.linspace(1.0, 4.0, T)[:, None] * np.ones(9)
        rel = np.linspace(4.0, 1.0, T)[:, None] * np.ones(10)
        Qbar = np.concatenate([cov, rel], axis=1)
        out = crb_select(Qbar, U_GRID)
        expected = np.log(np.linspace(1.0, 4.0, T)) / np.log(4.0)
        np.testing.assert_allclose(out["R_cov"], expected, rtol=1e-12)
        np.testing.assert_allclose(out["R_rel"], expected[::-1], rtol=1e-12)
        self.assertEqual(out["i_sel"], 2)
        self.assertEqual(float(out["R_max"].min()), float(np.maximum(out["R_cov"], out["R_rel"]).min()))

    def test_ties_pick_earliest(self) -> None:
        Qbar = np.array([[1.0] * 19, [2.0] * 19, [1.0] * 19])
        out = crb_select(Qbar, U_GRID)
        self.assertEqual(out["i_sel"], 0)


if __name__ == "__main__":
    unittest.main()
