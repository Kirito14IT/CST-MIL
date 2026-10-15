"""Regression tests for the isolated arithmetic-only MU recovery package."""

from __future__ import annotations

import importlib
from decimal import Decimal, localcontext
from pathlib import Path

from cst_mil.bench.resources import configure_cpu

configure_cpu(4)

import numpy as np
import pytest
from scipy import sparse
from sklearn.decomposition import NMF, _nmf


@pytest.fixture
def numerics():
    return importlib.import_module("cst_mil.research_followup.shorttext_nmf_stable.numerics")


def test_isolated_numerics_module_exists():
    importlib.import_module("cst_mil.research_followup.shorttext_nmf_stable.numerics")


def test_zero_locking_prevents_original_zero_times_infinity_failure(numerics):
    factor = np.asarray([[0.0, 0.25]])
    numerator = np.asarray([[0.003675536, 0.0]])
    denominator = np.asarray([[4.8183243e-316, 0.2]])
    with np.errstate(over="ignore", invalid="ignore"):
        old = factor * (numerator / denominator)
    assert np.isnan(old[0, 0])
    np.testing.assert_array_equal(numerics.stable_product(factor, numerator, denominator), [[0, 0]])


def test_positive_ratio_overflow_recovers_true_finite_product(numerics):
    factor, numerator, denominator = [
        np.asarray([[v]]) for v in (1e-320, 0.003675536, 4.8183243e-316)
    ]
    with np.errstate(over="ignore"):
        assert np.isinf(numerator / denominator).all()
    with localcontext() as context:
        context.prec = 100
        expected = float(
            Decimal.from_float(factor[0, 0])
            * Decimal.from_float(numerator[0, 0])
            / Decimal.from_float(denominator[0, 0])
        )
    result = numerics.stable_product(factor, numerator, denominator)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=0)


def test_normal_path_keeps_original_arithmetic_exactly(numerics):
    factor = np.asarray([[0.125, 3.0], [5.0, 0.0]])
    numerator = np.asarray([[0.25, 7.0], [0, 0.4]])
    denominator = np.asarray([[0.3, 1.5], [2.0, 0.1]])
    np.testing.assert_array_equal(
        numerics.stable_product(factor, numerator, denominator), factor * (numerator / denominator)
    )


def test_true_final_overflow_and_invalid_operands_still_fail(numerics):
    with pytest.raises(FloatingPointError):
        numerics.stable_product(
            np.asarray([[1e308]]), np.asarray([[1e308]]), np.asarray([[1e-308]])
        )
    with pytest.raises(FloatingPointError):
        numerics.stable_product(np.asarray([[1.0]]), np.asarray([[np.nan]]), np.asarray([[1.0]]))
    with pytest.raises(FloatingPointError):
        numerics.stable_product(np.asarray([[1.0]]), np.asarray([[1.0]]), np.asarray([[0.0]]))


def test_original_mu_update_fault_regression(numerics):
    x = sparse.csr_matrix([[1.0, 0]])
    w = np.asarray([[0.0, 1e-316]])
    h = np.asarray([[0.003675536, 0.5], [0, 1.0]])
    with np.errstate(over="ignore", invalid="ignore"):
        broken, *_ = _nmf._multiplicative_update_w(x, w.copy(), h, 2, 0, 0, 1)
    assert not np.isfinite(broken).all()
    repaired, *_ = numerics.stable_w(x, w.copy(), h, 2, 0, 0, 1)
    assert np.isfinite(repaired).all()
    np.testing.assert_array_equal(repaired, [[0, 0]])


def test_library_global_functions_are_unchanged_and_estimators_clone(numerics):
    original = (
        _nmf._multiplicative_update_w,
        _nmf._multiplicative_update_h,
        _nmf._fit_multiplicative_update,
        NMF._fit_transform,
    )
    x = sparse.csr_matrix([[1.0, 0, 0.2], [0, 1, 0.5], [0.1, 0.2, 1.0]])
    base = NMF(n_components=2, solver="mu", init="nndsvda", random_state=42, max_iter=10, tol=0)
    revised = numerics.StableMU(
        n_components=2, solver="mu", init="nndsvda", random_state=42, max_iter=10, tol=0
    )
    base.fit(x)
    revised.fit(x)
    np.testing.assert_array_equal(base.components_, revised.components_)
    assert original == (
        _nmf._multiplicative_update_w,
        _nmf._multiplicative_update_h,
        _nmf._fit_multiplicative_update,
        NMF._fit_transform,
    )


def test_only_frozen_frobenius_batch_mu_is_supported(numerics):
    x, w, h = sparse.csr_matrix([[1.0]]), np.asarray([[1.0]]), np.asarray([[1.0]])
    with pytest.raises(ValueError, match="Frobenius"):
        numerics.stable_w(x, w, h, 1, 0, 0, 1)
    with pytest.raises(ValueError, match="Frobenius"):
        numerics.stable_h(x, w, h, 2, 0, 0, 1, A=np.zeros_like(h), B=np.zeros_like(h))


def test_deployment_default_and_bindings_include_numerics_and_library_source():
    core = importlib.import_module("cst_mil.research_followup.shorttext_nmf_stable.core")
    execution = importlib.import_module("cst_mil.research_followup.shorttext_nmf_stable.execution")
    assert core.OUT.name == "nmf_stable_v1" and core.OUT.parent.name == "revisions"
    text = Path(execution.__file__).read_text(encoding="utf8")
    assert "numerics.py" in text and "Path(_nmf.__file__)" in text


def test_recovery_output_guard_and_complete_new_source_binding():
    core = importlib.import_module("cst_mil.research_followup.shorttext_nmf_stable.core")
    execution = importlib.import_module("cst_mil.research_followup.shorttext_nmf_stable.execution")
    assert core.guard_output(core.OUT) == core.OUT.resolve()
    with pytest.raises(ValueError, match="isolated revision"):
        core.guard_output(core.OUT.parent.parent)
    bound = execution._binding("scamspot", "nmf", core.OUT, {"cohort_fingerprint": "toy"}, {})
    folder = Path(execution.__file__).parent
    for source in folder.glob("*.py"):
        assert str(source.resolve()) in bound["sources"]
    assert str(Path(_nmf.__file__).resolve()) in bound["sources"]


def test_prepared_data_validation_remains_byte_identical_to_parent():
    stable_data = importlib.import_module("cst_mil.research_followup.shorttext_nmf_stable.data")
    original_data = importlib.import_module("cst_mil.research_followup.shorttext.data")
    assert Path(stable_data.__file__).read_bytes() == Path(original_data.__file__).read_bytes()


def test_fitted_nonfinite_components_are_refused_before_projection(monkeypatch):
    native = importlib.import_module("cst_mil.research_followup.shorttext_nmf_stable.native_topics")

    class InvalidFit:
        def __init__(self, n_components, **kwargs):
            self.n_components = n_components

        def fit(self, matrix):
            self.components_ = np.full((self.n_components, matrix.shape[1]), np.nan)

    monkeypatch.setattr(native, "NMF", InvalidFit)
    with pytest.raises(FloatingPointError, match="non-finite"):
        native.NMFTopicModel({"n_topics": 2}).fit(["alpha beta", "gamma delta"])


def test_checkpoint_restore_retains_class_local_solver(numerics, tmp_path):
    import joblib

    x = sparse.csr_matrix([[1.0, 0, 0.2], [0, 1, 0.5], [0.1, 0.2, 1.0]])
    model = numerics.StableMU(
        n_components=2, solver="mu", init="nndsvda", random_state=42, max_iter=10, tol=0
    ).fit(x)
    path = tmp_path / "toy.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    np.testing.assert_array_equal(model.transform(x), restored.transform(x))
