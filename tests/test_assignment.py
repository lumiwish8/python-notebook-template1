import pytest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import importlib.util

@pytest.fixture(scope="module")
def nb():
    """노트북을 실행하고 함수들을 모듈로 로드"""
    with open("assignment.ipynb") as f:
        notebook = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=120, kernel_name="python3")
    ep.preprocess(notebook)

    code = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )

    spec = importlib.util.spec_from_loader("nb_module", loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(code, module.__dict__)
    return module

class TestCalculateStats:
    def test_basic(self, nb):
        mean, std = nb.calculate_stats([1, 2, 3, 4, 5])
        assert abs(mean - 3.0) < 1e-6
        assert abs(std - 1.4142135) < 1e-4

    def test_single(self, nb):
        mean, std = nb.calculate_stats([7])
        assert abs(mean - 7.0) < 1e-6

class TestBayesPosterior:
    def test_basic(self, nb):
        result = nb.bayes_posterior(0.01, 0.9, 0.05)
        assert abs(result - 0.18) < 1e-6

    def test_equal(self, nb):
        result = nb.bayes_posterior(0.5, 0.8, 0.5)
        assert abs(result - 0.8) < 1e-6
