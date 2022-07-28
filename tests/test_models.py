import stan_py_example


def test_my_model():
    result = stan_py_example.run_my_model()
    assert "Mean" in result
