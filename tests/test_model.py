from sklearn.pipeline import Pipeline

from spam_classifier.model import build_pipelines


def test_returns_both_pipelines():
    pipes = build_pipelines()
    assert set(pipes) == {"naive_bayes", "logistic_regression"}


def test_are_pipelines_with_expected_steps():
    for pipe in build_pipelines().values():
        assert isinstance(pipe, Pipeline)
        assert [name for name, _ in pipe.steps] == ["tfidf", "clf"]


def test_can_fit_and_predict():
    pipe = build_pipelines()["naive_bayes"]
    pipe.fit(["win free money", "see you tomorrow", "free cash now"], [1, 0, 1])
    assert pipe.predict(["free money"]).shape == (1,)
