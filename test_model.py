from model import train_model


def test_model_training():
    model, X_test, y_test, accuracy = train_model()
    assert model is not None
    assert len(X_test) > 0


def test_model_accuracy():
    _, _, _, accuracy = train_model()
    assert 0 <= accuracy <= 1
    assert accuracy >= 0.80


def test_model_predictions():
    model, X_test, y_test, _ = train_model()
    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test)
    assert all(prediction in [0, 1, 2] for prediction in predictions)
