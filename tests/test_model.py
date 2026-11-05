"""Model tests: beats the trivial baseline, and behaves sensibly on cases we understand."""
import pandas as pd

from titanic_mlops.config import FEATURES, TARGET


def test_beats_majority_baseline(trained):
    pipe, _, test = trained
    acc = (pipe.predict(test[FEATURES]) == test[TARGET]).mean()
    baseline = max(test[TARGET].mean(), 1 - test[TARGET].mean())
    assert acc > baseline + 0.05, f"accuracy {acc:.3f} barely beats baseline {baseline:.3f}"


def test_known_direction_of_effects(trained):
    """Women in 1st class should get a higher survival probability than men in 3rd class."""
    pipe = trained[0]
    rows = pd.DataFrame([
        {"Pclass": 1, "Sex": "female", "Age": 30, "SibSp": 0, "Parch": 0, "Fare": 80.0, "Embarked": "S"},
        {"Pclass": 3, "Sex": "male", "Age": 30, "SibSp": 0, "Parch": 0, "Fare": 8.0, "Embarked": "S"},
    ])[FEATURES]
    p_woman_first, p_man_third = pipe.predict_proba(rows)[:, 1]
    assert p_woman_first > p_man_third


def test_missing_age_is_handled(trained):
    row = pd.DataFrame([{"Pclass": 2, "Sex": "male", "Age": None, "SibSp": 0, "Parch": 0, "Fare": 13.0,
                         "Embarked": None}])[FEATURES]
    p = trained[0].predict_proba(row)[0, 1]
    assert 0.0 <= p <= 1.0
