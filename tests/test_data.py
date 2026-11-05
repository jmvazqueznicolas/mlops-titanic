"""Data tests: the right version of the data, and every file honours the contract."""
import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from titanic_mlops.config import ROOT
from titanic_mlops.contract import validate
from titanic_mlops.data import expected_sha256, load_historical, sha256


def test_data_version_is_the_recorded_one():
    assert sha256() == expected_sha256()


def test_historical_data_passes_contract():
    validate(load_historical(), training=True)


def test_normal_batch_passes_contract():
    validate(pd.read_csv(ROOT / "data/batches/batch_01.csv"))


def test_schema_change_is_rejected():
    with pytest.raises(SchemaErrors):
        validate(pd.read_csv(ROOT / "data/batches/batch_02_schema_change.csv"))


def test_unexpected_column_is_rejected():
    df = pd.read_csv(ROOT / "data/batches/batch_01.csv").assign(Boat="13")   # the lifeboat would leak the label
    with pytest.raises(SchemaErrors):
        validate(df)
