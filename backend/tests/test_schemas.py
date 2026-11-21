import pytest
from pydantic import ValidationError

from app.schemas import SensorReading
from tests.conftest import SAMPLE


def test_valid_reading() -> None:
    assert SensorReading(**SAMPLE).engine_rpm == 700.0


@pytest.mark.parametrize(
    ("field", "value"), [("lub_oil_pressure", -1), ("engine_rpm", 20000), ("coolant_temp", 500)]
)
def test_out_of_range_rejected(field: str, value: float) -> None:
    with pytest.raises(ValidationError) as exc:
        SensorReading(**{**SAMPLE, field: value})
    assert field in str(exc.value)


def test_missing_and_extra_fields_rejected() -> None:
    data = dict(SAMPLE)
    data.pop("fuel_pressure")
    with pytest.raises(ValidationError):
        SensorReading(**data)
    with pytest.raises(ValidationError):
        SensorReading(**SAMPLE, turbo_boost=1.0)
