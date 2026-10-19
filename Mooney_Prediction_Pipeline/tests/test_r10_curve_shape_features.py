import numpy as np

from Mooney_Prediction_Pipeline.model_training.r10_curve_shape_features import (
    clean_curve,
    decode_hex_curve,
    resample,
)


def test_decode_marks_ffff_missing_without_cross_batch_fill():
    values = decode_hex_curve("00010002FFFF0004")
    assert values[:2].tolist() == [1.0, 2.0]
    assert np.isnan(values[2])
    cleaned, missing_rate = clean_curve(values)
    assert cleaned.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert missing_rate == 0.25


def test_decode_preserves_signed_16bit_negative_curve_observation():
    values = decode_hex_curve("FFFA0006")
    assert values.tolist() == [-6.0, 6.0]


def test_resample_preserves_endpoints():
    result = resample(np.array([10.0, 20.0, 30.0]), 5)
    assert result[0] == 10.0
    assert result[-1] == 30.0
