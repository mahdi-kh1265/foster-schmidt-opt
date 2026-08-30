from foster_eom.catalog.importers.s2p_coilcraft import decode_coilcraft_filename
import pytest

def test_decode_coilcraft_filename_dash():
    series, pn, value_h = decode_coilcraft_filename("016008C-10N")
    assert series == "016008C"
    assert value_h == pytest.approx(10e-9)

def test_decode_coilcraft_filename_no_separator():
    series, pn, value_h = decode_coilcraft_filename("0806SQ12N")
    assert series == "0806SQ"
    assert pn == "0806SQ12N"
    assert value_h == pytest.approx(12e-9)

    series, pn, value_h = decode_coilcraft_filename("1010VS111")
    assert series == "1010VS"
    assert value_h == pytest.approx(110e-9)

    series, pn, value_h = decode_coilcraft_filename("0806SQ5N5")
    assert series == "0806SQ"
    assert value_h == pytest.approx(5.5e-9)

def test_decode_coilcraft_filename_underscore():
    series, pn, value_h = decode_coilcraft_filename("1508_13N")
    assert series == "1508"
    assert value_h == pytest.approx(13e-9)

    series, pn, value_h = decode_coilcraft_filename("1508_5N5")
    assert series == "1508"
    assert value_h == pytest.approx(5.5e-9)

def test_decode_coilcraft_filename_unsupported():
    series, pn, value_h = decode_coilcraft_filename("GA3092-AL")
    assert value_h is None

    series, pn, value_h = decode_coilcraft_filename("HA4032")
    assert value_h is None
