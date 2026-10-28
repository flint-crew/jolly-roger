from __future__ import annotations

from argparse import ArgumentParser

import pytest
from capn_crunch import add_options_to_parser
from pydantic import ValidationError

from jolly_roger.tractor import TractorOptions, TukeyTractorOptions


def test_tractor_options():

    default_options = TukeyTractorOptions()
    default_options_tweak = default_options.with_options(output_column="DATA")  # type: ignore[arg-type]

    assert default_options.output_column == "CORRECTED_DATA"
    assert default_options_tweak.output_column == "DATA"

    _ = TukeyTractorOptions(target_objects=("sun", "SgrA", "CygA"))

    with pytest.raises(ValidationError):
        _ = TukeyTractorOptions(target_objects=9000)

    with pytest.raises(ValidationError):
        _ = TukeyTractorOptions(data_column=True)


def test_tukey_options_keep_cli_fields():
    """Splitting the shared options out must not change the tukey CLI"""
    assert set(TukeyTractorOptions.model_fields) == {
        "target_objects",
        "outer_width_ns",
        "tukey_width_ns",
        "data_column",
        "output_column",
        "copy_column_data",
        "dry_run",
        "make_plots",
        "number_of_plots",
        "overwrite",
        "chunk_size",
        "elevation_cut_deg",
        "ignore_nyquist_zone",
        "reverse_baselines",
        "flip_uvw_sign",
        "max_workers",
        "compare_to_field",
        "auto_size",
        "nth_sidelobe_null",
        "reweight",
        "weight_column",
        "no_guard_field",
        "dish_diameter_m",
        "guard_field_fraction",
        "object_minimum_flux",
        "peak_shift_search",
        "peak_shift_search_width_ns",
    }
    assert set(TractorOptions.model_fields) < set(TukeyTractorOptions.model_fields)
    assert "outer_width_ns" not in TractorOptions.model_fields


def test_field_is_guarded_by_default():
    assert not TukeyTractorOptions().no_guard_field
    assert TukeyTractorOptions(no_guard_field=True).no_guard_field


def test_guard_field_was_replaced():
    """The old guard_field option is rejected with an explanation"""
    with pytest.raises(ValidationError, match="replaced by no_guard_field"):
        TukeyTractorOptions(guard_field=True)  # type: ignore[call-arg]


def test_no_guard_field_flag():
    parser = add_options_to_parser(
        parser=ArgumentParser(), options_class=TukeyTractorOptions
    )

    assert not parser.parse_args([]).no_guard_field
    assert parser.parse_args(["--no-guard-field"]).no_guard_field
