from __future__ import annotations

import pytest
from pydantic import ValidationError

from jolly_roger.tractor import TukeyTractorOptions


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
