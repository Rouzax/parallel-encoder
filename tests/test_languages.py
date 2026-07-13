"""Tests for ISO 639 language code normalisation and matching."""

from __future__ import annotations

import pytest

from presets.languages import languages_match, normalize_language


class TestNormalizeLanguage:
    def test_iso_639_2_bibliographic_and_terminological_share_a_key(self):
        # Dutch: 'dut' is 639-2/B, 'nld' is 639-2/T. Matroska muxers write 'dut'.
        assert normalize_language("dut") == normalize_language("nld")

    def test_two_letter_code_shares_key_with_three_letter(self):
        assert normalize_language("nl") == normalize_language("dut")
        assert normalize_language("en") == normalize_language("eng")

    def test_case_insensitive(self):
        assert normalize_language("DUT") == normalize_language("dut")

    def test_strips_region_subtag(self):
        assert normalize_language("nl-NL") == normalize_language("dut")
        assert normalize_language("en_US") == normalize_language("eng")

    def test_unknown_code_normalises_to_itself(self):
        assert normalize_language("xyz") == "xyz"

    def test_empty_and_none_normalise_to_und(self):
        assert normalize_language("") == "und"
        assert normalize_language(None) == "und"


class TestLanguagesMatch:
    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("nld", "dut"),  # Dutch  (the reported bug)
            ("deu", "ger"),  # German
            ("fra", "fre"),  # French
            ("ces", "cze"),  # Czech
            ("ell", "gre"),  # Greek
            ("isl", "ice"),  # Icelandic
            ("zho", "chi"),  # Chinese
            ("fas", "per"),  # Persian
            ("ron", "rum"),  # Romanian
            ("slk", "slo"),  # Slovak
            ("nl", "dut"),   # 639-1 vs 639-2/B
            ("eng", "en"),
        ],
    )
    def test_equivalent_codes_match(self, a, b):
        assert languages_match(a, b)
        assert languages_match(b, a)

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("nld", "eng"),
            ("dut", "ger"),
            ("dut", "und"),
            ("dut", ""),
            ("eng", "por"),
        ],
    )
    def test_different_languages_do_not_match(self, a, b):
        assert not languages_match(a, b)
        assert not languages_match(b, a)

    def test_untagged_stream_never_matches_a_real_language(self):
        # Three files in the source library have audio streams with no
        # language tag; media_info reports these as 'und'.
        assert not languages_match("und", "dut")
        assert not languages_match("und", "eng")
