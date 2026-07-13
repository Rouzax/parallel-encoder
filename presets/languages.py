"""ISO 639 language code normalisation.

A language has up to three interchangeable codes, and different muxers write
different ones for the same track:

    ISO 639-1    2-letter   nl
    ISO 639-2/B  3-letter   dut    (bibliographic — what Matroska/mkvmerge write)
    ISO 639-2/T  3-letter   nld    (terminological — what preset configs tend to use)

FFmpeg's ``-map 0:a:m:language:X`` compares the tag as a literal string, so
``m:language:nld`` does not match a track tagged ``dut`` even though both mean
Dutch. Normalising both sides through :func:`normalize_language` before
comparing removes that trap.

Only the ~20 languages whose 639-2/B and 639-2/T codes actually differ need an
alias entry; for every other language the two codes are identical.
"""

from __future__ import annotations

# Each tuple groups the equivalent codes for one language:
#   (ISO 639-1, ISO 639-2/B, ISO 639-2/T)
# The 639-2/T code is used as the canonical key.
_LANGUAGE_GROUPS: tuple[tuple[str, str, str], ...] = (
    # Languages whose bibliographic and terminological codes differ.
    ("sq", "alb", "sqi"),  # Albanian
    ("hy", "arm", "hye"),  # Armenian
    ("eu", "baq", "eus"),  # Basque
    ("my", "bur", "mya"),  # Burmese
    ("zh", "chi", "zho"),  # Chinese
    ("cs", "cze", "ces"),  # Czech
    ("nl", "dut", "nld"),  # Dutch
    ("fr", "fre", "fra"),  # French
    ("ka", "geo", "kat"),  # Georgian
    ("de", "ger", "deu"),  # German
    ("el", "gre", "ell"),  # Greek
    ("is", "ice", "isl"),  # Icelandic
    ("mk", "mac", "mkd"),  # Macedonian
    ("mi", "mao", "mri"),  # Maori
    ("ms", "may", "msa"),  # Malay
    ("fa", "per", "fas"),  # Persian
    ("ro", "rum", "ron"),  # Romanian
    ("sk", "slo", "slk"),  # Slovak
    ("bo", "tib", "bod"),  # Tibetan
    ("cy", "wel", "cym"),  # Welsh
    # Common languages where 639-2/B and 639-2/T agree; listed so the
    # 2-letter code also resolves.
    ("en", "eng", "eng"),  # English
    ("es", "spa", "spa"),  # Spanish
    ("pt", "por", "por"),  # Portuguese
    ("it", "ita", "ita"),  # Italian
    ("sv", "swe", "swe"),  # Swedish
    ("da", "dan", "dan"),  # Danish
    ("no", "nor", "nor"),  # Norwegian
    ("fi", "fin", "fin"),  # Finnish
    ("pl", "pol", "pol"),  # Polish
    ("ru", "rus", "rus"),  # Russian
    ("ja", "jpn", "jpn"),  # Japanese
    ("ko", "kor", "kor"),  # Korean
    ("tr", "tur", "tur"),  # Turkish
    ("ar", "ara", "ara"),  # Arabic
    ("hi", "hin", "hin"),  # Hindi
    ("he", "heb", "heb"),  # Hebrew
    ("uk", "ukr", "ukr"),  # Ukrainian
    ("hu", "hun", "hun"),  # Hungarian
    ("bg", "bul", "bul"),  # Bulgarian
    ("hr", "hrv", "hrv"),  # Croatian
    ("sr", "srp", "srp"),  # Serbian
    ("th", "tha", "tha"),  # Thai
    ("vi", "vie", "vie"),  # Vietnamese
    ("id", "ind", "ind"),  # Indonesian
)

# Every known code (2-letter, 639-2/B, 639-2/T) -> canonical 639-2/T code.
_ALIASES: dict[str, str] = {
    code: group[2]
    for group in _LANGUAGE_GROUPS
    for code in group
}

# Tag used by ffprobe/media_info when a stream carries no language tag.
UNDETERMINED = "und"


def normalize_language(code: str | None) -> str:
    """Reduce a language code to a canonical key for comparison.

    Accepts ISO 639-1 (``nl``), ISO 639-2/B (``dut``) and ISO 639-2/T (``nld``)
    and maps all of them to the same key. A region subtag (``nl-NL``, ``en_US``)
    is stripped. Unknown codes normalise to themselves, so exact matching still
    works for languages not in the alias table.

    Args:
        code: A language code, or ``None``/empty for an untagged stream.

    Returns:
        The canonical code, or :data:`UNDETERMINED` when there is no code.
    """
    if not code:
        return UNDETERMINED

    # Strip a BCP-47 region subtag: "nl-NL" / "en_US" -> "nl" / "en".
    base = code.strip().lower().replace("_", "-").split("-", 1)[0]
    if not base:
        return UNDETERMINED

    return _ALIASES.get(base, base)


def languages_match(a: str | None, b: str | None) -> bool:
    """Report whether two language codes refer to the same language.

    Untagged streams (:data:`UNDETERMINED`) never match a real language, so a
    preset asking for Dutch will not silently pick up an untagged track.
    """
    norm_a = normalize_language(a)
    norm_b = normalize_language(b)
    if norm_a == UNDETERMINED or norm_b == UNDETERMINED:
        return False
    return norm_a == norm_b
