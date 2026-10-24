from nebulalm.data.normalize import normalize
from nebulalm.utils.hashing import sha256_text


def test_normalization_preserves_astronomy_notation():
    raw = "  α   Centauri\r\nβ Pictoris  &amp;  −2.5 km/s\n\n\nM_⊙   NGC 1300\tM31  "
    cleaned = normalize(raw)
    assert cleaned == "α Centauri\nβ Pictoris & −2.5 km/s\n\nM_⊙ NGC 1300 M31"
    again = normalize(cleaned)
    assert sha256_text(again) == sha256_text(cleaned)


def test_html_entities_decode_once():
    assert normalize("&lt;eos&gt;") == "<eos>"
    assert normalize(normalize("&lt;eos&gt;")) == "<eos>"
