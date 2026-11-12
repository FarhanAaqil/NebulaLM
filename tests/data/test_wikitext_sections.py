from nebulalm.data.parse import parse_dump_text, parse_wikitext


def test_dump_text_is_cut_at_the_first_trailing_heading():
    raw = (
        "Earth is the third planet.\n\nOrbit and turning\n\nEarth orbits the Sun.\n\n"
        "Related pages \n\n Solar System\n\nReferences\n\nOther websites \n\nPlanets\nTerrestrial planets"
    )
    text = parse_dump_text(raw)
    assert text.rstrip().endswith("Earth orbits the Sun.")
    assert "Orbit and turning" in text
    assert "Terrestrial planets" not in text


def test_dump_text_keeps_an_apparatus_word_inside_a_sentence_line():
    raw = "Mars has moons.\nNotes\nare kept here because the word is not a standalone heading."
    assert parse_dump_text(raw) == raw


def test_trailing_apparatus_sections_are_dropped_with_their_subsections():
    raw = (
        "'''Mars''' is the fourth planet.\n\n"
        "== Geology ==\nBasalt covers the plains.\n"
        "=== Volcanoes ===\nOlympus Mons is a shield volcano.\n"
        "== See also ==\n* [[Venus]]\n=== Lists ===\n* [[Phobos]]\n"
        "== References ==\n{{reflist}}\n"
        "== Orbit ==\nMars orbits at 1.52 AU.\n"
        "<gallery>\nFile:Mars.jpg|A caption that should vanish\n</gallery>\n"
        "== External links ==\n* [https://example.org NASA page]\n"
    )
    text = parse_wikitext(raw)
    assert "Olympus Mons is a shield volcano." in text
    assert "Mars orbits at 1.52 AU." in text
    assert "Venus" not in text
    assert "Phobos" not in text
    assert "caption that should vanish" not in text
    assert "NASA page" not in text
    assert "See also" not in text


def test_section_names_are_matched_case_insensitively_and_exactly():
    raw = "Intro text.\n== Notes on observation ==\nKept body.\n== NOTES ==\nDropped body.\n"
    text = parse_wikitext(raw)
    assert "Kept body." in text
    assert "Dropped body." not in text
