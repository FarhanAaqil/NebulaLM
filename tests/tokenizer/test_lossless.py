from pathlib import Path

from tokenizers.pre_tokenizers import ByteLevel

from nebulalm.tokenizer.train import train_byte_bpe

ROOT = Path(__file__).resolve().parents[2]


def test_byte_level_round_trip_and_literal_controls(tmp_path: Path):
    texts = [
        "Alpha Centauri is 4.37 light years away and β Pictoris keeps its Greek letter.",
        "Velocity -2.5 km/s, mass M_⊙, catalogs NGC 1300 and M31, coordinates 14h 39m.",
    ]
    tokenizer = train_byte_bpe(texts, 400, tmp_path, seed=17013, train_manifest_sha256="abc")
    assert tokenizer.tokenizer.model.unk_token is None
    for char in ByteLevel.alphabet():
        ids = tokenizer.encode(char)
        assert tokenizer.eos_id not in ids
        assert tokenizer.decode(ids) == char
    literal = "The prose contains <eos> and <pad> as words."
    ids = tokenizer.encode(literal)
    assert tokenizer.eos_id not in ids
    assert tokenizer.pad_id not in ids
    assert tokenizer.decode(ids) == literal
    with_eos = tokenizer.encode(literal, add_eos=True)
    assert with_eos[-1] == tokenizer.eos_id
    assert tokenizer.decode(with_eos) == literal
