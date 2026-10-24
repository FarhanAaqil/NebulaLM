"""Byte-level BPE training.

Literal "<eos>" and "<pad>" in prose are encoded as bytes. The packer appends
the eos id itself. Encoding does not go through Tokenizer.encode, because that
path consumes special-token strings.
"""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from nebulalm.utils.hashing import sha256_file, sha256_text

PAD_TOKEN = "<pad>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = (PAD_TOKEN, EOS_TOKEN)


class NebulaTokenizer:
    def __init__(self, tokenizer: Tokenizer, train_manifest_sha256: str) -> None:
        self.tokenizer = tokenizer
        self.train_manifest_sha256 = train_manifest_sha256
        pad = tokenizer.token_to_id(PAD_TOKEN)
        eos = tokenizer.token_to_id(EOS_TOKEN)
        if pad is None or eos is None:
            raise ValueError("tokenizer is missing <pad> or <eos>")
        if tokenizer.model.unk_token is not None:
            raise ValueError("tokenizer must not define an unknown token")
        self.pad_id = pad
        self.eos_id = eos

    @classmethod
    def from_file(cls, path: Path, train_manifest_sha256: str) -> "NebulaTokenizer":
        return cls(Tokenizer.from_file(str(path)), train_manifest_sha256)

    def encode(self, text: str, *, add_eos: bool = False) -> list[int]:
        normalizer = self.tokenizer.normalizer
        normalized = normalizer.normalize_str(text) if normalizer is not None else text
        pieces = self.tokenizer.pre_tokenizer.pre_tokenize_str(normalized)
        ids: list[int] = []
        for piece, _offset in pieces:
            for token in self.tokenizer.model.tokenize(piece):
                ids.append(token.id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], *, keep_special: bool = False) -> str:
        if keep_special:
            content = ids
        else:
            content = [token_id for token_id in ids if token_id not in {self.pad_id, self.eos_id}]
        return self.tokenizer.decode(content)

    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()


def train_byte_bpe(
    texts: list[str],
    vocab_size: int,
    output_dir: Path,
    *,
    seed: int,
    train_manifest_sha256: str,
) -> NebulaTokenizer:
    if vocab_size < 300:
        raise ValueError("vocab_size must leave room for the byte alphabet and special tokens")
    if not any(text.strip() for text in texts):
        raise ValueError("tokenizer training received no text")
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.normalizer = NFC()
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=ByteLevel.alphabet(),
        min_frequency=1,
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    config = {
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": tokenizer.get_vocab_size(),
        "seed": seed,
        "special_tokens": list(SPECIAL_TOKENS),
        "unk_token": None,
        "normalizer": "NFC",
        "pre_tokenizer": "ByteLevel",
        "add_prefix_space": False,
        "train_manifest_sha256": train_manifest_sha256,
        "training_text_sha256": sha256_text("\n".join(texts)),
        "document_count": len(texts),
    }
    (output_dir / "train_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sums = [
        f"{sha256_file(tokenizer_path)}  tokenizer.json",
        f"{sha256_file(output_dir / 'train_config.json')}  train_config.json",
    ]
    (output_dir / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    return NebulaTokenizer(tokenizer, train_manifest_sha256)
