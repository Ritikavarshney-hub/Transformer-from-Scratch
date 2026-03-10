"""
tokenizer.py  –  Simple word-level tokenizer for English → Sanskrit translation.
Sanskrit is tokenised at the word level as well (space-separated tokens).
No external libraries required beyond Python stdlib.
"""

import re
import json
import os
from collections import Counter
from typing import List, Dict, Optional


# ── special tokens ────────────────────────────────────────────────────────────
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]

PAD_ID = 0
UNK_ID = 1
SOS_ID = 2
EOS_ID = 3


def basic_tokenize(text: str) -> List[str]:
    """Lowercase and split on whitespace / punctuation."""
    text = text.lower().strip()
    # keep Devanagari, Latin letters, digits; split everything else
    tokens = re.findall(r'[\u0900-\u097f]+|[a-z]+|[0-9]+|[^\s\w]', text)
    return tokens if tokens else [UNK_TOKEN]


class Vocabulary:
    def __init__(self):
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        # seed with special tokens
        for tok in SPECIAL_TOKENS:
            self._add(tok)

    def _add(self, token: str) -> int:
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token
        return self.token2id[token]

    def build_from_texts(self, texts: List[str], min_freq: int = 1):
        counter: Counter = Counter()
        for text in texts:
            counter.update(basic_tokenize(text))
        for token, freq in counter.items():
            if freq >= min_freq:
                self._add(token)
        print(f"  Vocabulary size: {len(self)}")

    def encode(self, text: str, add_sos: bool = False, add_eos: bool = False) -> List[int]:
        tokens = basic_tokenize(text)
        ids = [self.token2id.get(t, UNK_ID) for t in tokens]
        if add_sos:
            ids = [SOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        special = {PAD_ID, UNK_ID, SOS_ID, EOS_ID}
        tokens = []
        for i in ids:
            if skip_special and i in special:
                continue
            tokens.append(self.id2token.get(i, UNK_TOKEN))
        return " ".join(tokens)

    def __len__(self):
        return len(self.token2id)

    # ── serialisation ─────────────────────────────────────────────────────────
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token2id, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        v = cls.__new__(cls)
        with open(path, "r", encoding="utf-8") as f:
            v.token2id = json.load(f)
        v.id2token = {int(i): t for t, i in v.token2id.items()}
        return v


def build_vocabs(train_en_path: str, train_sa_path: str,
                 save_dir: str = "checkpoints",
                 min_freq: int = 1):
    """Build and save src (EN) and tgt (SA) vocabularies from training data."""
    with open(train_en_path, encoding="utf-8") as f:
        en_lines = f.read().splitlines()
    with open(train_sa_path, encoding="utf-8") as f:
        sa_lines = f.read().splitlines()

    print("Building English vocabulary …")
    src_vocab = Vocabulary()
    src_vocab.build_from_texts(en_lines, min_freq=min_freq)

    print("Building Sanskrit vocabulary …")
    tgt_vocab = Vocabulary()
    tgt_vocab.build_from_texts(sa_lines, min_freq=min_freq)

    src_vocab.save(os.path.join(save_dir, "src_vocab.json"))
    tgt_vocab.save(os.path.join(save_dir, "tgt_vocab.json"))
    print(f"Saved vocabularies to {save_dir}/")
    return src_vocab, tgt_vocab
