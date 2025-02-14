from typing import List, Dict, Tuple, Iterable, Iterator, Optional
import regex as re
import pickle
from tqdm import tqdm

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
pattern = re.compile(PAT)


class Tokenizer:

    def __init__(self, vocab: Dict[str, int], merges: List[Tuple[bytes, bytes]], special_tokens: Optional[List[str]] = None):
        self.vocab = vocab # int -> bytes
        self.merges = merges

        self.special_tokens = [] if special_tokens is None else special_tokens

        # Add special tokens so we don't split them
        if self.special_tokens:
            for special_token in self.special_tokens:
                special_token_bytes = special_token.encode("utf-8")
                if special_token_bytes not in self.vocab.values():
                    self.vocab[len(self.vocab)] = special_token_bytes
    
        self.inverse_vocab = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()} # bytes -> int
    
        # Build merge rules with integer indices
        self.merges_int = {}
        for byte1, byte2 in merges:
            self.merges_int[(self.inverse_vocab[byte1], self.inverse_vocab[byte2])] = self.inverse_vocab[byte1 + byte2]


    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Optional[List[str]] = None, **kwargs):
        with open(vocab_filepath, "rb") as vf:
            vocab = pickle.load(vf)
        with open(merges_filepath, "rb") as mf:
            merges = pickle.load(mf)
        return cls(vocab, merges, special_tokens)
    

    def _split_text(self, text: str) -> List[str]:
        if not self.special_tokens:
            return [text]
        sorted_tokens = sorted((re.escape(t) for t in self.special_tokens), key=len, reverse=True)
        # Create pattern that captures the special tokens
        pattern = f'({"|".join(sorted_tokens)})'
        parts = re.split(pattern, text)
        # Filter out empty strings
        return [part for part in parts if part]

    def _update(self, ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        new_ids = []
        i = 0
        while i < len(ids):
            curr_pair = tuple(ids[i:i+2])
            if curr_pair == pair:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids
    
    def _encode_subword(self, chunk: str) -> List[int]:
        if chunk in self.special_tokens:
            return [self.special_tokens[chunk]]
        # Pretokenize 
        tokens = pattern.findall(chunk)
        ret = []
        for token in tokens:
            # turn it into bytes then look up in the vocab
            # We then take the sequence of vocabulary element merges created during BPE
            # training, and apply it to our pre-tokens in the same order of creation.
            token_bytes = token.encode("utf-8")
            encoded = [self.inverse_vocab[bytes([b])] for b in token_bytes]
            while len(encoded)>=2:
                pairs = {(encoded[i], encoded[i + 1]) for i in range(len(encoded) - 1)}
                merge_pair = min(pairs, key=lambda pair: self.merges_int.get(pair, float('inf')))
                if merge_pair not in self.merges_int:
                    break
                new_id = self.merges_int[merge_pair]
                encoded = self._update(encoded, merge_pair, new_id)
            ret.extend(encoded)
        return ret

    def encode(self, text: str, show_progress = False) -> List[int]:
        # break it up along special tokens so they don't get broken up
        chunks = self._split_text(text)
        encoded = []
        for chunk in tqdm(chunks, f"Encoding {len(chunks)}", disable = (not show_progress)):    
            if chunk in self.special_tokens:
                # add it directly
                encoded.append(self.inverse_vocab[chunk.encode("utf-8")])
            else:  
                # compute the merges that make up this text 
                encoded.extend(self._encode_subword(chunk))
        return encoded

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: List[int]) -> str:
        return b''.join([self.vocab[t] for t in ids]).decode('utf-8', errors='replace')
