import regex as re
from collections import Counter, defaultdict
import pickle
from tqdm import tqdm

def train_bpe(input_path, vocab_size, special_tokens):
    vocab = {}
    new_index = 0
    for token in special_tokens:
        vocab[new_index] = token.encode("utf-8")
        new_index += 1
    for x in range(1, 257):
        vocab[new_index] = bytes([x - 1])
        new_index += 1

    # Pretokenize the file.
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pattern = re.compile(PAT)
    # Word frequency table
    word_freq = Counter()

    with open(input_path, "r", encoding="utf8") as file:
        for line in tqdm(file, desc="Pre-tokenizing file"):
            for st in special_tokens:
                line = line.replace(st, "")
            tokens = pattern.findall(line)
            for token in tokens:
                token_bytes = token.encode("utf-8")
                # Represent token as a tuple of individual bytes.
                token_tuple = tuple(token_bytes[i:i+1] for i in range(len(token_bytes)))
                word_freq[token_tuple] += 1

    # Each word is stored as a dict with:
    #   'tokens': list of tokens
    #   'freq': frequency count from pretokenization.
    words = []
    for token_tuple, freq in word_freq.items():
        words.append({
            'tokens': list(token_tuple),
            'freq': freq
        })

    # pair_occurrences maps a pair (tuple of two tokens) to a set of occurrences where it appears in,
    # where each occurrence is (word_index, position) indicating where the pair occurs.
    pair_occurrences = defaultdict(set)
    # pair_freq stores the total frequency for each pair.
    pair_freq = defaultdict(int)
    for word_id, word in enumerate(words):
        tokens = word['tokens']
        for pos in range(len(tokens) - 1):
            pair = (tokens[pos], tokens[pos+1])
            pair_occurrences[pair].add((word_id, pos))
            pair_freq[pair] += word['freq']

    merges = []
    pbar = tqdm(total=vocab_size - len(vocab), desc="Training BPE")
    while len(vocab) < vocab_size and pair_freq:
        # Find the most frequent pair
        # (Potentially could use some priority queue/heap but updating that becomes tricky)
        curr_max = (0, None)
        for pair, count in pair_freq.items():
            if curr_max[1] is None or count > curr_max[0] or (count == curr_max[0] and pair > curr_max[1]):
                curr_max = (count, pair)

        max_pair = curr_max[1]
        max_count = curr_max[0]
        if max_pair is None or max_count == 0:
            break

        # Create the new token from the best pair.
        new_token = max_pair[0] + max_pair[1]
        vocab[new_index] = new_token
        new_index += 1
        merges.append(max_pair)

        # Get only the words that contain the max_pair.
        affected_occurrences = pair_occurrences[max_pair]
        # Remove the merged pair from our structures.
        del pair_occurrences[max_pair]
        del pair_freq[max_pair]

        # avoid updating the same word multiple times, group by word_id.
        affected_word_ids = {word_id for (word_id, pos) in affected_occurrences}
        for word_id in affected_word_ids:
            word = words[word_id]
            old_tokens = word['tokens']
            # rebuild the token list by merging all instances of max_pair in this word.
            new_tokens = []
            i = 0
            while i < len(old_tokens):
                if i < len(old_tokens) - 1 and (old_tokens[i], old_tokens[i+1]) == max_pair:
                    new_tokens.append(old_tokens[i] + old_tokens[i+1])
                    i += 2
                else:
                    new_tokens.append(old_tokens[i])
                    i += 1
            word['tokens'] = new_tokens

            # For this word, remove all old pair occurrences
            # It’s simpler to recompute the adjacent pairs for the entire word.
            # First, subtract this word’s contribution from all pairs it had.
            for pos in range(len(old_tokens) - 1):
                pair = (old_tokens[pos], old_tokens[pos+1])
                if (word_id, pos) in pair_occurrences[pair]:
                    pair_occurrences[pair].discard((word_id, pos))
                    pair_freq[pair] -= word['freq']
                    if pair_freq[pair] <= 0:
                        del pair_freq[pair]
                        if pair in pair_occurrences:
                            del pair_occurrences[pair]

            # add the new pairs from the updated token list.
            for pos in range(len(new_tokens) - 1):
                new_pair = (new_tokens[pos], new_tokens[pos+1])
                pair_occurrences[new_pair].add((word_id, pos))
                pair_freq[new_pair] += word['freq']

        pbar.update(1)
    pbar.close()
    return vocab, merges

# start = time.time()
# vo, me = train_bpe("tests/fixtures/corpus.en", 500, ["<|endoftext|>"])
# print(time.time() - start)
# print(vo,me)

# vo, me = train_bpe("data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
# vo, me = train_bpe("data/owt_train.txt", 32000, ["<|endoftext|>"])

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: dataset, vocab size, special tokens")
        exit()
    vo, me = train_bpe(sys.argv[0], int(sys.argv[1], sys.argv[2]))

    with open("vocab.txt", "w") as file:
                        for token, string in vo.items():
                            file.write(
                                str(token)
                                + " : "
                                + string.decode("utf8", errors="replace")
                                + "\n"
                            )
    with open("vocab.pkl", "wb") as file:
        pickle.dump(vo, file)
    with open("merges.txt", "w") as file:
        for merge in me:
            file.write(" ".join(map(str, merge)) + "\n")
    with open("merges.pkl", "wb") as file:
        pickle.dump(me, file)

