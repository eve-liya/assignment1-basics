import regex as re
from collections import Counter
import json
import pickle
import time


def train_bpe(input_path, vocab_size, special_tokens):
    start_time = time.time()
    vocab = dict()
    merges: list[tuple[bytes, bytes]]
    new_index = 0

    for token in special_tokens:
        vocab[new_index] = token.encode("utf-8")
        new_index += 1
    
    for x in range(1,257):
        vocab[x] = bytes([x - 1])
        new_index += 1

    merges: list[tuple[bytes, bytes]] = list()

    # Pretokenize
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pattern = re.compile(PAT)
    pretokenized = Counter()

    ## Process each line in the file
    with open(input_path, "r", encoding="utf8") as file:
        #TODO: add special token detection
        for text in file:
            for st in special_tokens:
                text = text.replace(st, "")
            pretokens = pattern.findall(text)
            for pretoken in pretokens:
                pretoken = pretoken.encode("utf-8")
                temp = []
                for x in range(1, len(pretoken) + 1):
                    temp.append(pretoken[x - 1:x])
                pretokenized[tuple(temp)] += 1

    # Find pair counts
    pair_counts = Counter()
    for tokens, count in pretokenized.items():
        for pair in zip(tokens, tokens[1:]):
            pair_counts[pair] += count

    while len(vocab) < vocab_size and len(pair_counts) > 0:

        curr_max = (0, None) 

        # might be better to do a priority queue... but it passes the tests...
        for pair, count in pair_counts.items():  # Iterate over items in the dictionary.
            if curr_max[1] is None or count > curr_max[0] or (count == curr_max[0] and pair > curr_max[1]):
                curr_max = (count, pair)

        vocab[new_index] = curr_max[1][0] + curr_max[1][1]
        new_index += 1

        merges.append(curr_max[1])

        to_add = []

        for word, counts in pretokenized.items():
            added = False
            for x in range(len(word) - 1):
                if (word[x], word[x + 1]) == curr_max[1]:
                    # if we are not at the end
                    if x < len(word) - 2:
                        if pair_counts.get((word[x + 1], word[x + 2]), 0) > 0:
                            pair_counts[(word[x + 1], word[x + 2])] -= counts
                        pair_counts[(word[x] + word[x + 1], word[x + 2])] += counts
                    # if we are not at the beginning
                    if x != 0:
                        pair_counts[(word[x - 1], word[x] + word[x + 1])] += counts
                        if pair_counts.get((word[x - 1], word[x]), 0) > 0:
                            pair_counts[(word[x - 1], word[x])] -= counts
                    # if we have already added this word to the list then remove the previous and recombine it
                    if added:
                        prev = to_add.pop()
                        to_add.append((prev[0][:x - 1] + tuple([word[x] + word[x + 1]]) + word[x + 2:], pretokenized[word], word))
                    else:
                        to_add.append((word[:x] + tuple([word[x] + word[x + 1]]) + word[x + 2:], pretokenized[word], word))
                        added = True

        # can't modify dict while iterating?
        for entry in to_add:
            pretokenized[entry[0]] = entry[1]
            del pretokenized[entry[2]]

        # The pair is merged so we can delete it
        del pair_counts[curr_max[1]]

    print(time.time() - start_time)
    return (vocab, merges)

vo, me = train_bpe("tests/fixtures/corpus.en", 500, ["<|endoftext|>"])

# print(vo,me)
"""
vo, me = train_bpe("data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])

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
"""
