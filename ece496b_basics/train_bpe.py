import regex as re
from collections import Counter


def train_bpe(input_path, vocab_size, special_tokens):

    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    new_index = 256

    vocab: dict[int, bytes] = {
        x: bytes([x]) for x in range(256)
    }
    merges: list[tuple[bytes, bytes]] = list()

    # Pretokenize
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretokenized = Counter()

    ## Process each line in the file
    with open(input_path, "r", encoding="utf8") as file:
        for line in file:
            pretokens = re.findall(PAT, line)
            for pretoken in pretokens:
                pretoken = tuple(pretoken.encode("utf-8"))
                pretokenized.update([pretoken])

    # print(len(pretokenized))
    for word, count in pretokenized.most_common(10):
        print(f"{word}: {count}")

    # Find pair counts
    pair_counts = Counter()
    for tokens, count in pretokenized.items():
        for pair in zip(tokens, tokens[1:]):
            pair_counts[pair] += count

    while len(vocab) < vocab_size:
        max_pair = pair_counts.most_common(1)[0][0]
        vocab[new_index] = bytes(max_pair)
        merges.append(max_pair)

        to_add = []

        """
        TODO:   the vocab should be storing bytes such as (32, 116), if it is in the form 8308 then that is a different character
                When we are adding the new entry to the pretokenized then originally it is like (32, 116, 111), then it shouldn't
                be (256, 111) since that is the index, instead ([32, 116], 111) ?? or (8308, 111) ?? not sure... need to read up on it

                Soo, vocab should contain just the bytes, merges should contain the bytes as well, so that might have to change. 
                Perhaps confirm what exactly the token is, probably just a string of bytes?
        """
        for word, counts in pretokenized.items():
            for x in range(len(word) - 2):
                if (word[x], word[x + 1]) == max_pair:
                    pair_counts[(bytes([word[x], word[x + 1]]), word[x + 2])] += counts
                    if x != 0:
                        pair_counts[(word[x - 1], bytes([word[x], word[x + 1]]))] += counts
                    to_add.append((word[:x] + tuple(bytes([word[x], word[x + 1]])) + word[x + 2:], pretokenized[word], word))

        # print(pair_counts.items())
        # Update word frequency tablee
        for entry in to_add:
            pretokenized[entry[0]] = entry[1]
            del pretokenized[entry[2]]

        del pair_counts[max_pair]
        new_index += 1

    for word, count in pretokenized.most_common(10):
        print(f"{word}: {count}")

    return vocab, merges

vo = train_bpe("../data/TinyStoriesV2-GPT4-valid.txt", 5000, "<|endoftext|>")

print(vo)

for entry, val in vo[0].items():
    print(val.decode("utf-8"))