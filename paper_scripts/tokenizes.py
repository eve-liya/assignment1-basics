import pickle

# Load the vocab dictionary
with open("bpe_tokenizers/owt_bpe/vocab.pkl", "rb") as vf:
    vocab = pickle.load(vf)

# Find the longest token from the values instead of keys
longest_token = max(vocab.values(), key=len)

print(f"Longest token: {longest_token} (Length: {len(longest_token)})")

decoded_once = longest_token.decode("utf-8")
print(decoded_once)


# import regex as re
# from collections import Counter
# from tqdm import tqdm

# # Define pretokenization regex pattern
# PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# pattern = re.compile(PAT)

# # Word frequency table
# word_freq = Counter()

# # Define special tokens to remove
# special_tokens = {"<|endoftext|>", "[UNK]", "[PAD]"}  # Example special tokens

# # Read and pretokenize the file
# input_path = "data/raw/TinyStoriesV2-GPT4-train.txt"  # Replace with your actual file path

# with open(input_path, "r", encoding="utf8") as file:
#     for line in tqdm(file, desc="Pre-tokenizing file"):
#         for st in special_tokens:
#             line = line.replace(st, "")

#         tokens = pattern.findall(line)
#         word_freq.update(tokens)  # Directly update counter with token list

# # Get the top N most frequent tokens
# N = 10  # Change this to the number of top words you want
# top_tokens = word_freq.most_common(N)

# # Print results
# print("\nTop pretokenized words:")
# for token, freq in top_tokens:
#     print(f"{token}: {freq}")
