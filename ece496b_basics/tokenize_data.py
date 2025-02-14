import sys
import numpy as np
import logging
from tqdm import tqdm
from ece496b_basics.tokenizer import Tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define dataset paths
datasets = {
    "tiny": {
        "train": "data/raw/TinyStoriesV2-GPT4-train.txt",
        "valid": "data/raw/TinyStoriesV2-GPT4-valid.txt",
        "vocab_filepath": "tinyStories/vocab.pkl",
        "merges_filepath": "tinyStories/merges.pkl",
        "special_tokens": ["<|endoftext|>"],
    },
    "owt": {
        "train": "data/raw/owt_train.txt",
        "valid": "data/raw/owt_valid.txt",
        "vocab_filepath": "owt_bpe/vocab.pkl",
        "merges_filepath": "owt_bpe/merges.pkl",
        "special_tokens": ["<|endoftext|>"],
    }
}

# Check for dataset selection
if len(sys.argv) < 2 or sys.argv[1] not in datasets:
    print("Usage: python script.py <tiny|owt>")
    sys.exit(1)

data_name = sys.argv[1]
data = datasets[data_name]

# Load tokenizer
tokenizer = Tokenizer.from_files(**data)
logging.info(f"Tokenizing {data_name}")
for split in ['train', 'valid']:
    with open(data[split]) as f:
        text = f.read()
    # Encode entire file, maybe this could be chunked but then there's the same 
    # issue with borders potentially cutting off special tokens
    # Or we could use the encode iterable not sure    
    encoded = tokenizer.encode(text, show_progress=True)

    # Save in batches
    total_batches = 1024
    batch_size = len(encoded) // total_batches
    arr = np.memmap(f'data/{data_name}/{split}.npy', dtype=np.uint16, mode='w+', shape=(len(encoded),))
    i = 0
    for batch_idx in tqdm(range(total_batches), desc=f'Saving {data_name} {split}.bin'):
        batch = encoded[i:i+batch_size]
        arr[i:i+batch_size] = batch
        i += batch_size
arr.flush()