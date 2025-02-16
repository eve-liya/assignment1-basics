import torch
import argparse
from ece496b_basics.transformer_model import softmax, Transformer_LM
from ece496b_basics.tokenizer import Tokenizer


def softmax_with_temperature(logits, temperature):
    return softmax(logits / temperature, dim=-1)

def generate_text(model: Transformer_LM, tokenizer: Tokenizer, prompt: str, 
                  max_length: int = 50, temperature: float = 1.0, p: float =0.9):
    # Tokenize the prompt and get the initial sequence
    tokens = tokenizer.encode(prompt)
    print(f'Generating from tokens: {tokens}')
    decoded = ''
    
    model.eval()
    
    with torch.no_grad():
        # While there are still tokens to generate and end-of-sequence token isn't reached
        while len(tokens) < max_length and not decoded.endswith('<|endoftext|>'):
            input_tensor = torch.tensor([tokens], dtype=torch.long, device=model.ln_final.weight.device)

            # Get the logits for the next token
            logits = model(input_tensor)
            logits = logits[0, -1]  # Get the last token's logits

            # Apply temperature scaling to logits
            probs = softmax_with_temperature(logits, temperature)

            # Apply top-p (nucleus) sampling if needed
            if p < 1.0:
                # Sort probabilities and calculate cumulative sum
                sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                cumulative_sorted_probs = torch.cumsum(sorted_probs, dim=-1)

                # Create a mask for top-p sampling
                nucleus_mask = cumulative_sorted_probs < p
                nucleus_mask[0] = nucleus_mask[0] | (~nucleus_mask.any())  # Ensure at least one token is chosen

                # Set low-probability tokens outside of the nucleus to zero
                non_nucleus_indices = sorted_indices[~nucleus_mask]
                probs[non_nucleus_indices] = 0.0

                # Renormalize probabilities
                probs /= probs.sum()

            # Sample the next token
            next_token = torch.multinomial(probs, 1).item()

            tokens.append(next_token)
            decoded = tokenizer.decode(tokens)

            # Print the decoded sequence so far
            # print(decoded)

            # Check if end-of-sequence token is generated
            if decoded.endswith('<|endoftext|>'):
                break
    
    return decoded

def parse_args():
    # at some point make this more modular...
    parser = argparse.ArgumentParser(description="Train Transformer LM")
    parser.add_argument("--checkpoint-path", required=True, type=str, help="Checkpoint path")
    parser.add_argument("--tokenizer-path", required=True, type=str, help="Tokenizer path")
    parser.add_argument("--checkpoint", required=True, type=str, help="Checkpoint path")


def main():
    # Load the pre-trained model
    model = Transformer_LM(d_model=512, num_heads=16, 
                           d_ff=2048, vocab_size=32000, context_length=256,
                           num_layers=4)
    model.load_state_dict(torch.load("checkpoints/checkpoint-owt-rtx5000.pth")['model_state'])  
    model.eval()  # Set the model to evaluation mode

    # Load the tokenizer
    tokenizer = Tokenizer.from_files('bpe_tokenizers/owt_bpe/vocab.pkl', 'bpe_tokenizers/owt_bpe/merges.pkl', ['<|endoftext|>'])

    prompt = "Once upon a time"
    max_length = 256
    temperature = 0.8
    p = 0.9

    generated_text = generate_text(model, tokenizer, prompt, max_length, temperature, p)

    # Print the output
    print("\nGenerated Text:\n", generated_text)

if __name__ == "__main__":
    main()