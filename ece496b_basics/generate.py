import torch
from ece496b_basics.transformer_model import softmax, Transformer_LM
from ece496b_basics.tokenizer import Tokenizer
def softmax_with_temperature(logits, temperature):
    """
    Apply softmax with temperature scaling to logits.
    logits: Tensor of shape (vocab_size,)
    temperature: float, temperature scaling factor
    Returns a probability distribution.
    """
    return softmax(logits / temperature, dim=-1)

def generate_text(model: Transformer_LM, tokenizer: Tokenizer, prompt, max_length=50, temperature=1.0, p=0.9):
    """
    Generate text from the model given a prompt.

    model: Pre-trained model (e.g., TransformerLM)
    tokenizer: Tokenizer object for encoding/decoding
    prompt: str, starting text to feed into the model
    max_length: int, maximum number of tokens to generate
    temperature: float, temperature scaling for softmax
    p: float, probability threshold for nucleus sampling (top-p)

    Returns the generated text as a string.
    """
    # Tokenize the prompt and get the initial sequence
    tokens = tokenizer.encode(prompt)
    print(f'Generating from tokens: {tokens}')
    decoded = ''
    
    # Set the model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        # While there are still tokens to generate and end-of-sequence token isn't reached
        while len(tokens) < max_length and not decoded.endswith('<|endoftext|>'):
            # Prepare the input tensor for the model
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

            # Append the next token to the sequence
            tokens.append(next_token)
            decoded = tokenizer.decode(tokens)

            # Print the decoded sequence so far
            # print(decoded)

            # Check if end-of-sequence token is generated
            if decoded.endswith('<|endoftext|>'):
                break
    
    return decoded

def main():
    # Load the pre-trained model
    model = Transformer_LM(d_model=512, num_heads=8, 
                           d_ff=2048, vocab_size=50000, context_length=128,
                           num_layers=6)
    model.load_state_dict(torch.load("bak.checkpoint.pth")['model_state'])  # Adjust path as needed
    model.eval()  # Set the model to evaluation mode

    # Load the tokenizer
    tokenizer = Tokenizer.from_files('tinyStories/vocab.pkl', 'tinyStories/merges.pkl', ['<|endoftext|>'])

    # Define the prompt and generation parameters
    prompt = "Once upon a time"
    max_length = 128
    temperature = 0.8
    p = 0.9

    # Generate text
    generated_text = generate_text(model, tokenizer, prompt, max_length, temperature, p)

    # Print the output
    print("\nGenerated Text:\n", generated_text)

if __name__ == "__main__":
    main()