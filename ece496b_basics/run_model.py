import argparse
import time
import torch
import logging
import numpy as np
import wandb  

from transformer_model import Transformer_LM, Parallel_Transformer_LM, Post_Norm_Transformer_LM, Norm_Ablation_Transformer_LM
from ece496b_basics.optimizing import AdamW, cross_entropy_loss, gradient_clipping, lr_cosine_schedule
from model_data import save_checkpoint, load_checkpoint, get_batch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer LM")
    
    parser.add_argument("--experiment_name", type=str, required=True, help="Name for logging")
    # Dataset
    parser.add_argument("--train_file", type=str, required=True, help="Path to training text file")
    parser.add_argument("--valid_file", type=str, required=True, help="Path to validation text file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for training")
    
    # Model hyperparameters
    parser.add_argument("--vocab_size", type=int, required=True, help="Vocabulary size")
    parser.add_argument("--d_model", type=int, default=512, help="Model hidden size")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of Transformer layers")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feedforward network dimension")  # Added missing argument
    parser.add_argument("--context_length", type=int, default=128, help="Context length")
    parser.add_argument("--attn_pdrop", type=float, default=0, help="Dropout probability for attention layer")
    parser.add_argument("--residual_pdrop", type=float, default=0, help="Dropout probability for transfomer layer")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")  # Added missing argument
    parser.add_argument("--total_iters", type=int, default=10000, help="Total training iterations")
    parser.add_argument("--warmup_iters", type=int, default=100, help="Warm up iterations")
    parser.add_argument("--lr_max", type=float, default=5e-4, help="Maximum learning rate")
    parser.add_argument("--lr_min", type=float, default=0.0, help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay coefficient")

    # Checkpointing & logging
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--save_interval", type=int, default=1000, help="Iterations between saving checkpoints")
    parser.add_argument("--log_interval", type=int, default=100, help="Iterations between logging training stats")
    parser.add_argument("--eval_interval", type=int, default=500, help="Iterations between validation")
    parser.add_argument("--eval_iters", type=int, default=100, help="Batches for evaluation")

    # Ablation Study
    parser.add_argument("--ablation", type=str, required=False, help="Which ablation to run")

    return parser.parse_args()

def evaluate(model: torch.nn.Module, dataset, config):
    """Run evaluation and return average loss."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(config.eval_iters):
            x, y = get_batch(dataset, config.batch_size, config.context_length, config.device)
            x, y = x.to(config.device), y.to(config.device)
            logits = model(x)
            loss = cross_entropy_loss(logits, y)
            total_loss += loss.item()
    model.train()
    return total_loss / config.eval_iters

def main():
    args = parse_args()
    logging.info(f"Training config: {args}")

    # Initialize WandB
    wandb.init(
        project= args.experiment_name,
        
        # log down the hyperparams
        config={
            "vocab_size": args.vocab_size,
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "d_ff": args.d_ff,
            "context_length": args.context_length,
            "batch_size": args.batch_size,
            "lr_max": args.lr_max,
            "lr_min": args.lr_min,
            "weight_decay": args.weight_decay,
            "total_iters": args.total_iters
        }
    )
    
    model = None
    match args.ablation:
        case "parallel":
            logging.info(f"Parallel Layer Abalation")
            model = Parallel_Transformer_LM(d_model=args.d_model, num_heads=args.num_heads, 
                                d_ff=args.d_ff, vocab_size=args.vocab_size, context_length=args.context_length,
                                num_layers=args.num_layers, attn_pdrop=args.attn_pdrop, residual_pdrop=args.residual_pdrop)
        case "no-norm":
            logging.info(f"Norm Abalation")
            model = Norm_Ablation_Transformer_LM(d_model=args.d_model, num_heads=args.num_heads, 
                                d_ff=args.d_ff, vocab_size=args.vocab_size, context_length=args.context_length,
                                num_layers=args.num_layers, attn_pdrop=args.attn_pdrop, residual_pdrop=args.residual_pdrop)
        case "post_norm":
            logging.info(f"Post Norm Abalation")
            model = Post_Norm_Transformer_LM(d_model=args.d_model, num_heads=args.num_heads, 
                                d_ff=args.d_ff, vocab_size=args.vocab_size, context_length=args.context_length,
                                num_layers=args.num_layers, attn_pdrop=args.attn_pdrop, residual_pdrop=args.residual_pdrop)
        case _:
            model = Transformer_LM(d_model=args.d_model, num_heads=args.num_heads, 
                                d_ff=args.d_ff, vocab_size=args.vocab_size, context_length=args.context_length,
                                num_layers=args.num_layers, attn_pdrop=args.attn_pdrop, residual_pdrop=args.residual_pdrop)

    model.to(args.device)

    train_dataset = np.memmap(args.train_file, dtype=np.uint16, mode='r')
    valid_dataset = np.memmap(args.valid_file, dtype=np.uint16, mode='r')

    optimizer = AdamW(model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay)

    # Load checkpoint if available
    iter_num = 0
    try:
        iter_num = load_checkpoint(args.checkpoint_path, model, optimizer)
        logging.info(f"Resumed from iteration {iter_num}")
    except FileNotFoundError:
        logging.info("No checkpoint found, starting fresh.")

    start_time = time.time()
    
    # Training loop
    while iter_num < args.total_iters:
        optimizer.zero_grad()

        # Get batch
        x, y = get_batch(train_dataset, args.batch_size, args.context_length, args.device)
        x, y = x.to(args.device), y.to(args.device)

        # Forward pass
        logits = model(x)
        loss = cross_entropy_loss(logits, y)

        # Backward pass
        loss.backward()

        # Gradient clipping
        gradient_clipping(model.parameters(), max_norm=1.0)

        # Learning rate scheduling
        lr = lr_cosine_schedule(iter_num, alpha_max=args.lr_max, alpha_min=args.lr_min, T_w=args.warmup_iters, T_c=args.total_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr  # Fixed learning rate update

        # Optimizer step
        optimizer.step()

        if iter_num % args.log_interval == 0:
            elapsed = time.time() - start_time
            logging.info(f"Iter {iter_num}: Train Loss = {loss.item():.4f}, LR = {lr:.6f}, Time/iter = {elapsed / args.log_interval:.4f}s")
            start_time = time.time()

            wandb.log({"train_loss": loss.item(), "learning_rate": lr, "time_per_iter": elapsed / args.log_interval})

        # Evaluation
        if iter_num % args.eval_interval == 0:
            val_loss = evaluate(model, valid_dataset, args)
            logging.info(f"Iter {iter_num}: Validation Loss = {val_loss:.4f}")

            wandb.log({"val_loss": val_loss})

        # Checkpoint every so often
        if iter_num % args.save_interval == 0:
            save_checkpoint(model, optimizer, iter_num, args.checkpoint_path)
            logging.info(f"Checkpoint saved at iteration {iter_num}")

        iter_num += 1

    wandb.finish()

if __name__ == "__main__":
    main()
