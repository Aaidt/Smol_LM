import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import time

from smol_lm import Smol_LM, get_batch, decode, encode, chars, device


@torch.no_grad()
def evaluate(
    model: Smol_LM, batch_size: int, context_length: int, num_batches: int = 20
) -> float:
    model.eval()
    total_loss = 0.0
    for _ in range(num_batches):
        x, y = get_batch("val", batch_size, context_length)
        _, loss = model(x, y)
        total_loss += loss.item()
    model.train()
    return total_loss / num_batches


def main():
    vocab_size = len(chars)
    d_model = 256
    n_heads = 8
    n_kv_heads = 2
    num_layers = 6
    max_seq_len = 256
    batch_size = 64
    learning_rate = 3e-4
    max_iters = 5000
    eval_interval = 500
    eval_iters = 20

    model = Smol_LM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")
    print(f"Device: {device}")
    print(f"Vocab size: {vocab_size}")
    print(f"Context length: {max_seq_len}")
    print(f"Training on Shakespeare ({len(chars)} unique chars)")
    print("-" * 60)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_iters)

    start_time = time.time()
    best_val_loss = float("inf")

    for step in range(max_iters + 1):
        x, y = get_batch("train", batch_size, max_seq_len)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % eval_interval == 0:
            val_loss = evaluate(model, batch_size, max_seq_len, eval_iters)
            elapsed = time.time() - start_time
            print(
                f"Step {step:5d} | train loss {loss.item():.4f} | val loss {val_loss:.4f} | elapsed {elapsed:.1f}s"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "smol_lm_best.pt")
                print(f"  -> saved best model (val_loss={val_loss:.4f})")

            # Generate sample text
            if step > 0:
                context = torch.tensor(
                    [encode("ROMEO: ")], dtype=torch.long, device=device
                )
                output = model.generate(context, max_new_tokens=200, temperature=0.8)
                generated = decode(output[0].tolist())
                print(f"  Sample:\n{generated}\n")
                print("-" * 60)

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s")
    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
