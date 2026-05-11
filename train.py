import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
from tqdm import tqdm
from model import (
    Transformer,
    make_src_mask,
    make_tgt_mask,
    SRC_VOCAB_PATH,
    TGT_VOCAB_PATH,
)

_DIR = os.path.dirname(os.path.abspath(__file__))


class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)
        smooth_val = self.smoothing / (self.vocab_size - 2)
        smooth_dist = torch.full_like(log_probs, smooth_val)
        smooth_dist[:, self.pad_idx] = 0.0
        smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        pad_mask = target == self.pad_idx
        smooth_dist[pad_mask] = 0.0
        loss = -(smooth_dist * log_probs).sum(dim=-1)
        return loss[~pad_mask].mean()


def run_epoch(
    data_iter: DataLoader,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    model.train() if is_train else model.eval()
    total_loss, n_batches = 0.0, 0

    with torch.set_grad_enabled(is_train):
        for src, tgt in tqdm(
            data_iter, desc=f"{'Train' if is_train else 'Val'} Ep{epoch_num}"
        ):
            src, tgt = src.to(device), tgt.to(device)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            src_mask = make_src_mask(src, model.pad_idx)
            tgt_mask = make_tgt_mask(tgt_in, model.pad_idx)

            logits = model(src, tgt_in, src_mask, tgt_mask)
            loss = loss_fn(
                logits.contiguous().view(-1, logits.size(-1)),
                tgt_out.contiguous().view(-1),
            )

            if is_train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler:
                    scheduler.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab: Dict[str, int],
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """Calculate BLEU score using Greedy Decoding."""
    from evaluate import load as load_metric
    import spacy

    bleu_metric = load_metric("bleu")
    idx2tok = {v: k for k, v in tgt_vocab.items()}
    src_idx2tok = {v: k for k, v in model.src_vocab.items()}
    sos, eos, pad = (
        tgt_vocab.get("<sos>", 2),
        tgt_vocab.get("<eos>", 3),
        tgt_vocab.get("<pad>", 1),
    )

    predictions, references = [], []
    model.eval()

    # We use a smaller subset or batching for BLEU evaluation
    with torch.no_grad():
        for src, tgt in tqdm(test_dataloader, desc="Evaluating BLEU (Greedy)"):
            src = src.to(device)
            for i in range(src.size(0)):
                # Convert src tensor back to string correctly and quickly
                src_sentence = " ".join(
                    [
                        src_idx2tok.get(id_.item(), "<unk>")
                        for id_ in src[i]
                        if id_.item() not in (sos, eos, pad)
                    ]
                )

                # Use inference
                pred_str = model.infer(src_sentence)

                ref = [
                    idx2tok.get(id_, "<unk>")
                    for id_ in tgt[i].tolist()
                    if id_ not in (sos, eos, pad)
                ]
                predictions.append(pred_str)
                references.append([" ".join(ref)])

    result = bleu_metric.compute(predictions=predictions, references=references)
    return result["bleu"] * 100


def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config": model.config,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"]


def run_training_experiment() -> None:
    import wandb
    from dataset import Multi30kDataset, collate_fn, PAD_IDX
    from lr_scheduler import NoamScheduler

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")
    # TUNED HYPERPARAMS: Reduced warmup, increased dropout
    cfg = dict(
        d_model=512,
        N=6,
        num_heads=8,
        d_ff=2048,
        dropout=0.1,
        warmup_steps=4000,
        num_epochs=30,
        batch_size=128,
        smoothing=0.1,
    )
    wandb.init(project="da6401-a3", config=cfg)

    train_ds = Multi30kDataset(split="train")
    val_ds = Multi30kDataset(
        split="validation", src_vocab=train_ds.src_vocab, tgt_vocab=train_ds.tgt_vocab
    )
    test_ds = Multi30kDataset(
        split="test", src_vocab=train_ds.src_vocab, tgt_vocab=train_ds.tgt_vocab
    )

    with open(SRC_VOCAB_PATH, "w") as f:
        json.dump(train_ds.src_vocab, f)
    with open(TGT_VOCAB_PATH, "w") as f:
        json.dump(train_ds.tgt_vocab, f)
    print(f"Vocab sizes: src={len(train_ds.src_vocab)}, tgt={len(train_ds.tgt_vocab)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )
    test_loader = DataLoader(
        test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=2
    )

    model = Transformer(
        src_vocab_size=len(train_ds.src_vocab),
        tgt_vocab_size=len(train_ds.tgt_vocab),
        d_model=cfg["d_model"],
        N=cfg["N"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
        tie_weights=True,
        gdrive_file_id=None,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamScheduler(
        optimizer, d_model=cfg["d_model"], warmup_steps=cfg["warmup_steps"]
    )
    loss_fn = LabelSmoothingLoss(
        len(train_ds.tgt_vocab), pad_idx=PAD_IDX, smoothing=cfg["smoothing"]
    )

    best_val_bleu = -1.0
    best_ckpt = os.path.join(_DIR, "best_checkpoint.pt")
    last_ckpt = os.path.join(_DIR, "checkpoint.pt")

    for epoch in range(cfg["num_epochs"]):
        train_loss = run_epoch(
            train_loader, model, loss_fn, optimizer, scheduler, epoch, True, device
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn, None, None, epoch, False, device
        )
        
        # Evaluate validation BLEU using greedy decoding for speed
        val_bleu = evaluate_bleu(model, val_loader, train_ds.tgt_vocab, device)

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_bleu": val_bleu,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        print(f"Epoch {epoch:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Val BLEU: {val_bleu:.2f}")

        save_checkpoint(model, optimizer, scheduler, epoch, last_ckpt)
        if val_bleu > best_val_bleu:
            best_val_bleu = val_bleu
            save_checkpoint(model, optimizer, scheduler, epoch, best_ckpt)
            print(f"  ✓ New best model saved (BLEU: {val_bleu:.2f})")

    load_checkpoint(best_ckpt, model)
    # Evaluate with Greedy Decoding for final score
    bleu = evaluate_bleu(model, test_loader, train_ds.tgt_vocab, device)
    wandb.log({"test_bleu": bleu})
    print(f"Final Test BLEU (Greedy): {bleu:.2f}")
    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()
