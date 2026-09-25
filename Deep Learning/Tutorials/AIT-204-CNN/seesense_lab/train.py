"""Part 4: train the CNN and save a checkpoint the serving code can load.

    python -m seesense_lab.train --epochs 5 --subset 10000
"""

import argparse
import json
import time
import uuid
from pathlib import Path

import torch
import torch.nn as nn

from . import CIFAR10_CLASSES
from .data import get_loaders
from .device import pick_device
from .model import SmallResNet, count_parameters


def build_optimizer_and_scheduler(model, lr=0.05, epochs=5, weight_decay=5e-4, use_scheduler=True):
    """Return (optimizer, scheduler). `scheduler` is None when `use_scheduler` is False."""
    # TODO 4.1: optimizer = SGD over model.parameters() with lr, momentum=0.9,
    #           nesterov=True, and the given weight_decay.
    # TODO 4.2: if use_scheduler, scheduler = CosineAnnealingLR(optimizer, T_max=epochs).
    #           (Tutorial section 13. train.py calls scheduler.step() once per epoch.)
    raise NotImplementedError


def run_epoch(model, loader, device, criterion, optimizer=None):
    """One pass over `loader`. Trains if `optimizer` is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)  # BatchNorm uses batch stats and Dropout is active only when True
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.size(0)
    return total_loss / seen, 100.0 * correct / seen


def shortcut_report(model, loader, device, patch_class):
    """Fraction of ALL-patched test images the model calls `patch_class`."""
    model.eval()
    hits = seen = 0
    with torch.no_grad():
        for x, _ in loader:
            pred = model(x.to(device)).argmax(1)
            hits += (pred == patch_class).sum().item()
            seen += pred.numel()
    return 100.0 * hits / seen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--subset", type=int, default=10000, help="train images to use (0 = all 50k)")
    ap.add_argument("--no-augment", action="store_true", help="Experiment A: turn augmentation off")
    ap.add_argument("--no-scheduler", action="store_true", help="Experiment B: constant learning rate")
    ap.add_argument("--shortcut", action="store_true", help="Experiment C: stamp a patch on every horse")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = pick_device(args.device)
    train_loader, test_loader, shortcut_loader = get_loaders(
        batch_size=args.batch_size, subset=args.subset or None,
        augment=not args.no_augment, shortcut=args.shortcut)

    model = SmallResNet().to(device)
    print(f"device={device}  params={count_parameters(model):,}  train_images={len(train_loader.dataset):,}")
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, lr=args.lr, epochs=args.epochs, use_scheduler=not args.no_scheduler)

    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc = run_epoch(model, train_loader, device, criterion, optimizer)
        te_loss, te_acc = run_epoch(model, test_loader, device, criterion)
        if scheduler is not None:
            scheduler.step()
        history.append(dict(epoch=epoch, lr=lr_now, train_loss=tr_loss, train_acc=tr_acc,
                            test_loss=te_loss, test_acc=te_acc))
        print(f"epoch {epoch:>2}/{args.epochs}  lr={lr_now:.4f}  "
              f"train {tr_loss:.3f}/{tr_acc:5.1f}%  test {te_loss:.3f}/{te_acc:5.1f}%  "
              f"gap={tr_acc - te_acc:+.1f}")
    elapsed = time.time() - start
    print(f"trained in {elapsed:.0f}s ({len(train_loader.dataset) * args.epochs / elapsed:.0f} img/s)")

    if shortcut_loader is not None:
        pct = shortcut_report(model, shortcut_loader, device, patch_class=7)
        print(f"SHORTCUT CHECK: {pct:.1f}% of all-patched test images predicted 'horse' "
              f"(a clean model would be near 10%)")

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    run_id = uuid.uuid4().hex[:8]
    torch.save(dict(state_dict={k: v.cpu() for k, v in model.state_dict().items()},
                    run_id=run_id, classes=CIFAR10_CLASSES, test_acc=history[-1]["test_acc"],
                    args=vars(args)), out / "model.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"saved {out / 'model.pt'} (run_id={run_id})")


if __name__ == "__main__":
    main()
