import argparse
import csv
import os
import random
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import torch
import torch.nn as nn

from dataset import get_cifar10, get_cifar100
from models import resnet


DATASET_CONFIG = {
    "cifar10": {
        "loader_fn": get_cifar10,
        "num_classes": 10,
    },
    "cifar100": {
        "loader_fn": get_cifar100,
        "num_classes": 100,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet18 on CIFAR")
    parser.add_argument(
        "--dataset",
        default="cifar10",
        choices=sorted(DATASET_CONFIG.keys()),
        help="Dataset to train on",
    )
    parser.add_argument(
        "--data_path",
        default=str(PROJECT_ROOT.parent / ".datasets"),
        help="Dataset root. Data will be read from <data_path>/<dataset>-data",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Path to save the best model state_dict. Defaults to models/resnet18_<dataset>.pth",
    )
    parser.add_argument(
        "--run_dir",
        default=None,
        help="Directory for logs and checkpoints. Defaults to artifacts/train_runs/resnet18_<dataset>",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint if it exists")
    parser.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="If > 0, save an extra checkpoint every N epochs",
    )
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device():
    # Prefer CUDA when available, then MPS for Apple silicon, otherwise CPU.
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def accuracy_from_logits(logits, labels):
    preds = logits.argmax(dim=1)
    return preds.eq(labels).sum().item(), labels.size(0)


def evaluate(model, loader, criterion, device, max_batches=0):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader, start=1):
            if max_batches > 0 and batch_idx > max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            correct, batch_size = accuracy_from_logits(logits, labels)
            total_correct += correct
            total_samples += batch_size
            total_loss += loss.item() * batch_size

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = 100.0 * total_correct / max(total_samples, 1)
    return avg_loss, avg_acc


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    epoch,
    epochs,
    print_every=100,
    max_batches=0,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    start_time = time.time()

    for step, (images, labels) in enumerate(loader, start=1):
        if max_batches > 0 and step > max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        correct, batch_size = accuracy_from_logits(logits, labels)
        total_correct += correct
        total_samples += batch_size
        total_loss += loss.item() * batch_size

        effective_total_steps = min(len(loader), max_batches) if max_batches > 0 else len(loader)
        if step % print_every == 0 or step == effective_total_steps:
            avg_loss = total_loss / max(total_samples, 1)
            avg_acc = 100.0 * total_correct / max(total_samples, 1)
            elapsed = time.time() - start_time
            print(
                f"Epoch [{epoch}/{epochs}] Step [{step}/{effective_total_steps}] "
                f"loss={avg_loss:.4f} acc={avg_acc:.2f}% elapsed={elapsed:.1f}s",
                flush=True,
            )

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = 100.0 * total_correct / max(total_samples, 1)
    epoch_time = time.time() - start_time
    return avg_loss, avg_acc, epoch_time


def save_checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def append_history_row(csv_path, row):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def resolve_run_paths(args):
    dataset_name = args.dataset.lower()
    output_path = (
        Path(args.output_path)
        if args.output_path
        else PROJECT_ROOT / "models" / f"resnet18_{dataset_name}.pth"
    )
    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else PROJECT_ROOT / "artifacts" / "train_runs" / f"resnet18_{dataset_name}"
    )
    return output_path, run_dir


def main():
    args = parse_args()
    seed_everything(args.seed)

    dataset_name = args.dataset.lower()
    dataset_config = DATASET_CONFIG[dataset_name]
    loader_fn = dataset_config["loader_fn"]

    device = select_device()
    output_path, run_dir = resolve_run_paths(args)
    checkpoint_path = run_dir / "last_checkpoint.pth"
    history_path = run_dir / "history.csv"

    print(f"Using device: {device}", flush=True)
    print(f"Dataset: {dataset_name}", flush=True)
    print(f"Training data root: {args.data_path}", flush=True)
    print(f"Best model path: {output_path}", flush=True)
    print(f"Run directory: {run_dir}", flush=True)

    train_loader = loader_fn(
        args.batch_size,
        data_root=args.data_path,
        train=True,
        val=False,
        num_workers=args.num_workers,
    )
    test_loader = loader_fn(
        args.eval_batch_size,
        data_root=args.data_path,
        train=False,
        val=True,
        num_workers=args.num_workers,
    )

    model = resnet.resnet18(num_classes=dataset_config["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 1
    best_acc = 0.0

    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint["best_acc"]
        print(
            f"Resumed from epoch {checkpoint['epoch']} with best_acc={best_acc:.2f}%",
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{args.epochs} lr={current_lr:.6f}", flush=True)

        train_loss, train_acc, epoch_time = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            args.epochs,
            print_every=args.print_every,
            max_batches=args.max_train_batches,
        )
        val_loss, val_acc = evaluate(
            model,
            test_loader,
            criterion,
            device,
            max_batches=args.max_eval_batches,
        )
        scheduler.step()

        print(
            f"Epoch {epoch} finished: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% "
            f"time={epoch_time:.1f}s",
            flush=True,
        )

        append_history_row(
            history_path,
            {
                "epoch": epoch,
                "lr": f"{current_lr:.8f}",
                "train_loss": f"{train_loss:.6f}",
                "train_acc": f"{train_acc:.4f}",
                "val_loss": f"{val_loss:.6f}",
                "val_acc": f"{val_acc:.4f}",
                "epoch_time_sec": f"{epoch_time:.2f}",
            },
        )

        checkpoint_payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc": best_acc,
            "args": vars(args),
        }
        save_checkpoint(checkpoint_path, checkpoint_payload)

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(run_dir / f"checkpoint_epoch_{epoch}.pth", checkpoint_payload)

        if val_acc >= best_acc:
            best_acc = val_acc
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)
            print(f"Saved new best model with val_acc={best_acc:.2f}% to {output_path}", flush=True)

        checkpoint_payload["best_acc"] = best_acc
        save_checkpoint(checkpoint_path, checkpoint_payload)

    print(f"Training complete. Best validation accuracy: {best_acc:.2f}%", flush=True)


if __name__ == "__main__":
    main()
