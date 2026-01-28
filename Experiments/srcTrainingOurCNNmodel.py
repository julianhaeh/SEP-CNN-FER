import time
import torch
from torch import nn
from torch.utils.data import DataLoader

from Data.clsOurDataset import OurDataset
from ModelArchitectures.clsOurCNNArchitecture import OurCNN

NUM_CLASSES = 6  # Angry, Disgust, Fear, Happy, Sad, Surprise


def train_emotion_ourcnn(
    epochs=55,
    batch_size=32,
    lr=3e-4,
    num_workers=8,
    save_path=f"standardcnn64_{time.strftime('%Y%m%d_%H%M%S')}.pth",
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Start mit Batchsize: {batch_size}, Learning Rate: {lr}, Workers: {num_workers}, Device: {device}")

    # Dataset / Loader
    train_ds = OurDataset(dataset="all", split="train")
    test_ds  = OurDataset(dataset="all", split="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Model
    model = OurCNN(
        num_classes=NUM_CLASSES,
        channels=(32, 64, 128, 256),
        dropout2d=0.0,
        dropout_fc=0.3,
        fc_dim=256,
    ).to(device)

    head = nn.Linear(128, NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    optimizer = torch.optim.SGD(
        list(model.parameters()) + list(head.parameters()),
        lr=lr, momentum=0.9, weight_decay=2.2e-4, nesterov=True
    )

    # LR schedule
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(0.6 * epochs), int(0.85 * epochs)],
        gamma=0.1,
    )

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()

        t0 = time.time()
        running_loss = 0.0
        correct = 0
        total = 0

        # ---- TRAIN ----
        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True)  # (B,1,64,64)
            y = torch.as_tensor(batch["label"], device=device).long()

            logits = model(x)                  # (B, NUM_CLASSES)
            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bsz = x.size(0)
            running_loss += loss.item() * bsz
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += bsz

        scheduler.step()

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        # ---- EVAL ----
        model.eval()
        test_loss_sum = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for batch in test_loader:
                x = batch["image"].to(device, non_blocking=True)
                y = torch.as_tensor(batch["label"], device=device).long()

                logits = model(x)
                loss = criterion(logits, y)

                bsz = x.size(0)
                test_loss_sum += loss.item() * bsz
                test_correct += (logits.argmax(dim=1) == y).sum().item()
                test_total += bsz

        test_loss = test_loss_sum / max(1, test_total)
        test_acc = test_correct / max(1, test_total)

        dt = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc*100:.2f}% | "
            f"test loss {test_loss:.4f} acc {test_acc*100:.2f}% | "
            f"time {dt:.1f}s"
        )

        # Save best checkpoint
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "test_acc": test_acc,
                },
                save_path,
            )

    print(f"Done. Best test acc: {best_acc*100:.2f}%")
    print(f"Saved best checkpoint to: {save_path}")


if __name__ == "__main__":
    train_emotion_ourcnn()
