import os
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from ModelArchitectures.clsMobileFaceNet import MobileFacenet
from Data.clsOurDataset import OurDataset

# from your_dataset_file import OurDataset
# from your_model_file import MobileFacenet, ArcMarginProduct

NUM_CLASSES = 6  # Angry, Disgust, Fear, Happy, Sad, Surprise

def train_emotion_mobilefacenet(
    epochs=20,
    batch_size=256,
    lr=0.05,
    num_workers=8,
    save_path="mobilefacenet_gray64_arcface.pth",
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = OurDataset(dataset="all", split="train")
    test_ds  = OurDataset(dataset="all", split="test")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )

    model = MobileFacenet().to(device) 
    head = nn.Linear(128, NUM_CLASSES).to(device)


    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        list(model.parameters()) + list(head.parameters()),
        lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True
    )

    # Optional: simple LR schedule
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(0.6 * epochs), int(0.85 * epochs)], gamma=0.1
    )

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        head.train()

        t0 = time.time()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True)  # (B,1,64,64)
            y = torch.as_tensor(batch["label"], device=device).long()

            emb = model(x)               # (B,128)
            logits = head(emb)              # (B,6)
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

        # Evaluate
        model.eval()
        head.eval()
        test_loss_sum = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for batch in test_loader:
                x = batch["image"].to(device, non_blocking=True)
                y = torch.as_tensor(batch["label"], device=device).long()

                emb = model(x)
                logits = head(emb)
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
                    "backbone_state_dict": model.state_dict(),
                    "head_state_dict": head.state_dict(),
                    "test_acc": test_acc,
                },
                save_path,
            )

    print(f"Done. Best test acc: {best_acc*100:.2f}%")
    print(f"Saved best checkpoint to: {save_path}")


if __name__ == "__main__":
    train_emotion_mobilefacenet()
