import sys
from pathlib import Path


# train.py  = test_contrast
BASE_DIR = Path(__file__).resolve().parent
# model dir
MODEL_DIR = BASE_DIR / "model"
WORK=r"D:\PythonProject3\ctsnet\work"
attention=r"D:\PythonProject3\External-Attention-pytorch"
# 插入搜索路径
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(attention))
import os

import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from CCNet import ccnet
from sf2net import SF2Net
from cnn import ResNet18,VGG16
from CompNet import CompNet
from CTSNet import CTSNet
from compnew_convmix import compnew

from PalmprintDataset import PalmprintDataset


# Set plot font to avoid Chinese rendering warnings
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("Palmprint Recognition Training Script")
    parser.add_argument("--dataset", type=str, required=True, help="dataset folder name under weights")
    parser.add_argument("--model", type=str, required=True,)
    parser.add_argument("--train-txt", type=str, required=True, help="path to train.txt")
    parser.add_argument("--val-txt", type=str, required=True, help="path to val.txt")
    parser.add_argument("--num-classes", type=int, default=600)
    parser.add_argument("--weight-root", type=str, default="./weights", help="root directory to save weights and logs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)   # 最大轮数固定80
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--img-size", type=int, nargs=2, default=[128, 128])
    parser.add_argument("--resume", type=str, default=None, help="checkpoint path for resume")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def build_model(name, num_classes):
    if name == "ccnet":
        return ccnet(num_classes)
    elif name == "compnet":
        return CompNet(num_classes)
    elif name == "sf2net":
        return SF2Net(num_classes)
    elif name == "ctsnet":
        return CTSNet(num_classes)
    elif name == "resnet18":
        return ResNet18(num_classes=num_classes)
    elif name == "vgg16":
        return VGG16(num_classes=num_classes)
    elif name =="compnew":
        return compnew(num_classes=num_classes)
    else:
        raise NotImplementedError(f"model {name} not supported")

def write_log(log_path, text):
    """Append text to log file and print to console."""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)


def plot_curves(epochs_list, train_loss_list, val_loss_list, val_acc_list, lr_list, save_dir):
    """Plot training curves: loss, validation accuracy and learning rate."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    ax1, ax2 = axes

    ax1.plot(epochs_list, train_loss_list, label="Train Loss", marker="o", markersize=3)
    ax1.plot(epochs_list, val_loss_list, label="Val Loss", marker="o", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_list, val_acc_list, label="Val Acc", color="orange", marker="o", markersize=3)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(epochs_list, lr_list, label="LR", color="green", linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2_twin.set_ylabel("Learning Rate")
    ax2.set_title("Val Acc & LR Curve")
    ax2.legend(loc="upper left")
    ax2_twin.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train model for one epoch, return average training loss."""
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Train")
    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # 修复：只接收两个返回值
        logits, feat = model(imgs, labels)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    return total_loss / len(loader)


def val_one_epoch(model, loader, criterion, device):
    """Evaluate model on validation set, return average loss and top-1 accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc="Val")
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits, feat = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            pred = torch.argmax(logits, dim=1)
            correct += torch.sum(pred == labels).item()
            total += labels.size(0)
    avg_loss = total_loss / len(loader)
    acc = correct / total
    return avg_loss, acc


if __name__ == "__main__":
    args = get_args()
    # Override device if cuda is unavailable
    DEVICE = args.device if torch.cuda.is_available() else "cpu"


    # Build experiment directory: ./weights/DatasetName/model
    train_txt_basename = os.path.splitext(os.path.basename(args.train_txt))[0]
    exp_root = os.path.join(args.weight_root, args.dataset, f"{args.model}")
    ckpt_dir = os.path.join(exp_root, "ckpts")
    vis_dir = os.path.join(exp_root, "visual")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    log_path = os.path.join(exp_root, "train_log.txt")

    write_log(log_path, f"Experiment dir: {exp_root}")
    write_log(log_path, f"Model: {args.model}, Train txt: {args.train_txt}, Val txt: {args.val_txt}")

    # Training augmentation pipeline
    train_transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    # Validation pipeline without random augmentation
    val_transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_dataset = PalmprintDataset(txt=args.train_txt, transform=train_transform, target_size=tuple(args.img_size))
    val_dataset = PalmprintDataset(txt=args.val_txt, transform=val_transform, target_size=tuple(args.img_size))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = build_model(args.model, args.num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 1
    best_val_acc = -1.0
    patience = 10        # 早停阈值
    patience_counter = 0 # 连续不提升计数器

    # Resume from checkpoint if path is provided and exists
    if args.resume is not None and os.path.exists(args.resume):
        write_log(log_path, f"Resume from {args.resume}")
        ckpt = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        if "best_val_acc" in ckpt:
            best_val_acc = ckpt["best_val_acc"]
        write_log(log_path, "Checkpoint loaded")

    # Lists for curve plotting
    ep_list = []
    tr_loss_list = []
    va_loss_list = []
    va_acc_list = []
    lr_list = []

    write_log(log_path, "epoch,train_loss,val_loss,val_acc,lr")

    for epoch in range(start_epoch, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        va_loss, va_acc = val_one_epoch(model, val_loader, criterion, DEVICE)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        ep_list.append(epoch)
        tr_loss_list.append(tr_loss)
        va_loss_list.append(va_loss)
        va_acc_list.append(va_acc)
        lr_list.append(current_lr)

        log_str = f"{epoch},{tr_loss:.6f},{va_loss:.6f},{va_acc:.6f},{current_lr:.6f}"
        write_log(log_path, log_str)
        write_log(log_path, f"Epoch {epoch} | TrainLoss:{tr_loss:.4f} | ValLoss:{va_loss:.4f} | ValAcc:{va_acc:.4f}")

        save_dict = {
            "model": model.state_dict(),
            "epoch": epoch,
            "best_val_acc": best_val_acc
        }

        # ====== 移除：每轮保存 epoch_{epoch}.pth ======
        # epoch_ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pth")
        # torch.save(save_dict, epoch_ckpt_path)

        # Update best checkpoint
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            patience_counter = 0
            best_path = os.path.join(ckpt_dir, "best.pth")
            # atomic save to avoid stream error
            tmp_best = best_path + ".tmp"
            try:
                torch.save(save_dict, tmp_best)
                os.replace(tmp_best, best_path)
                write_log(log_path, f"New best val acc {best_val_acc:.4f}, save best.pth")
            except Exception as e:
                write_log(log_path, f"WARNING: failed to save best.pth, error: {str(e)}")
                if os.path.exists(tmp_best):
                    os.remove(tmp_best)
        else:
            patience_counter += 1
            write_log(log_path, f"No improvement, patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                write_log(log_path, f"Early stop triggered after {patience} epochs without improvement.")
                break

    # ====== 训练结束后，保存最后一轮权重 last.pth ======
    last_path = os.path.join(ckpt_dir, "last.pth")
    tmp_last = last_path + ".tmp"
    try:
        torch.save(save_dict, tmp_last)
        os.replace(tmp_last, last_path)
        write_log(log_path, f"Saved last epoch checkpoint to last.pth (epoch {epoch})")
    except Exception as e:
        write_log(log_path, f"WARNING: failed to save last.pth, error: {str(e)}")
        if os.path.exists(tmp_last):
            os.remove(tmp_last)
    # Generate curve figure after training completes
    plot_curves(ep_list, tr_loss_list, va_loss_list, va_acc_list, lr_list, vis_dir)
    write_log(log_path, f"Training finished. Curves saved to {vis_dir}")