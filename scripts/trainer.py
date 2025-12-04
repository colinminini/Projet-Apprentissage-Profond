from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    """
    Training utility with live TensorBoard logging and per-epoch checkpoints.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: Optional[torch.device] = None,
        log_dir: str = "runs",
        checkpoint_dir: str = "checkpoints",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        grad_clip: Optional[float] = None,
        segments_per_clip: int = 10,
        clip_duration: float = 3.0,
    ) -> None:
        if segments_per_clip < 1:
            raise ValueError("segments_per_clip must be >= 1")

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.segments_per_clip = segments_per_clip
        self.clip_duration = clip_duration

        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.global_step = 0

    def train(self, dataloader: torch.utils.data.DataLoader, epoch: int) -> Tuple[float, float]:
        self.model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()

            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            bs = y.size(0)
            total_loss += loss.item() * bs
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_samples += bs

            self.writer.add_scalar("train/batch_loss", loss.item(), self.global_step)
            self.global_step += 1

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        self.writer.add_scalar("train/epoch_loss", avg_loss, epoch)
        self.writer.add_scalar("train/epoch_accuracy", avg_acc, epoch)
        return avg_loss, avg_acc

    @torch.no_grad()
    def test(self, dataloader: torch.utils.data.DataLoader, epoch: int) -> Tuple[float, float]:
        self.model.eval()

        # When training directly on full 30s clips, set segments_per_clip=1 to skip aggregation.
        if self.segments_per_clip == 1:
            total_loss, total_correct, total_samples = 0.0, 0, 0
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits, y)

                bs = y.size(0)
                total_loss += loss.item() * bs
                total_correct += (logits.argmax(dim=1) == y).sum().item()
                total_samples += bs

            avg_loss = total_loss / total_samples if total_samples else 0.0
            avg_acc = total_correct / total_samples if total_samples else 0.0

            self.writer.add_scalar("test/loss", avg_loss, epoch)
            self.writer.add_scalar("test/accuracy", avg_acc, epoch)
            return avg_loss, avg_acc

        total_loss, total_correct, total_clips = 0.0, 0, 0

        logits_buffer = []
        labels_buffer = []

        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)

            for sample_logits, label in zip(logits, y):
                logits_buffer.append(sample_logits)
                labels_buffer.append(label)

                if len(logits_buffer) == self.segments_per_clip:
                    stacked_labels = torch.stack(labels_buffer)
                    if (stacked_labels != stacked_labels[0]).any():
                        raise ValueError(
                            "Mismatched labels within an ensemble window. Ensure test_loader is not shuffled and "
                            "the dataset keeps 3s slices ordered by track."
                        )

                    clip_logits = torch.stack(logits_buffer).mean(dim=0, keepdim=True)
                    clip_label = stacked_labels[0].unsqueeze(0)

                    loss = self.criterion(clip_logits, clip_label)
                    total_loss += loss.item()
                    total_correct += (clip_logits.argmax(dim=1) == clip_label).sum().item()
                    total_clips += 1

                    logits_buffer.clear()
                    labels_buffer.clear()

        if logits_buffer:
            raise ValueError(
                f"Incomplete clip encountered with {len(logits_buffer)} segments. "
                "Ensure the test dataset length is divisible by the ensemble window, "
                "and that the DataLoader is not dropping the last incomplete batch."
            )

        avg_loss = total_loss / total_clips if total_clips else 0.0
        avg_acc = total_correct / total_clips if total_clips else 0.0

        self.writer.add_scalar("test/loss", avg_loss, epoch)
        self.writer.add_scalar("test/accuracy", avg_acc, epoch)
        return avg_loss, avg_acc

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        num_epochs: int,
        start_epoch: int = 1,
    ) -> None:
        for epoch in range(start_epoch, num_epochs + 1):
            train_loss, train_acc = self.train(train_loader, epoch)
            test_loss, test_acc = self.test(test_loader, epoch)

            if self.scheduler:
                self.scheduler.step()

            ckpt_path = self.save_checkpoint(
                epoch=epoch,
                train_loss=train_loss,
                test_loss=test_loss,
                train_acc=train_acc,
                test_acc=test_acc,
            )

            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f} | "
                f"test_loss={test_loss:.4f}, test_acc={test_acc:.2f} | "
                f"checkpoint={ckpt_path.name}"
            )

    def save_checkpoint(
        self,
        epoch: int,
        train_loss: float,
        test_loss: float,
        train_acc: float,
        test_acc: float,
    ) -> Path:
        ckpt_path = self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
            },
            ckpt_path,
        )
        return ckpt_path

def resume_from_checkpoint(trainer, train_loader, test_loader, device, start_epoch, total_epochs):
    """
    Reload a saved checkpoint and continue training with the provided Trainer instance.
    """
    ckpt_path = Path("checkpoints") / "CNN_exp" / f"epoch_{start_epoch-1:03d}.pt"
    ckpt = torch.load(ckpt_path, map_location=device)

    trainer.model.load_state_dict(ckpt["model_state_dict"])
    trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if ckpt.get("scheduler_state_dict") and trainer.scheduler:
        trainer.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    trainer.global_step = (start_epoch - 1) * len(train_loader)
    trainer.fit(train_loader, test_loader, num_epochs=total_epochs, start_epoch=start_epoch)
