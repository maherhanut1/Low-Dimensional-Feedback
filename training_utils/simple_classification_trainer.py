import torch
import os
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

class SimpleClassificationTrainer:
    def __init__(self,
                 model: torch.nn.Module,
                 optimizers,
                 schedulers,
                 train_loader,
                 val_loader,
                 num_epochs: int,
                 log_dir: str = 'runs/exp',
                 checkpoint_dir: str = 'checkpoints',
                 model_modify_fns=None,
                 model_modify_iters=None,
                 device=None):
        self.device = device if device is not None else (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = model.to(self.device)
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.writer = SummaryWriter(log_dir=log_dir)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.model_modify_fns = model_modify_fns if model_modify_fns is not None else []
        self.model_modify_iters = model_modify_iters
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        iteration = 0
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for batch in self.train_loader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                for opt in self.optimizers:
                    opt.zero_grad()
                loss.backward()
                for opt in self.optimizers:
                    opt.step()
                for sch in self.schedulers:
                    if hasattr(sch, 'step'):
                        sch.step()
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
                iteration += 1
                if self.model_modify_iters is not None and self.model_modify_iters > 0:
                    if iteration % self.model_modify_iters == 0:
                        for fn in self.model_modify_fns:
                            fn(self.model)
            train_loss = running_loss / total
            train_acc = correct / total
            val_loss, val_acc = self.evaluate()
            self.writer.add_scalar('train/loss', train_loss, epoch)
            self.writer.add_scalar('train/accuracy', train_acc, epoch)
            self.writer.add_scalar('val/loss', val_loss, epoch)
            self.writer.add_scalar('val/accuracy', val_acc, epoch)
            self.save_checkpoint(epoch)
            print(f"Epoch {epoch}/{self.num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    def evaluate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
        avg_loss = running_loss / total
        acc = correct / total
        return avg_loss, acc

    def save_checkpoint(self, epoch):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': [opt.state_dict() for opt in self.optimizers],
            'epoch': epoch
        }
        path = os.path.join(self.checkpoint_dir, f'checkpoint_{epoch}.pt')
        torch.save(checkpoint, path)
