import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
from typing import List, Callable, Tuple

class Trainer:
	def __init__(self,
				 model: torch.nn.Module,
				 optimizers: List[torch.optim.Optimizer],
				 schedulers: List[object],
				 train_loader,
				 test_loader,
				 metrics: List[Callable],
				 num_epochs: int,
				 loss_fns: List[Tuple[Callable, float]],
				 log_dir: str = 'runs/exp',
				 checkpoint_dir: str = 'checkpoints',
				 device: str = None,
				 model_modify_fns: List[Callable] = None,
				 model_modify_iters: int = None):
		self.device = device if device is not None else (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
		self.model = model.to(self.device)
		self.optimizers = optimizers
		self.schedulers = schedulers
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.metrics = metrics
		self.num_epochs = num_epochs
		self.loss_fns = loss_fns
		self.writer = SummaryWriter(log_dir=log_dir)
		self.checkpoint_dir = checkpoint_dir
		self.model_modify_fns = model_modify_fns if model_modify_fns is not None else []
		self.model_modify_iters = model_modify_iters
		os.makedirs(self.checkpoint_dir, exist_ok=True)

	def train(self):
		self.model.train()
		total_iterations = 0
		num_batches = len(self.train_loader)
		for epoch in range(self.num_epochs):
			print(f"Epoch {epoch+1}/{self.num_epochs}")
			pbar = tqdm(enumerate(self.train_loader), total=num_batches, desc=f"Epoch {epoch+1}")
			for batch_idx, batch in pbar:
				inputs, targets = batch
				inputs = inputs.to(self.device)
				targets = targets.to(self.device)
				outputs = self.model(inputs)
				# Weighted sum of all losses
				total_loss = 0.0
				for loss_fn, weight in self.loss_fns:
					total_loss = total_loss + weight * loss_fn(outputs, targets)
				for opt in self.optimizers:
					opt.zero_grad()
				total_loss.backward()
				for opt in self.optimizers:
					opt.step()
				for sch in self.schedulers:
					if hasattr(sch, 'step'):
						sch.step()
				total_iterations += 1
				# Call model_modify_fns every model_modify_iters iterations (if set and not zero)
				if self.model_modify_iters is not None and self.model_modify_iters > 0:
					if total_iterations % self.model_modify_iters == 0:
						for fn in self.model_modify_fns:
							fn(self)
				pbar.set_postfix({'loss': total_loss.item() if hasattr(total_loss, 'item') else total_loss})
			# End of epoch: evaluate and log
			self.log_tensorboard(epoch)
			self.save_checkpoint(epoch)
		print(f"Training complete: {self.num_epochs} epochs, {total_iterations} iterations.")
		# Final evaluation after all epochs
		self.log_tensorboard(self.num_epochs-1)
		self.save_checkpoint(self.num_epochs-1)
		self.writer.flush()
		self.writer.close()

	def save_checkpoint(self, iteration):
		checkpoint = {
			'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': [opt.state_dict() for opt in self.optimizers],
			'iteration': iteration
		}
		path = os.path.join(self.checkpoint_dir, f'checkpoint_{iteration}.pt')
		torch.save(checkpoint, path)

	def log_tensorboard(self, iteration):
		# Log learning rate (assume first optimizer and first param group)
		if self.optimizers:
			lr = self.optimizers[0].param_groups[0]['lr']
			self.writer.add_scalar('learning_rate', lr, iteration)

		# Log metrics and loss on evaluation (test) data
		eval_metrics = self._compute_metrics_and_loss(self.test_loader)
		for name, value in eval_metrics['metrics'].items():
			self.writer.add_scalar(f'eval/metric_{name}', value, iteration)
		self.writer.add_scalar('eval/total_loss', eval_metrics['total_loss'], iteration)

		# Log metrics and loss on training data
		train_metrics = self._compute_metrics_and_loss(self.train_loader)
		for name, value in train_metrics['metrics'].items():
			self.writer.add_scalar(f'train/metric_{name}', value, iteration)
		self.writer.add_scalar('train/total_loss', train_metrics['total_loss'], iteration)

	def _compute_metrics_and_loss(self, loader):
		self.model.eval()
		total_loss = 0.0
		total_batches = 0
		all_outputs = []
		all_targets = []
		metrics_results = {}
		with torch.no_grad():
			for batch in loader:
				inputs, targets = batch
				inputs = inputs.to(self.device)
				targets = targets.to(self.device)
				outputs = self.model(inputs)
				batch_loss = 0.0
				for loss_fn, weight in self.loss_fns:
					batch_loss = batch_loss + weight * loss_fn(outputs, targets)
				total_loss += batch_loss.item() if hasattr(batch_loss, 'item') else float(batch_loss)
				total_batches += 1
				all_outputs.append(outputs.detach().cpu())
				all_targets.append(targets.detach().cpu())
		# Concatenate all outputs and targets
		if all_outputs and all_targets:
			all_outputs = torch.cat(all_outputs, dim=0)
			all_targets = torch.cat(all_targets, dim=0)
		# Compute metrics using accumulated outputs and targets
		for metric_fn in self.metrics:
			try:
				value, name = metric_fn(all_outputs, all_targets)
			except TypeError:
				value, name = metric_fn(loader, self.model)
			metrics_results[name] = value
		avg_loss = total_loss / max(total_batches, 1)
		return {'total_loss': avg_loss, 'metrics': metrics_results}

	def evaluate(self):
		self.model.eval()
		with torch.no_grad():
			for metric_fn in self.metrics:
				value, name = metric_fn(self.test_loader, self.model)
				print(f"Metric [{name}]: {value}")

	@staticmethod
	def _weight_reset(m):
		if hasattr(m, 'reset_parameters'):
			m.reset_parameters()
