from torch.utils.tensorboard import SummaryWriter
import torch
from typing import List, Callable, Tuple




import os

class Trainer:
	def __init__(self,
				 model: torch.nn.Module,
				 optimizers: List[torch.optim.Optimizer],
				 schedulers: List[object],
				 train_loader,
				 test_loader,
				 metrics: List[Callable],
				 num_iterations: int,
				 evaluation_iterations: int,
				 loss_fns: List[Tuple[Callable, float]],
				 log_dir: str = 'runs/exp',
				 checkpoint_dir: str = 'checkpoints'):
		self.model = model
		self.optimizers = optimizers
		self.schedulers = schedulers
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.metrics = metrics
		self.num_iterations = num_iterations
		self.evaluation_iterations = evaluation_iterations
		self.loss_fns = loss_fns
		self.writer = SummaryWriter(log_dir=log_dir)
		self.checkpoint_dir = checkpoint_dir
		os.makedirs(self.checkpoint_dir, exist_ok=True)

	def train(self):
		self.model.train()
		iteration = 0
		train_iter = iter(self.train_loader)
		while iteration < self.num_iterations:
			print(f"Iteration {iteration+1}/{self.num_iterations}", end='\r')
			try:
				batch = next(train_iter)
			except StopIteration:
				train_iter = iter(self.train_loader)
				batch = next(train_iter)
			inputs, targets = batch
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
			iteration += 1
			if iteration % self.evaluation_iterations == 0:
				self.log_tensorboard(iteration)
				self.save_checkpoint(iteration)
		print(f"Iteration {self.num_iterations}/{self.num_iterations}")
		# Final evaluation after all iterations
		self.log_tensorboard(self.num_iterations)
		self.save_checkpoint(self.num_iterations)

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
		metrics_results = { }
		with torch.no_grad():
			for batch in loader:
				inputs, targets = batch
				outputs = self.model(inputs)
				batch_loss = 0.0
				for loss_fn, weight in self.loss_fns:
					batch_loss = batch_loss + weight * loss_fn(outputs, targets)
				total_loss += batch_loss.item() if hasattr(batch_loss, 'item') else float(batch_loss)
				total_batches += 1
			# Compute metrics (use the first batch for metrics, or loop again if needed)
			for metric_fn in self.metrics:
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
