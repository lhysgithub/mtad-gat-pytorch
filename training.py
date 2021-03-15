import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from datetime import datetime
import os


class Trainer:
	""" Trainer class for MTAD-GAT model.

	    :param model: MTAD-GAT model
	    :param optimizer: Optimizer used to minimize the loss function
	    :param window_size: Length of the input sequence
	    :param n_features: Number of input features
		:param n_epochs: Number of iterations/epochs
	    :param batch_size: Number of windows in a single batch
	    :param init_lr: Initial learning rate of the module
	    :param forecast_criterion: Loss to be used for forecasting.
	    :param recon_criterion: Loss to be used for reconstruction.
	    :param boolean use_cuda: To be run on GPU or not
	    :param dload: Download directory where models are to be dumped
	    :param log_dir: Directory where SummaryWriter logs are written to
	    """
	def __init__(self, model, optimizer, window_size, n_features,
				 n_epochs=200, batch_size=256, init_lr=0.001,
				 forecast_criterion=nn.MSELoss(),
				 recon_criterion=nn.MSELoss(),
				 use_cuda=True, dload='models/', log_dir='output/', print_every=1,
				 args_summary=""):

		self.model = model
		self.optimizer = optimizer
		self.window_size = window_size
		self.n_features = n_features
		self.n_epochs = n_epochs
		self.batch_size = batch_size
		self.init_lr = init_lr
		self.forecast_criterion = forecast_criterion
		self.recon_criterion = recon_criterion
		self.device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
		self.log_dir = log_dir
		self.print_every = print_every

		self.losses = {
			'train_total': [],
			'train_forecast': [],
			'train_recon': [],
			'val_total': [],
			'val_forecast': [],
			'val_recon': [],
		}
		self.epoch_times = []

		if self.device == 'cuda':
			self.model.cuda()

		self.id = datetime.now().strftime("%d%m%Y_%H%M%S")
		self.dload = f'{dload}/{self.id}'

		self.writer = SummaryWriter(f'{log_dir}/{self.id}')
		self.writer.add_text('args_summary', args_summary)


	def __repr__(self):
		return f'model={repr(self.model)}, w_size={self.window_size}, init_lr={self.init_lr}'


	def fit(self, train_loader, val_loader=None):
		""" Train model for self.n_epochs.
		Train and validation (if validation loader given) losses stored in self.losses

		:param train_loader: train loader of input data
		:param val_loader: validation loader of input data
		"""

		#init_train_loss = self.evaluate(train_loader)
		#print(f'Init total train loss: {init_train_loss[2]}')

		#if val_loader is not None:
		#	init_val_loss = self.evaluate(val_loader)
		#	print(f'Init total val loss: {init_val_loss[2]}')

		print(f'Training model for {self.n_epochs} epochs..')
		train_start = time.time()
		for epoch in range(self.n_epochs):
			epoch_start = time.time()
			self.model.train()
			forecast_b_losses = []
			recon_b_losses = []

			for x, y in train_loader:
				x = x.to(self.device)
				y = y.to(self.device)

				self.optimizer.zero_grad()
				preds, recons = self.model(x)
				if preds.ndim == 3:
					preds = preds.squeeze(1)
				if y.ndim == 3:
					y = y.squeeze(1)

				forecast_loss = torch.sqrt(self.forecast_criterion(y, preds))
				recon_loss = torch.sqrt(self.recon_criterion(x, recons))
				loss = forecast_loss + recon_loss

				loss.backward()
				self.optimizer.step()

				forecast_b_losses.append(forecast_loss.item())
				recon_b_losses.append(recon_loss.item())

			forecast_b_losses = np.array(forecast_b_losses)
			recon_b_losses = np.array(recon_b_losses)

			forecast_epoch_loss = np.sqrt((forecast_b_losses ** 2).mean())
			recon_epoch_loss = np.sqrt((recon_b_losses ** 2).mean())

			total_epoch_loss = forecast_epoch_loss + recon_epoch_loss

			self.losses['train_forecast'].append(forecast_epoch_loss)
			self.losses['train_recon'].append(recon_epoch_loss)
			self.losses['train_total'].append(total_epoch_loss)

			# Evaluate on validation set
			forecast_val_loss, recon_val_loss, total_val_loss = None, None, None
			if val_loader is not None:
				forecast_val_loss, recon_val_loss, total_val_loss = self.evaluate(val_loader)
				self.losses['val_forecast'].append(forecast_val_loss)
				self.losses['val_recon'].append(recon_val_loss)
				self.losses['val_total'].append(total_val_loss)

			self.write_loss(epoch)

			if total_val_loss <= self.losses['val_total'][-1]:
				self.save(f"{self.id}_model.pt")

			epoch_time = time.time() - epoch_start
			self.epoch_times.append(epoch_time)

			if epoch % self.print_every == 0:
				print(f'[Epoch {epoch + 1}] '
					  f'forecast_loss = {forecast_epoch_loss:.5f}, '
					  f'recon_loss = {recon_epoch_loss:.5f}, '
					  f'total_loss = {total_epoch_loss:.5f} ---- '
	
					  f'val_forecast_loss = {forecast_val_loss:.5f}, '
					  f'val_recon_loss = {recon_val_loss:.5f}, '
					  f'val_total_loss =  {total_val_loss:.5f}  '
					  
					  f'[{epoch_time:.1f}s]')

		# self.save(f"{self.id}-last_model")
		train_time = int(time.time()-train_start)
		self.writer.add_text('total_train_time', str(train_time))
		print(f'-- Training done in {train_time}s.')

	def evaluate(self, data_loader):
		""" Evaluate model

			:param data_loader: data loader of input data
			:return forecasting loss, reconstruction loss, total loss
		"""

		self.model.eval()

		forecast_losses = []
		recon_losses = []

		with torch.no_grad():
			for x, y in data_loader:
				x = x.to(self.device)
				y = y.to(self.device)

				y_hat, recons = self.model(x)
				if y_hat.ndim == 3:
					y_hat = y_hat.squeeze(1)
				if y.ndim == 3:
					y = y.squeeze(1)

				forecast_loss = torch.sqrt(self.forecast_criterion(y, y_hat))
				recon_loss = torch.sqrt(self.recon_criterion(x, recons))

				forecast_losses.append(forecast_loss.item())
				recon_losses.append(recon_loss.item())

		forecast_losses = np.array(forecast_losses)
		recon_losses = np.array(recon_losses)

		forecast_loss = np.sqrt((forecast_losses ** 2).mean())
		recon_loss = np.sqrt((recon_losses ** 2).mean())

		total_loss = forecast_loss + recon_loss

		return forecast_loss, recon_loss, total_loss

	def save(self, file_name):
		"""
		Pickles the model parameters to be retrieved later
		:param file_name: the filename to be saved as,`dload` serves as the download directory
		"""
		PATH = self.dload + '/' + file_name
		if os.path.exists(self.dload):
			pass
		else:
			os.mkdir(self.dload)
		torch.save(self.model.state_dict(), PATH)

	def load(self, PATH):
		"""
		Loads the model's parameters from the path mentioned
		:param PATH: Should contain pickle file
		"""

		self.model.load_state_dict(torch.load(PATH, map_location=self.device))

	def write_loss(self, epoch):
		for key, value in self.losses.items():
			self.writer.add_scalar(key, value[-1], epoch)
