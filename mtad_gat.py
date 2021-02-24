import torch
import torch.nn as nn

from modules import SpatialAttentionLayer, TemporalAttentionLayer, GRU, Forecasting_Model


class MTAD_GAT(nn.Module):
	def __init__(self,  num_nodes, window_size,
				 gru_n_layers=3,
				 gru_hid_dim=64,
				 forecasting_n_layers=3,
				 forecasting_hid_dim=32,
				 dropout=0.2,
				 alpha=0.2):
		super(MTAD_GAT, self).__init__()
		self.spatial_gat = SpatialAttentionLayer(num_nodes, window_size, dropout, alpha)
		self.temporal_gat = TemporalAttentionLayer(num_nodes, window_size, dropout, alpha)
		self.gru = GRU(num_nodes, gru_hid_dim, gru_n_layers, dropout)
		#self.forecasting_model = Forecasting_Model(gru_hid_dim*window_size, forecasting_hid_dim, num_nodes, forecasting_n_layers, dropout)
		self.forecasting_model = Forecasting_Model(gru_hid_dim, forecasting_hid_dim, num_nodes, forecasting_n_layers, dropout)
		#self.forecasting_model = Forecasting_Model(3 * num_nodes * window_size, forecasting_hid_dim, num_nodes, forecasting_n_layers, dropout)

		self.gru_init_h = None

	def set_gru_init_hidden(self, batch_size):
		self.gru_init_h = self.gru.init_hidden(batch_size)

	def forward(self, x):
		# x shape (b, n, k): b - batch size, n - window size, k - number of nodes/features

		# TODO: 1D convolution

		# h_spat = self.spatial_gat(x)
		#h_temp = self.temporal_gat(x)

		#h_cat = torch.cat([x, h_spat.T, h_temp], dim=1)
		#gru_out, _ = self.gru(h_cat.unsqueeze(1), self.gru_init_h)
		gru_out, _ = self.gru(x, self.gru_init_h)

		#print(f'gru shape: {gru_out.shape}')
		forecasting_in = gru_out[:, -1]
		#print(f'forecasting_in shape: {forecasting_in.shape}')

		# Flatten
		#forecasting_in = gru_out.reshape(x.shape[0], -1)
		predictions = self.forecasting_model(forecasting_in)

		# TODO: Reconstruction model

		return predictions

