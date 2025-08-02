import torch
import torch.nn as nn
from ..layers.Autoformer_EncDec import series_decomp


class Linear_extractor(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Linear_extractor, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.d_model
        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in
        self.enc_in = 1 if configs.CI else configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def reset_parameters(self, expert_index: int = 0):
        """
        Initialisiert die Gewichte der linearen Schichten neu.
        Dies ersetzt die fehlerhafte Zuweisung im Konstruktor.
        """
        if self.individual:
            for i in range(self.channels):
                self._reset_linear_layer(self.Linear_Seasonal[i], expert_index)
                self._reset_linear_layer(self.Linear_Trend[i], expert_index)
        else:
            self._reset_linear_layer(self.Linear_Seasonal, expert_index)
            self._reset_linear_layer(self.Linear_Trend, expert_index)

    def _reset_linear_layer(self, layer: nn.Linear, expert_index: int):
        """Setzt eine einzelne lineare Schicht auf den gewünschten Zustand zurück."""
        with torch.no_grad():
            # 1. Wende die ursprünglich beabsichtigte Initialisierung an
            layer.weight.fill_(1 / self.seq_len)
            # 2. Füge die deterministische Störung hinzu, um die Symmetrie zu brechen
            layer.weight[0, 0] += 0.001 * expert_index
            if layer.bias is not None:
                layer.bias.zero_()

    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

    def forecast(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def forward(self, x_enc):
        if x_enc.shape[0] == 0:
            return torch.empty((0, self.pred_len, self.enc_in)).to(x_enc.device)
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

