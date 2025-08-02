import torch
import torch.nn as nn
from einops import rearrange
from abc import ABC, abstractmethod
from typing import Tuple

@torch.jit.script
def _compiled_esn_loop(x: torch.Tensor, h: torch.Tensor, W_res: torch.Tensor, W_in: torch.Tensor, leak_rate: float) -> torch.Tensor:
    """
    JIT-kompilierte Funktion für die ESN-Zustandsaktualisierungsschleife.
    Dies vermeidet den Python-Interpreter-Overhead für jeden Zeitschritt.
    """
    # seq_len wird aus der Form des Input-Tensors abgeleitet
    for t in range(x.shape[1]):
        u_t = x[:, t, :]
        h = (1 - leak_rate) * h + leak_rate * torch.tanh(h @ W_res.T + u_t @ W_in.T)
    return h

@torch.jit.script
def _approximate_spectral_radius(W: torch.Tensor, num_iterations: int = 20) -> torch.Tensor:
    """
    Approximiert den Spektralradius (Betrag des größten Eigenwerts) einer Matrix
    effizient mit der Power-Iteration-Methode. Dies ist deutlich schneller als
    die Berechnung aller Eigenwerte mit `torch.linalg.eigvals`.
    """
    b_k = torch.randn(W.shape[1], device=W.device)
    for _ in range(num_iterations):
        # Berechne das Matrix-Vektor-Produkt
        b_k1 = W @ b_k
        # Berechne die Norm
        b_k1_norm = torch.norm(b_k1)
        # Normalisiere den Vektor für die nächste Iteration
        b_k = b_k1 / (b_k1_norm + 1e-9)

    spectral_radius = torch.norm(W @ b_k)
    return spectral_radius

class BaseReservoirExpert(nn.Module, ABC):
    """
    Abstrakte Basisklasse für ESN-Experten.

    Diese Klasse implementiert die gemeinsame Logik für alle ESN-Reservoire:
    - Erstellung der internen Reservoir-Matrix (W_res)
    - Anwendung von Sparsity und Skalierung des Spektralradius
    - Die Zustands-Update-Gleichung im Forward-Pass

    Spezifische Implementierungen (univariat, multivariat) müssen von dieser
    Klasse erben und die `forward`-Methode sowie die Initialisierung der
    Eingabe- und Ausgabeschichten vervollständigen.
    """
    def __init__(self, reservoir_size, d_model, spectral_radius, sparsity, leak_rate):
        super(BaseReservoirExpert, self).__init__()
        self.reservoir_size = reservoir_size
        self.d_model = d_model
        self.spectral_radius_target = spectral_radius
        self.sparsity = sparsity
        self.leak_rate = leak_rate

        self._create_reservoir_matrix()

    def _create_reservoir_matrix(self):
        """Erstellt und skaliert die interne Reservoir-Matrix W_res."""
        W_res = torch.randn(self.reservoir_size, self.reservoir_size)

        num_zeros = int(self.reservoir_size * self.reservoir_size * self.sparsity)
        zero_indices = torch.randperm(self.reservoir_size * self.reservoir_size)[:num_zeros]
        W_res.view(-1)[zero_indices] = 0

        # --- OPTIMIERUNG: Ersetze die langsame, exakte Eigenwertberechnung ---
        # durch die schnelle Power-Iteration-Approximation. Dies beschleunigt die
        # Modellinitialisierung erheblich, was besonders bei Optuna-Suchen mit
        # vielen Trials vorteilhaft ist.
        current_spectral_radius = _approximate_spectral_radius(W_res)
        if current_spectral_radius > 1e-9:
            W_res.mul_(self.spectral_radius_target / current_spectral_radius)

        self.register_buffer('W_res', W_res)

    def _update_reservoir_state(self, x: torch.Tensor, W_in: torch.Tensor) -> torch.Tensor:
        """Führt die Kern-Zustandsaktualisierung des Reservoirs durch."""
        batch_size, _, _ = x.shape
        h = torch.zeros(batch_size, self.reservoir_size, device=x.device)

        # Rufe die kompilierte Funktion auf, anstatt die Schleife in Python auszuführen.
        # Wir übergeben die Tensoren und den Skalar-Parameter `leak_rate`.
        h = _compiled_esn_loop(x, h, self.W_res, W_in, self.leak_rate)
        return h

    def reset_parameters(self):
        """Initialisiert die Reservoir-Matrix neu."""
        self._create_reservoir_matrix()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Muss von den Subklassen implementiert werden."""
        raise NotImplementedError


class UnivariateReservoirExpert(BaseReservoirExpert):
    """
    Ein ESN-Experte, der einen einzelnen Zeitreihenkanal verarbeitet.
    Input: [B, L, 1], Output: [B, D_Model]
    """
    def __init__(self, config):
        # Lese die spezifischen univariaten Hyperparameter aus der Konfiguration
        super().__init__(
            reservoir_size=config.reservoir_size_uni,
            d_model=config.d_model,
            spectral_radius=config.spectral_radius_uni,
            sparsity=config.sparsity_uni,
            leak_rate=getattr(config, 'leak_rate_uni', 1.0)
        )

        self.in_features = 1
        self.input_scaling = getattr(config, 'input_scaling', 1.0)

        # Spezifische Eingabegewichte für den univariaten Fall
        W_in = torch.randn(self.reservoir_size, self.in_features) * self.input_scaling
        self.register_buffer('W_in', W_in)

        # Spezifische, einfache Ausgabeschicht
        self.readout = nn.Linear(self.reservoir_size, self.d_model)

    def reset_parameters(self):
        """
        Initialisiert alle Gewichte dieses Experten neu:
        1. Ruft reset_parameters der Basisklasse auf, um W_res neu zu erstellen.
        2. Erstellt die Eingabematrix W_in neu.
        3. Initialisiert den trainierbaren Readout-Layer neu.
        """
        super().reset_parameters() # Initialisiert W_res neu
        W_in = torch.randn(self.reservoir_size, self.in_features) * self.input_scaling
        self.register_buffer('W_in', W_in)
        self.readout.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input-Tensor der Form `[Batch, SeqLen, 1]`.
        Returns:
            torch.Tensor: Feature-Vektor der Form `[Batch, d_model]`.
        """
        if x.shape[0] == 0:
            return torch.empty(0, self.d_model, device=x.device)

        # Führe die Zustandsaktualisierung der Basisklasse aus
        final_state = self._update_reservoir_state(x, self.W_in)

        # Projiziere den finalen Zustand durch die Ausgabeschicht
        output = self.readout(final_state)
        return output


class MultivariateReservoirExpert(BaseReservoirExpert):
    """
    Ein ESN-Experte, der alle Zeitreihenkanäle gleichzeitig verarbeitet.
    Verwendet einen "Structured Readout", um kanalspezifische Features zu erzeugen.
    Input: [B, L, N_Vars], Output: [B, D_Model, N_Vars]
    """
    def __init__(self, config):
        # Lese die spezifischen multivariaten Hyperparameter aus der Konfiguration
        super().__init__(
            reservoir_size=config.reservoir_size_multi,
            d_model=config.d_model,
            spectral_radius=config.spectral_radius_multi,
            sparsity=config.sparsity_multi,
            leak_rate=getattr(config, 'leak_rate_multi', 1.0)
        )

        self.n_vars = getattr(config, 'enc_in', 1)
        self.in_features = self.n_vars
        self.input_scaling = getattr(config, 'input_scaling_multi', 1.0)

        # Spezifische Eingabegewichte für den multivariaten Fall
        W_in = torch.randn(self.reservoir_size, self.in_features) * self.input_scaling
        self.register_buffer('W_in', W_in)

        # Spezifische "Structured Readout"-Ausgabeschicht
        self.readout = nn.Linear(self.reservoir_size, self.d_model * self.n_vars)

    def reset_parameters(self):
        """
        Initialisiert alle Gewichte dieses Experten neu:
        1. Ruft reset_parameters der Basisklasse auf, um W_res neu zu erstellen.
        2. Erstellt die Eingabematrix W_in neu.
        3. Initialisiert den trainierbaren Readout-Layer neu.
        """
        super().reset_parameters() # Initialisiert W_res neu
        W_in = torch.randn(self.reservoir_size, self.in_features) * self.input_scaling
        self.register_buffer('W_in', W_in)
        self.readout.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input-Tensor der Form `[Batch, SeqLen, N_Vars]`.
        Returns:
            torch.Tensor: Feature-Vektor der Form `[Batch, d_model, N_Vars]`.
        """
        if x.shape[0] == 0:
            return torch.empty(0, self.d_model, self.n_vars, device=x.device)

        # Führe die Zustandsaktualisierung der Basisklasse aus
        final_state = self._update_reservoir_state(x, self.W_in)

        # Projiziere den finalen Zustand durch die Ausgabeschicht
        output = self.readout(final_state)

        # Forme den Output um: [B, D*N] -> [B, D, N]
        output = rearrange(output, 'b (d n) -> b d n', n=self.n_vars)
        return output