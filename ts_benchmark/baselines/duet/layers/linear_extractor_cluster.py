import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from .distributional_router_encoder import encoder
from .expert_factory import create_experts
from .esn.reservoir_expert import UnivariateReservoirExpert, MultivariateReservoirExpert
from einops import rearrange
from .RevIN import RevIN  # Importiere RevIN


class SparseDispatcher(object):
    """
    Helper class to group inputs by expert and scatter outputs.
    (REVISED AND CORRECTED VERSION)
    """
    def __init__(self, num_experts, gates):
        """
        gates: a float32 tensor of shape [batch_size, num_experts]
        """
        self._gates = gates
        self._num_experts = num_experts

        # Find the top-k expert assignments for each batch item
        # top_k_gates has shape [batch_size, k]
        # top_k_indices has shape [batch_size, k]
        top_k_gates, top_k_indices = torch.topk(gates, k=int((gates > 0).sum().item() / gates.shape[0]), dim=1)

        # Create a tensor of batch indices, shape [batch_size, k]
        self._batch_idx = torch.arange(gates.shape[0], device=gates.device).unsqueeze(1).expand_as(top_k_indices)

        # Flatten these to create a list of (batch, expert) pairs that are active
        self._expert_idx_flat = top_k_indices.flatten() # Shape [batch_size * k]
        self._batch_idx_flat = self._batch_idx.flatten()  # Shape [batch_size * k]
        self._gates_flat = top_k_gates.flatten()         # Shape [batch_size * k]

        # Sort by expert index to group inputs for each expert
        self._sorted_expert_idx, self._perm = self._expert_idx_flat.sort(0)
        
        # Calculate the number of inputs for each expert
        self._part_sizes = torch.bincount(self._sorted_expert_idx, minlength=num_experts).tolist()

    def dispatch(self, inp):
        """
        Dispatches the input tensor to the experts.
        inp: a float32 tensor of shape [batch_size, seq_len, n_vars]
        Returns a list of tensors, one for each expert.
        """
        # Permute the flattened batch indices and gates according to the expert sort order
        perm_batch_idx = self._batch_idx_flat[self._perm]
        
        # Gather the inputs for all active experts using the permuted batch indices
        # inp_gathered has shape [batch_size * k, seq_len, n_vars]
        inp_gathered = inp[perm_batch_idx]
        
        # Split the gathered inputs into a list, one tensor per expert
        return torch.split(inp_gathered, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """
        Combines the outputs from the experts.
        expert_out: a list of tensors, one from each expert
        Returns a float32 tensor of shape [batch_size, d_model, n_vars]
        """
        # Concatenate the expert outputs, which are already grouped by expert
        stitched = torch.cat(expert_out, 0)

        # Create a result tensor of the correct shape, filled with zeros
        # Note: We derive the output shape from the first expert's output
        output_shape = [self._gates.shape[0]] + list(stitched.shape[1:])
        combined = torch.zeros(*output_shape, device=stitched.device, dtype=stitched.dtype)

        # Permute the expert outputs back to their original batch order
        # We need an inverse permutation for this
        _, self._inverse_perm = self._perm.sort(0)
        stitched_reordered = stitched[self._inverse_perm]

        # Get the corresponding gates for the reordered outputs
        gates_to_apply = self._gates_flat.unsqueeze(1).unsqueeze(2) if multiply_by_gates else 1.0

        # Multiply outputs by their gate values
        weighted_stitched = stitched_reordered * gates_to_apply

        # Use index_add_ to scatter the weighted outputs back to the correct batch items.
        # This correctly handles cases where a batch item is routed to multiple experts.
        combined.index_add_(0, self._batch_idx_flat, weighted_stitched)
        
        return combined


class Linear_extractor_cluster(nn.Module):
    def __init__(self, config):
        super(Linear_extractor_cluster, self).__init__()
        self.config = config # Store the config object
        self.noisy_gating = getattr(config, 'noisy_gating', True)

        # --- NEU: Korrekte Berechnung der Gesamtanzahl der Experten ---
        # Dies berücksichtigt die neue hybride Architektur.
        self.num_experts = (
            getattr(config, 'num_linear_experts', 0) +
            getattr(config, 'num_univariate_esn_experts', 0) +
            getattr(config, 'num_multivariate_esn_experts', 0)
        )
        # Fallback für alte Konfigurationen, um Abstürze zu vermeiden.
        if self.num_experts == 0 and hasattr(config, 'num_esn_experts'):
             self.num_experts = getattr(config, 'num_esn_experts', 0)

        self.input_size = config.seq_len 
        self.k = config.k

        # Die `expert_factory` kümmert sich um die korrekte Konfiguration der
        # verschiedenen Expertentypen (linear, uni-esn, multi-esn).
        self.experts = create_experts(config)

        

        self.gate = encoder(config, num_experts=self.num_experts, input_dim=config.d_model)
        self.noise = encoder(config, num_experts=self.num_experts, input_dim=config.d_model)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        self.revin = RevIN(config.enc_in, affine=True) # Initialize RevIN
        # Stelle sicher, dass k nicht größer als die Anzahl der Experten ist.
        if self.num_experts > 0:
            self.k = min(self.k, self.num_experts)
        else:
            self.k = 0

        # Projection layer for gating input if it's not explicitly provided
        # Assumes gating decision is based on the last time step's features
        self.gating_input_projection = nn.Linear(config.enc_in, config.d_model)

        # NEW: Expert identity embeddings
        self.expert_embedding_dim = getattr(config, 'expert_embedding_dim', 32) # Default to 32 if not specified
        self.expert_identity_embeddings = nn.Embedding(self.num_experts, self.expert_embedding_dim)

        # Adjust input dimension for gate and noise networks
        gating_input_dim = config.d_model + self.expert_embedding_dim
        self.gate = encoder(config, num_experts=self.num_experts, input_dim=gating_input_dim)
        self.noise = encoder(config, num_experts=self.num_experts, input_dim=gating_input_dim)


    def cv_squared(self, x):
        """
        Calculates the squared coefficient of variation.
        """
        eps = 1e-10
        if x.numel() < 2:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)

        mean = x.float().mean()

        if torch.isclose(mean, torch.tensor(0.0, device=mean.device), atol=eps):
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)

        mean_sq = (x.float() ** 2).mean()
        denominator = mean.detach()**2 + eps
        division_result = mean_sq / denominator
        cv_sq = division_result - 1
        
        result = torch.clamp(cv_sq, min=0.0)
        return result

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(
        self, clean_values, noisy_values, noise_stddev, noisy_top_values
    ):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k - 1

        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        safe_noise_stddev = torch.clamp(noise_stddev, min=1e-9)

        normal = Normal(0.0, 1.0)

        prob_if_in = normal.cdf((clean_values - threshold_if_in) / safe_noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / safe_noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, gating_input=None):
        # Use gating_input if provided, otherwise use x
        if gating_input is None:
            
            context_for_gating = self.gating_input_projection(x[:, -1, :])
            
        else:
            context_for_gating = gating_input

        batch_size = context_for_gating.shape[0]

        # Get expert identity embeddings
        expert_indices = torch.arange(self.num_experts, device=x.device)
        expert_identities = self.expert_identity_embeddings(expert_indices)
        # Expand expert_identities to match batch size: [num_experts, embedding_dim] -> [1, num_experts, embedding_dim]
        expert_identities_expanded = expert_identities.unsqueeze(0)

        # Expand context_for_gating to match num_experts: [batch_size, d_model] -> [batch_size, 1, d_model]
        context_for_gating_expanded = context_for_gating.unsqueeze(1)

        # Concatenate context with expert identities for each expert
        # Resulting shape: [batch_size, num_experts, d_model + embedding_dim]
        input_for_gating_conditioned = torch.cat(
            [context_for_gating_expanded.expand(-1, self.num_experts, -1),
             expert_identities_expanded.expand(context_for_gating.shape[0], -1, -1)],
            dim=-1
        )

        # Reshape for the gate and noise networks: [batch_size * num_experts, d_model + embedding_dim]
        input_for_gating_flat = input_for_gating_conditioned.view(-1, input_for_gating_conditioned.shape[-1])


        clean_logits_flat = self.gate(input_for_gating_flat)


        clean_logits_flat = torch.nan_to_num(clean_logits_flat)


        if self.noisy_gating and train:
           

            raw_noise_logits = self.noise(input_for_gating_flat)
            print(f"DEBUG: raw_noise_logits stats: mean={raw_noise_logits.mean().item():.6f}, std={raw_noise_logits.std().item():.6f}, min={raw_noise_logits.min().item():.6f}, max={raw_noise_logits.max().item():.6f}")


            raw_noise_logits = torch.nan_to_num(raw_noise_logits)


            noise_stddev_flat = torch.clamp(self.softplus(raw_noise_logits), min=1e-6) + self.config.noise_epsilon
            print(f"DEBUG: noise_stddev_flat stats: mean={noise_stddev_flat.mean().item():.6f}, std={noise_stddev_flat.std().item():.6f}, min={noise_stddev_flat.min().item():.6f}, max={noise_stddev_flat.max().item():.6f}")


            random_noise = torch.randn_like(clean_logits_flat)
            print(f"DEBUG: random_noise stats: mean={random_noise.mean().item():.6f}, std={random_noise.std().item():.6f}, min={random_noise.min().item():.6f}, max={random_noise.max().item():.6f}")

            noisy_logits_flat = clean_logits_flat + (random_noise * noise_stddev_flat)
            print(f"DEBUG: noisy_logits_flat stats: mean={noisy_logits_flat.mean().item():.6f}, std={noisy_logits_flat.std().item():.6f}, min={noisy_logits_flat.min().item():.6f}, max={noisy_logits_flat.max().item():.6f}")
        else:
            noisy_logits_flat = clean_logits_flat

      

        # Reshape noise_stddev to [batch_size, num_experts]
        if self.noisy_gating and train:
            noise_stddev = noise_stddev_flat.view(batch_size, self.num_experts).clone()
        else:
            noise_stddev = None

        # Reshape back to [batch_size, num_experts]
        clean_logits = clean_logits_flat.view(batch_size, self.num_experts).clone()
        noisy_logits = noisy_logits_flat.view(batch_size, self.num_experts).clone()

        self.clean_logits = clean_logits # Store for access in forward
        self.noisy_logits = noisy_logits if self.noisy_gating else None # Store for access in forward

        # Top K gating
        # Ensure k is not greater than num_experts
        top_logits, top_indices = noisy_logits.topk(min(self.k, self.num_experts), dim=1)
        top_k_gates = self.softmax(top_logits)

        gates = torch.zeros_like(noisy_logits) # Initialisiere mit Nullen
        gates.scatter_(1, top_indices, top_k_gates) # Verwende in-place scatter
            
        gates = gates + self.softmax(noisy_logits) - self.softmax(noisy_logits).detach()


        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load, clean_logits, noisy_logits if self.noisy_gating else None

    def forward(self, x, gating_input=None):
        # x hat die Form [B, L, N] (für CI=False) oder [(B*N), L, 1] (für CI=True)
        gates, load, clean_logits, noisy_logits = self.noisy_top_k_gating(x, self.training, gating_input=gating_input)
        

        # The noisy_logits tensor is returned from the gating function.
        # We apply softmax to get the dense probabilities for all experts.
        dense_gates = self.softmax(noisy_logits)
        # Importance is the sum of these probabilities across the batch for each expert.
        importance = dense_gates.sum(0)

        loss_importance = self.cv_squared(importance) + self.cv_squared(load)


        loss_importance = torch.clamp(loss_importance, max=1000.0)

        
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)

        expert_outputs = []
        for i in range(self.num_experts):
            expert = self.experts[i]
            expert_input = expert_inputs[i]

            # --- NEUE, KORREKTE HYBRIDE FORWARD-LOGIK ---
            # Fall 1: Univariater ESN-Experte. Erwartet kanalunabhängigen Input.
            if isinstance(expert, UnivariateReservoirExpert):
                # Input-Form: [B, L, N] -> [(B*N), L, 1]
                num_channels_in_input = expert_input.shape[-1]
                reshaped_input = rearrange(expert_input, 'b l n -> (b n) l 1')
                raw_expert_output = expert(reshaped_input) # Gibt [ (B*N), D ] zurück
                # Output-Form: [(B*N), D] -> [B, D, N]
                unified_output = rearrange(raw_expert_output, '(b n) d -> b d n', n=num_channels_in_input)
            # Fall 2: Multivariater ESN oder linearer Experte. Können direkt mit dem Input umgehen.
            # Beide geben bereits die korrekte 3D-Form [B, D, N] zurück.
            else:
                unified_output = expert(expert_input)
            
            expert_outputs.append(unified_output)

        y = dispatcher.combine(expert_outputs)

        # Collect selection counts per expert type
        expert_selection_counts = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.num_experts):
          expert_selection_counts[i] = torch.sum(gates[:, i] > 0)

        # --- KORRIGIERTE LOGIK: Gating-Gewichte pro Expertentyp sammeln ---
        linear_weights = []
        uni_esn_weights = []
        multi_esn_weights = []

        for i in range(self.num_experts):
            avg_weight_for_expert_i = torch.mean(gates[:, i])
            expert = self.experts[i]

            if isinstance(expert, UnivariateReservoirExpert):
                uni_esn_weights.append(avg_weight_for_expert_i)
            elif isinstance(expert, MultivariateReservoirExpert):
                multi_esn_weights.append(avg_weight_for_expert_i)
            else:
                linear_weights.append(avg_weight_for_expert_i)

        # Konvertiere die Listen in Tensoren
        avg_gate_weights_linear = torch.stack(linear_weights) if linear_weights else torch.tensor([], device=x.device)
        avg_gate_weights_uni_esn = torch.stack(uni_esn_weights) if uni_esn_weights else torch.tensor([], device=x.device)
        avg_gate_weights_multi_esn = torch.stack(multi_esn_weights) if multi_esn_weights else torch.tensor([], device=x.device)

        # Die Rückgabesignatur ändert sich auf 6 Werte.

        return y, loss_importance, avg_gate_weights_linear, avg_gate_weights_uni_esn, avg_gate_weights_multi_esn, expert_selection_counts, clean_logits, noisy_logits if self.noisy_gating else None
