import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
from pyro.nn import PyroModule
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel
import numpy as np
from pfnet.utilts import get_calculated_isotope_envelope, get_calculated_num_d


class FeatherPyroModel(PyroModule):
    def __init__(self, num_residues, cen_sigma=1.0, env_sigma=0.5, centroid_model=False, mask=None, device="cpu"):
        super().__init__()
        self.num_residues = num_residues
        self.pad_value = -100
        self._true_log_kex = None
        self.cen_sigma = torch.tensor(cen_sigma, device=device)
        self.env_sigma = torch.tensor(env_sigma, device=device)
        self.device = device
        self.mask = mask
        self.resolution_limits = None
        self.log_kch = None
        self.centroid_model = centroid_model

    def model(self, batch):
        # Global parameters
        # with pyro.plate("global_params", 1):
        #     #Global saturation level
        #     saturation = pyro.sample(
        #         "saturation",
        #         dist.Normal(saturation, 0.3)
        #     )

        #     # Global noise level
        #     sigma = pyro.sample(
        #         "sigma",
        #         dist.HalfNormal(scale=0.1)

        if self.mask is None:
            mask = self.get_mask()
        else:
            mask = self.mask
        
        log_kch = self.get_log_kch()   

        # Sample log_kex
        with pyro.plate("residues", self.num_residues, dim=-1):
            log_kex = pyro.sample("log_kex", dist.Uniform(-15, max(log_kch)))
            #log_kex = pyro.sample("log_kex", dist.Uniform(torch.full_like(log_kch, -15), log_kch))
            log_kex[mask[0]] = -15.0
            log_kex = log_kex.to(self.device)
        
        # with pyro.plate("residues", self.num_residues, dim=-1):
        #     log_P = pyro.sample("log_P", dist.Uniform(0, 14))
        #     #log_kex = pyro.sample("log_kex", dist.Uniform(torch.full_like(log_kch, -15), log_kch))
        #     log_P[mask[0]] = 14.0
        #     log_P = log_P.to(self.device)
            
        

        if self.centroid_model:
            pred_centroid, obs_centroid = self._hx_cen_forward(batch, log_kex)
            
        else:
            pred_envelope, obs_envelope, pred_centroid, obs_centroid = self._hx_forward(batch, log_kex)


        # Calculate expected deuteration
        with pyro.plate("timepoints_centroid"):

            # print(calculated_centroid.shape, observed_centroid.shape)
            pyro.sample(f"centroid", dist.Normal(loc=pred_centroid, scale=self.cen_sigma.expand(pred_centroid.shape)), obs=obs_centroid)

        if not self.centroid_model:
            
            with pyro.plate("timepoints_envelope"):
                # obs_envelope shape: (n, 20)
                # Create mask for valid peaks (values > 1e-5)
                # valid_idx = obs_envelope > 1e-5  # shape: (n, 20)
                # pyro.sample(
                #     "envelope",
                #     dist.Normal(
                #         loc=pred_envelope[valid_idx],  # flatten valid values
                #         scale=self.env_sigma.expand(torch.sum(valid_idx)),  # match the number of valid elements
                #     ),
                #     obs=obs_envelope[valid_idx],  # flatten valid observations
                # )
                sse = (obs_envelope - pred_envelope).pow(2).sum(dim=-1)

                pyro.sample(
                    "envelope",
                    dist.Normal(
                        loc=torch.zeros_like(sse),
                        scale=self.env_sigma.expand(sse.shape),  # match the number of valid elements
                    ),
                    obs=sse,  # flatten valid observations
                )
            

        return {"log_kex": log_kex}

    def get_mask(self):
        if self.mask is not None:
            return self.mask
        resolution_grouping = self._current_batch[1]['resolution_grouping'].to(self.device)

        nan_mask = resolution_grouping[:, :, 0] == 1  # non covered residues
        inf_mask = self._current_batch[1]['proline_mask'].bool()  # Prolines
        mask = (nan_mask) | (inf_mask)

        self.mask = mask

        return mask
    
    def get_log_kch(self):
        if self.log_kch is not None:
            return self.log_kch
        else:
            log_kch = self._current_batch[1]['log_kch'].squeeze(0)
            log_kch[torch.isinf(log_kch)] = 5.0
            self.log_kch = log_kch.to(self.device)
            return self.log_kch
    
    def get_resolution_limits(self):
        if self.resolution_limits is not None:
            return self.resolution_limits
        else:
            limits = self._current_batch[1]['resolution_limits'].squeeze(0)
            resid = torch.arange(1, self._current_batch[2][0][0]+1)
            padded_limits = []

            for res_i in resid:
                condition = (limits[:,0] <= res_i) & (res_i <= limits[:,1])
                if condition.sum() > 0:
                    padded_limits.append(limits[condition].squeeze(0))
                else:
                    padded_limits.append(torch.tensor([torch.tensor(res_i), torch.tensor(res_i)]))
                
            padded_limits = torch.stack(padded_limits)
            self.resolution_limits = padded_limits.to(self.device)
            return padded_limits.to(self.device)

    def _hx_forward(self, batch, log_kex):
        """
        Calculate the envelope and centroid for each peptide given the log_kex
        """
        # Calculate deuteration for each peptide
        peptide_data, residue_data, global_vars, _true_log_kex = batch
        start_pos = peptide_data["start_pos"].to(self.device)
        end_pos = peptide_data["end_pos"].to(self.device)
        time_points = peptide_data["time_point"].to(self.device)
        obs_envelope = peptide_data["probabilities"].to(self.device)
        obs_envelope_t0 = peptide_data["t0_isotope"].to(self.device)
        #back_ex = peptide_data["back_ex"].unsqueeze(-1).to(self.device)
        back_ex = 1-peptide_data["effective_deut_rent"].to(self.device)
        saturation = global_vars[:, -1].to(self.device)
        # resolution_grouping = residue_data['resolution_grouping'].to(self.device)

        mask = self.get_mask()
        pred_envelope = get_calculated_isotope_envelope(
            start_pos, end_pos, time_points, obs_envelope_t0, obs_envelope, log_kex.unsqueeze(0).to(self.device), back_ex, saturation
        )
        mass_num = torch.arange(pred_envelope.shape[-1], device=pred_envelope.device)
        obs_centroid_t0 = (obs_envelope_t0 * mass_num).sum(dim=-1)
        pred_centroid = (pred_envelope * mass_num).sum(dim=-1) - obs_centroid_t0
        obs_centroid = (obs_envelope * mass_num).sum(dim=-1) - obs_centroid_t0

        valid_tp_idx = time_points[0] != 0
        # valid_tp_num = sum(valid_tp_idx)

        # Pre-index tensors to avoid repeated indexing
        valid_pred_centroid = pred_centroid[0][valid_tp_idx]
        valid_obs_centroid = obs_centroid[0][valid_tp_idx]
        valid_pred_envelope = pred_envelope[0][valid_tp_idx]
        valid_obs_envelope = obs_envelope[0][valid_tp_idx]

        return valid_pred_envelope, valid_obs_envelope, valid_pred_centroid, valid_obs_centroid

    def _hx_cen_forward(self, batch, log_kex):
        """
        Calculate the envelope and centroid for each peptide given the log_kex
        """
        # Calculate deuteration for each peptide
        peptide_data, residue_data, global_vars, _true_log_kex = batch
        start_pos = peptide_data["start_pos"].to(self.device)
        end_pos = peptide_data["end_pos"].to(self.device)
        time_points = peptide_data["time_point"].to(self.device)
        num_d = peptide_data["num_d"].to(self.device)
        #back_ex = peptide_data["back_ex"].unsqueeze(-1).to(self.device)
        back_ex = 1-peptide_data["effective_deut_rent"].to(self.device)
        saturation = global_vars[:, -1].to(self.device)
        # resolution_grouping = residue_data['resolution_grouping'].to(self.device)

        mask = self.get_mask()
        pred_num_d = get_calculated_num_d(start_pos, end_pos, time_points, log_kex.unsqueeze(0).to(self.device), back_ex, saturation)

        valid_tp_idx = time_points[0] != 0
        # valid_tp_num = sum(valid_tp_idx)

        # Pre-index tensors to avoid repeated indexing
        valid_pred_num_d = pred_num_d[0][valid_tp_idx]
        valid_obs_num_d = num_d[0][valid_tp_idx]

        return valid_pred_num_d, valid_obs_num_d

    def potential_fn(self, params):
        """
        Computes the negative log probability for the model.
        """
        # Create a trace
        conditioned_model = poutine.condition(self.model, data=params)
        trace = poutine.trace(conditioned_model).get_trace(self._current_batch)

        return -trace.log_prob_sum().to(self.device)


class FeatherEmpiricalPyroModel(FeatherPyroModel):
    def __init__(self, num_residues, pfnet_pred, pfnet_uncertainty=None, cen_sigma=0.5, env_sigma=0.3, centroid_model=False, mask=None, device="cpu"):
        super().__init__(num_residues, cen_sigma, env_sigma, centroid_model, mask, device)
        self.pfnet_pred = pfnet_pred
        self.device = device
        self.centroid_model = centroid_model
        if pfnet_uncertainty is None:
            self.pfnet_uncertainty = torch.ones_like(pfnet_pred) * 4.0
        else:
            self.pfnet_uncertainty = pfnet_uncertainty

    def model(self, batch):


        mask = self.get_mask()

        # Sample log_kex
        with pyro.plate("residues", self.num_residues, dim=-1):
            log_kex = pyro.sample("log_kex", dist.Normal(self.pfnet_pred, self.pfnet_uncertainty))
            log_kex[mask[0]] = -15.0
            log_kex = log_kex.to(self.device)

        # hdx forward
        if self.centroid_model:
            pred_centroid, obs_centroid = self._hx_cen_forward(batch, log_kex)
        else:
            pred_envelope, obs_envelope, pred_centroid, obs_centroid = self._hx_forward(batch, log_kex)

        # Calculate expected deuteration
        with pyro.plate("timepoints_centroid"):

            # print(calculated_centroid.shape, observed_centroid.shape)
            pyro.sample(f"centroid", dist.Normal(loc=pred_centroid, scale=self.cen_sigma.expand(pred_centroid.shape)), obs=obs_centroid)

        if not self.centroid_model:

            with pyro.plate("timepoints_envelope"):
                # obs_envelope shape: (n, 20)
                # Create mask for valid peaks (values > 1e-5)
                # valid_idx = obs_envelope > 1e-5  # shape: (n, 20)

                # pyro.sample(
                #     "envelope",
                #     dist.Normal(
                #         loc=pred_envelope[valid_idx],  # flatten valid values
                #         scale=self.env_sigma.expand(torch.sum(valid_idx)),  # match the number of valid elements
                #     ),
                #     obs=obs_envelope[valid_idx],  # flatten valid observations
                # )


                sse = (obs_envelope - pred_envelope).pow(2).sum(dim=-1)

                pyro.sample(
                    "envelope",
                    dist.Normal(
                        loc=torch.zeros_like(sse),
                        scale=self.env_sigma.expand(sse.shape),  
                    ),
                    obs=sse,  # flatten valid observations
                )
            return {"log_kex": log_kex}


class FeatherMHKernel(MCMCKernel):
    def __init__(
        self,
        potential_fn,
        num_residues,
        pct_moves=25,
        temperature=10.0,
        grid_size=100,
        swap_stage=True,
        adjacency=5,
        target_accept_prob=None,
        adaptation_rate=0.02,
        device="cpu",
        grid_range=(-15.0, 5.0),
        feather_model=None,
        use_sampling_mask=True,
    ):
        super().__init__()

        self.num_residues = num_residues
        self.potential_fn = potential_fn
        self.pct_moves = pct_moves
        self.temperature = temperature
        self.swap_stage = swap_stage
        self._initial_params = None
        #self._cache = None
        # self._acceptance_ratio = 0.0
        #self._save_params = None
        self.feather_model = feather_model
        self.model = None
        self.grid_size = grid_size
        self.adjacency = adjacency
        # 设备设置
        self.device = torch.device(device)
        self.grid = torch.linspace(grid_range[0], grid_range[1], int(self.grid_size)).to(self.device)

        self.target_accept_prob = target_accept_prob
        self.adaptation_rate = adaptation_rate
        self._warmup_done = False
        self._acceptance_history = []
        self._steps_since_adaptation = 0
        self.use_sampling_mask = use_sampling_mask
        
        
    def _adapt_temperature(self, window_size=10):

        if not self._warmup_done and self.target_accept_prob is not None:
            # Add current acceptance ratio to history
            self._acceptance_history.append(self._acceptance_ratio)

            # Keep only the most recent window_size entries
            if len(self._acceptance_history) > window_size:
                self._acceptance_history.pop(0)

            # Only adapt temperature after collecting enough samples
            if len(self._acceptance_history) >= window_size:
                # Calculate mean acceptance ratio over window
                mean_acceptance = np.mean(self._acceptance_history)

                # Adapt temperature based on mean acceptance
                log_temp_delta = self.adaptation_rate * (self.target_accept_prob - mean_acceptance)
                self.temperature *= np.exp(log_temp_delta)

                # Clear history after adaptation
                self._acceptance_history = []

        #self.temperature = np.clip(self.temperature, 1e-8, 1e5)

    def grid_to_log_kex(self, indices):
        """Convert grid indices to log_kex values with bounds checking"""
        if torch.any(indices < 0) or torch.any(indices >= self.grid_size):
            raise ValueError(f"Grid indices must be between 0 and {self.grid_size-1}")
        return self.grid[indices]

    def log_kex_to_grid(self, log_kex):
        """Convert log_kex values to grid indices"""
        indices = torch.clamp(torch.round((log_kex - self.grid[0]) / (self.grid[-1] - self.grid[0]) * (self.grid_size - 1)), 0, self.grid_size - 1)
        return indices.to(dtype=torch.int64)

    def setup(self, warmup_steps, *args, **kwargs):
        self._warmup_steps = warmup_steps
        if self._initial_params is not None:
            self._save_params = list(self._initial_params.keys())

    def initialize(self, initial_params):
        """Initialize kernel state with initial parameters"""
        params = {k: v.clone() if torch.is_tensor(v) else v for k, v in initial_params.items()}

        mask = params["mask"].to(self.device)
        log_kex = params["log_kex"].to(self.device)
        log_kex[mask[0]] = -15.0
        params["log_kex"] = log_kex
    
        
        self._initial_params = params
        #self._cache = params
        #self._save_params = list(params.keys())
        return params
    
    @property
    def save_params(self):
        return ["log_kex"]
        # return self._save_params if self._save_params is not None else []

    def _compute_metropolis_acceptance(self, proposed_potential_val, current_potential_val, temperature=None):
        """Compute Metropolis acceptance ratio with numerical stability"""
        if temperature is None:
            temperature = self.temperature

        temperature = torch.tensor(temperature, device=self.device)

        log_accept_ratio = torch.tensor(-(proposed_potential_val - current_potential_val) / temperature, device=self.device).clamp(min=-100.0, max=100.0)

        return torch.exp(log_accept_ratio).item()

    def sample(self, params):
        """
        Performs one step of MH sampling with both swap and individual move phases.

        Args:
            params: Dictionary of current parameter values (e.g. {"log_kex": tensor})
        Returns:
            Dictionary of new parameter values
        """
        # if self._initial_params is None:
        #     self._initial_params = params
        #     self._cache = params

        # Extract and prepare current state
        initial_log_kex = params["log_kex"].to(self.device)
        initial_grid_idx = self.log_kex_to_grid(initial_log_kex)

        current_log_kex = params["log_kex"].to(self.device)
        current_grid_idx = self.log_kex_to_grid(current_log_kex)
        proposed_grid_idx = current_grid_idx.clone()

        log_kch = self._initial_params["log_kch"].to(self.device)
        log_kch_grid_idx = self.log_kex_to_grid(log_kch)
        mask = params["mask"].to(self.device)
        resolution_limits = self.feather_model.get_resolution_limits()
        
        # Number of residues and moves
        num_residues = current_log_kex.shape[-1]
        # num_moves = max(int(self.pct_moves * num_residues / 100), 1)

        # Get current potential
        with torch.no_grad():
            current_potential = self.potential_fn(params)
            if current_potential.numel() != 1:
                raise ValueError("potential_fn must return a scalar tensor")
            current_potential_val = current_potential.item()
            # print(current_potential_val)

        # Prepare residue indices
        if self.use_sampling_mask:
            sampling_mask = params["sampling_mask"]
            residue_indices = torch.arange(num_residues)[~sampling_mask].tolist()
        else:
            residue_indices = list(range(num_residues))
            
        num_moves = max(int(self.pct_moves * len(residue_indices) / 100), 1)
        np.random.seed(None) 
        np.random.shuffle(residue_indices)

        accepted = 0

        # --- Phase 1: Adjacent residue swaps ---
        # if self.swap_stage and np.random.rand() < self.pct_moves/100:
        if self.swap_stage:
            for i in range(num_moves):
                r1 = residue_indices[i]
                r2 = r1 + 1
                if r2 >= num_residues or mask[0][r1] or mask[0][r2]:
                    continue
                
                if self.use_sampling_mask:
                    if sampling_mask[r1] or sampling_mask[r2]:
                        continue
                    
                # Swap values
                old_val_r1 = proposed_grid_idx[..., r1].clone()
                old_val_r2 = proposed_grid_idx[..., r2].clone()
    
                # kch_r1 = self.grid_to_log_kex(log_kch_grid_idx[..., r1])
                # kch_r2 = self.grid_to_log_kex(log_kch_grid_idx[..., r2])
                # if old_val_r1 > kch_r2 or old_val_r2 > kch_r1:
                #     continue
                
                # skip the nan or Proline residues
                # if mask[0][r1] | mask[0][r2]:
                #     print(r1, r2, mask[0][r1], mask[0][r2])
                #     continue
                proposed_grid_idx[..., r1], proposed_grid_idx[..., r2] = old_val_r2, old_val_r1

                # Evaluate proposed state
                proposed_params = {"log_kex": self.grid_to_log_kex(proposed_grid_idx)}

                with torch.no_grad():
                    proposed_potential = self.potential_fn(proposed_params)
                    if proposed_potential.numel() != 1:
                        raise ValueError("potential_fn must return a scalar tensor")
                    proposed_potential_val = proposed_potential.item()

                # Metropolis acceptance
                accept_prob = self._compute_metropolis_acceptance(
                    proposed_potential_val, current_potential_val, temperature=torch.tensor(1e-10, device=self.device)
                    #proposed_potential_val, current_potential_val, temperature=self.temperature
                )

                if torch.rand(1).item() < accept_prob:
                    current_potential_val = proposed_potential_val
                    current_grid_idx = proposed_grid_idx.clone()
                else:
                    # Reject swap: revert
                    proposed_grid_idx[..., r1] = old_val_r1
                    proposed_grid_idx[..., r2] = old_val_r2

        # --- Phase 2: Individual residue moves ---
        for i in range(num_moves):
            r = residue_indices[i]
            old_val_r = proposed_grid_idx[..., r].clone()
            # skip the nan or Proline residues
            if mask[0][r]:
                #print(r, mask[0][r])
                continue

            # Propose a move
            delta = torch.randint(-self.adjacency, self.adjacency, (1,)).item()
            while delta == 0:  # Avoid no-move case
                delta = torch.randint(-self.adjacency, self.adjacency, (1,)).item()

            # proposed_grid_idx[..., r] = torch.clamp(current_grid_idx[..., r] + delta, 0, self.grid_size - 1)
            # proposed_grid_idx[..., r] = torch.clamp(current_grid_idx[..., r] + delta, 0, log_kch_grid_idx[..., r])
            max_idx = log_kch_grid_idx[..., r]  
            grid_len = max_idx + 1 
            proposed = current_grid_idx[..., r] + delta 
            proposed_grid_idx[..., r] = torch.remainder(proposed, grid_len)

            # Evaluate proposed state
            proposed_params = {"log_kex": self.grid_to_log_kex(proposed_grid_idx), **{k: v for k, v in params.items() if k != "log_kex"}}

            with torch.no_grad():
                proposed_potential = self.potential_fn(proposed_params)
                if proposed_potential.numel() != 1:
                    raise ValueError("potential_fn must return a scalar tensor")
                proposed_potential_val = proposed_potential.item()

            # Metropolis acceptance
            accept_prob = self._compute_metropolis_acceptance(proposed_potential_val, current_potential_val)

            if torch.rand(1).item() < accept_prob:
                current_potential_val = proposed_potential_val
                current_grid_idx = proposed_grid_idx.clone()
                accepted += 1
            else:
                # Reject move: revert
                proposed_grid_idx[..., r] = old_val_r

        # Update acceptance ratio (based on individual moves only)
        self._acceptance_ratio = accepted / num_moves
        #self._acceptance_ratio = (proposed_grid_idx != initial_grid_idx).sum().item() / num_moves
        
        self._adapt_temperature()

        # Return new parameters
        return {"log_kex": self.grid_to_log_kex(proposed_grid_idx), **{k: v for k, v in params.items() if k != "log_kex"}}

    @property
    def initial_params(self):
        return self._initial_params

    @initial_params.setter
    def initial_params(self, params):
        self._initial_params = params

    def logging(self):
        return {"acceptance_ratio": self._acceptance_ratio, "temperature": self.temperature}
