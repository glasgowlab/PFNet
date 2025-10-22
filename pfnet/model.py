import torch
import torch.nn as nn
import math
import lightning as L
import numpy as np
import torch
from pfnet.utilts import get_calculated_isotope_envelope, get_calculated_num_d, get_log_kex_inrange
from scipy.optimize import linear_sum_assignment


class AminoAcidEncoder:
    def __init__(self, aa_vocab=None, pad_char="-"):

        if aa_vocab is None:
            aa_vocab = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
        self.pad_char = pad_char
        if self.pad_char not in aa_vocab:
            aa_vocab.append(self.pad_char)

        self.char2idx = {char: i for i, char in enumerate(aa_vocab)}
        self.idx2char = {i: char for char, i in self.char2idx.items()}
        self.char2idx[self.pad_char] = -100  # Set padding index to -100
        self.idx2char[-100] = self.pad_char  # Add -100 mapping

    def encode(self, sequences, max_length=60):

        if max_length is None:
            max_length = max(len(seq) for seq in sequences)

        batch_size = len(sequences)
        # Fill with padding index -100
        encoded = torch.full((batch_size, max_length), fill_value=-100, dtype=torch.long)  # Use -100 as padding value

        for i, seq in enumerate(sequences):
            for j, char in enumerate(seq):
                encoded[i, j] = self.char2idx[char]

        return encoded

    def decode(self, encoded_tensor):

        decoded_seqs = []
        for row in encoded_tensor:
            chars = [self.idx2char[int(idx.item())] for idx in row]
            seq = "".join(chars).rstrip(self.pad_char)
            decoded_seqs.append(seq)
        return decoded_seqs


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, n_layers):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.n_layers = n_layers
        self.num_heads = num_heads

    def forward(self, src, src_key_padding_mask=None, need_weights=False):
        if not need_weights:
            return self.encoder(src, src_key_padding_mask=src_key_padding_mask)

        # Store attention weights from all layers
        attention_weights = []
        output = src

        for i in range(self.n_layers):
            # Get the encoder layer
            layer = self.encoder.layers[i]

            # Forward pass through self-attention
            residual = output
            new_output, attn_weights = layer.self_attn(
                output,
                output,
                output,
                key_padding_mask=src_key_padding_mask,
                need_weights=True,
                average_attn_weights=False,  # Get separate weights for each head
            )

            # Store attention weights
            attention_weights.append(attn_weights)  # Shape: [batch_size, num_heads, seq_len, seq_len]

            # Apply dropout and residual connection, followed by layer normalization
            new_output = layer.dropout1(new_output)
            output = layer.norm1(new_output + residual)

            # Forward pass through feed-forward network
            residual = output
            ffn_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(output))))
            ffn_output = layer.dropout2(ffn_output)
            output = layer.norm2(ffn_output + residual)

        return output, attention_weights

    def plot_attention_maps(self, attn_weights, layer_idx=0, head_idx=0, save_path=None, protein_index=0):

        import matplotlib.pyplot as plt
        import seaborn as sns

        # Get attention weights for specified layer and head and move to CPU
        weights = attn_weights[layer_idx].cpu()

        # Select specific head and convert to numpy
        attn_map = weights[protein_index, head_idx].detach().numpy()
        max_val = attn_map.max()

        # Create figure
        plt.figure(figsize=(10, 8))
        sns.heatmap(attn_map, cmap="viridis", cbar=True, vmin=0, vmax=max_val * 0.8)

        # Set title
        title = f"Self Attention Map - Layer {layer_idx}, Head {head_idx}"
        plt.title(title)

        # Set labels
        plt.xlabel("Key position")
        plt.ylabel("Query position")

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_heads, n_layers):
        super(TransformerDecoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.n_layers = n_layers
        self.num_heads = num_heads

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, need_weights=False):
        if not need_weights:
            return self.decoder(tgt, memory, tgt_mask=None, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)

        # Store attention weights from all layers
        self_attention_weights = []
        cross_attention_weights = []
        output = tgt  # Shape: [seq_len, batch_size, embed_dim]

        for i in range(self.n_layers):
            # Get the decoder layer
            layer = self.decoder.layers[i]

            ### Self-Attention SubLayer ###
            residual = output  # Save input for residual connection
            new_output, self_attn_weights = layer.self_attn(
                output,
                output,
                output,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                need_weights=True,
                average_attn_weights=False,  # Get separate weights for each head
            )
            self_attention_weights.append(self_attn_weights.cpu())  # Shape: [batch_size, num_heads, seq_len, seq_len]
            new_output = layer.dropout1(new_output)
            output = layer.norm1(new_output + residual)  # Residual connection + LayerNorm

            ### Cross-Attention SubLayer ###
            residual = output  # Save input for residual connection
            new_output, cross_attn_weights = layer.multihead_attn(
                output,
                memory,
                memory,
                key_padding_mask=memory_key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            cross_attention_weights.append(cross_attn_weights.cpu())  # Shape: [batch_size, num_heads, seq_len, seq_len]
            new_output = layer.dropout2(new_output)
            output = layer.norm2(new_output + residual)  # Residual connection + LayerNorm

            ### Feed-Forward SubLayer ###
            residual = output  # Save input for residual connection
            # Standard Transformer DecoderLayer uses linear1, activation, dropout, linear2
            ffn_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(output))))
            ffn_output = layer.dropout3(ffn_output)  # Ensure dropout3 exists
            output = layer.norm3(ffn_output + residual)  # Residual connection + LayerNorm

        return output, (self_attention_weights, cross_attention_weights)

    def plot_attention_maps(self, attn_weights, layer_idx=0, head_idx=0, is_self_attention=True, save_path=None, protein_index=0):
        """
        Plot attention maps for a specific layer and head

        Args:
            attn_weights: List of attention weights from all layers
            layer_idx: Which layer to visualize
            head_idx: Which attention head to visualize
            is_self_attention: Whether to plot self-attention or cross-attention
            save_path: Path to save the plot (if None, will show the plot)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Get attention weights for specified layer and head
        if isinstance(attn_weights, tuple):
            # For decoder, we have both self and cross attention
            weights = attn_weights[0 if is_self_attention else 1][layer_idx]
        else:
            # For encoder, we only have self attention
            weights = attn_weights[layer_idx]

        # Select specific head and convert to numpy
        attn_map = weights[protein_index, head_idx].detach().cpu().numpy()

        # Create figure
        plt.figure(figsize=(10, 8))
        sns.heatmap(attn_map, cmap="viridis", cbar=True)

        # Set title
        title = f"{'Self' if is_self_attention else 'Cross'} Attention Map - Layer {layer_idx}, Head {head_idx}"
        plt.title(title)

        # Set labels
        plt.xlabel("Key position")
        plt.ylabel("Query position")

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


class SinusoidalEncoder(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(SinusoidalEncoder, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, positions):
        """
        positions: tensor of shape (batch_size, sequence_length)
        Returns: tensor of shape (batch_size, sequence_length, d_model)
        """
        # Expand positions to match the number of dimensions needed
        positions = positions.unsqueeze(2)  # Shape: (batch_size, sequence_length, 1)

        # Calculate the div_term
        div_term = (
            torch.exp(torch.arange(0, self.d_model, 2, device=positions.device, dtype=torch.float32) * (-math.log(self.max_len) / self.d_model))
            .unsqueeze(0)
            .unsqueeze(0)
        )  # Shape: (1, 1, d_model/2)

        # Create the positional encoding matrix
        pe = torch.zeros(positions.size(0), positions.size(1), self.d_model, device=positions.device)

        # Apply sine to even indices and cosine to odd indices
        pe[:, :, 0::2] = torch.sin(positions * div_term)
        pe[:, :, 1::2] = torch.cos(positions * div_term)

        return pe


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoder, self).__init__()
        self.sinusoidal_encoder = SinusoidalEncoder(d_model, max_len)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, d_model))

        self.layer_norm_pos = nn.LayerNorm(d_model)

    def forward(self, positions):
        sinusoidal_embeddings = self.sinusoidal_encoder(positions)
        mlp_embeddings = self.mlp(sinusoidal_embeddings)
        return self.layer_norm_pos(mlp_embeddings)


class TimePointEmbedder(nn.Module):
    def __init__(self, d_model, max_time=65):
        super(TimePointEmbedder, self).__init__()

        assert d_model % 2 == 0
        self.d_model = d_model
        self.embed_dim = d_model // 2  # each type of encoding uses half of the dimension

        # discrete embedding for common time points
        self.steps = max_time * 2 + 1
        time_grid = torch.linspace(max_time * -1, max_time, steps=self.steps)
        self.register_buffer("time_grid", time_grid)
        self.embedding_layer = nn.Embedding(self.steps, self.embed_dim, padding_idx=0)

        # sin-cos encoding
        self.pos_encoder = PositionalEncoder(d_model=self.embed_dim, max_len=self.steps)

        # layer norm
        self.norm1 = nn.LayerNorm(self.embed_dim)

        self.fusion = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model))

    def forward(self, time_point, padding_mask=None):

        if padding_mask is not None:
            padding_mask = padding_mask.bool()
        else:
            padding_mask = time_point == -100

        # convert to log2
        eps = 1e-7
        time_point = torch.log2(time_point + eps)
        time_point = time_point.masked_fill(padding_mask, -100)

        # 1. discrete embedding
        time_dists = torch.square(time_point.unsqueeze(-1) - self.time_grid)
        closest_indices = torch.argmin(time_dists, dim=-1)
        discrete_embed = self.embedding_layer(closest_indices)  # [batch, seq_len, embed_dim]
        discrete_embed = self.norm1(discrete_embed)

        # 2. sin-cos encoding, already normalized
        continuous_embed = self.pos_encoder(time_point)  # [batch, seq_len, embed_dim]

        # 3. concat
        combined = torch.cat([discrete_embed, continuous_embed], dim=-1)  # [batch, seq_len, d_model]
        time_embed = self.fusion(combined)
        time_embed = time_embed.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        # fill every position with 0
        # time_embed = time_embed.masked_fill(torch.ones_like(time_embed, dtype=torch.bool), 0.0)

        return time_embed

class ConditionalBatchNorm1d(nn.Module):
    def __init__(self, num_features, condition_dim):
        super().__init__()
        self.num_features = num_features
        
        self.bn = nn.BatchNorm1d(num_features, affine=False)
        
        self.condition_mapper = nn.Linear(condition_dim, num_features * 2)
        
        nn.init.zeros_(self.condition_mapper.weight)
        nn.init.constant_(self.condition_mapper.bias[:num_features], 1)
        nn.init.zeros_(self.condition_mapper.bias[num_features:])

    def forward(self, x, c):

        normalized_x = self.bn(x)    

        gamma_beta = self.condition_mapper(c)
        gamma = gamma_beta[:, :self.num_features].unsqueeze(-1)
        beta = gamma_beta[:, self.num_features:].unsqueeze(-1)
        
        out = normalized_x * gamma + beta
        
        return out
    
    
class ProbabilityEncoder(nn.Module):
    def __init__(self, d_model, max_size=50, condition_embed_dim=128):
        super(ProbabilityEncoder, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  # kernel_size = 4, padding=0, stride = 2
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)  # out_channels=64
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        self.cond_norm1 = ConditionalBatchNorm1d(16, condition_embed_dim)
        self.cond_norm2 = ConditionalBatchNorm1d(32, condition_embed_dim)
        self.cond_norm3 = ConditionalBatchNorm1d(64, condition_embed_dim)
        
        # Define a fully connected layer
        self.sequential = nn.Sequential(nn.Linear(64 * max_size, d_model * 4), nn.ELU(), nn.Dropout(0.1), nn.Linear(d_model * 4, d_model))

        self.act = nn.ELU()

        # self.linear_back_ex = nn.Linear(1, max_size)
        self.condition_encoder = nn.Sequential(
            nn.Linear(max_size, 256),
            nn.ELU(),
            nn.Linear(256, condition_embed_dim)
        )

        self.layer_norm_prob = nn.LayerNorm(d_model)
        self.residual_proj = nn.Linear(max_size, d_model)

        self.d_model = d_model  # Added missing attribute

    def forward(self, probabilities, back_ex):
        # Convert probabilities to logits, handling edge cases
        # eps = 1e-7  # Small epsilon to prevent log(0) or log(1)
        # probabilities = torch.clamp(probabilities, eps, 1 - eps)
        # probabilities = torch.log(probabilities / (1 - probabilities))  # Shape: [B, tps, 20]

        # Store input for residual connection
        identity = probabilities  # Shape: [B, tps, 20]

        batch_size, tps, num_bins = probabilities.shape  # num_bins should be 20
        
        back_ex_flat = back_ex.view(batch_size * tps, -1)
        condition_vec = self.condition_encoder(back_ex_flat)

        # back_ex, expand to [batch_size, tps, d_model]
        # back_ex = self.linear_back_ex(back_ex)
        # back_ex = back_ex.masked_fill(torch.ones_like(back_ex, dtype=torch.bool), 0.0)
        # back_ex = self.back_ex_encoder(back_ex)
        # probabilities = probabilities + back_ex

        # Reshape for convolution
        x = probabilities.view(batch_size * tps, 1, num_bins)  # Shape: [B * tps, 1, 20]

        # Main conv path
        x = self.conv1(x)
        x = self.cond_norm1(x, condition_vec)
        x = self.act(x)
 
        x = self.conv2(x)
        x = self.cond_norm2(x, condition_vec)
        x = self.act(x)
        
        x = self.conv3(x)
        x = self.cond_norm3(x, condition_vec)
        x = self.act(x)
        
        # x = self.cond_norm3(x, timestep_embedding)
        x = x.flatten(1)  # Fixed flattening to keep batch dimension
        x = self.sequential(x)

        # Residual connection
        identity = identity.view(batch_size * tps, num_bins)  # Shape: [B * tps, 20]
        res = self.residual_proj(identity)  # Shape: [B * tps, d_model]

        # Add residual and reshape
        output = x + res  # Shape: [B * tps, d_model]
        output = output.view(batch_size, tps, self.d_model)  # Shape: [B, tps, d_model]
        output = self.layer_norm_prob(output)  # rms norm better for transformer

        # fill every position with 0
        # output = output.masked_fill(torch.ones_like(output, dtype=torch.bool), 0.0)

        return output


class PFNet(L.LightningModule):
    def __init__(self, d_model=128, num_heads=4, num_layers=4, pad_value=-100, learning_rate=1e-4, batch_size=8, num_recycling_cycles=3):
        super(PFNet, self).__init__()
        self.d_model = d_model
        self.pos_encoder = PositionalEncoder(d_model=int(d_model / 8), max_len=1000)
        self.time_point_embedder = TimePointEmbedder(d_model=int(d_model / 4))
        self.prob_encoder = ProbabilityEncoder(d_model=int(d_model / 2), max_size=50)
        self.saturation_encoder = nn.Linear(1, int(d_model / 8))
        self.log_kch_encoder = nn.Linear(1, int(d_model / 4))
        self.resolution_encoder = nn.Linear(4, int(d_model * 3 / 8))

        self.peptide_encoder_layers = TransformerEncoder(d_model, num_heads * 2, num_layers * 2)

        # self.residue_pos_enc = PositionalEncoding_peptide(d_model*2)
        self.residue_decoder_layers = TransformerDecoder(d_model, num_heads, num_layers)

        # Output layer
        self.regression_layer = nn.Linear(d_model, 1)
        # self.classification_layer = nn.Linear(d_model, 16)
        self.confidence_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1),
        )

        self.pad_value = pad_value
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_recycling_cycles = num_recycling_cycles
        
        self.recycled_prediction_encoder = nn.Sequential(
            nn.Linear(2, int(d_model / 8)), # (log_kex, confidence)
            nn.ReLU(),
            nn.LayerNorm(int(d_model/8))
        )
        
        self.envelope_error_encoder = nn.Sequential(
            nn.Linear(1, d_model // 4), 
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model)
        )

    def forward(self, x):
        peptide_data, residue_data, global_vars, log_kex = x

        # Peptide data

        start_pos = peptide_data["start_pos"]  # Shape: (num_peptide_time_points,)
        end_pos = peptide_data["end_pos"]  # Shape: (num_peptide_time_points,)
        time_points = peptide_data["time_point"]  # Shape: (num_peptide_time_points, d_model)
        obs_envelope = peptide_data["probabilities"]  # Shape: (num_peptide_time_points, 20)
        obs_envelope_t0 = peptide_data["t0_isotope"]  # Shape: (num_peptide_time_points, 20)
        #back_ex = peptide_data["back_ex"].unsqueeze(-1)  # Shape: (num_peptide_time_points, 1)
        back_ex = 1-peptide_data["effective_deut_rent"]# Shape: (num_peptide_time_points, 1)
        resolution_grouping = residue_data["resolution_grouping"]  # Shape: (num_peptide_time_points,)
        saturation = global_vars[:, -1]

        # get mask
        peptide_padding_mask, seq_mask, envelope_mask = self.get_mask(x)

        # get peptide token embeddings
        # start_pos_embed = self.layer_norm_start(self.start_pos_embedder(start_pos, )) # d_model*2
        # length_embed = self.layer_norm_length(self.length_embedder(length, )) #d_model
        # position_embed = self.input_projection(torch.cat([start_pos_embed, length_embed], dim=-1)) #d_model/2
        start_pos_embed = self.pos_encoder(start_pos)  # d_model/8
        end_pos_embed = self.pos_encoder(end_pos)  # d_model/8
        position_embed = torch.cat([start_pos_embed, end_pos_embed], dim=-1)  # d_model/4

        time_point_embed = self.time_point_embedder(
            time_points,
        )  # d_model/4
        token_embeddings = torch.cat([position_embed, time_point_embed], dim=-1)  # d_model/2

        # get full peptide embeddings
        probabilities_embeddings = self.prob_encoder(obs_envelope, back_ex)  # d_model/2
        peptide_embeddings = torch.cat([token_embeddings, probabilities_embeddings], dim=-1)  # d_model

        transformed_peptide_embeddings = self.peptide_encoder_layers(peptide_embeddings, src_key_padding_mask=peptide_padding_mask)
        transformed_peptide_embeddings = transformed_peptide_embeddings.masked_fill(peptide_padding_mask.unsqueeze(-1), 0)

        # get residue embeddings
        batch_size = global_vars.shape[0]
        max_seq_length = global_vars[:, 0].max()
        seq_positions = torch.arange(1, max_seq_length + 1, device=self.device)
        seq_positions = seq_positions.unsqueeze(0).expand(batch_size, -1)

        residue_pos_embeddings = self.pos_encoder(seq_positions)  # d_model/8
        saturation_embeddings = self.saturation_encoder(global_vars[:, -1].unsqueeze(1)).unsqueeze(1).expand(-1, seq_positions.shape[1], -1) # d_model/8
        log_kch = residue_data['log_kch'].unsqueeze(-1)
        log_kch[seq_mask] = -100
        log_kch_embeddings = self.log_kch_encoder(log_kch)  # d_model/4
        resolution_embeddings = self.resolution_encoder(resolution_grouping)  # d_model/2
        #residue_embeddings = torch.cat([residue_embeddings, saturation_embeddings, log_kch_embeddings, resolution_embeddings], dim=-1)  # d_model

        recycled_log_kex = torch.zeros_like(log_kch)
        recycled_confidence = torch.zeros_like(log_kch)
        
        last_cycle_error_embedding = torch.zeros_like(transformed_peptide_embeddings)

        all_cycle_predictions = []
                
        for cycle in range(self.num_recycling_cycles):
            recycled_input = torch.cat([recycled_log_kex, recycled_confidence], dim=-1)
            recycled_embeddings = self.recycled_prediction_encoder(recycled_input) 
            
            residue_embeddings = torch.cat([
                residue_pos_embeddings, 
                saturation_embeddings, 
                log_kch_embeddings, 
                resolution_embeddings,
                recycled_embeddings  
            ], dim=-1)

            current_memory = transformed_peptide_embeddings + last_cycle_error_embedding

            # get decoder output
            # tgt_key_padding_mask=None, memory_key_padding_mask=None):
            decoder_output = self.residue_decoder_layers(
                residue_embeddings,
                current_memory,
                tgt_key_padding_mask=seq_mask,
                memory_key_padding_mask=peptide_padding_mask,
            )

            decoder_output = decoder_output.masked_fill_(seq_mask.unsqueeze(-1), 0)
            pred_log_kex = self.regression_layer(decoder_output).squeeze(2)
            pred_log_kex_confidence = self.confidence_layer(decoder_output).squeeze(2)
            # output_classification = self.classification_layer(decoder_output)
            
            all_cycle_predictions.append((pred_log_kex, pred_log_kex_confidence))
            
             # detach from the graph so gradients don't flow back through multiple cycles
            recycled_log_kex = pred_log_kex.detach().unsqueeze(-1)
            recycled_confidence = pred_log_kex_confidence.detach().unsqueeze(-1)
            
            with torch.no_grad():
                pred_log_kex_inf = pred_log_kex.clone().detach()
                pred_log_kex_inf[seq_mask] = -float("inf")
                recyled_pred_envelope = get_calculated_isotope_envelope(
                    start_pos, end_pos, time_points, obs_envelope_t0, obs_envelope, pred_log_kex_inf, back_ex, saturation
                )
                
                recycled_envelope_error = (recyled_pred_envelope - obs_envelope).abs().sum(dim=-1).unsqueeze(-1).float()
                recycled_envelope_error[envelope_mask] = 0
                #recycled_envelope_error = (obs_envelope - recyled_pred_envelope).float()
            
            last_cycle_error_embedding = self.envelope_error_encoder(recycled_envelope_error.detach())
            last_cycle_error_embedding = last_cycle_error_embedding.masked_fill(peptide_padding_mask.unsqueeze(-1), 0)

        final_pred_log_kex, final_pred_log_kex_confidence = all_cycle_predictions[-1]

        # Calculate the final predicted envelope using the final kex prediction
        final_pred_log_kex_mask_inf = final_pred_log_kex.clone()
        final_pred_log_kex_mask_inf[seq_mask] = -float("inf")
        pred_envelope = get_calculated_isotope_envelope(
            start_pos, end_pos, time_points, obs_envelope_t0, obs_envelope, final_pred_log_kex_mask_inf, back_ex, saturation
        )
        
        # Return all cycle predictions for loss calculation, and the final envelope
        return all_cycle_predictions, pred_envelope, obs_envelope, seq_mask, envelope_mask
    
    # def setup(self, stage=None):
    #     if stage == "fit":
    #         self.forward = torch.compile(self.forward)
    
    def get_mask(self, x):
        peptide_data, residue_data, global_vars, log_kex = x
        peptide_padding_mask = (peptide_data["start_pos"] == -100).bool()
        residue_padding_mask = (log_kex == -100).bool()
        nan_mask = (residue_data["resolution_grouping"][:, :, 0] == 1).bool()  # non-peptide residues
        inf_mask = residue_data["proline_mask"].bool()
        seq_mask = residue_padding_mask | nan_mask | inf_mask
        
        t0_mask = peptide_data["time_point"] == 0
        envelope_mask = peptide_padding_mask | t0_mask
        
        return peptide_padding_mask, seq_mask, envelope_mask
    
    def get_single_pos(self, x):
        peptide_data, residue_data, global_vars, log_kex = x
        resolution_start_idx, resolution_end_idx = residue_data['resolution_limits'][:,:,0]-1, residue_data['resolution_limits'][:,:,1]-1
        target_shape = (residue_data['resolution_grouping'].shape[0], residue_data['resolution_grouping'].shape[1])
        device = resolution_start_idx.device
        is_single_mask = (resolution_start_idx == resolution_end_idx) & (resolution_start_idx >= 0)
        batch_indices, segment_indices = torch.where(is_single_mask)
        residue_indices = resolution_start_idx[batch_indices, segment_indices]
        single_pos = torch.zeros(target_shape, dtype=torch.float32, device=device)
        single_pos[batch_indices, residue_indices.long()] = 1.0
        single_pos = single_pos.bool()
        return single_pos

    def calc_loss(self, x, mass_spec_smooth_L1=True):
        peptide_data, residue_data, global_vars, log_kex = x
        log_kch = residue_data['log_kch'].unsqueeze(-1)

        all_cycle_predictions, pred_envelope, obs_envelope, seq_mask, envelope_mask = self.forward(x)
        
        total_mse_loss = 0
        total_confidence_loss = 0
        total_kch_loss = 0
        
        # Loop over predictions from each cycle
        for pred_log_kex, pred_log_kex_confidence in all_cycle_predictions:
            # Sort predictions and ground truth for this cycle
            log_kex_sorted, log_kch_sorted, pred_log_kex_sorted, conf_sorted = self._sort_log_kex(
                log_kex, log_kch, pred_log_kex, pred_log_kex_confidence, residue_data["resolution_limits"], residue_data["proline_mask"]
            )
                        
            mask = seq_mask

            # Calculate loss for this cycle        
            total_mse_loss += nn.MSELoss()(log_kex_sorted[~mask], pred_log_kex_sorted[~mask])
            total_kch_loss += torch.mean(nn.functional.relu(pred_log_kex_sorted[~mask] - log_kch_sorted[~mask].squeeze(-1)))
            
            abs_error = torch.abs(log_kex_sorted[~mask] - pred_log_kex_sorted[~mask])
            target_confidence = torch.exp(-0.2 * abs_error)
            total_confidence_loss += nn.SmoothL1Loss()(conf_sorted[~mask], target_confidence)

        # Average the losses over the cycles
        num_cycles = len(all_cycle_predictions)
        mse_loss = total_mse_loss / num_cycles
        confidence_loss = total_confidence_loss / num_cycles
        kch_loss = total_kch_loss / num_cycles

        envelope_loss, centroid_loss = self.calc_mass_spec_loss(obs_envelope, pred_envelope, envelope_mask, smooth_L1=mass_spec_smooth_L1)

        log_kex_mse_weight = 1
        log_kex_confidence_weight = 50
        envelope_loss_weight = 50
        centroid_loss_weight = 1
        kch_loss_weight = 1
                     
        # total loss
        total_loss = (
            mse_loss * log_kex_mse_weight
            + confidence_loss * log_kex_confidence_weight
            + envelope_loss * envelope_loss_weight
            + centroid_loss * centroid_loss_weight
            + kch_loss * kch_loss_weight
        )
        
        loss_dict = {
            "mse_loss": mse_loss,
            "confidence_loss": confidence_loss,
            "envelope_loss": envelope_loss,
            "centroid_loss": centroid_loss,
            "kch_loss": kch_loss,
        }

        return loss_dict, total_loss

        return mse_loss, classification_loss, total_loss

    def calc_mass_spec_loss(self, obs_envelope, pred_envelope, envelope_mask=None, smooth_L1=True):
        """
        calculate mass spec loss: envelope loss and centroid loss
        """

        mass_num = torch.arange(pred_envelope.shape[-1], device=pred_envelope.device)
        pred_centroid = (pred_envelope * mass_num).sum(dim=-1)
        obs_centroid = (obs_envelope * mass_num).sum(dim=-1)

        if smooth_L1:
            envelope_loss = nn.SmoothL1Loss(reduction="none")(pred_envelope[~envelope_mask], obs_envelope[~envelope_mask]).sum(dim=-1).mean()
            centroid_loss = nn.SmoothL1Loss()(pred_centroid[~envelope_mask], obs_centroid[~envelope_mask])
        else:
            envelope_loss = nn.L1Loss(reduction="none")(pred_envelope[~envelope_mask], obs_envelope[~envelope_mask]).sum(dim=-1).mean()
            centroid_loss = nn.L1Loss()(pred_centroid[~envelope_mask], obs_centroid[~envelope_mask])

        return envelope_loss, centroid_loss

    def _sort_log_kex(self, log_kex, log_kch, pred_log_kex, confidence, resolution_limits, pro_mask, only_pro_segment=False):
        """sort log_kex based on the resolution limits"""

        log_kex_clone = log_kex.clone()
        log_kch_clone = log_kch.clone()
        pred_log_kex_clone = pred_log_kex.clone()
        confidence_clone = confidence.clone()

        batch_size = log_kex.shape[1]
        batch_idx = torch.arange(resolution_limits.shape[0], device=resolution_limits.device).unsqueeze(1).expand(-1, resolution_limits.shape[1])
        start_idx = resolution_limits[:, :, 0] - 1 + batch_idx * batch_size
        end_idx = resolution_limits[:, :, 1] - 1 + batch_idx * batch_size

        # skip single residue
        start_idx = start_idx.view(-1)
        end_idx = end_idx.view(-1)
        single_residue_mask = start_idx == end_idx
        start_idx = start_idx[~single_residue_mask]
        end_idx = end_idx[~single_residue_mask]

        # from copy import deepcopy
        log_kex_sorted_flat = log_kex_clone.view(-1)
        log_kch_sorted_flat = log_kch_clone.view(-1)
        pred_log_kex_sorted_flat = pred_log_kex_clone.view(-1)
        confidence_sorted_flat = confidence_clone.view(-1)
        
        pro_mask_flat = pro_mask.view(-1)

        for s, e in zip(start_idx, end_idx):
            segment_has_pro = pro_mask_flat[s : e + 1].sum() > 0
            # skip segments without pro when only_pro_segment=True
            if only_pro_segment and not segment_has_pro:    
                continue
                
            # non-pro in current segment
            segment_mask = ~pro_mask_flat[s : e + 1].bool()
            
            if segment_mask.any(): 
                # Sort only non-pro positions, skip pro positions
                segment_slice = slice(s, e + 1)
                
            
                _ , truth_sort_indices = torch.sort(log_kex_sorted_flat[segment_slice][segment_mask], descending=True)
                
                # sort the truth
                log_kex_sorted_flat[segment_slice][segment_mask] = log_kex_sorted_flat[segment_slice][segment_mask][truth_sort_indices]
                log_kch_sorted_flat[segment_slice][segment_mask] = log_kch_sorted_flat[segment_slice][segment_mask][truth_sort_indices]
                
                # sort the pred
                _, pred_sort_indices = torch.sort(pred_log_kex_sorted_flat[segment_slice][segment_mask], descending=True)
                pred_log_kex_sorted_flat[segment_slice][segment_mask] = pred_log_kex_sorted_flat[segment_slice][segment_mask][pred_sort_indices]
                
                # sort the confidence
                confidence_sorted_flat[segment_slice][segment_mask] = confidence_sorted_flat[segment_slice][segment_mask][pred_sort_indices]

                
        # log_kex_sorted = log_kex_sorted_flat.view(log_kex.shape)
        log_kex_sorted = log_kex_sorted_flat.view(log_kex.shape)
        log_kch_sorted = log_kch_sorted_flat.view(log_kch.shape)
        pred_log_kex_sorted = pred_log_kex_sorted_flat.view(pred_log_kex.shape)
        confidence_sorted = confidence_sorted_flat.view(confidence.shape)

        return log_kex_sorted, log_kch_sorted, pred_log_kex_sorted, confidence_sorted

    def training_step(self, batch, batch_idx):
        loss_dict, total_loss = self.calc_loss(batch)

        self.log("train_loss", total_loss, batch_size=self.batch_size)
        self.log("train_confidence_loss", loss_dict["confidence_loss"], batch_size=self.batch_size)
        self.log("train_mse_loss", loss_dict["mse_loss"], batch_size=self.batch_size)
        self.log("train_envelope_loss", loss_dict["envelope_loss"], batch_size=self.batch_size)
        self.log("train_centroid_loss", loss_dict["centroid_loss"], batch_size=self.batch_size)
        self.log("train_kch_loss", loss_dict["kch_loss"], batch_size=self.batch_size)
        
        del batch
        torch.cuda.empty_cache()

        return total_loss

    def validation_step(self, batch, batch_idx):
        loss_dict, total_loss = self.calc_loss(batch)
        self.log("val_loss", total_loss, batch_size=self.batch_size, add_dataloader_idx=False)
        self.log("val_confidence_loss", loss_dict["confidence_loss"], batch_size=self.batch_size, add_dataloader_idx=False)
        self.log("val_mse_loss", loss_dict["mse_loss"], batch_size=self.batch_size, add_dataloader_idx=False)
        self.log("val_envelope_loss", loss_dict["envelope_loss"], batch_size=self.batch_size, add_dataloader_idx=False)
        self.log("val_centroid_loss", loss_dict["centroid_loss"], batch_size=self.batch_size, add_dataloader_idx=False)

        del batch
        torch.cuda.empty_cache()
        return total_loss

    def test_step(self, batch, batch_idx):
        loss_dict, total_loss = self.calc_loss(batch)
        self.log("test_loss", total_loss, batch_size=self.batch_size)
        self.log("test_confidence_loss", loss_dict["confidence_loss"], batch_size=self.batch_size)
        self.log("test_mse_loss", loss_dict["mse_loss"], batch_size=self.batch_size)
        self.log("test_envelope_loss", loss_dict["envelope_loss"], batch_size=self.batch_size)
        self.log("test_centroid_loss", loss_dict["centroid_loss"], batch_size=self.batch_size)
        del batch
        torch.cuda.empty_cache()
        return total_loss

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=5e-4)

        return optimizer
        # steps_per_epoch = self.trainer.estimated_stepping_batches//self.trainer.max_epochs
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer,
        #     T_0=steps_per_epoch * 4,
        #     T_mult=2,
        #     eta_min=self.learning_rate * 0.01
        # )

        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "interval": "step",
        #         "frequency": 1
        #     },
        # }


class PFNetCentroid(PFNet):
    def __init__(self, d_model=128, num_heads=4, num_layers=4, pad_value=-100, learning_rate=1e-4, batch_size=8, num_recycling_cycles=3):
        super(PFNetCentroid, self).__init__(d_model, num_heads, num_layers, pad_value, learning_rate, batch_size, num_recycling_cycles)
        
        del self.prob_encoder
        self.num_d_encoder = NumDEncoder(d_model=int(d_model / 2))

    def forward(self, x):
        peptide_data, residue_data, global_vars, log_kex = x

        # Peptide data
        start_pos = peptide_data["start_pos"]
        end_pos = peptide_data["end_pos"]
        time_points = peptide_data["time_point"]
        num_d = peptide_data["num_d"].unsqueeze(-1)
        # back_ex = peptide_data["back_ex"].unsqueeze(-1)
        back_ex = 1-peptide_data["effective_deut_rent"]
        resolution_grouping = residue_data["resolution_grouping"]
        saturation = global_vars[:, -1]
        log_kch = residue_data['log_kch'].unsqueeze(-1)

        # get mask
        peptide_padding_mask, seq_mask, num_d_mask = self.get_mask(x)

        # get peptide token embeddings
        start_pos_embed = self.pos_encoder(start_pos)
        end_pos_embed = self.pos_encoder(end_pos)
        position_embed = torch.cat([start_pos_embed, end_pos_embed], dim=-1)

        time_point_embed = self.time_point_embedder(time_points)
        token_embeddings = torch.cat([position_embed, time_point_embed], dim=-1)

        # get full peptide embeddings
        num_d_embeddings = self.num_d_encoder(num_d, back_ex)  # d_model/2
        peptide_embeddings = torch.cat([token_embeddings, num_d_embeddings], dim=-1)  # d_model

        transformed_peptide_embeddings = self.peptide_encoder_layers(peptide_embeddings, src_key_padding_mask=peptide_padding_mask)
        transformed_peptide_embeddings = transformed_peptide_embeddings.masked_fill(peptide_padding_mask.unsqueeze(-1), 0)

        # get residue embeddings
        batch_size = global_vars.shape[0]
        max_seq_length = global_vars[:, 0].max()
        seq_positions = torch.arange(1, max_seq_length + 1, device=self.device)
        seq_positions = seq_positions.unsqueeze(0).expand(batch_size, -1)

        residue_pos_embeddings = self.pos_encoder(seq_positions)
        saturation_embeddings = self.saturation_encoder(global_vars[:, -1].unsqueeze(1)).unsqueeze(1).expand(-1, seq_positions.shape[1], -1)
        log_kch[seq_mask] = -100
        log_kch_embeddings = self.log_kch_encoder(log_kch)
        resolution_embeddings = self.resolution_encoder(resolution_grouping)

        recycled_log_kex = torch.zeros_like(log_kch)
        recycled_confidence = torch.zeros_like(log_kch)
        
        last_cycle_error_embedding = torch.zeros_like(transformed_peptide_embeddings)
        
        all_cycle_predictions = []

        for cycle in range(self.num_recycling_cycles):
            recycled_input = torch.cat([recycled_log_kex, recycled_confidence], dim=-1)
            recycled_embeddings = self.recycled_prediction_encoder(recycled_input)

            residue_embeddings = torch.cat([
                residue_pos_embeddings,
                saturation_embeddings,
                log_kch_embeddings,
                resolution_embeddings,
                recycled_embeddings
            ], dim=-1)


            current_memory = transformed_peptide_embeddings + last_cycle_error_embedding

            decoder_output = self.residue_decoder_layers(
                residue_embeddings,
                current_memory,
                tgt_key_padding_mask=seq_mask,
                memory_key_padding_mask=peptide_padding_mask,
            )

            decoder_output = decoder_output.masked_fill_(seq_mask.unsqueeze(-1), 0)
            pred_log_kex = self.regression_layer(decoder_output).squeeze(2)
            pred_log_kex_confidence = self.confidence_layer(decoder_output).squeeze(2)

            all_cycle_predictions.append((pred_log_kex, pred_log_kex_confidence))

            recycled_log_kex = pred_log_kex.detach().unsqueeze(-1)
            recycled_confidence = pred_log_kex_confidence.detach().unsqueeze(-1)
            
            
            with torch.no_grad():
                pred_log_kex_inf = pred_log_kex.clone().detach()
                pred_log_kex_inf[seq_mask] = -float("inf")
                recycled_num_d = get_calculated_num_d(start_pos, end_pos, time_points, pred_log_kex_inf, back_ex, saturation).unsqueeze(-1)
                recycled_num_d_error = recycled_num_d - num_d
                recycled_num_d_error[num_d_mask] = 0
                
            last_cycle_error_embedding = self.envelope_error_encoder(recycled_num_d_error.detach())
            last_cycle_error_embedding = last_cycle_error_embedding.masked_fill(peptide_padding_mask.unsqueeze(-1), 0)

        pred_num_d = recycled_num_d

        return all_cycle_predictions, pred_num_d, num_d, seq_mask, num_d_mask

    def calc_loss(self, x, mass_spec_smooth_L1=True):
        peptide_data, residue_data, global_vars, log_kex = x
        log_kch = residue_data['log_kch'].unsqueeze(-1)

        all_cycle_predictions, pred_num_d, num_d, seq_mask, num_d_mask = self.forward(x)

        total_mse_loss = 0
        total_confidence_loss = 0
        total_kch_loss = 0

        # Loop over predictions from each cycle
        for pred_log_kex, pred_log_kex_confidence in all_cycle_predictions:
            # Sort predictions and ground truth for this cycle
            # log_kex_sorted, log_kch_sorted, pred_log_kex_sorted, conf_sorted = self._sort_log_kex(
            #     log_kex, log_kch, pred_log_kex, pred_log_kex_confidence, residue_data["resolution_limits"], residue_data["proline_mask"]
            # )
            log_kex_sorted = log_kex
            log_kch_sorted = log_kch
            pred_log_kex_sorted = pred_log_kex
            conf_sorted = pred_log_kex_confidence
            
            inf_mask = torch.isinf(log_kex_sorted)
            mask = seq_mask | inf_mask

            # Calculate loss for this cycle        
            total_mse_loss += nn.MSELoss()(log_kex_sorted[~mask], pred_log_kex_sorted[~mask])
            total_kch_loss += torch.mean(nn.functional.relu(pred_log_kex_sorted[~mask] - log_kch_sorted[~mask].squeeze(-1)))
            
            abs_error = torch.abs(log_kex_sorted[~mask] - pred_log_kex_sorted[~mask])
            target_confidence = torch.exp(-0.2 * abs_error)
            total_confidence_loss += nn.SmoothL1Loss()(conf_sorted[~mask], target_confidence)
            
        # Average the losses over the cycles
        num_cycles = len(all_cycle_predictions)
        mse_loss = total_mse_loss / num_cycles
        confidence_loss = total_confidence_loss / num_cycles
        kch_loss = total_kch_loss / num_cycles
        
        # Centroid loss
        if mass_spec_smooth_L1:
            centroid_loss = nn.SmoothL1Loss()(pred_num_d[~num_d_mask], num_d[~num_d_mask])
        else:
            centroid_loss = nn.L1Loss()(pred_num_d[~num_d_mask], num_d[~num_d_mask])

        log_kex_mse_weight = 1
        log_kex_confidence_weight = 5
        centroid_loss_weight = 1
        kch_loss_weight = 1          

        # total loss
        total_loss = (
            mse_loss * log_kex_mse_weight
            + confidence_loss * log_kex_confidence_weight
            + centroid_loss * centroid_loss_weight
            + kch_loss * kch_loss_weight
        )

        loss_dict = {
            "mse_loss": mse_loss,
            "confidence_loss": confidence_loss,
            "centroid_loss": centroid_loss,
            "kch_loss": kch_loss,
        }

        return loss_dict, total_loss

    def training_step(self, batch, batch_idx):
        loss_dict, total_loss = self.calc_loss(batch)

        self.log("train_loss", total_loss, batch_size=self.batch_size)
        self.log("train_confidence_loss", loss_dict["confidence_loss"], batch_size=self.batch_size)
        self.log("train_mse_loss", loss_dict["mse_loss"], batch_size=self.batch_size)
        self.log("train_centroid_loss", loss_dict["centroid_loss"], batch_size=self.batch_size)
        self.log("train_kch_loss", loss_dict["kch_loss"], batch_size=self.batch_size)

        del batch
        torch.cuda.empty_cache()

        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            loss_dict, total_loss = self.calc_loss(batch)
            self.log("val_loss", total_loss, batch_size=self.batch_size, add_dataloader_idx=False)
            self.log("val_confidence_loss", loss_dict["confidence_loss"], batch_size=self.batch_size, add_dataloader_idx=False)
            self.log("val_mse_loss", loss_dict["mse_loss"], batch_size=self.batch_size, add_dataloader_idx=False)
            self.log("val_centroid_loss", loss_dict["centroid_loss"], batch_size=self.batch_size, add_dataloader_idx=False)
            self.log("val_kch_loss", loss_dict["kch_loss"], batch_size=self.batch_size, add_dataloader_idx=False)
        elif dataloader_idx == 1:
            loss_dict, total_loss = self.calc_loss(batch, mass_spec_smooth_L1=False)
            self.log("laci_centroid_loss", loss_dict["centroid_loss"], batch_size=self.batch_size, add_dataloader_idx=False)
            self.log("laci_kch_loss", loss_dict["kch_loss"], batch_size=self.batch_size, add_dataloader_idx=False)

        del batch
        torch.cuda.empty_cache()
        return total_loss

    def test_step(self, batch, batch_idx):
        loss_dict, total_loss = self.calc_loss(batch)
        self.log("test_loss", total_loss, batch_size=self.batch_size)
        self.log("test_confidence_loss", loss_dict["confidence_loss"], batch_size=self.batch_size)
        self.log("test_mse_loss", loss_dict["mse_loss"], batch_size=self.batch_size)
        self.log("test_centroid_loss", loss_dict["centroid_loss"], batch_size=self.batch_size)
        self.log("test_kch_loss", loss_dict["kch_loss"], batch_size=self.batch_size)

        del batch
        torch.cuda.empty_cache()
        return total_loss


class NumDEncoder(nn.Module):
    def __init__(self, d_model, max_size=50):
        super(NumDEncoder, self).__init__()
        self.max_size = max_size
        self.d_model = d_model
        self.linear = nn.Linear(1, d_model)
        
        # self.linear_back_ex = nn.Linear(1, max_size)
        self.condition_encoder = nn.Sequential(
            nn.Linear(max_size, 256),
            nn.ELU(),
            nn.Linear(256, d_model)
        )
        
        self.cond_norm1 = ConditionalBatchNorm1d(1, d_model)
        self.act = nn.ELU()

    def forward(self, num_d, back_ex):
        
        batch_size, tps, _ = num_d.shape  
            
        back_ex_flat = back_ex.view(batch_size * tps, -1)
        condition_vec = self.condition_encoder(back_ex_flat)
        
        x = self.linear(num_d).view(batch_size * tps, 1, self.d_model) 
        x = self.cond_norm1(x, condition_vec)
        x = x.view(batch_size, tps, -1)
        x = self.act(x)
        
        return x