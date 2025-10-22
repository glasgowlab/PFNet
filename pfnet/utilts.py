import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import os
import tempfile
import glob
import json
import urllib.request
import traceback

from pigeon_feather.analysis import get_res_avg_logP, get_res_avg_logP_std, get_res_avg_log_kex
from pigeon_feather.tools import calculate_coverages
from pigeon_feather.data import HDXStatePeptideCompares
from pigeon_feather.hxio import load_HXMS_file


def get_calculated_isotope_envelope(start_pos, end_pos, time_points, envelope_t0, envelope, pred_log_kex, back_ex, saturation):

    # grab log_kex in range for each tps
    log_kex_inrange = get_log_kex_inrange(pred_log_kex, start_pos, end_pos)

    # expand time_points to the same shape as log_kex_inrange
    time_point_enc_expanded = time_points.unsqueeze(2).expand(-1, -1, log_kex_inrange.size(2))

    # calculate deut_uptake
    deut_uptake = 1 - torch.exp(-1 * (10**log_kex_inrange) * time_point_enc_expanded)
    #deut_uptake[time_point_enc_expanded == -100] = 0
    #deut_uptake = deut_uptake * (1 - back_ex) * saturation.unsqueeze(-1).unsqueeze(-1)
    deut_uptake[time_point_enc_expanded == -100] = 0
    deut_uptake = torch.nn.functional.pad(deut_uptake, (0, 50 - deut_uptake.size(-1))) 
    deut_uptake = deut_uptake * (1 - back_ex) * saturation.unsqueeze(-1).unsqueeze(-1)

    # flatten deut_uptake and t0_probabilities
    deut_uptake_flat = deut_uptake.view(-1, 50)
    envelope_t0_flat = envelope_t0.view(-1, 50)

    # Compute the probabilities without loops
    p_D_flat = batch_poisson_binomial_pmf(deut_uptake_flat)

    # caculated isotope envelope using log_kex_inrange
    caculated_envelope_flat = batch_convolve(envelope_t0_flat, p_D_flat)
    caculated_envelope = caculated_envelope_flat.view(envelope.shape)
    # set the padded values as probabilities
    caculated_envelope[envelope == -100] = -100
    # isotope_envelope_calculated = torch.tensor(isotope_envelope_calculated, dtype=torch.float32)

    return caculated_envelope


def get_calculated_num_d(start_pos, end_pos, time_points, pred_log_kex, back_ex, saturation):

    # grab log_kex in range for each tps
    log_kex_inrange = get_log_kex_inrange(pred_log_kex, start_pos, end_pos)

    # expand time_points to the same shape as log_kex_inrange
    time_point_enc_expanded = time_points.unsqueeze(2).expand(-1, -1, log_kex_inrange.size(2))

    # calculate deut_uptake
    deut_uptake = 1 - torch.exp(-1 * (10**log_kex_inrange) * time_point_enc_expanded)
    #deut_uptake = deut_uptake * (1 - back_ex) * saturation.unsqueeze(-1).unsqueeze(-1)
    deut_uptake[time_point_enc_expanded == -100] = 0
    deut_uptake = torch.nn.functional.pad(deut_uptake, (0, 50 - deut_uptake.size(-1))) 
    deut_uptake = deut_uptake * (1 - back_ex) * saturation.unsqueeze(-1).unsqueeze(-1)

    # flatten deut_uptake and t0_probabilities
    calculated_num_d = deut_uptake.sum(dim=-1)

    return calculated_num_d


def get_log_kex_inrange(pred_log_kex, start_pos, end_pos):

    # Ensure start_pos <= end_pos
    start_pos, end_pos = torch.min(start_pos, end_pos), torch.max(start_pos, end_pos)

    # Calculate lengths
    lengths = end_pos - start_pos + 1
    max_len = lengths.max().item()

    # Build index matrices
    batch_size, num_ranges = start_pos.size()
    batch_indices = torch.arange(batch_size).view(-1, 1, 1).to(device=start_pos.device)  # (batch_size, 1, 1)
    range_indices = torch.arange(max_len).view(1, 1, -1).to(device=start_pos.device)  # (1, 1, max_len)

    # Broadcast and calculate actual indices
    valid_mask = range_indices < lengths.unsqueeze(-1)  # Check if range is valid
    gather_indices = start_pos.unsqueeze(-1) + range_indices - 1  # Map start_pos to indices (note -1 since indices start at 0)

    # Restrict to valid range
    gather_indices = torch.where(valid_mask, gather_indices, torch.zeros_like(gather_indices))
    gather_indices = gather_indices.clamp(min=0, max=pred_log_kex.size(1) - 1)

    # Use gather to directly get subarrays
    padded_outputs = pred_log_kex[batch_indices, gather_indices]  # (batch_size, num_ranges, max_len)
    padded_outputs[~valid_mask] = float("-inf")

    return padded_outputs


def batch_poisson_binomial_pmf(p_arrays):
    """
    Poisson binomial PMF
    """
    device = p_arrays.device
    if isinstance(p_arrays, torch.Tensor):
        p_arrays = p_arrays.detach().cpu().numpy()

    batch_size, n = p_arrays.shape
    dp = np.zeros((batch_size, n + 1))
    dp[:, 0] = 1.0  # Base case: probability of 0 successes

    for k in range(n):
        p_k = p_arrays[:, k]  # Shape (batch_size,)
        q_k = 1 - p_k

        # Update dp in place using vectorized operations
        dp[:, 1:] = dp[:, 1:] * q_k[:, np.newaxis] + dp[:, :-1] * p_k[:, np.newaxis]
        dp[:, 0] *= q_k

    # Handle possible negative values due to numerical errors
    dp = np.maximum(dp, 0)

    # Normalize the PMFs to ensure they sum to 1
    dp /= dp.sum(axis=1, keepdims=True)

    # Convert back to torch.Tensor if needed
    pmfs = torch.from_numpy(dp).to(device=device)
    return pmfs


def batch_convolve(a_arrays, b_arrays):
    device = a_arrays.device
    batch_size, n1 = a_arrays.shape
    _, n2 = b_arrays.shape
    n = n1 + n2 - 1  # Length after convolution

    # Convert to numpy arrays
    a_arrays_np = a_arrays.detach().cpu().numpy()
    b_arrays_np = b_arrays.detach().cpu().numpy()

    # Compute FFTs along the last dimension
    A = np.fft.fft(a_arrays_np, n=n, axis=1)
    B = np.fft.fft(b_arrays_np, n=n, axis=1)

    # Multiply in frequency domain
    C = A * B

    # Inverse FFT to get the convolution result
    c_arrays = np.fft.ifft(C, axis=1).real
    c_arrays = c_arrays[:, :50]

    # force envelopes to be non-negative
    c_arrays[c_arrays < 0] = 0
    c_arrays = c_arrays / np.sum(c_arrays, axis=-1, keepdims=True)

    # Convert back to tensor on correct device
    c_arrays = torch.from_numpy(c_arrays).to(device=device)

    return c_arrays


def get_display_state_name(state_key, state_keys):
    """Get display name for state, handling duplicates."""
    original_state_names = ["_".join(key.split("_")[:-1]) for key in state_keys]
    has_duplicates = len(original_state_names) != len(set(original_state_names))
    
    if has_duplicates:
        return state_key
    else:
        return "_".join(state_key.split("_")[:-1])


def get_all_statics_info(hdxms_datas):
    """Generate comprehensive statistics for HDX-MS data."""
    if not isinstance(hdxms_datas, list):
        hdxms_datas = [hdxms_datas]

    state_names = list(set([state.state_name for data in hdxms_datas for state in data.states]))

    protein_sequence = hdxms_datas[0].protein_sequence

    # coverage statistics
    coverage = [calculate_coverages([data], state.state_name)
                for data in hdxms_datas for state in data.states]
    coverage = np.mean(np.array(coverage), axis=0)
    coverage_non_zero = 1 - np.count_nonzero(coverage == 0) / len(protein_sequence)

    # peptides and statistics
    all_peptides = [pep for data in hdxms_datas for state in data.states for pep in state.peptides]
    unique_peptides = set(pep.identifier for pep in all_peptides)
    avg_pep_length = np.mean([len(pep.sequence) for pep in all_peptides])

    # all tps
    all_tps = [tp for pep in all_peptides for tp in pep.timepoints if tp.deut_time != np.inf and tp.deut_time != 0.0]
    time_course = sorted(list(set([tp.deut_time for tp in all_tps])))

    def _group_and_average(numbers, threshold=50):
        numbers.sort()
        groups, current_group = [], [numbers[0]]
        for number in numbers[1:]:
            (current_group.append(number) if number - current_group[0] <= threshold else (groups.append(current_group), current_group := [number]))
        groups.append(current_group)
        return groups, [round(sum(group) / len(group), 1) for group in groups]

    groups, avg_timepoints = _group_and_average(time_course)

    # back exchange 
    peptides_with_exp = [pep for pep in all_peptides if pep.get_timepoint(np.inf) is not None]
    backexchange_rates = [1 - pep.max_d / pep.theo_max_d for pep in peptides_with_exp]
    if backexchange_rates == []:
        iqr_backexchange = np.nan
    else:    
        iqr_backexchange = np.percentile(backexchange_rates, 75) - np.percentile(backexchange_rates, 25)

    redundancy = np.mean(coverage)

    stats_text = (
        "=" * 60 + "\n" +
        " " * 20 + "HDX-MS Data Statistics\n" +
        "=" * 60 + "\n" +
        f"States names: {state_names}\n" +
        f"Time course (s): {avg_timepoints}\n" +
        f"Number of time points: {len(avg_timepoints)}\n" +
        f"Protein sequence length: {len(protein_sequence)}\n" +
        f"Average coverage: {coverage_non_zero:.2f}\n" +
        f"Number of unique peptides: {len(unique_peptides)}\n" +
        f"Average peptide length: {avg_pep_length:.1f}\n" +
        f"Redundancy (based on average coverage): {redundancy:.1f}\n" +
        f"Average peptide length to redundancy ratio: {avg_pep_length / redundancy:.1f}\n" +
        f"Backexchange average, IQR: {np.mean(backexchange_rates):.2f}, {iqr_backexchange:.2f}\n" +
        "=" * 60
    )
    
    return stats_text


def get_log_kex_plot(ana_objs, output_dir):
    """Generate log(kex) plot for all states."""
    first_key = list(ana_objs.keys())[0]
    seq_len = len(ana_objs[first_key].protein_sequence)
    num_len = int(np.ceil(seq_len / 150))

    fig, ax = plt.subplots(1, 1, figsize=(40*num_len, 8), sharey=True, sharex=True)

    state_keys = list(ana_objs.keys())
    
    for idx, (state_key, ana_obj) in enumerate(ana_objs.items()):
        state_name = get_display_state_name(state_key, state_keys)
        ana_obj.plot_kex_bar(
            ax=ax, resolution_indicator_pos=15-idx, label=state_name, show_seq=False,
        )
    
    ax.set_xlabel("Residue", fontsize=24)
    
    spacing = int(seq_len // 150 * 1 + 1)
    ax.set_xticks(ax.get_xticks()[::spacing])
    ax.set_xticklabels(ax.get_xticklabels(), fontdict={"fontsize": 24})
    seq_pos = 17
    for ii in range(0, seq_len, spacing):
        ax.text(ii, seq_pos, ana_objs[first_key].protein_sequence[ii], ha="center", va="center", fontsize=22)
        
    from matplotlib.colors import Normalize
    from matplotlib import cm
    coverage_max = np.nanmax(ana_objs[first_key].coverage)
    norm = Normalize(vmin=0, vmax=coverage_max)
    sm = cm.ScalarMappable(cmap=plt.cm.Blues, norm=norm)
    sm.set_array([])
    cbar_width_inch = 100 / 72
    cbar_height_inch = 20 / 72
    fig_width_inch, fig_height_inch = fig.get_size_inches()
    last_residue_pos = seq_len - 1
    ax_pos = ax.get_position()
    xlim = ax.get_xlim()
    last_residue_fig_pos = ax_pos.x0 + (last_residue_pos - xlim[0]) / (xlim[1] - xlim[0]) * ax_pos.width
    x_pos = last_residue_fig_pos - (cbar_width_inch / fig_width_inch)
    cbar_ax = fig.add_axes([
        x_pos,
        0.7,
        cbar_width_inch / fig_width_inch,
        cbar_height_inch / fig_height_inch
    ])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Coverage', fontsize=18, rotation=0)
    cbar.set_ticks([0, coverage_max])
    cbar.set_ticklabels(['0', f'{int(coverage_max)}'])
    cbar.ax.tick_params(labelsize=18)
    cbar.outline.set_visible(False)
    
    ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.7))

    fig.savefig(f"{output_dir}/pfnet_plots/log_kex_plot.png")
    plt.close()
    
    return f"{output_dir}/pfnet_plots/log_kex_plot.png"


def create_heatmap_compare(compare, colorbar_max, colormap="RdBu"):
    """Create heatmap for comparing two states."""
    import matplotlib.colors as col
    from matplotlib.patches import Rectangle
    from matplotlib import cm
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    if not compare.peptide_compares or len(compare.peptide_compares) == 0:
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.text(0.5, 0.5, 'No peptide comparison data available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=16)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.close()
        return fig
        
    if not compare.peptide_compares[0].peptide1_list or len(compare.peptide_compares[0].peptide1_list) == 0:
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.text(0.5, 0.5, 'No peptide data available for comparison', 
                ha='center', va='center', transform=ax.transAxes, fontsize=16)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.close()
        return fig

    with plt.style.context('default'):
        font_config = {"family": "Arial", "weight": "normal", "size": 14}
        axes_config = {"titlesize": 18, "titleweight": "bold", "labelsize": 16}
        
        fig, ax = plt.subplots(figsize=(20, 10))
        
        ax.tick_params(labelsize=font_config["size"])
        ax.set_title(
            compare.state1_list[0].state_name + "-" + compare.state2_list[0].state_name,
            fontsize=axes_config["titlesize"],
            fontweight=axes_config["titleweight"],
            fontfamily=font_config["family"]
        )
        
        colormap = cm.get_cmap(colormap)

        leftbound = compare.peptide_compares[0].peptide1_list[0].start - 10
        rightbound = compare.peptide_compares[-1].peptide1_list[0].end + 10
        ax.set_xlim(leftbound, rightbound)
        ax.xaxis.set_ticks(np.arange(round(leftbound, -1), round(rightbound, -1), 10))
        ax.set_ylim(-5, 110)
        ax.grid(axis="x")
        ax.yaxis.set_ticks([])

        norm = col.Normalize(vmin=-colorbar_max, vmax=colorbar_max)

        for i, peptide_compare in enumerate(compare.peptide_compares):
            for peptide in peptide_compare.peptide1_list:
                rect = Rectangle(
                    (peptide.start, (i % 20) * 5 + ((i // 20) % 2) * 2.5),
                    peptide.end - peptide.start,
                    4,
                    fc=colormap(norm(peptide_compare.deut_diff_avg)),
                )
                ax.add_patch(rect)

        cbar = fig.colorbar(cm.ScalarMappable(cmap=colormap, norm=norm), ax=ax)
        cbar.ax.tick_params(labelsize=axes_config["labelsize"])
        cbar.set_label('Deuteration difference (%)', fontsize=18, fontfamily="Arial")
        ax.set_xlabel('Residue', fontsize=18, fontfamily="Arial")

        fig.tight_layout()
        plt.close()

        return fig


def create_heatmap_single_state(hdxms_datas, colorbar_max, colormap="Greens"):
    """Create heatmap for single state."""
    import matplotlib.colors as col
    from matplotlib.patches import Rectangle
    from matplotlib import colormaps
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import seaborn as sns

    state_name = list(
        set([state.state_name for data in hdxms_datas for state in data.states])
    )
    if len(state_name) > 1:
        raise ValueError("More than one state name found")
    else:
        state_name = state_name[0]

    with plt.style.context('default'):
        font_config = {"family": "Arial", "weight": "normal", "size": 14}
        axes_config = {"titlesize": 18, "titleweight": "bold", "labelsize": 16}
        
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        
        ax.tick_params(labelsize=font_config["size"])
        ax.set_title(
            state_name,
            fontsize=axes_config["titlesize"], 
            fontweight=axes_config["titleweight"],
            fontfamily=font_config["family"]
        )

        colormap = sns.light_palette("seagreen", as_cmap=True)

        all_peptides = [
            pep for data in hdxms_datas for state in data.states for pep in state.peptides
        ]
        all_peptides.sort(key=lambda x: x.start)

        leftbound = all_peptides[0].start - 10
        rightbound = all_peptides[-1].end + 10
        ax.set_xlim(leftbound, rightbound)
        ax.xaxis.set_ticks(np.arange(round(leftbound, -1), round(rightbound, -1), 10))
        ax.set_ylim(-5, 110)
        ax.grid(axis="x")
        ax.yaxis.set_ticks([])

        norm = col.Normalize(vmin=0, vmax=colorbar_max)

        for i, peptide in enumerate(all_peptides):
            avg_d_percent = np.average(
                [tp.d_percent for tp in peptide.timepoints if tp.deut_time != np.inf]
            )
            rect = Rectangle(
                (peptide.start, (i % 20) * 5 + ((i // 20) % 2) * 2.5),
                peptide.end - peptide.start,
                4,
                fc=colormap(norm(avg_d_percent)),
            )
            ax.add_patch(rect)

        # coverage
        coverage = np.zeros(len(hdxms_datas[0].states[0].hdxms_data.protein_sequence))
        for pep in all_peptides:
            coverage[pep.start - 1 : pep.end] += 1
        height = 3
        for i in range(len(coverage)):
            color_intensity = (
                coverage[i] / 20
            )  # coverage.max()  # Normalizing the data for color intensity
            rect = patches.Rectangle(
                (i, 105), 1, height, color=plt.cm.Blues(color_intensity)
            )
            ax.add_patch(rect)

        from matplotlib import cm
        
        cbar = fig.colorbar(cm.ScalarMappable(cmap=colormap, norm=norm), ax=ax)
        cbar.ax.tick_params(labelsize=axes_config["labelsize"])
        cbar.set_label('Deuteration (%)', fontsize=18, fontfamily="Arial")
        
        ax.set_xlabel('Residue', fontsize=18, fontfamily="Arial")

        fig.tight_layout()
        plt.close()

        return fig


def get_heatmap(hdxms_data_list, output_dir):
    """Generate heatmaps for single or multiple states."""
    all_state_names = [state.state_name for data in hdxms_data_list for state in data.states]
    
    if len(hdxms_data_list) == 1:
        fig = create_heatmap_single_state(hdxms_data_list, colorbar_max=80)
        fig.savefig(f"{output_dir}/pfnet_plots/heatmap_{all_state_names[0]}.png")
    else:
        from itertools import product
        from pigeon_feather.data import HDXStatePeptideCompares

        if len(hdxms_data_list) >= 2:
            state1_name = hdxms_data_list[0].states[0].state_name
            state2_name = hdxms_data_list[1].states[0].state_name
            
            state1_list = [hdxms_data_list[0].states[0]]
            state2_list = [hdxms_data_list[1].states[0]]

            compare = HDXStatePeptideCompares(state1_list, state2_list)
            compare.add_all_compare()
    
            if len(compare.peptide_compares) > 0:
                heatmap_compare = create_heatmap_compare(compare, 20)
                heatmap_compare.savefig(f'{output_dir}/pfnet_plots/heatmap_{state1_name}_{state2_name}.png')
            else:
                print("Warning: No valid peptide comparisons found for heatmap generation")
    
    heatmaps = glob.glob(f"{output_dir}/pfnet_plots/heatmap_*.png")
    
    return heatmaps


def logPF_to_deltaG(ana_obj, logPF):
    """Convert logP value to deltaG in kJ/mol, local unfolding energy."""
    return 8.3145 * ana_obj.temperature * np.log(10) * logPF / 1000


def create_logP_df(ana_obj, index_offset, pfnet_confidence):
    """Create DataFrame with logP and related data."""
    df_logPF = pd.DataFrame()

    for res_i, _ in enumerate(ana_obj.results_obj.protein_sequence):
        res_obj_i = ana_obj.results_obj.get_residue_by_resindex(res_i)

        avg_logP, std_logP, SE_logP = get_res_avg_logP(res_obj_i)
        log_kex = get_res_avg_log_kex(res_obj_i) * -1

        df_i = pd.DataFrame(
            {
                "resid": [res_obj_i.resid - index_offset],
                "resname": [res_obj_i.resname],
                'avg_dG (kJ/mol)': [round(logPF_to_deltaG(ana_obj, avg_logP), 3)],
                'std_dG (kJ/mol)': [round(logPF_to_deltaG(ana_obj, std_logP), 3)],
                "avg_logP (log(sec^-1))": [round(avg_logP, 3)],
                "std_logP (log(sec^-1))": [round(std_logP, 3)],
                "log_kch (log(sec^-1))": [round(res_obj_i.log_k_init, 3)],
                "log_kex (log(sec^-1))":  [round(log_kex, 3)],
                "is_nan": [res_obj_i.is_nan()],
                "coverage": [ana_obj.coverage[res_i]],
                "PFNet_confidence": [float(pfnet_confidence[res_i])]
            }
        )
        
        if res_obj_i.is_nan():
            df_i["single_resolved"] = [np.nan]
            df_i["min_pep logPs"] = [np.nan]
            df_i["min_pep log_kex"] = [np.nan]
        else:
            df_i["single_resolved"] = [res_obj_i.mini_pep.if_single_residue()]
            df_i["min_pep logPs"] = [np.round(res_obj_i.clustering_results_logP, 3)]
            df_i["min_pep log_kex"] = [-1*np.round(res_obj_i.mini_pep.clustering_results_log_kex, 3)]
            
        df_logPF = pd.concat([df_logPF, df_i])
        df_logPF['is_nan'] = df_logPF['is_nan'].astype(bool)
        df_logPF['single_resolved'] = df_logPF['single_resolved'].astype(bool)

    df_logPF = df_logPF.reset_index(drop=True)

    return df_logPF


def get_csv_results(ana_objs, output_dicts, output_dir):
    """Generate CSV results for all analysis objects."""
    csv_path_list = []
    for idx, (state_key, ana_obj) in enumerate(ana_objs.items()):
        df_logPF = create_logP_df(ana_obj, 0, output_dicts[state_key]["results"]["pfnet_pred_log_kex_confidence"])
        csv_path = f"{output_dir}/pfnet_output/results_{state_key}.csv"
        df_logPF.to_csv(csv_path, index=False)
        csv_path_list.append(csv_path)

    return csv_path_list


def make_BFactorPlot(pdb_file, ana_objs, output_dir):
    """Generate BFactor plot for PDB visualization."""
    from pfnet.plot import BFactorPlot
    
    state_keys = list(ana_objs.keys())
    
    if len(state_keys) == 1:
        first_ana_obj = list(ana_objs.values())[0]
        bfactor_plot = BFactorPlot(
            first_ana_obj,
            pdb_file=pdb_file,
            plot_deltaG=True,
            temperature=float(first_ana_obj.temperature),
        )
        dg_pdb_path = f"{output_dir}/pfnet_plots/PFNet_dG.pdb"
        bfactor_plot.plot(dg_pdb_path)
    else:
        ana_obj_list = list(ana_objs.values())
        bfactor_plot = BFactorPlot(
            ana_obj_list[0],
            ana_obj_list[1],
            pdb_file=pdb_file,
            plot_deltaG=True,
            temperature=float(ana_obj_list[0].temperature),
        )
        
        file_state_name_0 = get_display_state_name(state_keys[0], state_keys)
        file_state_name_1 = get_display_state_name(state_keys[1], state_keys)
            
        dg_pdb_path = f"{output_dir}/pfnet_plots/PFNet_ddG_{file_state_name_0}-{file_state_name_1}.pdb"
        bfactor_plot.plot(dg_pdb_path)

    return dg_pdb_path


def get_summary(output_dicts, hdxms_data_list, output_dir):
    """Generate comprehensive summary of results."""
    SUMMARY = ""
    state_names = [state.state_name for data in hdxms_data_list for state in data.states]

    for idx, state_name in enumerate(state_names):
        results = output_dicts[f"{state_name}_{idx}"]["results"]
        
        stats_text = get_all_statics_info([hdxms_data_list[idx]])
        SUMMARY += stats_text
        
        SUMMARY += "\n"*2
        
        PFNET_HEADER = "=" * 60 + "\n" + " " * 20 + f"PFNet Results Summary for {state_name} \n" + "=" * 60 + "\n"
        SUMMARY += PFNET_HEADER
        
        single_num = sum(np.array(results['resolution_limits'])[:,0] == np.array(results['resolution_limits'])[:,1])
        
        non_covered_residues = (torch.tensor(results['resolution_grouping']) == torch.tensor([1, 0, 0, 0])).all(axis=1)
        non_covered_residues_num = non_covered_residues.sum().item()
        covered_residues_num = len(results['protein_sequence']) - non_covered_residues_num

        seq_mask = results["seq_mask"]
        highest_log_kex = torch.max(results["pfnet_pred_log_kex"][~seq_mask])
        lowest_log_kex = torch.min(results["pfnet_pred_log_kex"][~seq_mask])
        std_log_kex = torch.std(results["pfnet_pred_log_kex"][~seq_mask])
        
        high_confidence_residues = results["pfnet_pred_log_kex_confidence"] > 0.8
        high_confidence_residues_num = high_confidence_residues.sum().item()
        
        text = f"Highest log(kex) (PFNet): {highest_log_kex:.2f}\nLowest log(kex) (PFNet): {lowest_log_kex:.2f}\n"
        text += f"Std log(kex) (PFNet): {std_log_kex:.2f}\n"
        SUMMARY += text + "\n"
        
        SUMMARY += f"Number of single resolved residues: {single_num}\n"
        SUMMARY += f"Number of non covered residues: {non_covered_residues_num}\n"
        SUMMARY += f"Number of high confidence residues: {high_confidence_residues_num} ({high_confidence_residues_num/covered_residues_num*100:.2f}%)\n"
        SUMMARY += f"AE_mean_pfnet: {results['AE_mean_pfnet']:.2f}\n"
        SUMMARY += f"AE_median_pfnet: {results['AE_median_pfnet']:.2f}\n"
        SUMMARY += f"centroid model: {results['centroid_model']}\n"
        
        SUMMARY += "=" * 60 + "\n"*2
        
    summary_path = f"{output_dir}/summary.txt"
    with open(summary_path, 'w') as f:
        f.write(SUMMARY)
        
    return SUMMARY, summary_path


def load_pdb_data(pdb_id, pdb_file):
    """Load PDB data from ID or file path."""
    pdb_path = None
    if pdb_id:
        try:
            pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            temp_dir = tempfile.mkdtemp()
            pdb_path = os.path.join(temp_dir, f"{pdb_id}.pdb")
            urllib.request.urlretrieve(pdb_url, pdb_path)
            print(f"Downloaded PDB from {pdb_url} to {pdb_path}")
        except Exception as e:
            print(f"Error downloading PDB ID {pdb_id}: {e}")
            traceback.print_exc()
            return None
    elif pdb_file:
        pdb_path = pdb_file
        print(f"Using PDB file: {pdb_path}")
    
    return pdb_path
