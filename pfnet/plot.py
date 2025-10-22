import os
import re
import math
import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
from matplotlib.ticker import LogLocator, MultipleLocator, FormatStrFormatter
from pigeon_feather.plot import UptakePlot
from pigeon_feather.analysis import get_res_avg_logP, get_index_offset
import numpy as np
import MDAnalysis
import warnings

def pfnet_to_hdxmsdata(
    start_pos,
    end_pos,
    sequences,
    time_points,
    target_data,
    pred_data,
    back_ex,
    pred_state_name="PFNet",
    centroid_data=False,
    saturation=1.0,
):
    """Convert PFNet output to HDXMSData object.

    Args:
        start_pos: Start positions of peptides
        end_pos: End positions of peptides
        sequences: Peptide sequences
        time_points: Time points
        target_data: Target data (envelope or centroid)
        pred_data: Predicted data (envelope or centroid)
        back_ex: Back exchange values
        pred_state_name: Name for predicted state
        centroid_data: Whether input data is centroid (True) or envelope (False)
        saturation: HDX saturation level (default 1.0)
    """
    from pigeon_feather.data import HDXMSData, Peptide, ProteinState, Timepoint
    from pigeon_feather.tools import _add_max_d_to_pep

    batch_size = start_pos.shape[0]
    hdxms_data_list = []

    # Calculate predicted envelope and centroids if not centroid data
    if not centroid_data:
        # Calculate centroids from envelopes
        mass_num = torch.arange(target_data.shape[-1], device=target_data.device)
        centroid_calculated = (pred_data * mass_num).sum(dim=-1)
        centroid_target = (target_data * mass_num).sum(dim=-1)
    else:
        centroid_target = target_data
        centroid_calculated = pred_data

    for batch_idx in range(batch_size):
        # Get batch data
        start_pos_batch = start_pos[batch_idx].detach().cpu().numpy()
        end_pos_batch = end_pos[batch_idx].detach().cpu().numpy()
        sequences_batch = sequences[batch_idx]
        time_points_batch = time_points[batch_idx].detach().cpu().numpy()
        target_centroid_batch = centroid_target[batch_idx].detach().cpu().numpy()
        pred_centroid_batch = centroid_calculated[batch_idx].detach().cpu().numpy()
        back_ex_batch = back_ex[batch_idx].detach().cpu().numpy()

        if not centroid_data:
            target_envelope_batch = target_data[batch_idx].detach().cpu().numpy()
            pred_envelope_batch = pred_data[batch_idx].detach().cpu().numpy()

        hdxms_data = HDXMSData(
            protein_name="test",
            n_fastamides=0,
            protein_sequence="",
            saturation=saturation,
        )

        # Iterate over rows in the dataframe
        for tp_index in range(start_pos_batch.shape[0]):
            # Skip padding
            if start_pos_batch[tp_index] == -100:
                continue

            # Get or create protein states
            protein_state_target = None
            protein_state_pred = None
            for state in hdxms_data.states:
                if state.state_name == "Expt.":
                    protein_state_target = state
                elif state.state_name == pred_state_name:
                    protein_state_pred = state

            if not protein_state_target:
                protein_state_target = ProteinState("Expt.", hdxms_data=hdxms_data)
                hdxms_data.add_state(protein_state_target)
            if not protein_state_pred:
                protein_state_pred = ProteinState(pred_state_name, hdxms_data=hdxms_data)
                hdxms_data.add_state(protein_state_pred)

            # Get or create peptides
            peptide_target = None
            peptide_pred = None
            identifier = f"{start_pos_batch[tp_index]}-{end_pos_batch[tp_index]} {sequences_batch[tp_index]}"

            for pep in protein_state_target.peptides:
                if pep.identifier == identifier:
                    peptide_target = pep
                    break
            for pep in protein_state_pred.peptides:
                if pep.identifier == identifier:
                    peptide_pred = pep
                    break

            if not peptide_target:
                peptide_target = Peptide(
                    sequences_batch[tp_index],
                    start_pos_batch[tp_index],
                    end_pos_batch[tp_index],
                    protein_state_target,
                    n_fastamides=0,
                    RT=0,
                )
                protein_state_target.add_peptide(peptide_target)

            if not peptide_pred:
                peptide_pred = Peptide(
                    sequences_batch[tp_index],
                    start_pos_batch[tp_index],
                    end_pos_batch[tp_index],
                    protein_state_pred,
                    n_fastamides=0,
                    RT=0,
                )
                protein_state_pred.add_peptide(peptide_pred)

            # Add timepoint data
            timepoint_target = Timepoint(
                peptide_target,
                time_points_batch[tp_index],
                target_centroid_batch[tp_index],
                0,
                0,
            )
            timepoint_pred = Timepoint(
                peptide_pred,
                time_points_batch[tp_index],
                pred_centroid_batch[tp_index],
                0,
                0,
            )

            # Add envelope data if available
            if not centroid_data:
                timepoint_target.isotope_envelope = target_envelope_batch[tp_index]
                timepoint_pred.isotope_envelope = pred_envelope_batch[tp_index]

            peptide_target.add_timepoint(timepoint_target, allow_duplicate=True)
            peptide_pred.add_timepoint(timepoint_pred, allow_duplicate=True)

            # Add back exchange
            _add_max_d_to_pep(peptide_target, max_d=(1 - back_ex_batch[tp_index]) * peptide_target.theo_max_d, force=True)
            _add_max_d_to_pep(peptide_pred, max_d=(1 - back_ex_batch[tp_index]) * peptide_pred.theo_max_d, force=True)

        hdxms_data_list.append(hdxms_data)

    return hdxms_data_list


def plot_uptake_curves(hdxms_data_list, protein_name, outdir, time_window=None, centroid_model=False):
    # Get all peptides and identifiers
    all_peps = [peptide for hdxms_data in hdxms_data_list for state in hdxms_data.states for peptide in state.peptides]
    all_idfs = [pep for pep in all_peps if pep.unique_num_timepoints > 5]
    all_idfs = list(set([peptide.identifier for peptide in all_peps]))[:]
    all_idfs.sort(key=lambda x: tuple(map(int, re.findall(r"\d+-\d+", x)[0].split("-"))))

    # grab min and max timepoints
    all_tps = [tp.deut_time for pep in all_peps for tp in pep.timepoints if (tp.deut_time != 0) and (tp.deut_time != np.inf)]
    min_tp = np.log10(min(all_tps)) - 1
    max_tp = np.log10(max(all_tps)) + 1

    # Create output directory if needed
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    def idf_to_pep(idf):
        return [pep for pep in all_peps if pep.identifier == idf][0]

    # Plot settings
    num_subplots_per_figure = 300
    num_figures = math.ceil(len(all_idfs) / num_subplots_per_figure)

    all_idfs_subset = all_idfs[:]

    for fig_index in range(num_figures):
        # Select subset of identifiers for current figure
        selected_idf = all_idfs_subset[fig_index * num_subplots_per_figure : (fig_index + 1) * num_subplots_per_figure]
        num_col = math.ceil(len(selected_idf) / 5)

        fig, axs = plt.subplots(num_col, 5, figsize=(9 * 5, 8 * num_col))

        for i, idf in enumerate(selected_idf):
            if num_col == 1:
                ax = axs[i]
            else:
                ax = axs[i // 5, i % 5]

            pep = idf_to_pep(idf)
            try:

                ax.axhline(y=pep.max_d, color="lightgray", linestyle="--", linewidth=5)

                uptake = UptakePlot(
                    hdxms_data_list,
                    idf,
                    # states_subset=states_subset,
                    if_plot_fit=False,
                    figure=fig,
                    ax=ax,
                    calculate_num_d_from_envelope=False if centroid_model else True,
                )

                if time_window is not None:
                    ax.set_xlim(time_window[0], time_window[1])
                else:
                    ax.set_xlim(10**min_tp, 10**max_tp)

                ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
                
                if len(pep.sequence) < 20:
                    ax.set_title(pep.identifier)
                else:
                    idf = pep.identifier.split(" ")[0] + " " + pep.identifier.split(" ")[1][:9] + "..." + pep.identifier.split(" ")[1][-9:]
                    ax.set_title(idf)

                y_max = pep.theo_max_d / 0.9
                ax.set_ylim(-0.5, y_max + 0.5)

                exclude_states = ["state", "data_set_index", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
                handles, labels = ax.get_legend_handles_labels()
                new_labels = [label for label in labels if label not in exclude_states]
                new_handles = [handle for handle, label in zip(handles, labels) if label not in exclude_states]

                ax.legend(new_handles, new_labels, title="", title_fontsize="small", loc="best")

            except ValueError as e:
                print(e)
                pass

        # Layout adjustment and save
        fig.tight_layout()
        fig.savefig(f"{outdir}/{protein_name}_uptake_{fig_index}.pdf")



class BFactorPlot(object):
    def __init__(
        self,
        analysis_object_1,
        analysis_object_2=None,
        pdb_file=None,
        plot_deltaG=False,
        temperature=None,
        logP_threshold=3.0,
    ):
        """
        _summary_

        :param analysis_object_1: an instance of Analysis and data should be loaded
        :param analysis_object_2: an instance of Analysis and data should be loaded
        """
        self.analysis_object_1 = analysis_object_1
        self.analysis_object_2 = analysis_object_2

        if self.analysis_object_2 is not None:
            if analysis_object_1.protein_sequence != analysis_object_2.protein_sequence:
                warnings.warn(
                    "The protein sequences of the two analysis objects are different"
                )

        if pdb_file is None:
            raise ValueError("pdb_file is required")

        self.pdb_file = pdb_file
        self.index_offset = get_index_offset(analysis_object_1, pdb_file)

        self.plot_deltaG = plot_deltaG

        if plot_deltaG:
            if temperature is None:
                raise ValueError("temperature is required to convert logP to deltaG")
            self.temperature = temperature
        
        self.logP_threshold = logP_threshold

    def plot(self, out_file):
        """
        if analysis_object_2 is None, plot the logP values of the first analysis object
        if analysis_object_2 is not None, plot the difference of logP values between the two analysis objects

        :param out_file: output pdb file with B-factors set to logP values
        """

        if self.analysis_object_2 is None:
            self._plot_single_state(out_file)
        else:
            self._plot_two_states_diff(out_file)

    def _plot_single_state(self, out_file):
        u = MDAnalysis.Universe(self.pdb_file)
        u.atoms.tempfactors = 0.0

        for res_i, _ in enumerate(self.analysis_object_1.results_obj.protein_sequence):
            res = self.analysis_object_1.results_obj.get_residue_by_resindex(res_i)
            avg_logP, std_logP, avg_ = get_res_avg_logP(res)
            
            if res.if_off_log_kex_range_by_time_window:
                avg_logP = res._capped_log_P

            if std_logP > self.logP_threshold:
                avg_logP = np.nan
                continue
                # print(res.resid, res.resname ,avg_logP, std_logP, res.clustering_results_logP, res.std_within_clusters_logP)

            protein_res = u.select_atoms(
                f"protein and resid {res.resid - self.index_offset}"
            )
            if self.plot_deltaG:
                avg_deltaG = self._logPF_to_deltaG(avg_logP)
                protein_res.atoms.tempfactors = avg_deltaG

            else:
                protein_res.atoms.tempfactors = avg_logP
                
            # add confidence
            protein_res.atoms.occupancies = np.clip(res.pfnet_confidence, 0, 1)
        
        PROs = u.select_atoms(f"resname PRO")
        PROs.atoms.tempfactors = np.nan

        u.atoms.write(out_file)

    def _plot_two_states_diff(self, out_file):
        u = MDAnalysis.Universe(self.pdb_file)
        u.atoms.tempfactors = 0.0

        for res_i, _ in enumerate(self.analysis_object_1.results_obj.protein_sequence):
            res_obj_1 = self.analysis_object_1.results_obj.get_residue_by_resindex(
                res_i
            )
            res_obj_2 = self.analysis_object_2.results_obj.get_residue_by_resindex(
                res_i
            )

            avg_logP_1, std_logP_1, SE_logP_1 = get_res_avg_logP(res_obj_1)
            avg_logP_2, std_logP_2, SE_logP_2 = get_res_avg_logP(res_obj_2)
            
            
            # if res_obj_1.if_off_log_kex_range_by_time_window:
            #     avg_logP_1 = res_obj_1._capped_log_P

            # if res_obj_2.if_off_log_kex_range_by_time_window:
            #     avg_logP_2 = res_obj_2._capped_log_P
            diff_logP = avg_logP_1 - avg_logP_2
            combined_SE = np.sqrt(SE_logP_1 ** 2 + SE_logP_2 ** 2)

            # if not siginificant difference, 0.35 is the grid size, white
            if abs(diff_logP) < max(combined_SE, 0.35) and combined_SE <= self.logP_threshold:
                diff_logP = 0.0  
            # too noisy → gray
            elif combined_SE > self.logP_threshold:
                diff_logP = np.nan  

            protein_res = u.select_atoms(
                f"protein and resid {res_obj_1.resid - self.index_offset}"
            )

            if self.plot_deltaG:
                diff_deltaG = self._logPF_to_deltaG(diff_logP)
                protein_res.atoms.tempfactors = diff_deltaG

            else:
                protein_res.atoms.tempfactors = diff_logP

            # add confidence
            protein_res.atoms.occupancies = (np.clip(res_obj_1.pfnet_confidence, 0, 1)+np.clip(res_obj_2.pfnet_confidence, 0, 1))/2

        PROs = u.select_atoms(f"resname PRO")
        PROs.atoms.tempfactors = np.nan

        u.atoms.write(out_file)

    def _logPF_to_deltaG(self, logPF):
        """
        :param logPF: logP value
        :return: deltaG in kJ/mol, local unfolding energy
        """

        return 8.3145 * self.temperature * np.log(10) * logPF / 1000
    
    
import MDAnalysis

def get_index_offset(
    ana_obj,
    pdb_file,
):
    from pigeon_feather.tools import pdb2seq, find_peptide

    u = MDAnalysis.Universe(pdb_file)
    protein = u.select_atoms("protein")
    first_protein_res_index = protein.resids[0] - 1

    pdb_sequence = pdb2seq(pdb_file)
    seq = ana_obj.results_obj.protein_sequence

    index_offset = None
    win = 10 if len(seq) >= 10 else len(seq)

    for i in range(1, 9):
        _pos = int(len(seq) * i / 10)
        i_start = max(0, min(_pos - win // 2, len(seq) - win))
        pep = seq[i_start:i_start + win]
        if len(pep) < 6:
            continue
        try:
            pdb_start, pdb_end = find_peptide(pdb_sequence, pep)
            if pdb_start < 0:
                continue
            index_offset = i_start - (pdb_start + first_protein_res_index)
            break
        except Exception:
            continue

    if index_offset is None:
        raise ValueError("Failed to determine index_offset: peptide not found in PDB sequence")

    # check if the offset is correct
    for residue in protein.residues[20:30]:
        pdb_resname = MDAnalysis.lib.util.convert_aa_code(residue.resname)
        res = ana_obj.results_obj.get_residue_by_resid(residue.resid + index_offset).resname

        if pdb_resname != res:
            raise ValueError("HXMS data sequence and PDB sequence don't match!")

    return index_offset