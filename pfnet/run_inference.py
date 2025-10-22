import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pyopenms")
import torch
from pfnet.model import PFNet, PFNetCentroid
from pfnet.data import HDXDataset, custom_collate_fn
from torch.utils.data import DataLoader
from pfnet.utilts import get_calculated_isotope_envelope, get_calculated_num_d
from pfnet.bayesian import FeatherEmpiricalPyroModel, FeatherMHKernel
from pfnet.data import estimate_deut_rent
from pyro.infer.mcmc import MCMC
from pigeon_feather.analysis import Analysis, Results, MiniPep
from pigeon_feather.analysis import remove_outliers
from pigeon_feather.hxio import load_HXMS_file
from pfnet.hxio import datadict_to_hdxmsdata
from sklearn.cluster import KMeans
from copy import deepcopy
import numpy as np
import json
from pfnet.hxio import hxms_data_to_grouped_dict
import math
from copy import deepcopy
import os
import warnings


def predict(
    input,
    output_json=None,
    loaded_model=None,
    centroid_model=False,
    refinement=False,
    refine_steps=200,
    refine_cen_sigma=0.5,
    refine_env_sigma=0.3,
    refine_single_pos_conf_threshold=0.8,
    refine_non_single_pos_conf_threshold=0.9,
    uptake_plots=False,
    plots_dir=None,
    benchmark=False,
    device="cpu",
):
    torch.set_grad_enabled(False)
    model_type = "centroid" if centroid_model else "envelope"
    
    results_dict = {"results": {}, "analysis_objs": {}, "hdxms_data_objs": {}}

    # load model and data
    pfnet_model = _load_model(loaded_model, centroid_model)
    dataset, batch, hdxms_data = _load_data(input, centroid_model)
    results_dict["hdxms_data_objs"]["hdxms_data_expt"] = hdxms_data

    # run prediction
    pred_log_kex, pred_confidence, _log_kex, pred_exp, obs_exp, seq_mask, _ = forward_chunked_batches(pfnet_model, batch, centroid_model=centroid_model)

    results_dict["results"]["pfnet_pred_log_kex"] = pred_log_kex[0]
    results_dict["results"]["pfnet_pred_log_kex_confidence"] = pred_confidence[0]
    results_dict["results"]["seq_mask"] = seq_mask[0]
    results_dict["results"]["_benchmark_log_kex"] = _log_kex[0]

    # run analysis
    _log_kex_posterior_samples = pred_log_kex.repeat(1, 1).detach().cpu().numpy() * -1
    _log_kex_posterior_samples[:, seq_mask[0]] = np.nan
    analysis_pfnet, posterior_mean_pfnet, posterior_std_pfnet = _create_analysis_obj(pfnet_model, hdxms_data, _log_kex_posterior_samples, pred_confidence[0], batch)
    results_dict["analysis_objs"]["analysis_pfnet"] = analysis_pfnet
    # add pAE to analysis_pfnet
    pAE = np.log(results_dict["results"]["pfnet_pred_log_kex_confidence"])/-0.2
    pSTD = pAE * np.sqrt(np.pi/2)
    
    for mini_pep in analysis_pfnet.results_obj.mini_peps:
        mini_pep.std_within_clusters_log_kex = np.array([])  # reset to empty array
    
    for res_i in range(len(analysis_pfnet.results_obj.protein_sequence)):
        res_obj = analysis_pfnet.results_obj.get_residue_by_resindex(res_i)
        if not res_obj.is_nan():
            res_obj.mini_pep.std_within_clusters_log_kex = np.append(
                res_obj.mini_pep.std_within_clusters_log_kex, 
                pSTD[res_i]
            )
        
    # get fit error
    pred_hdxms_data_pfnet, ae_pfnet = _get_AE(batch, dataset, posterior_mean_pfnet, centroid_model, seq_mask)
    pred_hdxms_data_pfnet.protein_name = "PFNet"
    pred_hdxms_data_pfnet.states[0].state_name = "PFNet"  
    results_dict["hdxms_data_objs"]["hdxms_data_pfnet"] = pred_hdxms_data_pfnet
    fit_error = {}        
    fit_error["ae_pfnet"] = ae_pfnet

    if refinement:
        #raise ValueError("Refinement is not recommended!")
        # run refinement
        single_pos = pfnet_model.get_single_pos(batch)[0]
        sampling_mask = ((single_pos) & (pred_confidence[0] > refine_single_pos_conf_threshold)) | ((~single_pos) & (pred_confidence[0] > refine_non_single_pos_conf_threshold))
        
        posterior_samples, posterior_mean_refined, posterior_std_refined = run_empirical_bayesian(
            refine_steps, refine_cen_sigma, refine_env_sigma, pred_log_kex, batch, seq_mask, centroid_model, device="cpu",
            sampling_mask=sampling_mask
        )

        # run analysis
        _log_kex_posterior_samples = np.array(posterior_samples["log_kex"]) * -1
        _log_kex_posterior_samples[:, seq_mask[0]] = np.nan
        analysis_pfnet_refined, posterior_mean_refined, posterior_std_refined = _create_analysis_obj(pfnet_model, hdxms_data, _log_kex_posterior_samples, pred_confidence[0], batch)

        results_dict["results"]["pfnet_posterior_log_kex_samples"] = _log_kex_posterior_samples
        results_dict["results"]["pfnet_posterior_log_kex_mean"] = posterior_mean_refined[0]
        results_dict["results"]["pfnet_posterior_log_kex_std"] = posterior_std_refined[0]
        results_dict["results"]["sampling_mask"] = sampling_mask
        results_dict["analysis_objs"]["analysis_pfnet_refined"] = analysis_pfnet_refined

        # get fit error
        pred_hdxms_data_refined, ae_pfnet_refined = _get_AE(batch, dataset, posterior_mean_refined, centroid_model, seq_mask)
        pred_hdxms_data_refined.protein_name = "PFNet+PF"
        pred_hdxms_data_refined.states[0].state_name = "PFNet+PF"
        results_dict["hdxms_data_objs"]["hdxms_data_pfnet_refined"] = pred_hdxms_data_refined
        fit_error["ae_pfnet_refined"] = ae_pfnet_refined

    # add results to results_dict
    results_dict["results"]["protein_sequence"] = dataset[0]["protein_sequence"]
    results_dict["results"]["temperature"] = np.array(dataset[0]["temperature"], dtype=np.float64).round(decimals=3).tolist()
    results_dict["results"]["pH"] = np.array(dataset[0]["pH"], dtype=np.float64).round(decimals=3).tolist()
    results_dict["results"]["saturation"] = np.array(dataset[0]["saturation"], dtype=np.float64).round(decimals=3).tolist()
    results_dict["results"]["resolution_grouping"] = batch[1]["resolution_grouping"][0].detach().numpy().tolist()
    results_dict["results"]["resolution_limits"] = batch[1]["resolution_limits"][0].detach().numpy().tolist()
    results_dict["results"]["AE_mean_pfnet"] = float(f"{np.mean(ae_pfnet[model_type]):.3f}")
    results_dict["results"]["AE_median_pfnet"] = float(f"{np.median(ae_pfnet[model_type]):.3f}")
    results_dict["AE_pfnet"] = ae_pfnet
    
    if refinement:
        results_dict["results"]["AE_mean_pfnet_refined"] = float(f"{np.mean(ae_pfnet_refined[model_type]):.3f}")
        results_dict["results"]["AE_median_pfnet_refined"] = float(f"{np.median(ae_pfnet_refined[model_type]):.3f}")
    if benchmark:
        # get fit error
        pred_hdxms_data_benchmark, ae_pfnet_benchmark = _get_AE(batch, dataset, _log_kex, centroid_model, seq_mask)
        pred_hdxms_data_benchmark.protein_name = "PF"
        pred_hdxms_data_benchmark.states[0].state_name = "PF" 
        results_dict["hdxms_data_objs"]["hdxms_data_pfnet_benchmark"] = pred_hdxms_data_benchmark
        fit_error["ae_pfnet_benchmark"] = ae_pfnet_benchmark
        # log the fit error
        results_dict["results"]["AE_mean_benchmark"] = float(f"{np.mean(ae_pfnet_benchmark[model_type]):.3f}")
        results_dict["results"]["AE_median_benchmark"] = float(f"{np.median(ae_pfnet_benchmark[model_type]):.3f}")
    
    results_dict["results"]["centroid_model"] = centroid_model
    
    
    json_dict = {}
    for key, value in results_dict["results"].items():
        if isinstance(value, np.ndarray):
            numpy_value = np.nan_to_num(np.array(value, dtype=np.float64), nan=-100).round(decimals=3).tolist()
            json_dict[key] = numpy_value
        elif isinstance(value, torch.Tensor):
            numpy_value = value.detach().cpu().numpy()
            json_dict[key] = np.array(numpy_value, dtype=np.float64).round(decimals=3).tolist()
        else:
            json_dict[key] = value

    if output_json is not None:
        with open(output_json, "w") as f:
            json.dump(json_dict, f, indent=2, separators=(",", ": "), sort_keys=False, ensure_ascii=False)
            
    if uptake_plots:
        print("plotting uptake plots")
        # generate plots
        hdxms_data_list = [v for k, v in results_dict["hdxms_data_objs"].items()]
        _generate_plots(hdxms_data_list, plots_dir, fit_error, centroid_model)

    return results_dict


def _load_model(loaded_model, centroid_model):
    if loaded_model is not None:
        return loaded_model

    if centroid_model:
        print("loading PFNetCentroid model")
        checkpoint_path = f"{os.path.dirname(__file__)}/../model/PFNetCentroid.ckpt"
        model = PFNetCentroid.load_from_checkpoint(checkpoint_path)
    else:
        print("loading PFNet model")
        checkpoint_path = f"{os.path.dirname(__file__)}/../model/PFNet.ckpt"
        model = PFNet.load_from_checkpoint(checkpoint_path)

    model.eval()
    model.to(device="cpu")
    return model


def _load_data(input, centroid_model):
    if isinstance(input, dict):
        dataset = [input]
    elif input.endswith(".hxms"):
        hxms_data = load_HXMS_file(input, n_fastamides=1)
        all_peptides = [peptide for state in hxms_data.states for peptide in state.peptides]
        
        skip_peptides = []
        for peptide in all_peptides:
            if len(peptide.identifier.split(" ")[1]) > 50:
                skip_peptides.append(peptide.identifier)
                continue
            try:
                backex_tp, deut_rent = estimate_deut_rent(peptide, if_true_max_d=False)
                peptide.deut_rent = deut_rent   
            except Exception as e:
                print("Error in estimate_deut_rent:", peptide.identifier)
                print("max_d:", peptide.max_d, "theo_max_d:", peptide.theo_max_d)
                skip_peptides.append(peptide.identifier)
        print(f"Skipping {len(skip_peptides)} peptides")
        hxms_data.states[0].peptides = [pep for pep in all_peptides if pep.identifier not in skip_peptides]
        
        dataset = HDXDataset(hxms_data_to_grouped_dict(hxms_data), centroid_data=centroid_model)
    elif input.endswith(".pt"):
        dataset = HDXDataset(input, centroid_data=centroid_model)
    else:
        raise ValueError("Invalid input type")
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    batch = next(iter(data_loader))
    hdxms_data = datadict_to_hdxmsdata(dataset[0], protein_name="Expt.", centroid_data=centroid_model)

    return dataset, batch, hdxms_data


def _create_analysis_obj(pfnet_model, hdxms_data, pred_log_kex_samples, pfnet_confidence, batch):
    analysis_pfnet = PFNetAnalysis(hdxms_data.states, hdxms_data.temperature, hdxms_data.pH)
    analysis_pfnet.feather_samples = pred_log_kex_samples
    analysis_pfnet.clustering_results()

    posterior_mean = np.zeros(analysis_pfnet.results_obj.n_residues)
    posterior_std = np.zeros(analysis_pfnet.results_obj.n_residues)
    for mini_pep in analysis_pfnet.results_obj.mini_peps:
        # print(mini_pep.start, mini_pep.end, mini_pep.clustering_results_log_kex)
        posterior_mean[mini_pep.start - 1 : mini_pep.end] = mini_pep.clustering_results_log_kex
        posterior_std[mini_pep.start - 1 : mini_pep.end] = mini_pep.std_within_clusters_log_kex

    seq_mask = np.logical_or(np.isinf(posterior_mean), posterior_mean == 0.0)
    posterior_mean = posterior_mean * -1
    posterior_mean[seq_mask] = -100
    posterior_std[seq_mask] = -100
    
    # add confidence to analysis_pfnet
    for res_idx, res_obj in enumerate(analysis_pfnet.results_obj.residues):
        res_obj.pfnet_confidence = pfnet_confidence[res_idx]

    # __empty_tensor = torch.zeros(1, analysis_pfnet.results_obj.n_residues)
    # _, _, posterior_mean, _ = pfnet_model._sort_log_kex(__empty_tensor, __empty_tensor, torch.tensor(posterior_mean).unsqueeze(0), __empty_tensor, batch[1]["resolution_limits"])
    # _, _, posterior_std, _ = pfnet_model._sort_log_kex(__empty_tensor, __empty_tensor, torch.tensor(posterior_std).unsqueeze(0), __empty_tensor, batch[1]["resolution_limits"])

    posterior_mean = torch.tensor(posterior_mean).unsqueeze(0)
    posterior_std = torch.tensor(posterior_std).unsqueeze(0)

    return analysis_pfnet, posterior_mean, posterior_std


def _generate_plots(hdxms_data_list, outdir, fit_error, centroid_model):
    from pfnet.plot import plot_uptake_curves

    if outdir is None:
        outdir = os.path.join(os.getcwd(), "pfnet_plots")
    os.makedirs(outdir, exist_ok=True)
    plot_uptake_curves(hdxms_data_list, protein_name="PFNet", outdir=outdir, centroid_model=centroid_model)

    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plot_models = ["centroid", "envelope"] if not centroid_model else ["centroid"]

    # plt.style.use("ggplot")
    colors = ["#43a2ca", "#2ca25f", "#756bb1", "#f768a1"]
    
    for plot_model in plot_models:

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(fit_error["ae_pfnet"][plot_model], bins=50, label="PFNet", ax=ax, alpha=0.5, binwidth=0.1, color=colors[0])
        if "ae_pfnet_benchmark" in fit_error:
            sns.histplot(fit_error["ae_pfnet_benchmark"][plot_model], bins=50, label="PF", ax=ax, alpha=0.5, binwidth=0.1, color=colors[1])
        if "ae_pfnet_refined" in fit_error:
            sns.histplot(fit_error["ae_pfnet_refined"][plot_model], bins=50, label="PFNet+PF", ax=ax, alpha=0.5, binwidth=0.1, color=colors[2])
        
        # annotate the mean and median
        ax.text(
            0.7, 0.95,
            f"Median: {np.median(fit_error['ae_pfnet'][plot_model]):.3f}",
            ha="left", va="top", transform=ax.transAxes, fontsize=10
        )
        ax.text(
            0.7, 0.88,
            f"Mean: {np.mean(fit_error['ae_pfnet'][plot_model]):.3f}",
            ha="left", va="top", transform=ax.transAxes, fontsize=10
        )
        
        ax.legend(fontsize=12)
        ax.set_xlabel(f"AE ({plot_model})", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.tick_params(labelsize=12)
        fig.tight_layout()
        plt.close()
        fig.savefig(os.path.join(outdir, f"ae_histogram_{plot_model}.png"), dpi=300)


def _get_AE(batch, dataset, pred_log_kex, centroid_model, seq_mask):
    
    peptide_data, residue_data, global_vars, log_kex = batch
    start_pos = peptide_data["start_pos"]
    end_pos = peptide_data["end_pos"]
    time_points = peptide_data["time_point"]
    #back_ex = peptide_data["back_ex"].unsqueeze(-1)
    back_ex = 1-peptide_data["effective_deut_rent"]
    saturation = global_vars[:, -1]
    t0_mask = peptide_data["time_point"] == 0
    
    pred_log_kex_mask_inf = pred_log_kex.clone()
    pred_log_kex_mask_inf[seq_mask] = -float("inf")
    
    ae_pfnet = {}
    if not centroid_model:
        obs_envelope = peptide_data["probabilities"]
        obs_envelope_t0 = peptide_data["t0_isotope"]

        pred_envelope = get_calculated_isotope_envelope(
            start_pos, end_pos, time_points, obs_envelope_t0, obs_envelope, pred_log_kex_mask_inf, back_ex, saturation
        )

        pred_data_dict = {k: v for k, v in dataset[0].items() if k != "isotope_envelope"}
        pred_data_dict["isotope_envelope"] = pred_envelope[0]
        pred_hdxms_data = datadict_to_hdxmsdata(pred_data_dict, centroid_data=centroid_model)
        ae_pfnet_envelope = (pred_envelope[~t0_mask] - obs_envelope[~t0_mask]).abs().sum(-1).flatten().detach().cpu().numpy()
        
        ae_pfnet_centroid = (
            pred_envelope[~t0_mask] @ torch.arange(50, dtype=pred_envelope.dtype, device=pred_envelope.device)
            - obs_envelope[~t0_mask] @ torch.arange(50, dtype=obs_envelope.dtype, device=obs_envelope.device)
        ).abs().flatten().detach().cpu().numpy()
        
        ae_pfnet["envelope"] = ae_pfnet_envelope
        ae_pfnet["centroid"] = ae_pfnet_centroid
    else:
        obs_num_d = peptide_data["num_d"]
        
        pred_num_d = get_calculated_num_d(start_pos, end_pos, time_points, pred_log_kex_mask_inf, back_ex, saturation)
        pred_data_dict = {k: v for k, v in dataset[0].items() if k != "num_d"}
        pred_data_dict["num_d"] = pred_num_d[0]
        pred_hdxms_data = datadict_to_hdxmsdata(pred_data_dict, centroid_data=centroid_model)
        ae_pfnet_centroid = (pred_num_d[~t0_mask] - obs_num_d[~t0_mask]).abs().flatten().detach().cpu().numpy()
        ae_pfnet["centroid"] = ae_pfnet_centroid

    return pred_hdxms_data, ae_pfnet



def run_empirical_bayesian(refine_steps, refine_cen_sigma, refine_env_sigma, pfnet_pred, batch, seq_mask, centroid_model=False,sampling_mask=None, device="cpu"):

    feather_input_batch = deepcopy(batch)
    with torch.inference_mode():
        pfnet_pred = pfnet_pred[0].detach().to(device).clone()

        num_residues = int(batch[2][0][0])
        feather_model = FeatherEmpiricalPyroModel(
            num_residues,
            pfnet_pred,
            cen_sigma=refine_cen_sigma,
            env_sigma=refine_env_sigma,
            device=device,
            #centroid_model=centroid_model,
            centroid_model=True,
            mask=seq_mask,
        )

        feather_model._current_batch = feather_input_batch


        kernel = FeatherMHKernel(
            potential_fn=feather_model.potential_fn,
            num_residues=num_residues,
            pct_moves=25,
            temperature=0.1,
            grid_size=200,
            swap_stage=True,
            adjacency=10,
            target_accept_prob=0.2,
            adaptation_rate=0.85,
            device=device,
            grid_range=(-15.0, max(feather_model.get_log_kch())),
            feather_model=feather_model,
        )
        
        # single_pos = (feather_input_batch[1]['resolution_grouping'][:, :, 2:4] == 0).all(dim=-1)[0]

        kernel.initialize({"log_kex": pfnet_pred.detach().to(device).clone(),
                           "mask": seq_mask,
                           "log_kch": feather_model.get_log_kch(),
                           "sampling_mask": sampling_mask})

        mcmc = MCMC(
            kernel,
            num_samples=refine_steps,
            warmup_steps=50,
        )

        mcmc.run(feather_input_batch)

        posterior_samples = mcmc.get_samples()
        posterior_mean = np.mean(posterior_samples["log_kex"].detach().cpu().numpy(), axis=0)
        posterior_std = np.std(posterior_samples["log_kex"].detach().cpu().numpy(), axis=0)

        return posterior_samples, posterior_mean, posterior_std


def split_batch(batch, chunk_size, chunk_padding):
    """
    split batch into small chunks with 300 residues, based on start_pos and end_pos
    """

    peptide_data, residue_data, global_vars, log_kex = batch
    start_pos = peptide_data["start_pos"]  # Shape: (num_peptide_time_points,)
    end_pos = peptide_data["end_pos"]  # Shape: (num_peptide_time_points,)

    # min_pos, max_pos = int(min(start_pos.flatten())), int(max(end_pos.flatten()))
    min_pos, max_pos = 1, int(global_vars[0][0])
    num_chunks = math.ceil((max_pos - min_pos) / chunk_size)
    chunks = []
    for i in range(num_chunks):
        start = max(1, i * chunk_size + 1)
        end = min(max_pos, (i + 1) * chunk_size)
        chunks.append([start, end])

    chunked_batches = []
    for chunk_idx, chunk in enumerate(chunks):

        tp_idx = (start_pos >= chunk[0] - chunk_padding) & (end_pos <= chunk[1] + chunk_padding)
        peptide_data_chunk = {k: v[tp_idx].unsqueeze(0) for k, v in peptide_data.items()}

        res_start_idx = int(max(0, chunk[0] - chunk_padding - 1))
        res_end_idx = int(min(global_vars[0][0], chunk[1] + chunk_padding))

        residue_data_chunk = {
            "resolution_grouping": residue_data["resolution_grouping"][:, res_start_idx:res_end_idx],
            "proline_mask": residue_data["proline_mask"][:, res_start_idx:res_end_idx],
            "log_kch": residue_data['log_kch'][:, res_start_idx:res_end_idx],
            "resolution_limits": residue_data["resolution_limits"][:, res_start_idx:res_end_idx],
        }

        log_kex_chunk = log_kex[:, res_start_idx:res_end_idx]

        global_vars_chunk = torch.tensor([log_kex_chunk.shape[1], global_vars[0][1]]).unsqueeze(0)

        if chunk_idx != 0:
            peptide_data_chunk["start_pos"] = peptide_data_chunk["start_pos"] - (chunk[0] - chunk_padding) + 1
            peptide_data_chunk["end_pos"] = peptide_data_chunk["end_pos"] - (chunk[0] - chunk_padding) + 1
            
        if not tp_idx.any():
            chunked_batches.append([[], residue_data_chunk, global_vars_chunk, log_kex_chunk])
        else:
            chunked_batches.append([peptide_data_chunk, residue_data_chunk, global_vars_chunk, log_kex_chunk])

    return chunks, chunked_batches


def forward_chunked_batches(model, x, chunk_size=250, chunk_padding=50, centroid_model=False, if_sort_log_kex=False):

    batch = [deepcopy(i) for i in x]
    chunks, chunked_batches = split_batch(batch, chunk_size=chunk_size, chunk_padding=chunk_padding)
    protein_length = int(batch[2][0][0])

    pred_log_kex = []
    pred_log_kex_confidence = []

    for chunk_idx, chunk in enumerate(chunks):
        
        if chunked_batches[chunk_idx][0] != []:
            #pred_log_kex_chunk, pred_log_kex_confidence_chunk, _, _, _, _ = model.forward(chunked_batches[chunk_idx])
            all_cycle_predictions, _, _, _, _ = model.forward(chunked_batches[chunk_idx])
            pred_log_kex_chunk, pred_log_kex_confidence_chunk = all_cycle_predictions[-1]
        else:
            pred_log_kex_chunk = torch.full((1, chunk_size+chunk_padding*2), -100.0)
            pred_log_kex_confidence_chunk = torch.full((1, chunk_size+chunk_padding*2), -100.0) 

        if chunk_idx == 0:
            start_idx = 0
            end_idx = min(chunk_size, protein_length)
        elif chunk_idx == len(chunks) - 1:
            start_idx = chunk_padding
            # end_idx = int(min(chunk_size, protein_length % chunk_size) + chunk_padding)
            remaining_length = protein_length - chunk_size * (len(chunks) - 1)
            end_idx = chunk_padding + remaining_length
        else:
            start_idx = chunk_padding
            end_idx = start_idx + chunk_size

        pred_log_kex.append(pred_log_kex_chunk[:, start_idx:end_idx])
        pred_log_kex_confidence.append(pred_log_kex_confidence_chunk[:, start_idx:end_idx])

        # print(chunk_idx,chunk, start_idx, end_idx, pred_log_kex_chunk.shape, pred_log_kex_confidence_chunk[:, start_idx:end_idx].shape)

    pred_log_kex = torch.concat(pred_log_kex, dim=-1)
    pred_log_kex_confidence = torch.concat(pred_log_kex_confidence, dim=-1)

    peptide_data, residue_data, global_vars, log_kex = batch

    start_pos = peptide_data["start_pos"]
    end_pos = peptide_data["end_pos"]
    time_points = peptide_data["time_point"]
    back_ex = 1-peptide_data["effective_deut_rent"]
    saturation = global_vars[:, -1]
    log_kch = residue_data['log_kch'].unsqueeze(-1)
    if not centroid_model:
        obs_envelope = peptide_data["probabilities"]
        obs_envelope_t0 = peptide_data["t0_isotope"]
    else:
        obs_num_d = peptide_data["num_d"].unsqueeze(-1)

    peptide_padding_mask, seq_mask, envelope_mask = model.get_mask(batch)
    pred_log_kex_mask_inf = pred_log_kex.clone()
    pred_log_kex_mask_inf[seq_mask] = -float("inf")

    if not centroid_model:
        pred_envelope = get_calculated_isotope_envelope(
            start_pos, end_pos, time_points, obs_envelope_t0, obs_envelope, pred_log_kex_mask_inf, back_ex, saturation
        )
    else:
        pred_num_d = get_calculated_num_d(start_pos, end_pos, time_points, pred_log_kex_mask_inf, back_ex, saturation)

    # log_kex[seq_mask] = -100
    # pred_log_kex[seq_mask] = -100
    # pred_log_kex_confidence[seq_mask] = -100

    # if_sort_log_kex = True
    # if if_sort_log_kex:
    #     # position irrelevant within the resolution limits, all segment
    #     log_kex, log_kch, pred_log_kex, pred_log_kex_confidence = model._sort_log_kex(log_kex, log_kch, pred_log_kex, pred_log_kex_confidence, residue_data["resolution_limits"])
    # else:
    # pro_mask = residue_data["proline_mask"]
    # log_kex, log_kch, pred_log_kex, pred_log_kex_confidence = model._sort_log_kex(log_kex, log_kch, pred_log_kex_mask_inf, pred_log_kex_confidence, residue_data["resolution_limits"],                                                                           only_pro_segment=False, pro_mask=pro_mask)

    # mask =  torch.isinf(pred_log_kex)
    mask = seq_mask
    log_kex[mask] = -100
    pred_log_kex[mask] = -100
    pred_log_kex_confidence[mask] = -100
    
    if not centroid_model:
        return pred_log_kex, pred_log_kex_confidence, log_kex, pred_envelope, obs_envelope, seq_mask, envelope_mask
    else:
        return pred_log_kex, pred_log_kex_confidence, log_kex, pred_num_d, obs_num_d, seq_mask, envelope_mask


class PFNetAnalysis(Analysis):
    def __init__(self, states, temperature, pH, **kwargs):
        super().__init__(states, temperature, pH, **kwargs)

    def clustering_results(self):

        results = Results(self)

        for k, v in self.maximum_resolution_limits.items():
            try:
                mini_pep = results.get_mini_pep(v[0], v[1])
            except:
                mini_pep = MiniPep(v[0], v[1])
                results.add_mini_pep(mini_pep)

            # print(k,v, clustering_a_mini_pep(k, v[0], v[1]))
            res = results.get_residue_by_resid(k)
            mini_pep.add_residue(res)
            mini_pep.set_clustering_results(self._clustering_a_mini_pep(k, v[0], v[1]))

        self.results_obj = results

        # for res in results.residues:
        #     if hasattr(res, "mini_pep"):
        #         res.clustering_results_logP = np.array([res.log_k_init + i for i in res.mini_pep.clustering_results_log_kex])  # log_kex is negative
        #         res.std_within_clusters_logP = res.mini_pep.std_within_clusters_log_kex
        #         res.SE_within_clusters_logP = res.mini_pep.SE_within_clusters_log_kex

    def _clustering_a_mini_pep(self, k, v0, v1):

        # v0 and v1 are 1-based
        # Apply remove_outliers to each column and collect results
        # cleaned_data = [remove_outliers(self.bayesian_hdx_df[col]) for col in self.bayesian_hdx_df.iloc[:,v0-1:v1].columns]
        # cleaned_data = [remove_outliers(self.bayesian_hdx_df_log_kex[col]) for col in self.bayesian_hdx_df_log_kex.iloc[:,v0-1:v1].columns]
        cleaned_data = [remove_outliers(self.feather_samples[:, col]) for col in range(v0 - 1, v1)]

        initial_centers = np.array([np.mean(col[:20]) for col in cleaned_data if col.size != 0]).reshape(-1, 1)
        num_clusters = initial_centers.shape[0]

        # add 0 if Proline in the mini_pep
        num_Ps = self.protein_sequence[v0 - 1 : v1].count("P")

        if num_clusters == 0:
            if num_Ps > 0:
                return np.array([np.inf] * num_Ps), np.array([np.inf] * num_Ps), np.array([np.inf] * num_Ps)
            num_nan = v1 - v0 + 1
            return np.array([np.nan] * num_nan), np.array([np.nan] * num_nan), np.array([np.nan] * num_nan)

        pool_values = np.concatenate(cleaned_data).flatten()
        pool_values = pool_values[~np.isnan(pool_values)]
        
        # # rm data larger than log_kch， kex >= kch, -kex <= -kch
        # mini_pep_log_kch = [res.log_k_init*-1 for res in mini_pep.residues]
        # pool_values = pool_values[pool_values <= min(mini_pep_log_kch)]
        
        if len(pool_values) != 0:
            k_cluster = KMeans(n_clusters=num_clusters, random_state=0, init=initial_centers, n_init="auto").fit(pool_values.reshape(-1, 1))
            #sorted_indices = np.argsort(k_cluster.cluster_centers_.flatten())

            std_within_cluster = np.zeros(num_clusters)
            data_within_cluster = np.zeros(num_clusters, dtype=object)
            for i in range(num_clusters):
                cluster_mask = k_cluster.labels_ == i
                cluster_points = pool_values[cluster_mask]
                center = k_cluster.cluster_centers_[i]
                std_within_cluster[i] = np.std(cluster_points)
                data_within_cluster[i] = cluster_points

            # cluster_centers = k_cluster.cluster_centers_.flatten()[sorted_indices]
            # std_within_cluster = std_within_cluster[sorted_indices]
            # data_within_cluster = data_within_cluster[sorted_indices]
            cluster_centers = k_cluster.cluster_centers_.flatten()
            std_within_cluster = std_within_cluster
            data_within_cluster = data_within_cluster

            if num_Ps > 0:

                for seq_i, seq in enumerate(self.protein_sequence[v0 - 1 : v1]):
                    if seq == "P":
                        cluster_centers = np.insert(cluster_centers, seq_i, np.inf)
                        std_within_cluster = np.insert(std_within_cluster, seq_i, 0)
                        data_within_cluster = np.insert(data_within_cluster, seq_i, np.inf)

                return cluster_centers, std_within_cluster, data_within_cluster
            else:
                return cluster_centers, std_within_cluster, data_within_cluster
