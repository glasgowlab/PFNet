import argparse
import os
import glob

from pfnet.run_inference import predict
from pfnet.utilts import (
    get_display_state_name,
    get_all_statics_info,
    get_log_kex_plot,
    get_heatmap,
    get_csv_results,
    make_BFactorPlot,
    get_summary,
    load_pdb_data
)
from pigeon_feather.hxio import load_HXMS_file


def parse_args():
    parser = argparse.ArgumentParser(description="Run PFNet inference on HDX-MS data with comprehensive output generation")
    parser.add_argument("--input", type=str, required=True, help="Input file or directory")
    parser.add_argument("--input2", type=str, help="Second input file (optional, for comparison)")
    parser.add_argument("--output_json", type=str, default="pfnet_output.json", help="Output JSON file")
    parser.add_argument("--model_type", type=str, choices=["envelope", "centroid"], default="envelope", help="Model type")
    parser.add_argument("--refine", action="store_true", help="Enable refinement")
    parser.add_argument("--refine_steps", type=int, default=200, help="Number of refinement steps")
    parser.add_argument("--refine_cen_sigma", type=float, default=0.5, help="Centroid sigma for refinement")
    parser.add_argument("--refine_env_sigma", type=float, default=0.3, help="Envelope sigma for refinement")
    parser.add_argument("--refine_single_pos_conf_threshold", type=float, default=0.8, help="Single position confidence threshold for refinement")
    parser.add_argument("--refine_non_single_pos_conf_threshold", type=float, default=0.9, help="Non-single position confidence threshold for refinement")
    parser.add_argument("--plot", action="store_true", help="Generate uptake plots")
    parser.add_argument("--outputs_dir", type=str, default="pfnet_outputs", help="Directory for outputs")
    
    # New comprehensive output options
    parser.add_argument("--pdb_id", type=str, help="PDB ID to download for structure visualization")
    parser.add_argument("--pdb_file", type=str, help="Path to PDB file for structure visualization")
    parser.add_argument("--generate_all", action="store_true", help="Generate all outputs (equivalent to enabling all generate_* flags)")
    parser.add_argument("--generate_summary", action="store_true", default=True, help="Generate comprehensive summary")
    parser.add_argument("--generate_csv", action="store_true", default=True, help="Generate CSV results")
    parser.add_argument("--generate_log_kex_plot", action="store_true", default=True, help="Generate log(kex) plot")
    parser.add_argument("--generate_heatmaps", action="store_true", default=True, help="Generate heatmaps")
    parser.add_argument("--generate_bfactor_plot", action="store_true", default=True, help="Generate BFactor plot for PDB")
    parser.add_argument("--output_dir", type=str,default="pfnet_outputs", help="Output directory")
    
    return parser.parse_args()


def main():
    """Main entry point for pfnet CLI"""
    args = parse_args()

    if args.generate_all:
        args.generate_summary = True
        args.generate_csv = True
        args.generate_log_kex_plot = True
        args.generate_heatmaps = True
        args.generate_bfactor_plot = True
        args.plot = True

    output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/pfnet_output", exist_ok=True)
    os.makedirs(f"{output_dir}/pfnet_plots", exist_ok=True)

    print(f"Output directory: {args.output_dir}")

    input_files = [args.input]
    if args.input2:
        input_files.append(args.input2)

    print(f"Processing {len(input_files)} input file(s): {input_files}")

    hdxms_data_list = [load_HXMS_file(input_file) for input_file in input_files]
    global state_names
    state_names = [state.state_name for data in hdxms_data_list for state in data.states]
    print(f"State names: {state_names}")

    output_dicts = {}
    for idx, input_file in enumerate(input_files):
        print(f"Running PFNet prediction for {state_names[idx]}...")
        output_dict_i = predict(
            input=input_file,
            output_json=f"{output_dir}/pfnet_output/results_{state_names[idx]}_{idx}.json",
            centroid_model=(args.model_type == "centroid"),
            refinement=args.refine,
            refine_steps=args.refine_steps,
            refine_cen_sigma=args.refine_cen_sigma,
            refine_env_sigma=args.refine_env_sigma,
            refine_single_pos_conf_threshold=args.refine_single_pos_conf_threshold,
            refine_non_single_pos_conf_threshold=args.refine_non_single_pos_conf_threshold,
            uptake_plots=args.plot,
            plots_dir=f"{output_dir}/pfnet_plots",
            benchmark=False
        )
        output_dicts[f"{state_names[idx]}_{idx}"] = output_dict_i

        if args.plot:
            for plot_file in glob.glob(f"{output_dir}/pfnet_plots/PFNet_uptake_?.pdf"):
                os.rename(plot_file, plot_file.replace("PFNet_uptake_", f"PFNet_uptake_{state_names[idx]}_"))

            for envelope_type in ["centroid", "envelope"]:
                plot_file = f"{output_dir}/pfnet_plots/ae_histogram_{envelope_type}.png"
                if os.path.exists(plot_file):
                    os.rename(plot_file, plot_file[:-4] + f"_{state_names[idx]}.png")

    analysis_objs = {f"{state_names[idx]}_{idx}": output_dicts[f"{state_names[idx]}_{idx}"]["analysis_objs"][f"analysis_pfnet"]
                    for idx in range(len(state_names)) if f"{state_names[idx]}_{idx}" in output_dicts}

    results = {}

    if args.generate_log_kex_plot:
        print("Generating log(kex) plot...")
        try:
            log_kex_plot = get_log_kex_plot(analysis_objs, output_dir)
            results['log_kex_plot'] = log_kex_plot
            print(f"Log(kex) plot saved: {log_kex_plot}")
        except Exception as e:
            print(f"Error generating log(kex) plot: {e}")
            results['log_kex_plot'] = None

    if args.generate_csv:
        print("Generating CSV results...")
        try:
            csv_results_path_list = get_csv_results(analysis_objs, output_dicts, output_dir)
            results['csv_files'] = csv_results_path_list
            print(f"CSV files generated: {csv_results_path_list}")
        except Exception as e:
            print(f"Error generating CSV results: {e}")
            results['csv_files'] = []

    if args.generate_summary:
        print("Generating summary...")
        try:
            summary, summary_path = get_summary(output_dicts, hdxms_data_list, output_dir)
            results['summary'] = summary
            results['summary_path'] = summary_path
            print(f"Summary saved: {summary_path}")
        except Exception as e:
            print(f"Error generating summary: {e}")
            results['summary'] = ""
            results['summary_path'] = None

    if args.generate_heatmaps:
        print("Generating heatmaps...")
        try:
            heatmaps = get_heatmap(hdxms_data_list, output_dir)
            results['heatmaps'] = heatmaps
            print(f"Heatmaps generated: {heatmaps}")
        except Exception as e:
            print(f"Error generating heatmaps: {e}")
            results['heatmaps'] = []

    dg_pdb_path = None
    if args.generate_bfactor_plot and (args.pdb_id or args.pdb_file):
        print("Generating BFactor plot...")
        try:
            pdb_path = load_pdb_data(args.pdb_id, args.pdb_file)
            if pdb_path:
                dg_pdb_path = make_BFactorPlot(pdb_path, analysis_objs, output_dir)
                results['dg_pdb_path'] = dg_pdb_path
                print(f"BFactor plot saved: {dg_pdb_path}")
            else:
                print("No PDB file available for BFactor plot generation")
        except Exception as e:
            print(f"Error generating BFactor plot: {e}")
            results['dg_pdb_path'] = None

    json_files = glob.glob(f"{output_dir}/pfnet_output/results_*.json")
    ae_histograms = sorted(glob.glob(f"{output_dir}/pfnet_plots/ae_histogram_*.png"))
    uptake_plots = glob.glob(f"{output_dir}/pfnet_plots/PFNet_uptake_*.pdf")


    if results.get('summary'):
        # print("\nSUMMARY CONTENT:")
        print("-" * 40)
        print(results['summary'])
        print("-" * 40)


if __name__ == "__main__":
    main()
