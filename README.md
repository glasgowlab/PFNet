# PFNet

PFNet is a machine learning model that determines $\Delta G_{op}$ for arbitrarily large proteins and complexes from conventional peptide-level hydrogen exchange/mass spectrometry (HX/MS) datasets.

## Installation

PFNet uses Pixi for dependency management and supports both CPU and GPU environments across multiple platforms.

### prerequisites

```bash
# install pixi
curl -fsSL https://pixi.sh/install.sh | sh
```

### installation options
```bash
# clone the repository
git clone https://github.com/glasgowlab/PFNet
cd PFNet

# install cpu version (default, supports linux/macos)
pixi install

# or install gpu version (linux only)
pixi install -e cuda
```

### platform support

- **linux**: cpu and gpu (cuda) support (gpu mainly for training, cpu inference is already fast)
- **macos intel**: cpu support only
- **macos apple silicon**: cpu support only

### running pfnet

#### in project directory
```bash
# cpu version
pixi run pfnet --input examples/EEHEEEEHEE_rd4_0871.hxms --generate_all

# gpu version (linux only)
pixi run -e cuda pfnet --input examples/EEHEEEEHEE_rd4_0871.hxms --generate_all
```

#### from any directory
```bash
# run from anywhere by specifying the manifest path
pixi run --manifest-path /path/to/PFNet/pyproject.toml pfnet --input examples/EEHEEEEHEE_rd4_0871.hxms --generate_all
```

## Quick Start

### A typical HX/MS workflow

![HX/MS workflow](.hxms_workflow.svg)
**Tools:**
- [PIGEON](https://github.com/glasgowlab/PIGEON-FEATHER) - GitHub
- [PFLink](https://huggingface.co/spaces/glasgow-lab/PFLink) - Hugging Face
- [PFNet](https://huggingface.co/spaces/glasgow-lab/PFNet) - Hugging Face

### Input Data Format

PFNet accepts HXMS format as input, which is a unified, lightweight, scalable, and human-readable file format for HX/MS data. The HXMS format preserves the isotopic mass envelopes for all peptides, captures the full experimental time-course including the fully deuterated control samples, and contains all other key information. HXMS files can be generated using [PFLink](https://huggingface.co/spaces/glasgow-lab/PFLink), which supports exports from BioPharma Finder, HDExaminer, DynamX, and HDX Workbench. 

### basic usage
run pfnet on a single hx/ms data file:

```bash
pixi run pfnet --input examples/EEHEEEEHEE_rd4_0871.hxms --generate_all
```

### comparison analysis
compare two protein states:

```bash
pixi run pfnet --input examples/ecDHFR_APO.hxms --input2 examples/ecDHFR_MTX.hxms --generate_all
```

### with structure visualization
generate bfactor plots for pdb visualization:

```bash
pixi run pfnet --input examples/ecDHFR_APO.hxms --pdb_id 6XG5 --generate_all
```

### gpu acceleration (linux only)
note: gpu is primarily used for training the model. inference is already very fast on cpu, so gpu acceleration provides minimal speedup for typical use cases.

```bash
pixi run -e cuda pfnet --input examples/EEHEEEEHEE_rd4_0871.hxms --generate_all
```

## Command Line Interface

PFNet provides a comprehensive command-line interface accessible through `pixi run pfnet`:

### input arguments
```
--input INPUT                input hdx-ms data file (required)
--input2 INPUT2              second input file for comparison (optional)
--output_dir OUTPUT_DIR      output directory (default: pfnet_outputs)
--model_type {envelope,centroid}  model type (default: envelope)
```

### refinement options
```
--refine                     enable bayesian refinement
--refine_steps STEPS         number of refinement steps (default: 200)
--refine_cen_sigma SIGMA     centroid sigma for refinement (default: 0.5)
--refine_env_sigma SIGMA     envelope sigma for refinement (default: 0.3)
--refine_single_pos_conf_threshold THRESHOLD  single position confidence threshold (default: 0.8)
--refine_non_single_pos_conf_threshold THRESHOLD  non-single position confidence threshold (default: 0.9)
```

### output generation options
```
--generate_all              generate all outputs (recommended)
--generate_summary          generate summary (default: True)
--generate_csv              generate csv results (default: True)
--generate_log_kex_plot     generate log(kex) plot (default: True)
--generate_heatmaps         generate heatmaps (default: True)
--generate_bfactor_plot     generate bfactor plot for pdb (default: True)
--plot                      generate uptake plots
```

### structure visualization
```
--pdb_id PDB_ID             pdb id to download for structure visualization
--pdb_file PDB_FILE         path to pdb file for structure visualization
```

## Output Files

PFNet generates comprehensive outputs organized in the following structure:

```
output_dir/
├── pfnet_output/
│   ├── results_[state]_[idx].json    # Raw prediction results
│   └── results_[state]_[idx].csv     # Residue-level data
├── pfnet_plots/
│   ├── log_kex_plot.png              # Log(kex) visualization
│   ├── heatmap_[state].png          # Single state heatmap
│   ├── heatmap_[state1]_[state2].png # Comparison heatmap
│   ├── PFNet_uptake_[state]_[idx].pdf # Uptake plots
│   ├── ae_histogram_[model]_[state].png # Absolute error histograms
│   └── PFNet_$\Delta G_{op}$.pdb                 # BFactor plot (single state)
│       └── PFNet_$\Delta\Delta G_{op}$_[state1]-[state2].pdb # BFactor plot (comparison)
└── summary.txt                       # Comprehensive analysis summary
```

### Output Descriptions

- **JSON files**: Raw prediction results including kex values, confidence scores, and model metadata
- **CSV files**: Residue-level data with columns for:
  - Residue information (ID, name)
  - Predicted values ($\Delta G_{op}$, logP, log_kex)
  - Confidence scores and coverage
  - Single-residue resolution status
- **Log(kex) plots**: Bar plots showing exchange rates across the protein sequence
- **Heatmaps**: Visual representation of deuteration levels or differences between states
- **Uptake plots**: Uptake plots showing experimental vs predicted uptake
- **BFactor plots**: PDB files colored by predicted stability ($\Delta G_{op}$) or stability differences ($\Delta\Delta G_{op}$), $\Delta G_{op}$ or $\Delta\Delta G_{op}$ were stored in the BFactor column and the confidence is stored in the occupancy column 
- **Summary**: Statistics of the input data and analysis results


## examples

### example 1: basic analysis
```bash
pixi run pfnet --input examples/EEHEEEEHEE_rd4_0871.hxms --generate_all
```

### example 2: state comparison
```bash
pixi run pfnet --input examples/EEHEEEEHEE_rd4_0871.hxms --input2 examples/state2.hxms --generate_all
```

### example 3: with structure
```bash
pixi run pfnet --input examples/EEHEEEEHEE_rd4_0871.hxms --pdb_id 1A2B --generate_all
```

### example 4: custom output directory
```bash
pixi run pfnet --input examples/EEHEEEEHEE_rd4_0871.hxms --output_dir ./my_analysis --generate_all
```

### example 5: run from any directory
```bash
# from any directory, specify the full path to pyproject.toml
pixi run --manifest-path /path/to/PFNet/pyproject.toml pfnet --input examples/EEHEEEEHEE_rd4_0871.hxms --generate_all
```

### setting up a global alias (optional)
for convenience, you can create a global alias to run pfnet from anywhere:

```bash
# add to your ~/.zshrc or ~/.bashrc
alias pfnet='pixi run --manifest-path /path/to/PFNet/pyproject.toml pfnet'

# then reload your shell
source ~/.zshrc  # or source ~/.bashrc

# now you can run pfnet from anywhere
pfnet --input examples/EEHEEEEHEE_rd4_0871.hxms --generate_all
```

## model types

pfnet offers two model variants:

- **envelope model** (`--model_type envelope`): default model that utilizes the full isotope envelopes (recommended)
- **centroid model** (`--model_type centroid`): simplified model that only uses centroid uptake value for the prediction (only use it if there is no envelope data)

## visualization in pymol

to visualize the $\Delta G_{op}$ values in pymol using the output pdb file, use the following commands:

```python
spectrum b, white_orange, minimum=0, maximum=80;
select nans, not(b=0 or b>0 or b<0); color grey80, nans;
color gray50, not polymer.protein
util.cnc 
set valence, 0
```

note: nans are prolines, residues not covered, or noisy data. 

## acknowledgments

pfnet builds upon the pigeon-feather library for hx/ms data processing and analysis.
