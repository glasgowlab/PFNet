from pigeon_feather.data import HDXMSData, Peptide, ProteinState, Timepoint
from pigeon_feather.tools import custom_pad, _add_max_d_to_pep
from pfnet.data import get_resolution_grouping
import numpy as np
import torch
from datetime import datetime


def hxms_data_to_grouped_dict(hxms_data):
    data_dict = {}
    all_peptides = [peptide for state in hxms_data.states for peptide in state.peptides]
    all_peptides.sort(key=lambda x: (x.start, x.end))

    for i, peptide in enumerate(all_peptides):

        pep_dict = {}
        
        # raw_start = int(peptide.identifier.split(" ")[0].split("-")[0])
        # raw_end = int(peptide.identifier.split(" ")[0].split("-")[1])
        # raw_sequence = peptide.identifier.split(" ")[1]

        pep_dict["start"] = peptide.raw_start
        pep_dict["end"] = peptide.raw_end
        pep_dict["sequence"] = peptide.raw_sequence
        pep_dict["time"] = [tp.deut_time for tp in peptide.timepoints if tp.deut_time != np.inf]
        pep_dict["num_d"] = [tp.num_d for tp in peptide.timepoints if tp.deut_time != np.inf]
        pep_dict["isotope"] = [custom_pad(tp.isotope_envelope[:50], 50) for tp in peptide.timepoints if tp.deut_time != np.inf]
        # t0 isotope envelope
        pep_dict["t0_isotope"] = [custom_pad(peptide.get_timepoint(0).isotope_envelope[:50], 50) for tp in peptide.timepoints if tp.deut_time != np.inf]
        pep_dict["max_d"] = peptide.max_d
        pep_dict["theo_max_d"] = peptide.theo_max_d
        pep_dict['effective_deut_rent'] = custom_pad(peptide.deut_rent, 50)

        data_dict[f"peptide_{i}"] = pep_dict

    # protein features
    data_dict["num_peptides"] = len(all_peptides)
    data_dict["protein_sequence"] = hxms_data.protein_sequence
    data_dict["temperature"] = hxms_data.temperature
    data_dict["pH"] = hxms_data.pH
    from hdxrate import k_int_from_sequence
    data_dict["log_kch"] =  np.log10(k_int_from_sequence(hxms_data.protein_sequence, hxms_data.temperature, hxms_data.pH, d_percentage=hxms_data.saturation*100))
    data_dict["log_kex"] = np.zeros(len(hxms_data.protein_sequence))
    data_dict["log_P"] = np.zeros(len(hxms_data.protein_sequence))
    data_dict["saturation"] = hxms_data.saturation
    data_dict["resolution_limits"], data_dict["resolution_grouping"] = get_resolution_grouping(hxms_data)

    return data_dict


def datadict_to_hdxmsdata(
    data_dict,
    protein_name="SIM",
    n_fastamides=1,
    centroid_data=False,
):

    saturation = float(data_dict["saturation"])
    protein_sequence = data_dict["protein_sequence"]

    hdxms_data = HDXMSData(
        protein_name,
        n_fastamides,
        protein_sequence=protein_sequence,
        saturation=saturation,
        temperature=data_dict["temperature"],
        pH=data_dict["pH"],
    )

    for start, end, time, num_d, envelope, max_d, deut_rent in zip(
        data_dict["start"], data_dict["end"], data_dict["time_points"], data_dict["num_d"], data_dict["isotope_envelope"], data_dict["max_d"], data_dict['effective_deut_rent']
    ):
        # Check if protein state exists
        protein_state = None
        for state in hdxms_data.states:
            if state.state_name == protein_name:
                protein_state = state
                break

        # If protein state does not exist, create and add to HDXMSData
        if not protein_state:
            protein_state = ProteinState(protein_name, hdxms_data=hdxms_data)
            hdxms_data.add_state(protein_state)

        #raw_start = int(start-n_fastamides)
        raw_start = int(start)
        raw_end = int(end)
        raw_sequence = protein_sequence[raw_start -1: raw_end]

        # Check if peptide exists in current state
        peptide = None
        for pep in protein_state.peptides:
            # identifier = f"{row['Start']+n_fastamides}-{row['End']} {row['Sequence'][n_fastamides:]}"
            identifier = f"{raw_start}-{raw_end} {raw_sequence}"
            if pep.identifier == identifier:
                peptide = pep
                break

        # If peptide does not exist, create and add to ProteinState
        if not peptide:
            # skip if peptide is less than 4 residues
            if len(raw_sequence) < 4:
                continue
            peptide = Peptide(
                raw_sequence,
                raw_start,
                raw_end,
                protein_state,
                n_fastamides=n_fastamides,
                RT=None,
            )
            # peptide = Peptide(row['Sequence'], row['Start'], row['End'], protein_state)
            protein_state.add_peptide(peptide)

        # Add timepoint data to peptide
        timepoint = Timepoint(
            peptide,
            float(time),
            float(num_d),
            0,
            0,
        )
        timepoint.deut_rent = deut_rent
        if not centroid_data:
            if isinstance(envelope, torch.Tensor):
                envelope = envelope.detach().numpy()
            timepoint.isotope_envelope = envelope
        peptide.add_timepoint(timepoint, allow_duplicate=True)

        if timepoint.deut_time == 0.0:
           # _add_max_d_to_pep(peptide, max_d=(1 - float(back_ex)) * peptide.theo_max_d, force=True)
            _add_max_d_to_pep(peptide, max_d=max_d, force=True)

    return hdxms_data

