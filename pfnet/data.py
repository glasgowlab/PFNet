import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from pigeon_feather.data import SimulatedData, HDXMSData, ProteinState, Peptide, Timepoint
from pigeon_feather.tools import custom_pad, _add_max_d_to_pep, event_probabilities, normlize
from pigeon_feather.spectra import get_theoretical_isotope_distribution
import pyopenms as oms
import os
import glob
import signal
from contextlib import contextmanager
from pfnet.model import AminoAcidEncoder
import numpy as np
from scipy.optimize import brentq 
from hdxrate import k_int_from_sequence
from collections import defaultdict
import math 

class HDXDataset(Dataset):

    def __init__(self, input, data_start_idx=0, data_end_idx=10, centroid_data=False, iso_env_peaks=50, permutate_tps=False):

        self.cache = {}
        self.centroid_data = centroid_data
        self.iso_env_peaks = iso_env_peaks
        self.permutate_tps = permutate_tps
        # if input is a file, load it
        if isinstance(input, dict):
            self.files = [input]
            self.cache = {0: input}
        elif os.path.isfile(input):
            self.input_file = input
            self.files = [self.input_file]
        else:
            self.data_dir = input
            self.files = sorted(
                glob.glob(os.path.join(self.data_dir, "*.pt")),
                key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1]),
            )[data_start_idx:data_end_idx]

    def load_protein_file(self, file_path):
        try:
            return torch.load(file_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        if idx in self.cache:
            data_dict = self.cache[idx]
        else:
            # Load the specific protein file
            data_dict = self.load_protein_file(self.files[idx])
            if data_dict is None:
                # skip it and load 0 index
                return self.__getitem__(0)
            self.cache[idx] = data_dict

        number_of_peptides = torch.tensor(data_dict["num_peptides"], dtype=torch.long)

        # Flatten lists and convert to tensors
        starts = []
        ends = []
        times = []
        num_d = []
        max_d = []
        isotopes = []
        t0_isotopes = []
        back_exs = []
        sequences = []
        effective_deut_rents = []

        for i in range(number_of_peptides):
            peptide = data_dict[f"peptide_{i}"]
            num_time_points = len(peptide["time"])
            starts.extend([peptide["start"]] * num_time_points)
            ends.extend([peptide["end"]] * num_time_points)
            sequences.extend([peptide["sequence"]] * num_time_points)
            times.extend(peptide["time"])
            num_d.extend(peptide["num_d"])
            effective_deut_rents.extend([peptide["effective_deut_rent"]] * num_time_points)
            t0_index = peptide["time"].index(0)
            max_d.extend([peptide["max_d"]] * num_time_points)
            theo_max_d = peptide["theo_max_d"]
            back_exs.extend([1 - peptide["max_d"] / theo_max_d] * num_time_points)
            if not self.centroid_data:
                isotopes.extend(peptide["isotope"])
                t0_isotopes.extend([peptide["isotope"][t0_index]] * num_time_points)
            else:
                isotopes.extend([np.zeros(self.iso_env_peaks)] * num_time_points)
                t0_isotopes.extend([np.zeros(self.iso_env_peaks)] * num_time_points)

        start = torch.tensor(np.array(starts), dtype=torch.int16)
        end = torch.tensor(np.array(ends), dtype=torch.int16)
        sequence = np.array(sequences)
        time_points = torch.tensor(np.array(times), dtype=torch.float32)
        num_d = torch.tensor(np.array(num_d), dtype=torch.float32)
        max_d = torch.tensor(np.array(max_d), dtype=torch.float32)
        isotope_envelope = torch.tensor(np.array(isotopes), dtype=torch.float32).view(-1, self.iso_env_peaks)
        t0_isotope = torch.tensor(np.array(t0_isotopes), dtype=torch.float32).view(-1, self.iso_env_peaks)
        back_ex = torch.tensor(np.array(back_exs), dtype=torch.float32)
        effective_deut_rent = torch.tensor(np.array(effective_deut_rents), dtype=torch.float32)
        saturation = torch.tensor(data_dict["saturation"], dtype=torch.float32)
        log_kex = torch.tensor(data_dict["log_kex"], dtype=torch.float32)
        log_kch = torch.tensor(data_dict["log_kch"], dtype=torch.float32)
        resolution_limits = torch.tensor(data_dict["resolution_limits"], dtype=torch.int32)
        resolution_grouping = torch.tensor(data_dict["resolution_grouping"], dtype=torch.float32)
        miss_id_bool = torch.zeros_like(time_points)

        if self.permutate_tps:
            permutated_idx = torch.randperm(time_points.shape[0])

            # only for peptide data
            time_points = time_points[permutated_idx]
            start = start[permutated_idx]
            end = end[permutated_idx]
            sequence = sequence[permutated_idx]
            isotope_envelope = isotope_envelope[permutated_idx]
            t0_isotope = t0_isotope[permutated_idx]
            back_ex = back_ex[permutated_idx]
            num_d = num_d[permutated_idx]
            max_d = max_d[permutated_idx]
            effective_deut_rent = effective_deut_rent[permutated_idx]
            miss_id_bool = miss_id_bool[permutated_idx]

        return {
            "time_points": time_points,
            "start": start,
            "end": end,
            "log_kex": log_kex,
            "log_kch": log_kch,
            "back_ex": back_ex,
            "sequence": sequence,
            "num_d": num_d,
            "max_d": max_d,
            "effective_deut_rent": effective_deut_rent,
            "isotope_envelope": isotope_envelope,
            "t0_isotope": t0_isotope,
            "saturation": saturation,
            "resolution_limits": resolution_limits,
            "resolution_grouping": resolution_grouping,
            "protein_sequence": data_dict["protein_sequence"],
            "temperature": data_dict["temperature"],
            "pH": data_dict["pH"],
            "miss_id_bool": miss_id_bool,
            "proline_mask": torch.tensor([1 if res == "P" else 0 for res in data_dict["protein_sequence"]]),
        }


class OnlineHDXDataset(Dataset):

    def __init__(
        self,
        data_start_idx=0,
        data_end_idx=10,
        base_seed=42,
        permutate_tps=True,
        noise_level=None,
        length_range=(100, 400),
        generate_miss_id=False,
        iso_env_peaks=50,
        centroid_data=False
    ):
        self.data_start_idx = data_start_idx
        self.data_end_idx = data_end_idx
        self.base_seed = base_seed
        self.idxs = range(self.data_start_idx, self.data_end_idx)
        self.cache = {}
        self.permutate_tps = permutate_tps
        self.noise_level = noise_level
        self.length_range = length_range
        self.generate_miss_id = generate_miss_id
        self.iso_env_peaks = iso_env_peaks
        self.centroid_data = centroid_data

    def __len__(self):
        return len(self.idxs)

    def _generate_data(self, i):
        try:
            with timeout(30):
                return generate_single_data(i, self.base_seed, length_range=self.length_range, noise_level=self.noise_level, centroid_data=self.centroid_data)
        except ValueError:
            for j in range(200):
                try:
                    with timeout(10):
                        return generate_single_data(j, 0, length_range=self.length_range, noise_level=self.noise_level, centroid_data=self.centroid_data)
                except ValueError:
                    continue  # try next
            raise ValueError(f"Generating data failed for index {i}")

    def __getitem__(self, idx):

        idx = self.idxs[idx]

        if idx in self.cache:
            data_dict = self.cache[idx]
        else:
            data_dict = self._generate_data(idx)
            self.cache[idx] = data_dict

        number_of_peptides = torch.tensor(data_dict["num_peptides"], dtype=torch.long)

        # Flatten lists and convert to tensors
        starts = []
        ends = []
        times = []
        num_d = []
        max_d = []
        isotopes = []
        t0_isotopes = []
        back_exs = []
        sequences = []
        effective_deut_rents = []

        for i in range(number_of_peptides):
            peptide = data_dict[f"peptide_{i}"]
            num_time_points = len(peptide["time"])
            starts.extend([peptide["start"]] * num_time_points)
            ends.extend([peptide["end"]] * num_time_points)
            sequences.extend([peptide["sequence"]] * num_time_points)
            times.extend(peptide["time"])
            num_d.extend(peptide["num_d"])
            effective_deut_rents.extend([peptide["effective_deut_rent"]] * num_time_points)
            isotopes.extend(peptide["isotope"])
            t0_isotopes.extend(peptide["t0_isotope"])
            max_d.extend([peptide["max_d"]] * num_time_points)
            theo_max_d = peptide["theo_max_d"]
            back_exs.extend([1 - peptide["max_d"] / theo_max_d] * num_time_points)

        start = torch.tensor(np.array(starts), dtype=torch.int16)
        end = torch.tensor(np.array(ends), dtype=torch.int16)
        sequence = np.array(sequences)
        time_points = torch.tensor(np.array(times), dtype=torch.float32)
        num_d = torch.tensor(np.array(num_d), dtype=torch.float32)
        max_d = torch.tensor(np.array(max_d), dtype=torch.float32)
        isotope_envelope = torch.tensor(np.array(isotopes), dtype=torch.float32).view(-1, self.iso_env_peaks)
        t0_isotope = torch.tensor(np.array(t0_isotopes), dtype=torch.float32).view(-1, self.iso_env_peaks)
        back_ex = torch.tensor(np.array(back_exs), dtype=torch.float32)
        effective_deut_rent = torch.tensor(np.array(effective_deut_rents), dtype=torch.float32)
        saturation = torch.tensor(data_dict["saturation"], dtype=torch.float32)
        log_kex = torch.tensor(data_dict["log_kex"], dtype=torch.float32)
        log_kch = torch.tensor(data_dict["log_kch"], dtype=torch.float32)
        resolution_limits = torch.tensor(data_dict["resolution_limits"], dtype=torch.int32)
        resolution_grouping = torch.tensor(data_dict["resolution_grouping"], dtype=torch.float32)

        if self.permutate_tps:
            permutated_idx = torch.randperm(time_points.shape[0])

            # only for peptide data
            time_points = time_points[permutated_idx]
            start = start[permutated_idx]
            end = end[permutated_idx]
            sequence = sequence[permutated_idx]
            isotope_envelope = isotope_envelope[permutated_idx]
            t0_isotope = t0_isotope[permutated_idx]
            back_ex = back_ex[permutated_idx]
            num_d = num_d[permutated_idx]
            max_d = max_d[permutated_idx]
            effective_deut_rent = effective_deut_rent[permutated_idx]

        dataset = {
            "time_points": time_points,
            "start": start,
            "end": end,
            "log_kex": log_kex,
            "log_kch": log_kch,
            "num_d": num_d,
            "max_d": max_d,
            "effective_deut_rent": effective_deut_rent,
            "back_ex": back_ex,
            "sequence": sequence,
            "isotope_envelope": isotope_envelope,
            "t0_isotope": t0_isotope,
            "saturation": saturation,
            "resolution_limits": resolution_limits,
            "resolution_grouping": resolution_grouping,
            "protein_sequence": data_dict["protein_sequence"],
            "temperature": data_dict["temperature"],
            "pH": data_dict["pH"],
            "proline_mask": torch.tensor([1 if res == "P" else 0 for res in data_dict["protein_sequence"]]),
        }

        if self.generate_miss_id:
            dataset = replace_peptides_with_miss_id(dataset, mz_threshold=0.1, group_replacement_chance=0.8)
        else:
            dataset["miss_id_bool"] = torch.zeros_like(dataset["time_points"])

        return dataset


class PFNetSimulatedData(SimulatedData):

    def __init__(
        self,
        length=100,
        seed=42,
        noise_level=0,
        saturation=1.0,
        random_backexchange=False,
        timepoints=None,
        drop_timepoints=True,
        temperature=293.0,
        pH=7.0,
        logP_range=(2, 12),
        iso_env_peaks=50,
    ):

        super().__init__()
        self.seed = seed
        random.seed(seed)
        self.length = length
        self.gen_seq()
        self.gen_logP(logP_range)
        self.cal_k_init()
        self.cal_k_ex()

        if timepoints is None:
            self.timepoints = generate_tps()
        else:
            self.timepoints = timepoints

        self.noise_level = noise_level

        self.saturation = saturation
        self.random_backexchange = random_backexchange
        self.drop_timepoints = drop_timepoints
        self.temperature = temperature
        self.pH = pH
        self.iso_env_peaks = iso_env_peaks
        
    def cal_k_ex(self):
        'calculate exchange rate for each residue'
        self.log_k_ex = self.log_k_init - self.logP
        self.log_k_ex[self.log_k_ex==np.inf] = 100


    def gen_peptides_simple(self, min_len=5, max_len=12, num_peptides=30):
        'generate random peptides from the protein sequence without complex restraints on the coverage'
        random.seed(self.seed)
        sequence_length = len(self.sequence)
        peptides = []
        
        # Generate peptides until we have the required number
        attempt = 0
        max_attempts = num_peptides * 100  # prevent infinite loop
        
        while len(peptides) < num_peptides and attempt < max_attempts:
            # Random start position
            start = random.randint(0, sequence_length - min_len)
            # Random length within range
            pep_len = random.randint(min_len, max_len)
            # Ensure we don't exceed sequence length
            end = min(start + pep_len, sequence_length)
            
            peptide = self.sequence[start:end]
            
            # Only add peptides with valid length and avoid duplicates
            if len(peptide) >= min_len and peptide not in peptides:
                peptides.append(peptide)
            
            attempt += 1
        
        # Sort peptides by their position in the sequence
        self.peptides = sorted(peptides, key=lambda x: self.sequence.find(x))

    def convert_to_hdxms_data(self, centroid_data=False):
        'convert the simulated data to a HDXMSData object'
        
        hdxms_data = HDXMSData("simulated_data", protein_sequence=self.sequence, saturation=self.saturation, temperature=self.temperature, pH=self.pH)
        protein_state = ProteinState("SIM", hdxms_data=hdxms_data)
        hdxms_data.add_state(protein_state)

        # calculate incorporation
        self.calculate_incorporation()
        
        back_ex_dict = {}
        
        for peptide in self.peptides:
            start = self.sequence.find(peptide) + 1
            end = start + len(peptide) - 1

            peptide_obj = Peptide(peptide, start, end, protein_state, n_fastamides=1)

            if self.random_backexchange:
                if peptide_obj.identifier in back_ex_dict:
                    true_back_exchange = back_ex_dict[peptide_obj.identifier]
                    noised_back_exchange = np.clip(true_back_exchange + np.random.normal(0.0, self.noise_level/5), 0.0, 0.8)
                    _add_max_d_to_pep(peptide_obj, max_d=(1-noised_back_exchange)*peptide_obj.theo_max_d)
                    peptide_obj.true_max_d = (1-true_back_exchange)*peptide_obj.theo_max_d
                else:
                    true_back_exchange = random.uniform(0.0, 0.8)
                    noised_back_exchange = np.clip(true_back_exchange + np.random.normal(0.0, self.noise_level/5), 0.0, 0.8)
                    _add_max_d_to_pep(peptide_obj, max_d=(1-noised_back_exchange)*peptide_obj.theo_max_d)
                    back_ex_dict[peptide_obj.identifier] = true_back_exchange
                    peptide_obj.true_max_d = (1-true_back_exchange)*peptide_obj.theo_max_d
            else:
                true_back_exchange = 0.0
  
            _, noised_deut_rent = estimate_deut_rent(peptide_obj, T=273.15, pH=2.3, quench_saturation=0.455, if_true_max_d=False)
            _, true_deut_rent = estimate_deut_rent(peptide_obj, T=273.15, pH=2.3, quench_saturation=0.455, if_true_max_d=True)
            peptide_obj.deut_rent = noised_deut_rent
            #peptide_obj.deut_rent = true_deut_rent

            try:
                protein_state.add_peptide(peptide_obj, allow_duplicate=True)
                
                # make sure 0 timepoint is included and is the first timepoint
                if self.drop_timepoints:
                    timepoints = self.timepoints.copy()
                    timepoints = random.sample(list(timepoints), k=int(0.8*len(timepoints)))
                else:
                    timepoints = self.timepoints.copy()
                
                if 0 not in timepoints:
                    timepoints = np.insert(timepoints, 0, 0)
                    
                # sort timepoints
                timepoints.sort()
                
                noise_multiplier = 1.0
                # pep_length = peptide_obj.end - peptide_obj.start + 1
                # if (pep_length <= 5 or pep_length >= 15) and random.uniform(0, 1) < 0.3:
                #     noise_multiplier = 1.1
                            
                for tp_i, tp in enumerate(timepoints):
                    # tp_raw_deut = self.incorporations[
                    #     peptide_obj.start - 1 : peptide_obj.end
                    # ][:, tp_i]
                    #tp_raw_deut = self.incorporations[tp][peptide_obj.start - 1 : peptide_obj.end]
                    tp_raw_deut = self.incorporations[tp][peptide_obj.start - 1 - peptide_obj.n_fastamides: peptide_obj.end]
                    
                    #add sidechain exchange noise, 5% of theo_max_d, only positive
                    if tp_i != 0 and random.uniform(0, 1) < 0.3:
                        tp_raw_deut = tp_raw_deut + random.uniform(0, self.noise_level/10) 
                    
                    #tp_raw_deut = tp_raw_deut * (1 - true_back_exchange) * self.saturation
                    # print(tp_raw_deut.shape, peptide_obj.deut_rent.shape)
                    tp_raw_deut = tp_raw_deut * true_deut_rent * self.saturation
                    pep_incorp = sum(tp_raw_deut)
                    # random_stddev = peptide_obj.theo_max_d * self.noise_level * random.uniform(-1, 1)
                    random_stddev = 0
                    tp_obj = Timepoint(
                        peptide_obj,
                        tp,
                        pep_incorp + random_stddev,
                        random_stddev,
                    )
                    
                    if centroid_data:
                        tp_obj.isotope_envelope = np.zeros(30)
                        peptide_obj.add_timepoint(tp_obj)
                        continue
                    
                    if tp == 0.0:
                        t0_theo = get_theoretical_isotope_distribution(tp_obj)[1]
                        
                    p_D = event_probabilities(tp_raw_deut)

                    isotope_envelope = np.convolve(t0_theo, p_D)
                    # isotope_noise = np.array(
                    #     [
                    #         random.uniform(-1, 1) * self.noise_level * peak
                    #         for peak in isotope_envelope
                    #     ]
                    # )
                    isotope_noise = np.array(
                        [
                            random.uniform(-1, 1) * self.noise_level/10 +  random.uniform(-1, 1) * self.noise_level * peak 
                            for peak in isotope_envelope
                        ]
                    )
                    
                    if tp_i != 0:
                        isotope_noise += np.array([random.uniform(-1, 1) * self.noise_level/10 * (np.log10(tp)/18) for _ in isotope_envelope])
                    
                    isotope_envelope = isotope_envelope + isotope_noise * noise_multiplier
                    isotope_envelope[isotope_envelope < 0] = 0
                    isotope_envelope = normlize(isotope_envelope)
                    
                    tp_obj.isotope_envelope = isotope_envelope
                     
                    # update num_d
                    mass_num = np.arange(len(isotope_envelope))
                    if tp_i == 0:
                        t0_centroid = np.sum(isotope_envelope * mass_num)

                    tp_obj.num_d = np.sum(isotope_envelope * mass_num) - t0_centroid

                    peptide_obj.add_timepoint(tp_obj)

            except Exception as e:
                print(e)
                continue
            
        #print(back_ex_dict)

        self.hdxms_data = hdxms_data

        # peptide_obj.add_timepoint(tp, self.incorporations[start:end])

class UnionFind:

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx = self.find(x)
        ry = self.find(y)
        if rx != ry:
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1


def custom_collate_fn(batch):
    # Separate the batch into individual components
    pad_value = -100

    aa_encoder = AminoAcidEncoder()
    sequences = [aa_encoder.encode(sequences=item["sequence"]) for item in batch]

    # Extract components from batch
    time_points_batch = [item["time_points"] for item in batch]
    start_batch = [item["start"] for item in batch]
    end_batch = [item["end"] for item in batch]
    log_kex_batch = [item["log_kex"] for item in batch]
    log_kch_batch = [item["log_kch"] for item in batch]
    num_d_batch = [item["num_d"] for item in batch]
    max_d_batch = [item["max_d"] for item in batch]
    isotope_envelope_batch = [item["isotope_envelope"] for item in batch]
    t0_isotope_batch = [item["t0_isotope"] for item in batch]
    saturation_batch = [item["saturation"] for item in batch]
    back_ex_batch = [item["back_ex"] for item in batch]
    proline_mask_batch = [item["proline_mask"] for item in batch]
    resolution_grouping_batch = [item["resolution_grouping"] for item in batch]
    resolution_limits_batch = [item["resolution_limits"] for item in batch]
    miss_id_bool_batch = [item["miss_id_bool"] for item in batch]
    effective_deut_rent_batch = [item["effective_deut_rent"] for item in batch]

    # length_batch = [item['length'] for item in batch]

    # Pad sequences
    time_points_padded = pad_sequence(time_points_batch, batch_first=True, padding_value=pad_value)
    start_padded = pad_sequence(start_batch, batch_first=True, padding_value=pad_value)
    end_padded = pad_sequence(end_batch, batch_first=True, padding_value=pad_value)
    sequence_padded = pad_sequence(sequences, batch_first=True, padding_value=pad_value)
    log_kex_padded = pad_sequence(log_kex_batch, batch_first=True, padding_value=pad_value)
    log_kch_padded = pad_sequence(log_kch_batch, batch_first=True, padding_value=pad_value)
    num_d_padded = pad_sequence(num_d_batch, batch_first=True, padding_value=pad_value)
    max_d_padded = pad_sequence(max_d_batch, batch_first=True, padding_value=pad_value)
    isotope_envelope_padded = pad_sequence(isotope_envelope_batch, batch_first=True, padding_value=pad_value)
    t0_isotope_padded = pad_sequence(t0_isotope_batch, batch_first=True, padding_value=pad_value)
    back_ex_padded = pad_sequence(back_ex_batch, batch_first=True, padding_value=pad_value)
    effective_deut_rent_padded = pad_sequence(effective_deut_rent_batch, batch_first=True, padding_value=pad_value)
    resolution_grouping_padded = pad_sequence(resolution_grouping_batch, batch_first=True, padding_value=pad_value)
    proline_mask_padded = pad_sequence(proline_mask_batch, batch_first=True, padding_value=pad_value)
    resolution_limits_padded = pad_sequence(resolution_limits_batch, batch_first=True, padding_value=pad_value)
    miss_id_bool_padded = pad_sequence(miss_id_bool_batch, batch_first=True, padding_value=pad_value)
    # length_padded = pad_sequence(length_batch, batch_first=True, padding_value=pad_value)

    # Stack saturation since it's fixed size
    saturation_stacked = torch.stack(saturation_batch)

    peptide_data = {
        "time_point": time_points_padded,
        "start_pos": start_padded,
        "end_pos": end_padded,
        "sequence": sequence_padded,
        "num_d": num_d_padded,
        "max_d": max_d_padded,
        "probabilities": isotope_envelope_padded,
        "t0_isotope": t0_isotope_padded,
        "back_ex": back_ex_padded,
        "effective_deut_rent": effective_deut_rent_padded,
        #    'theo_max_d': theo_max_d_padded,
        "miss_id_bool": miss_id_bool_padded,
    }

    residue_data = {
        "resolution_grouping": resolution_grouping_padded,
        "resolution_limits": resolution_limits_padded,
        "proline_mask": proline_mask_padded,
        "log_kch": log_kch_padded,
    }

    # Create global vars tensor with sequence length and saturation
    global_vars = torch.stack([torch.ones_like(saturation_stacked) * log_kex_padded.shape[1], saturation_stacked], dim=1)

    return peptide_data, residue_data, global_vars, log_kex_padded


@contextmanager
def timeout(seconds):

    def signal_handler(signum, frame):
        raise ValueError("Timed out!")

    # Set the signal handler and a timeout
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)


def generate_tps(min_num=6, max_num=12, lower_range=(0, 2), upper_range=(4, 6), noise_fraction=0.1):
    while True:
        num_tps = random.randint(min_num, max_num)
        lower_bound = random.uniform(*lower_range)
        upper_bound = random.uniform(*upper_range)

        if upper_bound <= lower_bound:
            continue  # ensure upper_bound is greater than lower_bound

        # generate uniformly distributed timepoints
        tps = np.linspace(lower_bound, upper_bound, num=num_tps)

        # calculate step size and add noise
        step = (upper_bound - lower_bound) / (num_tps - 1)
        noise = np.random.uniform(-noise_fraction * step, noise_fraction * step, size=num_tps)
        tps += noise

        # ensure timepoints are within bounds and sorted
        tps = np.clip(tps, lower_bound, upper_bound)
        tps.sort()

        if len(tps) >= min_num:
            break

    tps = np.insert(10**tps, 0, 0)

    return tps


def get_resolution_grouping(input_data):

    def if_in_segment(resid, segment_tuple):
        if resid >= segment_tuple[0] and resid <= segment_tuple[1]:
            return 1
        else:
            return 0

    from pigeon_feather.analysis import Analysis
    from pigeon_feather.data import HDXMSData

    if isinstance(input_data, PFNetSimulatedData):
        states = [state for state in input_data.hdxms_data.states]
    elif isinstance(input_data, HDXMSData):
        states = [state for state in input_data.states]
    else:
        raise ValueError("Invalid input type")

    analysis = Analysis(states, temperature=293.0, pH=7.0)
    analysis.maximum_resolution_limits

    binary_groupings = []

    for res_idx, res in enumerate(analysis.protein_sequence):

        resid = res_idx + 1

        if resid not in analysis.maximum_resolution_limits.keys():

            # binary_grouping = [if_empty, if_NC_term, if_in_before, if_in_after]
            binary_grouping = [1, 0, 0, 0]

        elif resid - 1 not in analysis.maximum_resolution_limits.keys() or resid + 1 not in analysis.maximum_resolution_limits.keys():

            # binary_grouping = [if_empty, if_NC_term, if_in_before, if_in_after]
            binary_grouping = [0, 1, 0, 0]

        else:

            before_res = analysis.maximum_resolution_limits[resid - 1]
            # current_segment = analysis.maximum_resolution_limits[resid]
            after_res = analysis.maximum_resolution_limits[resid + 1]

            binary_grouping = [0, 0, if_in_segment(resid, before_res), if_in_segment(resid, after_res)]

        binary_groupings.append(binary_grouping)

    resolution_limits = sorted(list(set(analysis.maximum_resolution_limits.values())))

    return resolution_limits, binary_groupings


from hdxrate import k_int_from_sequence
from pigeon_feather.tools import calculate_simple_deuterium_incorporation
import numpy as np
from scipy.optimize import brentq 
from functools import lru_cache  

@lru_cache(maxsize=500) 
def get_full_theo_max_d(sequence, saturation):
    'theoretical max deuterium incorporated in the peptide'
    num_prolines = sequence.count("P")
    theo_max_d = len(sequence) - num_prolines
    return theo_max_d * saturation


def sim_data_to_grouped_dict(simulated_data):
    data_dict = {}
    all_peptides = [peptide for state in simulated_data.hdxms_data.states for peptide in state.peptides]
    all_peptides.sort(key=lambda x: (x.start, x.end))
    iso_env_peaks = simulated_data.iso_env_peaks

    for i, peptide in enumerate(all_peptides):

        pep_dict = {}

        #pep_dict["start"] = peptide.start
        pep_dict["start"] = peptide.start - peptide.n_fastamides
        pep_dict["end"] = peptide.end
        # pep_dict["sequence"] = peptide.sequence
        pep_dict["sequence"] = peptide.identifier.split(' ')[1]
        pep_dict["time"] = [tp.deut_time for tp in peptide.timepoints if tp.deut_time != np.inf]
        pep_dict["num_d"] = [tp.num_d for tp in peptide.timepoints if tp.deut_time != np.inf]
        pep_dict["isotope"] = [custom_pad(tp.isotope_envelope[:iso_env_peaks], iso_env_peaks) for tp in peptide.timepoints if tp.deut_time != np.inf]
        # t0 isotope envelope
        pep_dict["t0_isotope"] = [custom_pad(peptide.get_timepoint(0).isotope_envelope[:iso_env_peaks], iso_env_peaks) for tp in peptide.timepoints if tp.deut_time != np.inf]
        pep_dict["max_d"] = peptide.max_d
        pep_dict["theo_max_d"] = peptide.theo_max_d
        
        # estimate the deuterium retention based on the instrisic rate
        # backex_tp, deut_rent = estimate_deut_rent(peptide, T=273.15, pH=2.3, quench_saturation=0.455)
        # pep_dict["effective_deut_rent"] = custom_pad(deut_rent[:50], 50)
        pep_dict["effective_deut_rent"] = custom_pad(peptide.deut_rent[:50], 50)    

        data_dict[f"peptide_{i}"] = pep_dict

    # protein features
    data_dict["num_peptides"] = len(all_peptides)
    data_dict["protein_sequence"] = simulated_data.sequence
    data_dict["temperature"] = simulated_data.temperature
    data_dict["pH"] = simulated_data.pH
    data_dict["log_kch"] = simulated_data.log_k_init
    data_dict["log_kex"] = simulated_data.log_k_ex
    data_dict["log_P"] = simulated_data.logP
    data_dict["saturation"] = simulated_data.saturation
    data_dict["resolution_limits"], data_dict["resolution_grouping"] = get_resolution_grouping(simulated_data)

    # add Proline mask
    # data_dict["proline_mask"] = torch.tensor([1 if res == "P" else 0 for res in simulated_data.sequence])

    return data_dict


def generate_single_data(
    i, base_seed=42, length=None, length_range=(100, 400), num_peptides=None, tps=None, logP_range=(2, 12), noise_level=None, output_dir=None, centroid_data=False, prefix=""
):

    if output_dir is not None:
        output_path = os.path.join(output_dir, f"simdata{f'_{prefix}' if prefix else ''}_{i}.pt")
        if os.path.exists(output_path) and os.path.getsize(output_path) != 0:
            return output_path

    # set random seed
    random.seed(base_seed**5 + i)
    np.random.seed(base_seed**5 + i)
    torch.manual_seed(base_seed**5 + i)

    if tps is None:

        random_float = random.random()
        if random_float < 0.6:
            # normal tps
            tps = generate_tps(min_num=6, max_num=12, lower_range=(0, 2), upper_range=(4, 6))
        elif 0.60 <= random_float < 0.70:
            # full range tps
            tps = generate_tps(min_num=12, max_num=18, lower_range=(-6, -4), upper_range=(16, 18))
        elif 0.70 <= random_float < 0.85:
            # short tps
            tps = generate_tps(min_num=12, max_num=14, lower_range=(-6, -4), upper_range=(4, 6))
        elif 0.85 <= random_float < 1.00:
            # long tps
            tps = generate_tps(min_num=12, max_num=14, lower_range=(0, 2), upper_range=(16, 18))
        #tps = generate_tps(min_num=12, max_num=18, lower_range=(-6, -4), upper_range=(16, 18))

        # tps = generate_tps(min_num=6, max_num=12, lower_range=(0,2), upper_range=(4,6), noise_fraction=0.1)

    if length is None:
       #length = random.randint(*length_range)
       length = random.randint(20, 100)
    if num_peptides is None:
        num_peptides = int(random.uniform(0.7, 1.3) * length)
    if noise_level is None:
        noise_level = random.uniform(0.1, 0.3)
    saturation = random.uniform(0.7, 1.0)
    random_backexchange = True
    temperature = random.uniform(283.15, 308.15)  # 10-35 C
    pH = random.uniform(6.5, 8.5)  # 6.5-8.5

    simulated_data = PFNetSimulatedData(
        length=length,
        seed=i,
        noise_level=noise_level,
        saturation=saturation,
        random_backexchange=random_backexchange,
        timepoints=tps,
        temperature=temperature,
        pH=pH,
        logP_range=logP_range,
        iso_env_peaks=50,
    )

    random_float = random.random()
    if random_float < 0.8:
        max_len = random.randint(15, 30)
    else:
        max_len = random.randint(30, 48)
    max_len = 15
    simulated_data.gen_peptides(num_peptides=num_peptides, min_len=3, max_len=max_len)
    # simulated_data.gen_peptides_simple(num_peptides=num_peptides, min_len=3, max_len=max_len)
    simulated_data.convert_to_hdxms_data()

    data_dict = sim_data_to_grouped_dict(simulated_data)

    if output_dir is not None:
        torch.save(data_dict, output_path)

    return data_dict


def collapse_groups_with_threshold_union_find(peptides, values, threshold):
    """
    Given a list of peptides and their values, use union-find to cluster peptides
    within the threshold, then return only those groups with more than one peptide
    and within the threshold.
    """
    # Sort peptides by mass
    sorted_peptides = sorted(peptides, key=lambda p: values[p])
    masses = [values[p] for p in sorted_peptides]

    n = len(peptides)
    uf = UnionFind(n)

    # Use a sliding window approach to find neighbors within mz_threshold
    left = 0
    for right in range(n):
        # Move left pointer until mass difference is within threshold
        while masses[right] - masses[left] > threshold:
            left += 1
        # Union all within threshold
        for i in range(left, right):
            if masses[right] - masses[i] <= threshold:
                uf.union(right, i)

    # Extract groups
    groups_map = {}
    for i in range(n):
        root = uf.find(i)
        if root not in groups_map:
            groups_map[root] = []
        groups_map[root].append(sorted_peptides[i])

    # Filter groups by threshold and size
    final_groups = []
    for grp in groups_map.values():
        if len(grp) > 1:
            grp_masses = [values[g] for g in grp]
            if max(grp_masses) - min(grp_masses) <= threshold:
                final_groups.append(grp)
    return final_groups


def get_similar_mass_groups(protein_dataset, mz_threshold, group_replacement_chance):
    peptides = protein_dataset["sequence"]
    unique_peptides = list(set(peptides))

    # Precompute masses
    peptides_mz_dic = {}
    for peptide in unique_peptides:
        peptide_obj = oms.AASequence.fromString(peptide)
        mono_mz = peptide_obj.getMonoWeight()
        peptides_mz_dic[peptide] = mono_mz

    collapsed_groups = collapse_groups_with_threshold_union_find(unique_peptides, peptides_mz_dic, mz_threshold)

    # Randomly keep some groups
    groups_to_keep_after_chance = [g for g in collapsed_groups if random.random() <= group_replacement_chance]
    return groups_to_keep_after_chance


def replace_peptides_with_miss_id(protein_dataset, mz_threshold, group_replacement_chance):

    mimic_dataset = dict(protein_dataset)

    groups_to_keep_after_chance = get_similar_mass_groups(protein_dataset, mz_threshold, group_replacement_chance)

    # Convert lists to arrays if large for faster lookups (optional)
    seq = mimic_dataset["sequence"]
    tp = mimic_dataset["time_points"]
    iso_env = mimic_dataset["isotope_envelope"]
    back_ex = mimic_dataset["back_ex"]
    deut_rent = mimic_dataset["effective_deut_rent"]
    miss_id_bool = torch.zeros_like(tp)
    
    peptide_to_indices = defaultdict(list)
    for i, peptide_seq in enumerate(protein_dataset["sequence"]):
        peptide_to_indices[peptide_seq].append(i)
    

    for group in groups_to_keep_after_chance:
        mimic = group[random.randint(0, len(group) - 1)]
        # Find all entries for the mimic peptide
        #indexs_of_mimic = [i for i, x in enumerate(seq) if x == mimic]
        indexs_of_mimic = peptide_to_indices[mimic]

        # Collect timepoints, probabilities, back_ex
        unique_tp = {}
        for idx in indexs_of_mimic:
            # Store the first occurrence of a timepoint
            # If time_point is repeated, it should map to the same iso_env/back_ex
            if tp[idx] not in unique_tp:
                unique_tp[float(tp[idx])] = (iso_env[idx], back_ex[idx], deut_rent[idx])

        # For the other peptides in the group
        for not_mimic in [x for x in group if x != mimic]:
            #indexs_of_not_mimic = [i for i, x in enumerate(seq) if x == not_mimic]
            indexs_of_not_mimic = peptide_to_indices[not_mimic]
            for idx in indexs_of_not_mimic:
                if float(tp[idx]) in unique_tp:
                    iso_env[idx], back_ex[idx], deut_rent[idx] = unique_tp[float(tp[idx])]
                    miss_id_bool[idx] = 1
                else:
                    valid_tps = [t for t in unique_tp.keys() if t != 0]
                    if not valid_tps:
                        valid_tps = list(unique_tp.keys())
                    chosen_tp = random.choice(valid_tps)
                    iso_env[idx], back_ex[idx], deut_rent[idx] = unique_tp[chosen_tp]
                    tp[idx] = chosen_tp
                    miss_id_bool[idx] = 1

    # Assign arrays back to mimic_dataset if needed
    mimic_dataset["sequence"] = seq
    mimic_dataset["time_points"] = tp
    mimic_dataset["isotope_envelope"] = iso_env
    mimic_dataset["back_ex"] = back_ex
    mimic_dataset["miss_id_bool"] = miss_id_bool
    mimic_dataset["effective_deut_rent"] = deut_rent
    return mimic_dataset


def estimate_deut_rent(peptide, T=273.15, pH=2.3, quench_saturation=1, bounds=(0, 1e8), tol=0.1, if_true_max_d=True):
    '''
    estimate the deuterium retention based on the instrisic rate
    input: peptide, a Peptide object
    output: tp, deut_rent, a tuple of the time point and the deuterium retention
    '''
    # full seq
        # saturation in exchange reaction
    saturation = peptide.protein_state.hdxms_data.saturation
        
    raw_seq = peptide.identifier.split(' ')[1]
    res_duet = np.array([1 if aa != 'P' else 0 for aa in raw_seq])

    if if_true_max_d:
        max_d = peptide.true_max_d 
    else:
        max_d = peptide.max_d 

    kk = k_int_from_sequence(raw_seq, T, pH, exchange_type='DH', d_percentage=quench_saturation*100)
    kk[kk==np.inf] = 1e10
    
    def f(tt, res_duet):
        deut_rent = np.exp(-kk * tt)
        calc_max_d = (deut_rent*res_duet).sum() * saturation
        return calc_max_d - max_d

    backex_tp = brentq(f, *bounds, xtol=tol, args=(res_duet))
    deut_rent = np.exp(-kk * backex_tp)
    deut_rent[res_duet==0] = 0.0

    return backex_tp, deut_rent
