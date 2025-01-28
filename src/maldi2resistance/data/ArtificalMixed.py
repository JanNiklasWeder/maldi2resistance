import math
import random
from typing import Union, Optional

import numpy
import torch
from torch.utils.data import Dataset, Subset
from tqdm.auto import tqdm

from maldi2resistance.data.driams import Driams


class ArtificialMixedInfection (Dataset):
    real_cross_infections =[
        ("Staphylococcus epidermidis", "Staphylococcus hominis"),
        ("Escherichia coli", "Escherichia coli"),
        ("Staphylococcus capitis", "Staphylococcus aureus"),
        ("Staphylococcus aureus", "Enterococcus faecium"),
        ("Enterococcus faecium", "Staphylococcus haemolyticus"),
        ("Enterococcus faecium", "Streptococcus mitis"),
        ("Staphylococcus haemolyticus", "Streptococcus mitis"),
        ("Enterococcus faecalis", "Enterococcus faecalis"),
        ("Klebsiella pneumoniae", "Klebsiella pneumoniae"),
        ("Staphylococcus caprae", "Streptococcus salivarius"),
        ("Proteus mirabilis", "Streptococcus dysgalactiae"),
        ("Streptococcus parasanguinis", "Streptococcus pneumoniae"),
        ("Streptococcus parasanguinis", "Cutibacterium acnes"),
        ("Streptococcus pneumoniae", "Cutibacterium acnes"),
        ("Klebsiella oxytoca", "Staphylococcus warneri"),
        ("Pseudomonas aeruginosa", "Staphylococcus pettenkoferi"),
        ("Streptococcus anginosus", "Streptococcus constellatus"),
        ("Candida albicans", "Corynebacterium striatum"),
        ("Streptococcus pyogenes", "Micrococcus luteus"),
        ("Bacteroides fragilis", "Enterobacter cloacae"),
        ("Streptococcus agalactiae", "Staphylococcus lugdunensis"),
        ("Staphylococcus cohnii", "Staphylococcus petrasii"),
    ]

    def __init__(self, dataset: Union[Driams, Subset],
                        generator_seed = None,
                        use_percent_of_data: Optional[float] = None,
                        mix_ratio: Optional[float] = 0.5):
        """

        :param dataset:
        :param generator_seed:
        :param use_percent_of_data: Define this number to use random mixtures 
            that are not based on real observations. According to the specified
            percentage in comma notation, two random splits are created from
            the original data set and combined at random.
            Thus, when using 0.4, 40% of the original data is mixed,
            i.e. two splits of 20% each are merged. This means that a maximum
            of one dataset containing 50% of the data points of the original dataset can be created.
        """
        generator = numpy.random.default_rng(seed=generator_seed)
        gen = torch.Generator()
        self.data = []

        if isinstance(dataset, Subset):
            indices = dataset.indices
            driams = dataset.dataset
        else:
            indices = None
            driams = dataset
            
        if use_percent_of_data is not None:
            assert 0 < use_percent_of_data <= 1
            n_datapoints = math.floor(use_percent_of_data * len(dataset) / 2)
            left_datapoints = len(dataset) - n_datapoints - n_datapoints

            split_1, split_2, _ = torch.utils.data.random_split(dataset, [n_datapoints, n_datapoints, left_datapoints], generator= None if generator_seed is None else gen.manual_seed(generator_seed))

        for first, second in self.real_cross_infections:
            first_spectra = driams.filter4species(indices=indices, species=[first])
            second_spectra = driams.filter4species(indices=indices, species=[second])

            # we run the loop once if we combine randomly instead of for each real combination
            if use_percent_of_data is not None:
                first_spectra = split_1
                second_spectra = split_2

            used_first = set()
            used_second = set()

            with tqdm(total=min([len(first_spectra), len(second_spectra)]), leave= False) as pbar:
                while len(first_spectra) != len(used_first) and len(second_spectra) != len(used_second):
                    idx_first = generator.integers(0, len(first_spectra))
                    idx_second= generator.integers(0, len(second_spectra))

                    while idx_first in used_first:
                        idx_first +=1
                        if idx_first >= len(first_spectra):
                            idx_first = 0

                    used_first.add(idx_first)

                    while idx_second in used_second:
                        idx_second +=1
                        if idx_second >= len(second_spectra):
                            idx_second = 0

                    used_second.add(idx_second)

                    spectrum_first, label_first = first_spectra[idx_first]
                    spectrum_second, label_second = second_spectra[idx_second]

                    combined_spectrum = torch.add(mix_ratio * spectrum_first, (1-mix_ratio) * spectrum_second)
                    #combined_spectrum = torch.div(combined_spectrum, 2)

                    combined_label = torch.add(label_first, label_second)
                    combined_label = torch.div(combined_label, 2)
                    combined_label[combined_label == 0.5] = 1

                    self.data.append(
                        (combined_spectrum, combined_label)
                    )
                    pbar.update(1)
                pbar.close()

            if use_percent_of_data is not None:
                break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
