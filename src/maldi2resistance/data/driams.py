import warnings
from pathlib import Path
from typing import Union, Optional, Literal, List, Tuple, Iterator, Dict

import numpy as np
import polars
import os
import pandas as pd
import torch


from maldi_learn.vectorization import BinningVectorizer
from polars.type_aliases import SchemaDefinition
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import Dataset, Subset, ConcatDataset
from tqdm.auto import tqdm

from maldi2resistance.data.AntibioticFingerprint import FingerprintLookup


class Driams(Dataset):
    def __init__(
        self,
        root_dir: Union[str, Path],
        antibiotics: Union[list[str], set[str], None] = None,
        years: list[Literal[2015, 2016, 2017, 2018]] = [2015, 2016, 2017, 2018],
        sites: list[Literal["DRIAMS-A", "DRIAMS-B", "DRIAMS-C", "DRIAMS-D"]] = [
            "DRIAMS-A",
            "DRIAMS-B",
            "DRIAMS-C",
            "DRIAMS-D",
        ],
        remove_nan_how: Optional[Literal["any", "all"]] = "all",
        cutoff_value: Optional[int] = 1000,
        cutoff_value_positive: Optional[int] = 200,
        cutoff_value_negative: Optional[int] = 200,
        label_includes_species=False,
        returned_bins: Optional[Tuple[int, int]] = None,
        replace_drug_names: Optional[Dict[str, str]] = None,
        bin_size:int = 1,
        leave_progress_bar = False,
    ):
        """

        :param root_dir:
        :param antibiotics:
        :param years:
        :param sites:
        :param remove_nan_how:
        :param cutoff_value:
        :param cutoff_value_positive:
        :param cutoff_value_negative:
        :param label_includes_species:
        :param returned_bins: Parameter der festlegt, welcher bereich des Spektrums zurückgegeben wird.
            Der angegebene bereich wird auf die Bins angewendet, nicht auf die Tatsächliche m/z range.
            (0,3) sorgt dafür, dass nur die ersten drei Bins zurückgegeben werden.
        """

        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        self.__root_dir = root_dir
        self.__meta = {}
        self.__species = []
        self.__keys = []
        self.__idx_table = {}
        self.__n_data_points = 0
        self.__loading_type: Literal["memory", "file"] = "memory"
        # self.loading_type = self.__loading_type
        self.__returned_bins = returned_bins
        self.__leave_progress_bar = leave_progress_bar

        self.label_includes_species = label_includes_species

        for site in sites:
            for year in years:
                try:
                    key = (site, year)
                    self.__meta[key] = pd.read_csv(
                        root_dir / f"{site}/id/{year}/{year}_clean.csv",
                        low_memory= False,
                    )
                    self.__keys.append(key)
                except FileNotFoundError as e:
                    warnings.warn(
                        message=f"Metafile for Site ({site}) and Year ({year}) combination not found!"
                    )
                    continue


        selected_antibiotics: set
        if antibiotics is None:
            selected_antibiotics = set()

            for key in self.__keys:
                meta = self.__meta[key]

                # use every antibiotic

                meta_info_header = [
                    "",
                    "Unnamed: 0.1",
                    "code",
                    "species",
                    "laboratory_species",
                    "combined_code",
                    "genus",
                    "Unnamed: 0",
                ]
                meta_infos = meta.columns.isin(meta_info_header)

                selected_antibiotics.update(meta.loc[:, ~meta_infos].columns.to_list())
        else:

            cutoff_value = None
            cutoff_value_negative = None
            cutoff_value_positive = None

            if isinstance(antibiotics, list):
                selected_antibiotics = set(antibiotics)

            else:
                selected_antibiotics = antibiotics

        if replace_drug_names is not None:
            selected_antibiotics = selected_antibiotics.union(
                set(replace_drug_names.keys())
            ).union(set(replace_drug_names.values()))

        self.__selected_antibiotics: List[str] = sorted(selected_antibiotics)

        species_set = set()
        added_antibiotics = set()
        removed_antibiotics = set()
        for key in self.__keys:
            meta = self.__meta[key]

            selected_columns = meta.loc[
                :, meta.columns.isin(self.selected_antibiotics)
            ].copy()
            allowed_values = selected_columns.isin(["S", "I", "R", "-"])
            selected_columns[~allowed_values] = float("nan")

            replace_dict = {"-": float("nan"), "": float("nan"), "R": 1, "I": 1, "S": 0}
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=FutureWarning)
                selected_columns.replace(replace_dict, inplace=True)
            selected_columns.infer_objects(copy=False)

            meta.loc[:, meta.columns.isin(self.selected_antibiotics)] = selected_columns

            for antibiotic in self.selected_antibiotics:
                if antibiotic not in meta.columns:
                    meta[antibiotic] = float("nan")

            if replace_drug_names is not None:
                for old, new in tqdm(
                    replace_drug_names.items(), desc=f"Merging Antibiotics for {key}",
                    leave= self.__leave_progress_bar,
                ):
                    # nothing to move over
                    if old not in meta.columns:
                        continue

                    # rename old column to new if new doesnt exist but old does
                    if new not in meta.columns:
                        meta.rename(columns={old: new}, inplace=True)
                        if (
                            old in self.__selected_antibiotics
                            and new not in self.__selected_antibiotics
                        ):
                            added_antibiotics.add(new)
                        continue

                    # else merge both columns row wise
                    for index, row in meta.iterrows():
                        # if in target is nan copy value from old column without checking
                        if np.isnan(row[new]):
                            meta.loc[index, new] = row[old]
                            continue

                        # if they are equal target column has already correct value
                        if row[old] == row[new]:
                            continue

                        # if they are unequal target column is set to nan/ value removed
                        if row[old] != row[new]:
                            meta.loc[index, new] = float("NaN")

                    removed_antibiotics.add(old)
                    meta.drop(old, axis=1, inplace=True)

            if remove_nan_how is not None:
                available_antibiotics = [
                    antibiotic
                    for antibiotic in self.__selected_antibiotics
                    if antibiotic in meta.columns
                ]
                meta.dropna(
                    inplace=True,
                    ignore_index=True,
                    how=remove_nan_how,
                    subset=available_antibiotics,
                )

            length = len(meta.index)
            self.__idx_table[(self.__n_data_points, self.__n_data_points + length)] = (
                key
            )
            self.__n_data_points += length
            species_set.update(meta["species"])

        for antibiotic in removed_antibiotics:
            self.__selected_antibiotics.remove(antibiotic)

        self.__selected_antibiotics = sorted(
            set(self.__selected_antibiotics).union(added_antibiotics)
        )

        # UMG data has nan that is interpreted as float => cast to string
        species_set = set([str(x) for x in species_set])
        self.__species = sorted(species_set)

        # binning !make sure to end up with ints to prevent issues
        min_bin = 2000
        max_bin = 20000
        n_potential_bins = (max_bin - min_bin)
        n_bins = n_potential_bins / bin_size
        if n_potential_bins % bin_size != 0:
            raise ValueError(f"A bin size of {bin_size} does not result in an integer for the number "
                             f"of bins ({n_bins}). Please adjust the bin size accordingly.")
        n_bins = int(n_bins)

        self.__n_bins = n_bins
        self.__bin_size = bin_size
        self.transform = BinningVectorizer(n_bins, min_bin=min_bin, max_bin=max_bin)

        self.__spectrum_polars_schema: SchemaDefinition = {
            "mass.spectra.": float,
            "intensity.spectra.": float,
        }

        # create stats for positive and negative classes for each antibiotic
        stat_intermediate = {}

        for antibiotic in self.selected_antibiotics[:]:
            series_list = []
            for key in self.__keys:
                try:
                    series_list.append(self.__meta[key][antibiotic])
                except KeyError:
                    continue

            series = pd.concat(series_list)
            positive = (series == 1).sum()
            negative = (series == 0).sum()
            n_sum = series.count()

            if cutoff_value is not None and n_sum < cutoff_value:
                self.__selected_antibiotics.remove(antibiotic)
                continue

            if cutoff_value_negative is not None and negative < cutoff_value_negative:
                self.__selected_antibiotics.remove(antibiotic)
                continue

            if cutoff_value_positive is not None and positive < cutoff_value_positive:
                self.__selected_antibiotics.remove(antibiotic)
                continue

            stat_intermediate[antibiotic] = {
                "positive": positive,
                "negative": negative,
                "n_sum": n_sum,
            }

        self.__label_stats = pd.DataFrame(stat_intermediate)

        if self.label_includes_species:
            self.lb = LabelBinarizer()
            self.lb.fit(self.species)
            self.species_hash = {}

            for species in self.species:
                species_one_hot = torch.from_numpy(self.lb.transform([species]))
                species_one_hot = species_one_hot.squeeze()

                self.species_hash[species] = species_one_hot

    def get_stats(self, subset:Subset = None):
        all_labels = []

        if subset is None:
            subset = self

        for data in tqdm(subset, desc="Calculating stats", leave=False):
            if self.label_includes_species:
                 _, y, _ = data
            else:
                _, y = data

            all_labels.append(y)

        all_labels = torch.stack(all_labels)
        stat_intermediate = {}

        for pos, antibiotic in enumerate(self.selected_antibiotics):
            labels = all_labels[:, pos]
            labels = torch.nan_to_num(labels, nan=2)
            frequency = torch.bincount(labels.int(), minlength=2)
            stat_intermediate[antibiotic] = {
                "positive": frequency[1].item(),
                "negative": frequency[0].item(),
                "n_sum": frequency[0].item() + frequency[1].item(),
            }
        return pd.DataFrame(stat_intermediate)


    def get_meta_data(
        self,
        site: Literal["DRIAMS-A", "DRIAMS-B", "DRIAMS-C", "DRIAMS-D"],
        year: Literal[2015, 2016, 2017, 2018],
    ):
        return self.__meta[(site, year)]

    def __len__(self):
        return self.__n_data_points

    def __getitem__(self, idx):
        if idx >= self.__n_data_points:
            raise IndexError

        site_key = None
        for (lower, upper), value in self.__idx_table.items():
            if lower <= idx < upper:
                site_key = value
                local_idx = idx - lower
                break

        meta = self.__meta[site_key]
        site, year = site_key

        code, species = meta.loc[local_idx, ["code", "species"]]

        tensor = self.__use_for_loading(site, year, code)

        labels = meta.loc[local_idx, self.selected_antibiotics]
        labels = labels.astype("float32")
        labels = torch.from_numpy(labels.values)

        if self.label_includes_species:
            species_one_hot = self.species_hash[species]

            return tensor, labels, species_one_hot

        return tensor, labels

    @property
    def n_bins(self) -> int:
        return self.__n_bins
    @property
    def selected_antibiotics(self) -> List[str]:
        return self.__selected_antibiotics

    @property
    def species(self) -> List[str]:
        return self.__species

    @property
    def label_stats(self) -> pd.DataFrame:
        return self.__label_stats

    @property
    def loading_type(self) -> Literal["memory", "file"]:
        return self.__loading_type

    @loading_type.setter
    def loading_type(self, value: Literal["memory", "file"]):
        path = self.__root_dir / "maldi2resistance/preprocessed" / f"binSize@{self.__bin_size}"

        if not (path / "finished").is_file():
            raise FileNotFoundError(f"No preprocessed data found for bin size of {self.__bin_size}. Please use the preprocess method.")

        self.__loading_type = value

        if value == "memory":
            self.__create_memory_hashmap()
            self.__use_for_loading = self.__load_from_memory

        elif value == "file":
            self.__use_for_loading = self.__load_from_disk

    def __create_memory_hashmap(self):
        self.__data = {}

        for idx in tqdm(range(len(self)), desc="Loading Spectra into Memory", leave= self.__leave_progress_bar):
            site_key = None
            for (lower, upper), value in self.__idx_table.items():
                if lower <= idx < upper:
                    site_key = value
                    local_idx = idx - lower
                    break

            meta = self.__meta[site_key]
            site, year = site_key

            code = meta.loc[local_idx, "code"]

            save_path = self.__root_dir / "maldi2resistance/preprocessed" / f"binSize@{self.__bin_size}"

            ref = f"{site}/{year}/{code}"
            file = save_path / f"{ref}.npy"

            spectra = np.float32(np.load(file))
            if self.__returned_bins is not None:
                lower, upper = self.__returned_bins
                spectra = spectra[lower:upper]

            self.__data[ref] = torch.from_numpy(spectra)

    def __load_from_memory(self, site, year, code) -> torch.Tensor:
        ref = f"{site}/{year}/{code}"

        return self.__data[ref]

    def __load_from_disk(self, site, year, code) -> torch.Tensor:
        ref = f"{site}/{year}/{code}"
        save_path = self.__root_dir / "maldi2resistance/preprocessed/" / f"binSize@{self.__bin_size}" / f"{ref}.npy"

        spectra = np.float32(np.load(save_path))
        if self.__returned_bins is not None:
            lower, upper = self.__returned_bins
            spectra = spectra[lower:upper]

        return torch.from_numpy(spectra)

    def __get_site_key(
        self, idx: int
    ) -> Tuple[
        Tuple[
            Literal["DRIAMS-A", "DRIAMS-B", "DRIAMS-C", "DRIAMS-D"],
            Literal[2015, 2016, 2017, 2018],
        ],
        int,
    ]:
        for (lower, upper), value in self.__idx_table.items():
            if lower <= idx < upper:
                site_key = value
                local_idx = idx - lower
                break

        return site_key, local_idx

    def get_site_year_key(
        self, indice: int
    ) -> Tuple[
        Literal["DRIAMS-A", "DRIAMS-B", "DRIAMS-C", "DRIAMS-D"],
        Literal[2015, 2016, 2017, 2018],
        str,
    ]:
        site_key, local_idx = self.__get_site_key(indice)
        (site, year) = site_key

        code = self.__meta[site_key].loc[local_idx, "code"]

        return site, year, code

    def filter4species(self, indices: Optional[List[int]], species: List[str]) -> Subset:
        filtered_indices: List[int] = []

        if indices is None:
            indices = range(len(self))

        for idx in indices:
            site_key, local_idx = self.__get_site_key(idx)

            idx_species = self.__meta[site_key].loc[local_idx, "species"]
            if idx_species in species:
                filtered_indices.append(idx)

        return Subset(dataset=self, indices=filtered_indices)

    def getK_fold(
        self, n_splits, shuffle: bool, random_state=None
    ) -> Iterator[Tuple[Subset, Subset]]:
        k_fold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

        for train_idx, test_idx in k_fold.split(self):
            train_subset = Subset(dataset=self, indices=train_idx)
            test_subset = Subset(dataset=self, indices=test_idx)

            yield train_subset, test_subset

    def preprocess(self, save_path: Union[Path, str, None] = None):
        if save_path is None:
            save_path = self.__root_dir / "maldi2resistance/preprocessed" / f"binSize@{self.__bin_size}"
        elif isinstance(save_path, str):
            save_path = Path(save_path)

        for idx in tqdm(range(len(self)), leave=self.__leave_progress_bar):
            site_key, local_idx = self.__get_site_key(idx=idx)

            meta = self.__meta[site_key]
            site, year = site_key

            code:str = meta.loc[local_idx, "code"]

            # remove .txt if in code
            if code.endswith(".txt"):
                path = os.path.join(
                    f"{self.__root_dir}/{site}/preprocessed/{year}/{code}"
                )
            else:
                path = os.path.join(
                    f"{self.__root_dir}/{site}/preprocessed/{year}/{code}.txt"
                )

            if site == "UMG":
                spectrum = polars.read_csv(path, separator=" ", comment_prefix="#", has_header=False, new_columns=["mass.spectra.", "intensity.spectra."], dtypes = [polars.Float64, polars.Float64])
            else:
                spectrum = polars.read_csv(path, separator=" ", comment_prefix="#")

            min_range = min(spectrum["mass.spectra."])
            min_range = max(min_range, self.transform.min_bin)
            max_range = max(spectrum["mass.spectra."])
            max_range = min(max_range, self.transform.max_bin)

            bin_edges_ = np.linspace(min_range, max_range, self.transform.n_bins + 1)

            vec = np.histogram(
                spectrum["mass.spectra."],
                bins=bin_edges_,
                weights=spectrum["intensity.spectra."],
            )[0]

            temp_path = save_path / f"{site}/{year}"
            temp_path.mkdir(exist_ok=True, parents=True)

            np.save(temp_path / code, vec)

        with open(save_path / "finished", "w") as file:
            file.write("")

    def _repr_html_(self):
        head = ""
        body_susceptible = ""
        body_resistances = ""
        body_sum = ""

        for antibiotic in self.selected_antibiotics:
            series_list = []
            for key in self.__keys:
                try:
                    series_list.append(self.__meta[key][antibiotic])
                except KeyError:
                    continue

            series = pd.concat(series_list)

            head += f"<th> {antibiotic} </th>"

            body_resistances += f"<td> {(series == 1).sum()} </td>"
            body_susceptible += f"<td> {(series == 0).sum()} </td>"
            body_sum += f"<td> {series.count()} </td>"

        return f"""
        <table>
            <thead>
                <tr>
                    <th>Antibiotic:</th>
                    {head}
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Number resistant:</td>
                    {body_resistances}
                </tr>
                <tr>
                    <td>Number susceptible:</td>
                    {body_susceptible}
                </tr>
                <tr>
                    <td>Number data points:</td>
                    {body_sum}
                </tr>
            </tbody>
        </table>
                """


class DriamsSingleAntibiotic(Dataset):
    def __init__(self,
                 driams:Union[Driams,Subset[Driams]],
                 use_morganFingerprint4Drug:bool = False,
                 prepeare4chemprop = False,
                 ):

        self.subset = None
        if isinstance(driams, Subset):
            self.subset = driams
            driams = driams.dataset
        elif isinstance(driams, ConcatDataset):
            self.subset = driams

            if isinstance(driams.datasets[0], Subset):
                driams = driams.datasets[0].dataset
            else:
                driams = driams.datasets[0]

        else:
            self.subset = driams
            driams = driams

        self.driams = driams
        self.data = []
        self.fp_lookup = FingerprintLookup()
        from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
        self.featurizer = SimpleMoleculeMolGraphFeaturizer()
        self.spectra_lookup = {}
        self.mol_lookup = {}

        if prepeare4chemprop:
            from chemprop.data import MoleculeDatapoint
            self.__return_from_idx = self.__get4Chemprop

            for antibiotic_name in self.driams.selected_antibiotics:
                smiles = self.fp_lookup.drug_name2Smiles[antibiotic_name]
                mol = MoleculeDatapoint.from_smi(smiles)
                self.mol_lookup[antibiotic_name] = mol


        for number, (x, y) in enumerate(tqdm(self.subset, leave=False, desc=f"Create single label Dataset")):
            self.spectra_lookup[number] = x

            for pos, label in enumerate(y):
                if torch.isnan(label):
                    continue

                if use_morganFingerprint4Drug:
                    antibiotic_name = self.driams.selected_antibiotics[pos]
                    fingerprint = self.fp_lookup.get_MorganFingerprint(antibiotic_name)

                    self.data.append((number, label, torch.Tensor(fingerprint)))
                    continue

                if prepeare4chemprop:
                    antibiotic_name = self.driams.selected_antibiotics[pos]

                    self.data.append((number, label, antibiotic_name))
                    continue

                self.data.append((number, label, torch.tensor(pos)))

    def __get4Chemprop(self, idx):
        number, label, drug_name = self.data[idx]

        return (self.spectra_lookup[number], label, self.mol_lookup[drug_name])

    def __return_from_idx(self, idx):
        number, label, drug = self.data[idx]
        return (self.spectra_lookup[number], label, drug)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return self.__return_from_idx(idx)

    def getK_fold(
        self, n_splits, shuffle: bool, random_state=None
    ) -> Iterator[Tuple[Subset, Subset]]:
        k_fold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

        for train_idx, test_idx in k_fold.split(self):
            train_subset = Subset(dataset=self, indices=train_idx)
            test_subset = Subset(dataset=self, indices=test_idx)

            yield train_subset, test_subset
