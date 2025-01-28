import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Union

import torch
from matplotlib import pyplot as plt
from torchmetrics.classification import BinaryPrecisionRecallCurve, BinaryAveragePrecision
from torchmetrics.utilities.compute import auc


@dataclass
class PrecisionRecallResult:
    precision: torch.Tensor
    recall: torch.Tensor
    thresholds: torch.Tensor
    auc: torch.Tensor
    positive_percentage: float
    n_positives: int
    n_negatives: int


class MultiLabelPRNan:
    def __init__(self, thresholds=None):
        self.__fig, self.__ax = plt.subplots()
        self.thresholds = thresholds
        self.__results: Dict[str, PrecisionRecallResult] = {}
        self.__allAUC: List[int] = []
        self.__micro = None
        self.__macro = None

    def compute(
        self,
        prediction: torch.tensor,
        label: torch.tensor,
        class_names,
        device="cpu",
        create_csv: Union[str, Path, None] = None,
        skip_list: list = None,
    ):
        if isinstance(create_csv, str):
            create_csv = Path(create_csv)

        auc_pr_sum = 0
        test_label_all = torch.tensor([]).to(device)
        output_all = torch.tensor([]).to(device)
        prediction = prediction.to(device)
        label = label.to(device)
        self.__allAUC: List[int] = []
        n_non_none_results = 0

        for i in range(len(class_names)):
            metric = BinaryPrecisionRecallCurve(thresholds=self.thresholds)
            auPR = BinaryAveragePrecision(thresholds=self.thresholds)

            antibiotic = class_names[i]
            if (skip_list != None) and (antibiotic in skip_list):
                continue

            mask = torch.isnan(label[:, i])
            test_labels_part = label[~mask][:, i]

            if test_labels_part.numel() == 0:
                self.__results[antibiotic] = None
                continue

            occurrences = torch.bincount(test_labels_part.int())
            values = torch.numel(test_labels_part)

            try:
                positives = occurrences[1].item() / values
            except IndexError:
                # no positives
                self.__results[antibiotic] = None
                continue

            prediction_part = prediction[~mask][:, i]

            test_label_all = torch.cat((test_label_all, test_labels_part))
            output_all = torch.cat((output_all, prediction_part))

            metric.update(prediction_part, test_labels_part.int())

            precision, recall, thresholds = metric.compute()
            aucPC = auPR(prediction_part, test_labels_part.int())
            self.__allAUC.append(aucPC.item())

            auc_pr_sum += aucPC.item()

            self.__results[antibiotic] = PrecisionRecallResult(
                precision=precision,
                recall=recall,
                thresholds=thresholds,
                auc=aucPC,
                positive_percentage=positives,
                n_positives=occurrences[1].item(),
                n_negatives=occurrences[0].item(),
            )
            n_non_none_results += 1

        auPR = BinaryAveragePrecision(thresholds=self.thresholds)

        occurrences = torch.bincount(test_label_all.int())
        values = torch.numel(test_label_all.int())

        try:
            positives = occurrences[1].item() / values
        except IndexError:
            # no positives
            positives = "Nan"

        self.__macro = auc_pr_sum / n_non_none_results
        self.__micro = auPR(output_all, test_label_all.int()).item()

        if create_csv is not None:
            create_csv.parent.mkdir(parents=True, exist_ok=True)

            with open(create_csv, "w") as csvfile:
                columns = [
                    "class",
                    "PrecisionRecallAUC",
                    "FrequencyPositiveClass",
                    "n_positives",
                    "n_negatives",
                ]
                writer = csv.DictWriter(csvfile, fieldnames=columns)
                writer.writeheader()

                for c in class_names:
                    if (skip_list != None) and (c in skip_list):
                        continue

                    result = self.__results[c]

                    if result is None:
                        writer.writerow(
                            {
                                "class": c,
                                "PrecisionRecallAUC": "NaN",
                                "FrequencyPositiveClass": "NaN",
                                "n_positives": "NaN",
                                "n_negatives": "NaN",
                            }
                        )
                    else:
                        writer.writerow(
                            {
                                "class": c,
                                "PrecisionRecallAUC": result.auc.item(),
                                "FrequencyPositiveClass": result.positive_percentage,
                                "n_negatives": result.n_negatives,
                                "n_positives": result.n_positives,
                            }
                        )

                writer.writerow(
                    {
                        "class": "micro",
                        "PrecisionRecallAUC": self.__micro,
                        "FrequencyPositiveClass": positives,
                    }
                )
                writer.writerow(
                    {
                        "class": "macro",
                        "PrecisionRecallAUC": self.__macro,
                        "FrequencyPositiveClass": "NaN",
                    }
                )

        return self.__micro, self.__macro

    def __call__(
        self, best: int = 3, worst: int = 3, class_names: List[str] = [], device="cpu"
    ):
        all_auc = sorted(self.__allAUC, reverse=True)
        selected_auc = all_auc[:best] + all_auc[-worst:]

        n_lines = 0
        for antibiotic, result in self.__results.items():
            if result is None:
                continue

            metric = BinaryPrecisionRecallCurve(thresholds=self.thresholds)

            curve = (result.precision, result.recall, result.thresholds)

            if result.auc in selected_auc or antibiotic in class_names:
                self.__fig, self.__ax = metric.plot(
                    curve=curve, score=result.auc, ax=self.__ax
                )
                new_label = f"[{result.auc:.3f}] {antibiotic}"
                self.__ax.get_lines()[n_lines].set_label(new_label)
                n_lines += 1

        plt.legend()
        return self.__fig, self.__ax
