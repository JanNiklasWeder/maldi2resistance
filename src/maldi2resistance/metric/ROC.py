import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Union

import torch
from matplotlib import pyplot as plt
from torchmetrics.classification import BinaryROC, BinaryAUROC


@dataclass
class ROCResult:
    fpr: torch.Tensor
    tpr: torch.Tensor
    thresholds: torch.Tensor
    auc: torch.Tensor
    n_positives: int
    n_negatives: int


class MultiLabelRocNan:
    def __init__(self, thresholds=None):
        self.__fig, self.__ax = plt.subplots()
        self.thresholds = thresholds
        self.__results: Dict[str, ROCResult] = {}
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

        auc_roc_sum = []
        test_label_all = torch.tensor([]).to(device)
        output_all = torch.tensor([]).to(device)
        prediction = prediction.to(device)
        label = label.to(device)
        self.__allAUC = []

        for i in range(len(class_names)):
            metric = BinaryROC(thresholds=self.thresholds)
            auRoc = BinaryAUROC(thresholds=self.thresholds)

            antibiotic = class_names[i]
            if (skip_list != None) and (antibiotic in skip_list):
                continue

            # create masked for NaNs and remove those out of the tensor
            mask = torch.isnan(label[:, i])
            test_labels_part = label[~mask][:, i]
            test_labels_part = test_labels_part.int()
            output_part = prediction[~mask][:, i]

            if test_labels_part.numel() == 0:
                self.__results[antibiotic] = None
                continue

            test_label_all = torch.cat((test_label_all, test_labels_part))
            output_all = torch.cat((output_all, output_part))

            occurrences = torch.bincount(test_labels_part.int())

            auc_roc = auRoc(output_part, test_labels_part)
            self.__allAUC.append(auc_roc)

            auc_roc_sum.append(auc_roc.item())

            metric.update(output_part, test_labels_part)
            fpr, tpr, thresholds = metric.compute()

            try:
                self.__results[antibiotic] = ROCResult(
                    fpr=fpr,
                    tpr=tpr,
                    thresholds=thresholds,
                    auc=auc_roc,
                    n_positives=occurrences[1].item(),
                    n_negatives=occurrences[0].item(),
                )
            except IndexError:
                self.__results[antibiotic] = None
                continue

        auRoc = BinaryAUROC(thresholds=self.thresholds)

        self.__macro = sum(auc_roc_sum)/len(auc_roc_sum)
        self.__micro = auRoc(output_all, test_label_all.int()).item()

        if create_csv is not None:
            create_csv.parent.mkdir(parents=True, exist_ok=True)

            with open(create_csv, "w") as csvfile:
                columns = ["class", "ROCAUC", "n_positives", "n_negatives"]
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
                                "ROCAUC": "NaN",
                                "n_positives": "NaN",
                                "n_negatives": "NaN",
                            }
                        )
                    else:
                        writer.writerow(
                            {
                                "class": c,
                                "ROCAUC": result.auc.item(),
                                "n_positives": result.n_positives,
                                "n_negatives": result.n_negatives,
                            }
                        )

                writer.writerow({"class": "micro", "ROCAUC": self.__micro})
                writer.writerow({"class": "macro", "ROCAUC": self.__macro})

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

            metric = BinaryROC(thresholds=self.thresholds)

            curve = (result.fpr, result.tpr, result.thresholds)

            if result.auc in selected_auc or antibiotic in class_names:
                self.__fig, self.__ax = metric.plot(
                    curve=curve, score=result.auc, ax=self.__ax
                )
                new_label = f"[{result.auc:.3f}] {antibiotic}"
                self.__ax.get_lines()[n_lines].set_label(new_label)
                n_lines += 1

        plt.legend()
        return self.__fig, self.__ax
