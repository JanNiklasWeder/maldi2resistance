import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torchmetrics.classification
import torch.nn as nn

import sys, os, copy
sys.path.append('../')
from maldi2resistance.data.ArtificalMixed import ArtificialMixedInfection
from maldi2resistance.data.driams import Driams
from maldi2resistance.model.MultilabelResMLP import MultilabelResMLP
from maldi2resistance.model.MLP import AeBasedMLP
from maldi2resistance.metric.ROC import MultiLabelRocNan
from maldi2resistance.metric.PrecisionRecall import MultiLabelPRNan

DEVICE = torch.device("cuda")
EPOCH = 30
batch_size = 128
LATENT_SIZE = 512
Modelname = 'MLP'
SEED = int(sys.argv[1])
torch.manual_seed(SEED)

output_dir = f'result_ratio_{Modelname}_DRIAMS-UMG_E{EPOCH}_B{batch_size}_L{LATENT_SIZE}'
os.makedirs(output_dir, exist_ok=True)

driams = Driams(
    root_dir="/scratch1/users/park11/driams",
    #sites=['DRIAMS-B']
    # antibiotics= ['Ciprofloxacin', 'Ceftriaxone', "Cefepime", "Piperacillin-Tazobactam", "Tobramycin"]
)
driams.loading_type = "memory"

umg = Driams(
    root_dir="/scratch1/users/park11/micro_ms",
    bin_size=1,
    sites=["UMG"],
    years=[2020, 2021],
    antibiotics=driams.selected_antibiotics,
)
umg.loading_type = "memory"

train_size = int(0.8 * len(driams))
test_size = len(driams) - train_size

umg_train_size = int(0.8 * len(umg))
umg_test_size = len(umg) - umg_train_size

gen = torch.Generator()
gen.manual_seed(SEED)

def save_results(_output, _test_labels, _driams, _output_suffix):
    target_antibiotics = copy.deepcopy(_driams.selected_antibiotics)

    skip_list = ['Aztreonam', 'Benzylpenicillin', 'Clarithromycin', 'Polymyxin B']

    ml_roc = MultiLabelRocNan()
    print(ml_roc.compute(_output, _test_labels, target_antibiotics, create_csv=f"./{output_dir}/ROC_{_output_suffix}_Seed_{SEED}.csv", skip_list = skip_list))
    fig_, ax_ = ml_roc()
    plt.savefig(f"./{output_dir}/ROC_{_output_suffix}_Seed_{SEED}.png", transparent=True, format= "png", bbox_inches = "tight")

    ml_pr = MultiLabelPRNan()
    print(ml_pr.compute(_output, _test_labels, target_antibiotics, create_csv=f"./{output_dir}/PR_{_output_suffix}_Seed_{SEED}.csv", skip_list = skip_list))
    fig_, ax_ = ml_pr()
    plt.savefig(f"./{output_dir}/PR_{_output_suffix}_Seed_{SEED}.png", transparent=True, format= "png", bbox_inches = "tight")


def main():
    umg_train_dataset, umg_test_dataset = torch.utils.data.random_split(umg, [umg_train_size, umg_test_size], generator= gen)
    driams_train_dataset, driams_test_dataset = torch.utils.data.random_split(driams, [train_size, test_size], generator= gen)

    train_dataset = torch.utils.data.ConcatDataset([driams_train_dataset, umg_train_dataset])
    #test_dataset = torch.utils.data.ConcatDataset([driams_test_dataset, umg_test_dataset])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model_filename = f'./{output_dir}/{Modelname}_driams_Train_Mixed2_Epochs_{EPOCH}_Seed_{SEED}.pt'
    
    if os.path.exists(model_filename):
        model = torch.jit.load(model_filename)
        print(f'model {model_filename} is successfully loaded')
    else:
        if Modelname == 'MLP':
            model = AeBasedMLP(input_dim=18000, output_dim=len(driams.selected_antibiotics), hidden_dim=4096, latent_dim=LATENT_SIZE)
        elif Modelname == 'ResMLP':
            model = MultilabelResMLP(input_dim=18000, output_dim=len(driams.selected_antibiotics), hidden_dim=LATENT_SIZE)
        else:
            exit('Model name error')

        model.to(DEVICE)
        model_state = copy.deepcopy(model.state_dict())
        model.train()

        optimizer = Adam(model.parameters(), lr=1e-3, amsgrad = True)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
        loss_per_batch = []

        criterion = nn.BCELoss()

        class_weights_negative = torch.tensor((1 - (driams.label_stats.loc["negative"] / driams.label_stats.loc["n_sum"])).values, device=DEVICE)
        class_weights_positive = torch.tensor((1 - (driams.label_stats.loc["positive"] / driams.label_stats.loc["n_sum"])).values, device=DEVICE)

        print("Start training ...")
        for epoch in tqdm(range(EPOCH)):
            overall_loss = 0
        
            for batch_idx, (x, y) in enumerate(train_loader):
                       
                x = x.view(batch_size, 18000)
                x = x.to(DEVICE)

                split1,split2 = torch.chunk(x, 2)
                combined = torch.add(split1, split2)
                combined_features = torch.div(combined, 2)
                
                x = torch.cat((x, combined_features), dim = 0)
                
                y = y.view(batch_size, len(driams.selected_antibiotics))
                y = y.to(DEVICE)
                
                split1,split2 = torch.chunk(y, 2)
                combined = torch.add(split1, split2)
                combined_labels = torch.div(combined, 2)
                combined_labels[combined_labels > 0.3] =1
                
                y = torch.cat((y, combined_labels), dim = 0)
                
                positive_weight = torch.clone(y)
                negative_weight = torch.clone(y)
                negative_weight[negative_weight == 1] = -1
                negative_weight[negative_weight == 0] = 1
                negative_weight[negative_weight == -1] = 0
                negative_weight = class_weights_negative * negative_weight[:, None]
                positive_weight = class_weights_positive * positive_weight[:, None]
                
                weight = torch.add(positive_weight, negative_weight)
                weight = torch.nan_to_num(weight, 0)
                weight = weight[:,0, :]
                
                weight.to(DEVICE)
                y = torch.nan_to_num(y, 0)
                
                optimizer.zero_grad()

                # output, mean, log_var = model(x)
                output = model(x)

                #loss = loss_function(y, output, mean, log_var)
                loss = F.binary_cross_entropy(output, y, weight=weight)
                loss = criterion(output, y)
                current_loss_value = loss.item()
                loss_per_batch.append(current_loss_value)
                
                overall_loss += current_loss_value
                
                loss.backward()
                optimizer.step()

            scheduler.step()
            with tqdm.external_write_mode():
                print(f"\tAverage Loss: {overall_loss / (batch_idx*batch_size):.6f} \tLearning rate: {scheduler.get_last_lr()[0]:.6f}")
    
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_filename)

        print("Train finish\n Test begins")


    model.eval()
    for mix_ratio in [x/10.0 for x in range(1,6)]:
        driams_test_dataset_mixed = ArtificialMixedInfection(driams_test_dataset, generator_seed=SEED, use_percent_of_data=1, mix_ratio=mix_ratio)
        umg_test_dataset_mixed = ArtificialMixedInfection(umg_test_dataset, generator_seed=SEED, use_percent_of_data=1, mix_ratio=mix_ratio)
        
        #
        # single+mixed test run
        output_suffix = f'{Modelname}_train_mixed2_test_single_mixed_{mix_ratio}'
        test = torch.utils.data.ConcatDataset([driams_test_dataset, driams_test_dataset_mixed, umg_test_dataset, umg_test_dataset_mixed])

        test_loader = DataLoader(test, batch_size=len(test), shuffle=True)
        test_features, test_labels = next(iter(test_loader))
        test_features = test_features.to(DEVICE)
        test_labels = test_labels.to(DEVICE)

        output = model(test_features)
        output = torch.tensor(output)

        save_results(output, test_labels, driams, output_suffix)

        #
        # mixed test run
        output_suffix = f'{Modelname}_train_mixed2_test_mixed_{mix_ratio}'
        test = torch.utils.data.ConcatDataset([driams_test_dataset_mixed, umg_test_dataset_mixed])

        test_loader = DataLoader(test, batch_size=len(test), shuffle=True)
        test_features, test_labels = next(iter(test_loader))
        test_features = test_features.to(DEVICE)
        test_labels = test_labels.to(DEVICE)

        output = model(test_features)
        output = torch.tensor(output)

        save_results(output, test_labels, driams, output_suffix)
        
main()
