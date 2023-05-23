import torch
import pickle
from torch import nn
from torch.nn import functional as F

DATASETS = {
            'cifar10_densenet40'      : ['datasets/probs_densenet40_c10_logits.p',],
            'cifar10_wideresnet32'    : ['datasets/probs_resnet_wide32_c10_logits.p',],
            'cifar10_resnet110'       : ['datasets/probs_resnet110_c10_logits.p',],
            'cifar10_resnet110sd'     : ['datasets/probs_resnet110_SD_c10_logits.p',],
            'cifar100_densenet40'     : ['datasets/probs_densenet40_c100_logits.p',],
            'cifar100_wideresnet32'   : ['datasets/probs_resnet_wide32_c100_logits.p',],
            'cifar100_resnet110'      : ['datasets/probs_resnet110_c100_logits.p',],
            'cifar100_resnet110sd'    : ['datasets/probs_resnet110_SD_c100_logits.p',],
            'imagenet_resnet152'      : ['datasets/probs_resnet152_imgnet_logits.p',],
            'imagenet_densenet161'    : ['datasets/probs_densenet161_imgnet_logits.p',],

            'LT_cifar10_densenet40'   : ['datasets/probs_densenet40_c10_LT_logits.p',],
            'LT_cifar10_wideresnet28' : ['datasets/probs_resnet_wide28_c10_LT_logits.p',],
            'LT_cifar10_resnet110'    : ['datasets/probs_resnet110_c10_LT_logits.p',],
            'LT_cifar10_resnet110sd'  : ['datasets/probs_resnet110_SD_c10_LT_logits.p',],
            'LT_cifar100_densenet40'  : ['datasets/probs_densenet40_c100_LT_logits.p',],
            'LT_cifar100_wideresnet28': ['datasets/probs_resnet_wide28_c100_LT_logits.p',],
            'LT_cifar100_resnet110'   : ['datasets/probs_resnet110_c100_LT_logits.p',],
            'LT_cifar100_resnet110sd' : ['datasets/probs_resnet110_SD_c100_LT_logits.p',],

            'cifar10'                 : ['datasets/probs_densenet40_c10_logits.p',
                                         'datasets/probs_resnet_wide32_c10_logits.p',
                                         'datasets/probs_resnet110_c10_logits.p',
                                         'datasets/probs_resnet110_SD_c10_logits.p',],
            'cifar100'                : ['datasets/probs_densenet40_c100_logits.p',
                                         'datasets/probs_resnet_wide32_c100_logits.p',
                                         'datasets/probs_resnet110_c100_logits.p',
                                         'datasets/probs_resnet110_SD_c100_logits.p',],
            'imagenet'                : ['datasets/probs_resnet152_imgnet_logits.p',
                                         'datasets/probs_densenet161_imgnet_logits.p',],
                                      
            'LT_cifar10'              : ['datasets/probs_densenet40_c10_LT_logits.p',
                                         'datasets/probs_resnet_wide28_c10_LT_logits.p',
                                         'datasets/probs_resnet110_c10_LT_logits.p',
                                         'datasets/probs_resnet110_SD_c10_LT_logits.p',],
            'LT_cifar100'             : ['datasets/probs_densenet40_c100_LT_logits.p',
                                         'datasets/probs_resnet_wide28_c100_LT_logits.p',
                                         'datasets/probs_resnet110_c100_LT_logits.p',
                                         'datasets/probs_resnet110_SD_c100_LT_logits.p',],

            'all'                     : ['datasets/probs_densenet40_c10_logits.p',
                                         'datasets/probs_resnet_wide32_c10_logits.p',
                                         'datasets/probs_resnet110_c10_logits.p',
                                         'datasets/probs_resnet110_SD_c10_logits.p',
                                         'datasets/probs_densenet40_c100_logits.p',
                                         'datasets/probs_resnet_wide32_c100_logits.p',
                                         'datasets/probs_resnet110_c100_logits.p',
                                         'datasets/probs_resnet110_SD_c100_logits.p',
                                         'datasets/probs_resnet152_imgnet_logits.p',
                                         'datasets/probs_densenet161_imgnet_logits.p',],

            'LT'                      : ['datasets/probs_densenet40_c10_LT_logits.p',
                                         'datasets/probs_resnet_wide28_c10_LT_logits.p',
                                         'datasets/probs_resnet110_c10_LT_logits.p',
                                         'datasets/probs_resnet110_SD_c10_LT_logits.p',
                                         'datasets/probs_densenet40_c100_LT_logits.p',
                                         'datasets/probs_resnet_wide28_c100_LT_logits.p',
                                         'datasets/probs_resnet110_c100_LT_logits.p',
                                         'datasets/probs_resnet110_SD_c100_LT_logits.p',],

            'ALL'                     : ['datasets/probs_densenet40_c10_logits.p',
                                         'datasets/probs_resnet_wide32_c10_logits.p',
                                         'datasets/probs_resnet110_c10_logits.p',
                                         'datasets/probs_resnet110_SD_c10_logits.p',
                                         'datasets/probs_densenet40_c100_logits.p',
                                         'datasets/probs_resnet_wide32_c100_logits.p',
                                         'datasets/probs_resnet110_c100_logits.p',
                                         'datasets/probs_resnet110_SD_c100_logits.p',
                                         'datasets/probs_resnet152_imgnet_logits.p',
                                         'datasets/probs_densenet161_imgnet_logits.p',
                                         'datasets/probs_densenet40_c10_LT_logits.p',
                                         'datasets/probs_resnet_wide28_c10_LT_logits.p',
                                         'datasets/probs_resnet110_c10_LT_logits.p',
                                         'datasets/probs_resnet110_SD_c10_LT_logits.p',
                                         'datasets/probs_densenet40_c100_LT_logits.p',
                                         'datasets/probs_resnet_wide28_c100_LT_logits.p',
                                         'datasets/probs_resnet110_c100_LT_logits.p',
                                         'datasets/probs_resnet110_SD_c100_LT_logits.p',],
            }

CALIBRATOR = {
    'TS' : 'temperature_scaler',
    'ETS': 'ensemble_temperature_scaler',
    'CTS': 'class_based_temperature_scaler',
    'PTS': 'parameterized_temperature_scaler',
}

LOSS = {
    'CE' : 'cross_entropy_loss',
    'LS' : 'label_smoothing_loss',
    'FL' : 'focal_loss',
    'CL' : 'scaling_classwise_training_loss'
}

class expected_caibration_error(nn.Module):
    def __init__(self, n_bins=15):
        super(expected_caibration_error, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        
        ece = torch.zeros(1, device=logits.device)
        acc = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                acc += accuracies[in_bin].float().mean() * prop_in_bin
        return ece * 100, acc * 100
    
def dataloader(args):
    with open(args.data, 'rb') as f:
        (y_probs_val, y_val), (y_probs_test, y_test) = pickle.load(f)
        
    #     print("{}".format(file.split('/')[-1].split('.p')[0]))
    #     print("y_probs_val : {} | y_val : {} | y_probs_test : {} | y_true_test : {}".format(y_probs_val.shape,y_val.shape,y_probs_test.shape,y_test.shape))

    args.n_class = y_probs_val.shape[1] # using for class_based_temperature_scalining

    valid_logits = torch.tensor(y_probs_val).cuda()
    valid_labels = torch.tensor(y_val).long().view(-1).cuda()
    test_logits = torch.tensor(y_probs_test).cuda()
    test_labels = torch.tensor(y_test).long().view(-1).cuda()
        
    return (valid_logits, valid_labels), (test_logits, test_labels)

def dataset_mapping(data):
    return DATASETS[data]
    
def calibrator_mapping(cal):
    return CALIBRATOR[cal]
    
def loss_mapping(loss):
    return LOSS[loss]
