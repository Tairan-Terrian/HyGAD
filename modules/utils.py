import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np
from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_score, average_precision_score
from scikitplot.helpers import binary_ks_curve

Tensor = torch.tensor

def fixed_augmentation(graph, seed_nodes, sampler, aug_type: str, p: float = None):
    assert aug_type in ['drophidden', 'none']
    with graph.local_scope():
        if aug_type == 'drophidden':
            input_nodes, output_nodes, blocks = sampler.sample_blocks(graph, seed_nodes)
        else:
            input_nodes, output_nodes, blocks = sampler.sample_blocks(graph, seed_nodes)
        return input_nodes, output_nodes, blocks


def eval_auc_roc(pred, target):
    scores = roc_auc_score(target, pred)
    return scores


def eval_auc_pr(pred, target):
    scores = average_precision_score(target, pred)
    return scores


def eval_ks_statistics(target, pred):
    scores = binary_ks_curve(target, pred)[3]
    return scores


def find_best_f1(probs, labels):
    best_f1, best_thre = -1., -1.
    thres_arr = np.linspace(0.05, 0.95, 19)
    for thres in thres_arr:
        preds = np.zeros_like(labels)
        preds[probs > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre


def eval_pred(pred: Tensor, target: Tensor):
    s_pred = pred.cpu().detach().numpy()
    s_target = target.cpu().detach().numpy()

    auc_roc = roc_auc_score(s_target, s_pred)
    auc_pr = average_precision_score(s_target, s_pred)
    ks_statistics = eval_ks_statistics(s_target, s_pred)

    best_f1, best_thre = find_best_f1(s_pred, s_target)
    p_labels = (s_pred > best_thre).astype(int)
    accuracy = np.mean(s_target == p_labels)
    recall = recall_score(s_target, p_labels)
    precision = precision_score(s_target, p_labels)

    return auc_roc, auc_pr, ks_statistics, accuracy, \
        recall, precision, best_f1, best_thre


def nll_loss(pred, target, pos_w: float = 1.0):
    weight_tensor = torch.tensor([1., pos_w]).to(pred.device)
    loss_value = F.nll_loss(pred, target.long(), weight=weight_tensor)

    return loss_value, to_np(loss_value)


def nll_loss_raw(pred: Tensor, target: Tensor, pos_w,
                 reduction: str = 'mean'):
    weight_tensor = torch.tensor([1., pos_w]).to(pred.device)
    loss_value = F.nll_loss(pred, target.long(), weight=weight_tensor,
                            reduction=reduction)

    return loss_value


def l2_regularization(model):
    l2_reg = torch.tensor(0., requires_grad=True)
    for key, value in model.named_parameters():
        if len(value.shape) > 1 and 'weight' in key:
            l2_reg = l2_reg + torch.sum(value ** 2) * 0.5
    return l2_reg

def build_mlp(in_dim: int, out_dim: int, p: float, hid_dim: int=64, final_act: bool=True):
    mlp_list = []

    mlp_list.append(CustomLinear(in_dim, hid_dim, bias=True))
    mlp_list.append(nn.ELU())
    mlp_list.append(nn.Dropout(p=p))
    mlp_list.append(nn.LayerNorm(hid_dim))
    mlp_list.append(CustomLinear(hid_dim, out_dim, bias=True))
    if final_act:
        mlp_list.append(nn.ELU())
        mlp_list.append(nn.Dropout(p=p))

    return nn.Sequential(*mlp_list)

def to_np(x):
    return x.cpu().detach().numpy()


def store_model(my_model, args):
    file_path = os.path.join('model-weights',
                             args['data-set'] + '.pth')
    torch.save(my_model.state_dict(), file_path)



class CustomLinear(nn.Linear):
    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)


