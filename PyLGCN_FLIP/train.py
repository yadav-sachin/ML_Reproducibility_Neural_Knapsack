from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy
from models import LGCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="Disables CUDA training."
)
parser.add_argument(
    "--fastmode",
    action="store_true",
    default=False,
    help="Validate during training pass.",
)
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument(
    "--epochs", type=int, default=1000, help="Number of epochs to train."
)
parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate.")
parser.add_argument(
    "--weight_decay",
    type=float,
    default=5e-4,
    help="Weight decay (L2 loss on parameters).",
)
parser.add_argument(
    "--dataset",
    type=int,
    default=0,
    help="Graph Dataset (0 for cora and 1 for citeseer).",
)
parser.add_argument("--hidden", type=int, default=16, help="Number of hidden units.")
parser.add_argument(
    "--dropout", type=float, default=0.5, help="Dropout rate (1 - keep probability)."
)

args = parser.parse_args()
args.cuda = args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data from the required dataset
dataset = "cora" if args.dataset == 0 else "citeseer"
if dataset == "cora":
    adj, features, labels, idx_train, idx_val, idx_test = load_data()
else:
    adj, features, labels, idx_train, idx_val, idx_test = load_data(
        path="./data/citeseer/", dataset="citeseer"
    )


# Model and optimizer
model = LGCN(
    nfeat=features.shape[1],
    nhid=args.hidden,
    nclass=labels.max().item() + 1,
    dropout=args.dropout,
)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

# A custom round function
# On forward-pass, round function is applied
# In backward-pass, the round function does'nt exist and the gradients are backpropagated without change
class my_round_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


# Test the model with the new adjacency matrix with some edges flipped from original
def test(flip_model, new_adj):
    flip_model.eval()
    output = flip_model(features, new_adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print(
        "Test set results:",
        "loss= {:.4f}".format(loss_test.item()),
        "accuracy= {:.4f}".format(acc_test.item()),
    )


def train_flip():
    n_nodes = adj.shape[0]
    n_edges = n_nodes * n_nodes
    budget = 0.005

    T = 1.0
    # slope annealing parameters
    # change rate, step annealing step size
    r, s = 1.004, 50
    my_round = my_round_func.apply

    # Parameters of Neural Knapsack
    e = torch.randn((n_nodes, n_nodes), requires_grad=True)
    if args.cuda:
        with torch.no_grad():
            e = e.cuda()

    torch.nn.init.xavier_uniform_(e)
    learning_rate = 5

    for epoch in range(args.epochs):

        optimizer.zero_grad()
        e.requires_grad = True

        with torch.no_grad():
            e.grad = torch.zeros_like(e)

        # Straigth-throgh estimator
        x0 = T * e
        x1 = torch.sigmoid(x0)
        x2 = my_round(x1)

        # Create the new adjacency matrix and get output from model on this modified graph
        new_A = torch.abs(x2 - adj)
        output = model(features, new_A)

        loss_train = F.nll_loss(output[idx_train], labels[idx_train])

        loss_train.backward(retain_graph=True)
        optimizer.step()

        acc_val = accuracy(output[idx_val], labels[idx_val])
        acc_train = accuracy(output[idx_train], labels[idx_train])

        # o' is the gradient of training loss of model w.r.t. e (neural knapsack parameters)
        o_dash = e.grad.clone().detach()
        e.grad.zero_()
        optimizer.zero_grad()

        cost_overrun = (torch.sum(x2) / n_edges) - budget

        cost_overrun.backward(retain_graph=False)

        with torch.no_grad():
            # b' is the gradient of cost overrun w.r.t. e (neural knapsack parameters)
            b_dash = e.grad.clone().detach()

            # Conditions for beta
            if cost_overrun <= 0:
                beta = 0
            else:
                ob = torch.sum(o_dash * b_dash)
                oo = torch.sum(o_dash * o_dash)
                bb = torch.sum(b_dash * b_dash)
                term1 = ob / (cost_overrun * bb)
                term2 = oo / (cost_overrun * ob)
                if ob <= 0:
                    beta = 0
                elif ob > 0 and term2 > term1:
                    beta = (term1 + term2) / 2
                else:
                    beta = term1 + (1e-6)

            # Adaptive Gradient Ascent
            # Update the Neural Knapsack Parameters
            del_l = o_dash - beta * cost_overrun * b_dash
            e = e + learning_rate * del_l

            if epoch % 20 == 0:
                print(
                    "Epoch: ", epoch, "accuracy: {:.4f}".format(acc_val.item()),
                )

            T = (r ** (epoch // s)) * T


train_flip()
