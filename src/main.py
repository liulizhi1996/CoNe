import argparse

import torch

from src.utils.data_utils import get_data
from src.utils.misc_utils import set_seed, str2bool
from src.models.model import CoNe
from src.train import train
from src.inference import test


def select_embedding(args, num_nodes, device):
    if args.train_node_embedding:
        emb = torch.nn.Embedding(num_nodes, args.hidden_channels).to(device)
        torch.nn.init.xavier_uniform_(emb.weight)
    elif args.pretrained_node_embedding:
        weight = torch.load(args.pretrained_node_embedding)
        emb = torch.nn.Embedding.from_pretrained(weight)
        emb.weight.requires_grad = False
    else:
        emb = None
    return emb


def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    data = get_data(args)
    emb = select_embedding(args, data.num_nodes, device)

    model = CoNe(data.num_features, args.hidden_channels, args.num_layers,
                 node_embedding=emb, attn_type=args.attn_type, num_heads=args.num_heads,
                 gnn_dropout=args.gnn_dropout, attn_dropout=args.attn_dropout,
                 mlp_dropout=args.mlp_dropout).to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: args.lr_decay ** e)

    for epoch in range(args.epochs):
        loss = train(model, optimizer, data, args, device)
        if (epoch + 1) % args.eval_steps == 0:
            test_res = test(model, data, args, device)
            to_print = f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Test: {test_res:.4f}'
            print(to_print)
        scheduler.step()


def parse_args():
    # Data settings
    parser = argparse.ArgumentParser(description='CoNe')
    parser.add_argument('--dataset_name', type=str, default='Neural',
                        choices=['Neural', 'Celegans', 'Netscience', 'Pblog', 'UCsocial', 'Condmat',
                                 'Astro', 'Collaboration', 'Congress', 'Usair'])
    parser.add_argument('--test_pct', type=float, default=0.1,
                        help='the percentage of supervision edges to be used for test. These edges will not appear '
                             'in the training or validation sets for either supervision or message passing')
    parser.add_argument('--seed', type=int, default=23, help='seed for reproducibility')
    parser.add_argument('--walk_length', type=int, default=64, help='number of random walk steps for RWPE')
    parser.add_argument('--rwpe_use_weight', type=str2bool, default=False,
                        help='whether to consider edge weights for RWPE')
    # Model settings
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--gnn_dropout', type=float, default=0.0)
    parser.add_argument('--attn_dropout', type=float, default=0.0)
    parser.add_argument('--mlp_dropout', type=float, default=0.0)
    parser.add_argument('--attn_type', type=str, default='sga')
    parser.add_argument('--num_heads', type=int, default=1)
    # Training settings
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay for optimization')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--train_node_embedding', action='store_true',
                        help="also train free-parameter node embeddings together with GNN")
    parser.add_argument('--pretrained_node_embedding', type=str, default=None,
                        help="load pretrained node embeddings as additional node features")
    # Testing settings
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=1000000,
                        help='eval batch size should be largest the GPU memory can take - the '
                             'same is not necessarily true at training time')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
