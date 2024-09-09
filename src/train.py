import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def train(model, optimizer, data, args, device):
    model.train()

    idx_loader = DataLoader(range(data.edge_index.size(-1)),
                            args.batch_size, shuffle=True)
    total_loss = 0
    for idx in idx_loader:
        optimizer.zero_grad()

        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        edge_weight = data.edge_weight.to(device)
        edge_label_index = data.edge_index[:, idx].to(device)
        edge_label = data.edge_weight[idx].to(device)

        out = model(x, edge_index, edge_weight, edge_label_index)
        loss = get_loss(out, edge_label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * args.batch_size

        del x, edge_index, edge_weight, edge_label_index, edge_label
        torch.cuda.empty_cache()

    return total_loss / data.edge_index.size(-1)


def get_loss(y_pred, y_true):
    return F.mse_loss(y_pred.view(-1), y_true.to(torch.float))
