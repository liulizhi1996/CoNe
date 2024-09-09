import torch
from torch.utils.data import DataLoader

from src.utils.eval_utils import evaluate_rmse


@torch.no_grad()
def test(model, data, args, device):
    model.eval()
    test_pred, test_true = get_preds(model, data, args, device)
    result = evaluate_rmse(test_pred, test_true)
    return result


@torch.no_grad()
def get_preds(model, data, args, device):
    idx_loader = DataLoader(range(data.test_edge_index.size(-1)),
                            args.eval_batch_size, shuffle=False)
    y_pred, y_true = [], []
    for idx in idx_loader:
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        edge_weight = data.edge_weight.to(device)
        edge_label_index = data.test_edge_index[:, idx].to(device)
        edge_label = data.test_edge_weight[idx].to(device)

        out = model(x, edge_index, edge_weight, edge_label_index)
        y_true.append(edge_label.view(-1).cpu().to(torch.float))
        y_pred.append(out.view(-1).cpu())

        del x, edge_index, edge_weight, edge_label_index, edge_label
        torch.cuda.empty_cache()

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    return pred, true
