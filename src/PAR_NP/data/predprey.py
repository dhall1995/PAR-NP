import torch


def add_indicators_and_concat(x_list, indicator_values=None):
    if indicator_values is None:
        indicator_values = torch.tensor([0, 1])
    x_list = [
        torch.cat([x, torch.full(x.shape, indicator_values[idx])], dim=1)
        for idx, x in enumerate(x_list)
    ]
    return torch.cat(x_list, dim=-1)


def pre_process_predprey_batch(batch):
    x_contexts = {"pred": batch["contexts"][0][0], "prey": batch["contexts"][1][0]}
    y_contexts = {"pred": batch["contexts"][0][1], "prey": batch["contexts"][1][1]}
    x_target = {"pred": batch["xt"][0][0], "prey": batch["xt"][1][0]}
    pred_ind, prey_ind = 0, 1
    x_context = add_indicators_and_concat(
        [torch.tensor(x_contexts[animal]) for animal in ["pred", "prey"]],
        indicator_values=torch.tensor([pred_ind, prey_ind]),
    )
    x_target = add_indicators_and_concat(
        [torch.tensor(x_target[animal]) for animal in ["pred", "prey"]],
        indicator_values=torch.tensor([pred_ind, prey_ind]),
    )
    y_context = torch.cat(
        [torch.tensor(y_contexts["pred"]), torch.tensor(y_contexts["prey"])], dim=-1
    )
    y_target = torch.cat(
        [torch.tensor(batch["yt"][0]), torch.tensor(batch["yt"][1])], dim=-1
    )

    return x_context, y_context, x_target, y_target
