import torch

def churn(pred_1, pred_2):
    """
    Shows the disagreement between two model's prediction. Higher the churn, more the disagreement. Bounds: [0,1]. Takes to prediction arrays as inputs and retuns churn.
    """
    assert torch.tensor(pred_1).shape == torch.tensor(pred_2).shape, "Size mismatch between pred_1 and pred_2"

    match = torch.unique((pred_1 == pred_2), return_counts=True)
    disagreement = match[1][match[0] == False]
    churn = disagreement / torch.numel(pred_1)

    return churn

pred_1 = torch.randint(0, 100, (1000,))
pred_2 = torch.randint(0, 100, (1000,))
churn(pred_1, pred_2)
