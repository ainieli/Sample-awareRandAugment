import torch
import torch.nn.functional as F


class DRASearchLoss(torch.nn.Module):

    def __init__(self):
        super(DRASearchLoss, self).__init__()
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, pred, target, search=False, lb_smooth=0.0):
        if search:
            aug_loss = 0
            ori_loss = 0
            if pred[0] is not None:
                # For augmented prediction
                y = torch.zeros(pred[0].shape).to(pred[0].device).scatter_(-1, target[..., None], 1.0)
            else:
                # For original prediction
                y = torch.zeros(pred[1].shape).to(pred[1].device).scatter_(-1, target[..., None], 1.0)
                
            if lb_smooth > 0:
                y = lb_smooth / y.shape[1] * torch.ones(y.shape).to(y.device) + (1 - lb_smooth) * y
                
            if pred[0] is not None:
                aug_loss = -(F.log_softmax(pred[0], dim=-1) * y).sum(dim=-1).mean()
                
            if pred[1] is not None:
                ori_loss = self.ce(pred[1], target)

            loss = aug_loss + ori_loss
        else:
            loss = self.ce(pred[0], target)

        return loss


