import torch


def accuracy(predicted, groundtruth, thresh=0.5):
    """Accuracy on single label classification
    Args:
        predicted (torch.Tensor): batch of probability distributions on classes
        groundtruth (torch.Tensor): batch of probability distributions on classes
    """
    predicted = predicted > thresh
    correct = torch.sum(predicted.float() == groundtruth).float()
    score = correct / groundtruth.numel()
    return score.item()


def precision(predicted, groundtruth, thresh=0.5):
    """Precision on single label classification
    Args:
        predicted (torch.Tensor): batch of probability distributions on classes
        groundtruth (torch.Tensor): batch of probability distributions on classes
    """
    positives = predicted > thresh
    true_positives = torch.sum(positives[positives].float() == groundtruth[positives]).float()
    precision = true_positives.div(positives.sum())
    return precision.item()


def recall(predicted, groundtruth, thresh=0.5):
    """Recall on single label classification
    Args:
        predicted (torch.Tensor): batch of probability distributions on classes
        groundtruth (torch.Tensor): batch of probability distributions on classes
    """
    predicted = predicted > thresh
    positives = groundtruth == 1
    true_positives = torch.sum(predicted[positives].float() == groundtruth[positives]).float()
    recall = true_positives.div(positives.sum())
    return recall.item()
