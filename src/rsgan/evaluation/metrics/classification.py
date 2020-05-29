import torch


def accuracy(predicted, groundtruth, thresh=0.5):
    """Accuracy on single label classification
    Args:
        predicted (torch.Tensor): batch of probability distributions on classes
        groundtruth (torch.Tensor): batch of probability distributions on classes
    """
    predicted = predicted > thresh
    correct = (predicted.float() == groundtruth).sum().float()
    score = correct / len(groundtruth)
    return score.item()


def precision(predicted, groundtruth, thresh=0.5):
    """Precision on single label classification
    Args:
        predicted (torch.Tensor): batch of probability distributions on classes
        groundtruth (torch.Tensor): batch of probability distributions on classes
    """
    predicted = predicted > thresh
    positives = torch.nonzero(predicted).flatten()
    true_positives = torch.sum(predicted[positives].float() == groundtruth[positives]).float()
    precision = true_positives / len(positives)
    return precision.item()


def recall(predicted, groundtruth, thresh=0.5):
    """Recall on single label classification
    Args:
        predicted (torch.Tensor): batch of probability distributions on classes
        groundtruth (torch.Tensor): batch of probability distributions on classes
    """
    predicted = predicted > thresh
    positives = torch.nonzero(groundtruth == 1).flatten()
    true_positives = torch.sum(predicted[positives].float() == groundtruth[positives]).float()
    recall = true_positives / len(positives)
    return recall.item()
