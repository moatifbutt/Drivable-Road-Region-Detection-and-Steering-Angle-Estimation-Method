import numpy as np
import torch

def batch_pix_accuracy(predict, target, labeled):
    pixel_labeled = labeled.sum()
    pixel_correct = ((predict == target) * labeled).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()

def batch_iou(predictions, target, num_classes, labeled):
    predictions = predictions * labeled.long()
    intersection = predictions * (predictions == target).long()

    area_inter = torch.histc(intersection.float(), bins=num_classes, max=num_classes, min=1)
    area_pred = torch.histc(predictions.float(), bins=num_classes, max=num_classes, min=1)
    area_lab = torch.histc(target.float(), bins=num_classes, max=num_classes, min=1)
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
    return area_inter.cpu().numpy(), area_union.cpu().numpy()

def eval_metric(output, target, num_classes):
    _, predictions = torch.max(output.data, 1)
    predictions = predictions + 1
    target = target + 1

    labeled = (target > 0) * (target <= num_classes)
    correct, num_labeled = batch_pix_accuracy(predictions, target, labeled)
    inter, union = batch_iou(predictions, target, num_classes, labeled)
    return [np.round(correct, 5), np.round(num_labeled, 5), np.round(inter, 5), np.round(union, 5)]