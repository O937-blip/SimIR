import sys

import  numpy as np
import torch.nn as nn
import torch
from skimage import measure
import  numpy
from detectron2.evaluation.evaluator import DatasetEvaluator
import logging
from detectron2.utils.comm import all_gather, is_main_process, synchronize
logger = logging.getLogger(__name__)
from collections import OrderedDict

class SirstEvaluator(DatasetEvaluator):
    """Computes pixAcc and mIoU metric scores
    """
    def __init__(self, nclass=1, bins=10):  #bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
        self.nclass = nclass
        self.bins = bins
        self._cpu_device = torch.device("cpu")
        self.tp_arr = torch.zeros(self.bins+1,device=self._cpu_device)
        self.pos_arr = torch.zeros(self.bins+1,device=self._cpu_device)
        self.fp_arr = torch.zeros(self.bins+1,device=self._cpu_device)
        self.neg_arr = torch.zeros(self.bins+1,device=self._cpu_device)
        self.class_pos=torch.zeros(self.bins+1,device=self._cpu_device)
        self.total = 0
        self.FA = np.zeros(self.bins + 1)
        self.PD = np.zeros(self.bins + 1)
        self.target = np.zeros(self.bins + 1)

    def process(self, batch, results):
        preds = results['preds']
        labels = results['labels']
        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels, self.nclass)
        correct = correct.to(self.total_correct)
        labeled = labeled.to(self.total_correct)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union
        for iBin in range(self.bins+1):
            score_thresh = (iBin + 0.0) / self.bins
            # print(iBin, "-th, score_thresh: ", score_thresh)
            i_tp, i_pos, i_fp, i_neg,i_class_pos = cal_tp_pos_fp_neg(preds, labels, self.nclass,score_thresh)
            i_tp = i_tp.to(self.tp_arr)
            i_pos = i_pos.to(self.tp_arr)
            i_neg = i_neg.to(self.tp_arr)
            i_class_pos = i_class_pos.to(self.tp_arr)
            i_fp = i_fp.to(self.tp_arr)
            self.tp_arr[iBin]   += i_tp
            self.pos_arr[iBin]  += i_pos
            self.fp_arr[iBin]   += i_fp
            self.neg_arr[iBin]  += i_neg
            self.class_pos[iBin]+=i_class_pos
            self.h = preds.size(2)
            self.w = preds.size(3)

            score_thresh = (iBin + 0.0) / self.bins
            predits = np.array((preds.sigmoid() > score_thresh).cpu()).astype('int64')
            predits = np.reshape(predits, (self.h, self.w))  # 512
            labelss = np.array((labels).cpu()).astype('int64')  # P
            labelss = np.reshape(labelss, (self.h, self.w))  # 512
            image = measure.label(predits, connectivity=2)
            coord_image = measure.regionprops(image)
            label = measure.label(labelss, connectivity=2)
            coord_label = measure.regionprops(label)

            self.target[iBin] += len(coord_label)
            self.image_area_total = []
            self.image_area_match = []
            self.distance_match = []
            self.dismatch = []

            for K in range(len(coord_image)):
                area_image = np.array(coord_image[K].area)
                self.image_area_total.append(area_image)

            for i in range(len(coord_label)):
                centroid_label = np.array(list(coord_label[i].centroid))
                for m in range(len(coord_image)):
                    centroid_image = np.array(list(coord_image[m].centroid))
                    distance = np.linalg.norm(centroid_image - centroid_label)
                    area_image = np.array(coord_image[m].area)
                    if distance < 3:
                        self.distance_match.append(distance)
                        self.image_area_match.append(area_image)

                        del coord_image[m]
                        break

            self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
            self.FA[iBin] += np.sum(self.dismatch)
            self.PD[iBin] += len(self.distance_match)
        self.total = self.total + 1



    def evaluate(self):
        synchronize()
        self.tp_arr = sum(all_gather(self.tp_arr))
        self.fp_arr = sum(all_gather(self.fp_arr))
        self.pos_arr = sum(all_gather(self.pos_arr))
        self.neg_arr = sum(all_gather(self.neg_arr))
        self.class_pos = sum(all_gather(self.class_pos))
        self.total_correct = sum(all_gather(self.total_correct))
        self.total_label = sum(all_gather(self.total_label))
        self.total_inter = sum(all_gather(self.total_inter))
        self.total_union = sum(all_gather(self.total_union))
        self.FA = sum(all_gather(self.FA))
        self.PD = sum(all_gather(self.PD))
        self.target  = sum(all_gather(self.target))
        self.total = sum(all_gather(self.total))
        if not is_main_process():
            return
        tp_rates    = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates    = self.fp_arr / (self.neg_arr + 0.001)

        recall      = self.tp_arr / (self.pos_arr   + 0.001)
        precision   = self.tp_arr / (self.class_pos + 0.001)

        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()

        Final_FA = self.FA / ((self.h* self.w) * self.total)  # 512
        Final_PD = self.PD / self.target

        #results = {}
        #results["tp_rates"] = str(tp_rates)

        results = OrderedDict({"tp_rates": tp_rates,"fp_rates": fp_rates,"recall": recall, "precision": precision,"pixAcc": pixAcc, "mIoU": mIoU,'PD':Final_PD,'FA':Final_FA})
        logger.info(results)

        return results

    def reset(self):

        self.tp_arr   = torch.zeros([11],device=self._cpu_device)
        self.pos_arr  = torch.zeros([11],device=self._cpu_device)
        self.fp_arr   = torch.zeros([11],device=self._cpu_device)
        self.neg_arr  = torch.zeros([11],device=self._cpu_device)
        self.class_pos = torch.zeros([11],device=self._cpu_device)
        self.total_inter = torch.tensor(0,device=self._cpu_device)
        self.total_union = torch.tensor(0,device=self._cpu_device)
        self.total_correct = torch.tensor(0,device=self._cpu_device)
        self.total_label = torch.tensor(0,device=self._cpu_device)

        self.n = 0
        self.FA = np.zeros([self.bins + 1])
        self.PD = np.zeros([self.bins + 1])


def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):

    predict = (torch.sigmoid(output) > score_thresh).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    intersection = predict * ((predict == target).float())

    tp = intersection.sum()
    fp = (predict * ((predict != target).float())).sum()
    tn = ((1 - predict) * ((predict == target).float())).sum()
    fn = (((predict != target).float()) * (1 - predict)).sum()
    pos = tp + fn
    neg = fp + tn
    class_pos= tp+fp

    return tp, pos, fp, neg, class_pos

def batch_pix_accuracy(output, target):

    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).float()
    pixel_labeled = (target > 0).float().sum()
    pixel_correct = (((predict == target).float())*((target > 0)).float()).sum()



    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):

    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())

    area_inter, _  = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred,  _  = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab,   _  = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union     = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union







