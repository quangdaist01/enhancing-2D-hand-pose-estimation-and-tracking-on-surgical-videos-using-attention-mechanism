import numpy as np
import json
import os
import sys

import eval_helpers
from eval_helpers import Joint


def computeMetrics(scores_all, labels_all, num_gt_all):
    ap_all = np.zeros((num_gt_all.shape[0] + 1, 1))
    rec_all = np.zeros((num_gt_all.shape[0] + 1, 1))
    pre_all = np.zeros((num_gt_all.shape[0] + 1, 1))
    # iterate over joints
    for j in range(num_gt_all.shape[0]):
        scores = np.zeros([0, 0], dtype=np.float32)
        labels = np.zeros([0, 0], dtype=np.int8)
        # iterate over images
        for imgidx in range(num_gt_all.shape[1]):
            scores = np.append(scores, scores_all[j][imgidx])
            labels = np.append(labels, labels_all[j][imgidx])
        # compute recall/precision values
        n_gt = sum(num_gt_all[j, :])
        precision, recall, scores_sorted_idxs = eval_helpers.computeRPC(scores, labels, n_gt)
        if len(precision) > 0:
            ap_all[j] = eval_helpers.VOCap(recall, precision) * 100
            pre_all[j] = precision[len(precision) - 1] * 100
            rec_all[j] = recall[len(recall) - 1] * 100

    # mean cuối cùng ở phần từ cuối cùng
    idxs = np.argwhere(~np.isnan(ap_all[:num_gt_all.shape[0], 0]))
    ap_all[num_gt_all.shape[0]] = ap_all[idxs, 0].mean()
    idxs = np.argwhere(~np.isnan(rec_all[:num_gt_all.shape[0], 0]))
    rec_all[num_gt_all.shape[0]] = rec_all[idxs, 0].mean()
    idxs = np.argwhere(~np.isnan(pre_all[:num_gt_all.shape[0], 0]))
    pre_all[num_gt_all.shape[0]] = pre_all[idxs, 0].mean()

    return ap_all, pre_all, rec_all


def evaluateAP(gt_frames_all, pred_frames_all, output_dir, bSaveAll=True, bSaveSeq=False):
    dist_thresh = 0.5

    seqidxs = []
    for imgidx in range(len(gt_frames_all)):
        seqidxs += [gt_frames_all[imgidx]["seq_id"]]
    seqidxs = np.array(seqidxs)

    seqidxsUniq = np.unique(seqidxs)
    nSeq = len(seqidxsUniq)

    names = Joint().name
    names['21'] = 'total'

    if bSaveSeq:
        for si in range(nSeq):
            print("seqidx: %d/%d" % (si + 1, nSeq))

            # extract frames IDs for the sequence
            imgidxs = np.argwhere(seqidxs == seqidxsUniq[si])
            seq_name = gt_frames_all[imgidxs[0, 0]]["seq_name"]

            gt_frames = [gt_frames_all[imgidx] for imgidx in imgidxs.flatten().tolist()]
            pred_frames = [pred_frames_all[imgidx] for imgidx in imgidxs.flatten().tolist()]

            # assign predicted poses to GT poses
            # scores: part detections scores
            # labels: positive/negative labels
            # nGT: number of annotated joints per image
            scores, labels, num_gt, _ = eval_helpers.assignGTmulti(gt_frames, pred_frames, dist_thresh)

            # compute average precision (AP), precision and recall per part
            ap, precision, recall = computeMetrics(scores, labels, num_gt)
            metricsSeq = {'ap': ap.flatten().tolist(), 'pre': precision.flatten().tolist(), 'rec': recall.flatten().tolist(), 'names': names}

            filename = output_dir + '/' + seq_name + '_AP_metrics.json'
            print('saving results to', filename)
            eval_helpers.writeJson(metricsSeq, filename)

    # assign predicted poses to GT poses
    scoresAll, labelsAll, nGTall, _ = eval_helpers.assignGTmulti(gt_frames_all, pred_frames_all, dist_thresh)

    # compute average precision (AP), precision and recall per part
    apAll, preAll, recAll = computeMetrics(scoresAll, labelsAll, nGTall)
    if bSaveAll:
        metrics = {'ap': apAll.flatten().tolist(), 'pre': preAll.flatten().tolist(), 'rec': recAll.flatten().tolist(), 'names': names}
        filename = output_dir + '/total_AP_metrics.json'
        print('saving results to', filename)
        eval_helpers.writeJson(metrics, filename)

    return apAll, preAll, recAll
