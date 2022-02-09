import numpy as np
from shapely import geometry
import sys
import os
import json
import glob
from convert import convert_videos

MIN_SCORE = -9999
MAX_TRACK_ID = 10000


class Joint:
    def __init__(self):
        self.count = 21

        self.wrist = 0
        self.thumb_k = 1
        self.thumb_b = 2
        self.thumb_m = 3
        self.thumb_t = 4
        self.index_k = 5
        self.index_b = 6
        self.index_m = 7
        self.index_t = 8
        self.middle_k = 9
        self.middle_b = 10
        self.middle_m = 11
        self.middle_t = 12
        self.ring_k = 13
        self.ring_b = 14
        self.ring_m = 15
        self.ring_t = 16
        self.pinky_k = 17
        self.pinky_b = 18
        self.pinky_m = 19
        self.pinky_t = 20

        self.name = {}
        self.name[self.wrist] = "wrist"
        self.name[self.thumb_k] = "thumb_k"
        self.name[self.thumb_b] = "thumb_b"
        self.name[self.thumb_m] = "thumb_m"
        self.name[self.thumb_t] = "thumb_t"
        self.name[self.index_k] = "index_k"
        self.name[self.index_b] = "index_b"
        self.name[self.index_m] = "index_m"
        self.name[self.index_t] = "index_t"
        self.name[self.middle_k] = "middle_k"
        self.name[self.middle_b] = "middle_b"
        self.name[self.middle_m] = "middle_m"
        self.name[self.middle_t] = "middle_t"
        self.name[self.ring_k] = "ring_k"
        self.name[self.ring_b] = "ring_b"
        self.name[self.ring_m] = "ring_m"
        self.name[self.ring_t] = "ring_t"
        self.name[self.pinky_k] = "pinky_k"
        self.name[self.pinky_b] = "pinky_b"
        self.name[self.pinky_m] = "pinky_m"
        self.name[self.pinky_t] = "pinky_t"

        self.symmetric_joint = {}
        self.symmetric_joint[self.name[self.wrist]] = -1
        self.symmetric_joint[self.name[self.thumb_k]] = -1
        self.symmetric_joint[self.name[self.thumb_b]] = -1
        self.symmetric_joint[self.name[self.thumb_m]] = -1
        self.symmetric_joint[self.name[self.thumb_t]] = -1
        self.symmetric_joint[self.name[self.index_k]] = -1
        self.symmetric_joint[self.name[self.index_b]] = -1
        self.symmetric_joint[self.name[self.index_m]] = -1
        self.symmetric_joint[self.name[self.index_t]] = -1
        self.symmetric_joint[self.name[self.middle_k]] = -1
        self.symmetric_joint[self.name[self.middle_b]] = -1
        self.symmetric_joint[self.name[self.middle_m]] = -1
        self.symmetric_joint[self.name[self.middle_t]] = -1
        self.symmetric_joint[self.name[self.ring_k]] = -1
        self.symmetric_joint[self.name[self.ring_b]] = -1
        self.symmetric_joint[self.name[self.ring_m]] = -1
        self.symmetric_joint[self.name[self.ring_t]] = -1
        self.symmetric_joint[self.name[self.pinky_k]] = -1
        self.symmetric_joint[self.name[self.pinky_b]] = -1
        self.symmetric_joint[self.name[self.pinky_m]] = -1
        self.symmetric_joint[self.name[self.pinky_t]] = -1


def getPointGTbyID(points, pidx):
    point = []
    for i in range(len(points)):
        if points[i]["id"] != None and points[i]["id"][0] == pidx:  # if joint id matches
            point = points[i]
            break

    return point


def getHeadSize(x1, y1, x2, y2):
    # (Note by Nate - 9/10/2020) Too large of a "headSize" caused many false matches when
    # calculating shortest distances
    # Tough to get some good reference measurement. I can go for the tip of the thumb
    # which is about ~0.2 the size of the bounding box for a full hand.

    headSize = 0.2 * np.linalg.norm(np.subtract([x2, y2], [x1, y1]));
    return headSize


def formatCell(val, delim):
    return "{:>5}".format("%1.1f" % val) + delim


def getHeader():
    strHeader = "&"
    strHeader += " Wrist &"
    strHeader += " Thumb &"
    strHeader += " Index &"
    strHeader += " Middle &"
    strHeader += " Ring &"
    strHeader += " Pinky &"
    strHeader += " Total%s" % ("\\" + "\\")
    return strHeader


def getMotHeader():
    strHeader = "&"
    strHeader += " MOTA &"
    strHeader += " MOTA &"
    strHeader += " MOTA &"
    strHeader += " MOTA &"
    strHeader += " MOTA &"
    strHeader += " MOTA &"
    strHeader += " MOTA &"
    strHeader += " MOTP &"
    strHeader += " Prec &"
    strHeader += " Rec  %s\n" % ("\\" + "\\")
    strHeader += "&"
    strHeader += " Wrist &"
    strHeader += " Thumb &"
    strHeader += " Index &"
    strHeader += " Middle &"
    strHeader += " Ring &"
    strHeader += " Pinky &"
    strHeader += " Total&"
    strHeader += " Total&"
    strHeader += " Total&"
    strHeader += " Total%s" % ("\\" + "\\")

    return strHeader


def getCum(vals):
    cum = [];
    -1
    cum += [(vals[[Joint().wrist], 0].mean())]
    cum += [(vals[[Joint().thumb_k, Joint().thumb_b, Joint().thumb_m, Joint().thumb_t], 0].mean())]
    cum += [(vals[[Joint().index_k, Joint().index_b, Joint().index_m, Joint().index_t], 0].mean())]
    cum += [(vals[[Joint().middle_k, Joint().middle_b, Joint().middle_m, Joint().middle_t], 0].mean())]
    cum += [(vals[[Joint().ring_k, Joint().ring_b, Joint().ring_m, Joint().ring_t], 0].mean())]
    cum += [(vals[[Joint().pinky_k, Joint().pinky_b, Joint().pinky_m, Joint().pinky_t], 0].mean())]
    for i in range(Joint().count, len(vals)):
        cum += [vals[i, 0]]
    return cum


def getFormatRow(cum):
    row = "&"
    for i in range(len(cum) - 1):
        row += formatCell(cum[i], " &")
    row += formatCell(cum[len(cum) - 1], (" %s" % "\\" + "\\"))
    return row


def printTable(vals, motHeader=False):
    cum = getCum(vals)
    row = getFormatRow(cum)
    if motHeader:
        header = getMotHeader()
    else:
        header = getHeader()
    print(header)
    print(row)
    return header + "\n", row + "\n"


def printTableTracking():
    cum = getCum(vals)
    row = getFormatRow(cum)
    print(getHeader())
    print(row)
    return getHeader() + "\n", row + "\n"


# compute recall/precision curve (RPC) values
def computeRPC(scores, labels, total_positive):
    precision = np.zeros(len(scores))
    recall = np.zeros(len(scores))
    num_positive = 0;

    idxsSort = np.array(scores).argsort()[::-1]
    labelsSort = labels[idxsSort];

    for sidx in range(len(idxsSort)):
        if labelsSort[sidx] == 1:
            num_positive += 1
        # recall: how many true positives were found out of the total number of positives?
        recall[sidx] = 1.0 * num_positive / total_positive
        # precision: how many true positives were found out of the total number of samples?
        precision[sidx] = 1.0 * num_positive / (sidx + 1)

    return precision, recall, idxsSort


# compute Average Precision using recall/precision values
def VOCap(rec, prec):
    mpre = np.zeros([1, 2 + len(prec)])
    mpre[0, 1:len(prec) + 1] = prec
    mrec = np.zeros([1, 2 + len(rec)])
    mrec[0, 1:len(rec) + 1] = rec
    mrec[0, len(rec) + 1] = 1.0

    for i in range(mpre.size - 2, -1, -1):
        mpre[0, i] = max(mpre[0, i], mpre[0, i + 1])

    i = np.argwhere(~np.equal(mrec[0, 1:], mrec[0, :mrec.shape[1] - 1])) + 1
    i = i.flatten()

    # compute area under the curve
    ap = np.sum(np.multiply(np.subtract(mrec[0, i], mrec[0, i - 1]), mpre[0, i]))

    return ap


def get_data_dir():
    dataDir = "./"
    return dataDir


def help(msg=''):
    sys.stderr.write(msg + '\n')
    exit()


def process_arguments(argv):
    mode = 'multi'

    if len(argv) > 3:
        mode = str.lower(argv[3])
    elif len(argv) < 3 or len(argv) > 4:
        help()

    gt_file = argv[1]
    pred_file = argv[2]

    if not os.path.exists(gt_file):
        help('Given ground truth directory does not exist!\n')

    if not os.path.exists(pred_file):
        help('Given prediction directory does not exist!\n')

    return gt_file, pred_file, mode


def process_arguments_server(argv):
    'multi'

    print(len(argv))
    assert (len(argv) == 10, "Wrong number of arguments")

    gt_dir = argv[1]
    pred_dir = argv[2]
    mode = str.lower(argv[3])
    evaltrack = argv[4]
    shortname = argv[5]
    chl = argv[6]
    shortname_uid = argv[7]
    shakey = argv[8]
    timestamp = argv[9]
    if not os.path.exists(gt_dir):
        help('Given ground truth does not exist!\n')

    if not os.path.exists(pred_dir):
        help('Given prediction does not exist!\n')

    return gt_dir, pred_dir, mode, evaltrack, shortname, chl, shortname_uid, shakey, timestamp


def load_data(argv):
    dataDir = get_data_dir()

    gt_file, pred_file, mode = process_arguments(argv)
    gtFilename = dataDir + gt_file
    predFilename = dataDir + pred_file

    # load ground truth (GT)
    with open(gtFilename) as data_file:
        data = json.load(data_file)
    gtFramesAll = data

    # load predictions
    with open(predFilename) as data_file:
        data = json.load(data_file)
    prFramesAll = data

    return gtFramesAll, prFramesAll


def cleanupData(gtFramesAll, prFramesAll):
    # remove all GT frames with empty annorects and remove corresponding entries from predictions
    imgidxs = []
    for imgidx in range(len(gtFramesAll)):
        if len(gtFramesAll[imgidx]["annorect"]) > 0:
            imgidxs += [imgidx]
    gtFramesAll = [gtFramesAll[imgidx] for imgidx in imgidxs]
    prFramesAll = [prFramesAll[imgidx] for imgidx in imgidxs]

    # remove all gt rectangles that do not have annotations
    for imgidx in range(len(gtFramesAll)):
        gtFramesAll[imgidx]["annorect"] = removeRectsWithoutPoints(gtFramesAll[imgidx]["annorect"])
        prFramesAll[imgidx]["annorect"] = removeRectsWithoutPoints(prFramesAll[imgidx]["annorect"])

    return gtFramesAll, prFramesAll


def removeIgnoredPointsRects(rects, polyList):
    ridxs = list(range(len(rects)))
    for ridx in range(len(rects)):
        points = rects[ridx]["annopoints"][0]["point"]
        pidxs = list(range(len(points)))
        for pidx in range(len(points)):
            pt = geometry.Point(points[pidx]["x"][0], points[pidx]["y"][0])
            bIgnore = False
            for poidx in range(len(polyList)):
                poly = polyList[poidx]
                if poly.contains(pt):
                    bIgnore = True
                    break
            if bIgnore:
                pidxs.remove(pidx)
        points = [points[pidx] for pidx in pidxs]
        if len(points) > 0:
            rects[ridx]["annopoints"][0]["point"] = points
        else:
            ridxs.remove(ridx)
    rects = [rects[ridx] for ridx in ridxs]
    return rects


def removeIgnoredPoints(gtFramesAll, prFramesAll):
    []
    for imgidx in range(len(gtFramesAll)):
        if ("ignore_regions" in gtFramesAll[imgidx].keys() and
                len(gtFramesAll[imgidx]["ignore_regions"]) > 0):
            regions = gtFramesAll[imgidx]["ignore_regions"]
            polyList = []
            for ridx in range(len(regions)):
                points = regions[ridx]["point"]
                pointList = []
                for pidx in range(len(points)):
                    pt = geometry.Point(points[pidx]["x"][0], points[pidx]["y"][0])
                    pointList += [pt]
                poly = geometry.Polygon([[p.x, p.y] for p in pointList])
                polyList += [poly]

            rects = prFramesAll[imgidx]["annorect"]
            prFramesAll[imgidx]["annorect"] = removeIgnoredPointsRects(rects, polyList)
            rects = gtFramesAll[imgidx]["annorect"]
            gtFramesAll[imgidx]["annorect"] = removeIgnoredPointsRects(rects, polyList)

    return gtFramesAll, prFramesAll


def rectHasPoints(rect):
    return (("annopoints" in rect.keys()) and
            (len(rect["annopoints"]) > 0 and len(rect["annopoints"][0]) > 0) and
            ("point" in rect["annopoints"][0].keys()))


def removeRectsWithoutPoints(rects):
    idxsPr = []
    for ridxPr in range(len(rects)):
        if rectHasPoints(rects[ridxPr]):
            idxsPr += [ridxPr];
    rects = [rects[ridx] for ridx in idxsPr]
    return rects


def load_data_dir(argv):
    gt_dir, pred_dir, mode = process_arguments(argv)
    if not os.path.exists(gt_dir):
        help('Given GT directory ' + gt_dir + ' does not exist!\n')
    if not os.path.exists(pred_dir):
        help('Given prediction directory ' + pred_dir + ' does not exist!\n')
    filenames = glob.glob(gt_dir + "/*.json")
    gtFramesAll = []
    prFramesAll = []

    for i in range(len(filenames)):
        # load each annotation json file
        with open(filenames[i]) as data_file:
            data = json.load(data_file)
        if not "annolist" in data:
            data = convert_videos(data)[0]
        gt = data["annolist"]
        for imgidx in range(len(gt)):
            gt[imgidx]["seq_id"] = i
            gt[imgidx]["seq_name"] = os.path.basename(filenames[i]).split('.')[0]
            for idx_gt in range(len(gt[imgidx]["annorect"])):
                if "track_id" in gt[imgidx]["annorect"][idx_gt].keys():
                    # adjust track_ids to make them unique across all sequences
                    assert (gt[imgidx]["annorect"][idx_gt]["track_id"][0] < MAX_TRACK_ID)
                    gt[imgidx]["annorect"][idx_gt]["track_id"][0] += i * MAX_TRACK_ID
        gtFramesAll += gt
        gtBasename = os.path.basename(filenames[i])
        predFilename = pred_dir + gtBasename

        if not os.path.exists(predFilename):
            raise IOError('Prediction file ' + predFilename + ' does not exist')

        # load predictions
        with open(predFilename) as data_file:
            data = json.load(data_file)
        if not "annolist" in data:
            data = convert_videos(data)[0]
        pr = data["annolist"]
        if len(pr) != len(gt):
            raise Exception('# prediction frames %d != # GT frames %d for %s' % (len(pr), len(gt), predFilename))
        for imgidx in range(len(pr)):
            track_id_frame = []
            for ridxPr in range(len(pr[imgidx]["annorect"])):
                if "track_id" in pr[imgidx]["annorect"][ridxPr].keys():
                    track_id = pr[imgidx]["annorect"][ridxPr]["track_id"][0]
                    track_id_frame += [track_id]
                    # adjust track_ids to make them unique across all sequences
                    assert (track_id < MAX_TRACK_ID)
                    pr[imgidx]["annorect"][ridxPr]["track_id"][0] += i * MAX_TRACK_ID
            track_id_frame_unique = np.unique(np.array(track_id_frame)).tolist()
            if len(track_id_frame) != len(track_id_frame_unique):
                raise Exception('Non-unique tracklet IDs found in frame %s of prediction %s' % (pr[imgidx]["image"][0]["name"], predFilename))
        prFramesAll += pr

    gtFramesAll, prFramesAll = cleanupData(gtFramesAll, prFramesAll)

    gtFramesAll, prFramesAll = removeIgnoredPoints(gtFramesAll, prFramesAll)

    return gtFramesAll, prFramesAll


def writeJson(val, fname):
    with open(fname, 'w') as data_file:
        json.dump(val, data_file)


def assignGTmulti(gt_frames_all, pred_frames_all, dish_thresh):
    assert (len(gt_frames_all) == len(pred_frames_all))

    num_joint = Joint().count
    # part detection scores
    scores_all = {}
    # positive / negative labels
    labels_all = {}
    # number of annotated GT joints per image
    num_GT_all = np.zeros([num_joint, len(gt_frames_all)])
    for pidx in range(num_joint):
        scores_all[pidx] = {}
        labels_all[pidx] = {}
        for img_idx in range(len(gt_frames_all)):
            scores_all[pidx][img_idx] = np.zeros([0, 0], dtype=np.float32)
            labels_all[pidx][img_idx] = np.zeros([0, 0], dtype=np.int8)

    # number of GT poses
    num_GT_hands = np.zeros((len(gt_frames_all), 1))
    # number of predicted poses
    num_pred_hands = np.zeros((len(gt_frames_all), 1))

    # container to save info for computing MOT metrics
    MOT_all = {}

    for img_idx in range(len(gt_frames_all)):
        # distance between predicted and GT joints
        dist = np.full((len(pred_frames_all[img_idx]["annorect"]), len(gt_frames_all[img_idx]["annorect"]), num_joint), np.inf)
        # score of the predicted joint
        scores = np.full((len(pred_frames_all[img_idx]["annorect"]), num_joint), np.nan)
        # body joint prediction exist
        has_pred = np.zeros((len(pred_frames_all[img_idx]["annorect"]), num_joint), dtype=bool)
        # body joint is annotated
        has_gt = np.zeros((len(gt_frames_all[img_idx]["annorect"]), num_joint), dtype=bool)

        track_idx_gt = []
        track_idx_pred = []
        idxs_pred = []
        for idx_gt in range(len(pred_frames_all[img_idx]["annorect"])):
            if (("annopoints" in pred_frames_all[img_idx]["annorect"][idx_gt].keys()) and
                    ("point" in pred_frames_all[img_idx]["annorect"][idx_gt]["annopoints"][0].keys())):
                idxs_pred += [idx_gt];
        pred_frames_all[img_idx]["annorect"] = [pred_frames_all[img_idx]["annorect"][idx] for idx in idxs_pred]

        num_pred_hands[img_idx, 0] = len(pred_frames_all[img_idx]["annorect"])
        num_GT_hands[img_idx, 0] = len(gt_frames_all[img_idx]["annorect"])
        # iterate over GT poses
        for idx_gt in range(len(gt_frames_all[img_idx]["annorect"])):
            # GT pose
            rect_gt = gt_frames_all[img_idx]["annorect"][idx_gt]
            if "track_id" in rect_gt.keys():
                track_idx_gt += [rect_gt["track_id"][0]]
            points_gt = []
            if len(rect_gt["annopoints"]) > 0:
                points_gt = rect_gt["annopoints"][0]["point"]
            # iterate over all possible body joints
            for idx_gt in range(num_joint):
                # GT joint in LSP format
                point_gt = getPointGTbyID(points_gt, idx_gt)
                if len(point_gt) > 0:
                    has_gt[idx_gt, idx_gt] = True

        # iterate over predicted poses
        for idx_gt in range(len(pred_frames_all[img_idx]["annorect"])):
            # predicted pose
            rect_pred = pred_frames_all[img_idx]["annorect"][idx_gt]
            if "track_id" in rect_pred.keys():
                track_idx_pr += [rect_pred["track_id"][0]]
            points_pred = rect_pred["annopoints"][0]["point"]
            for idx_gt in range(num_joint):
                # predicted joint in LSP format
                point_pred = getPointGTbyID(points_pred, idx_gt)
                if len(point_pred) > 0:
                    if not ("score" in point_pred.keys()):
                        # use minimum score if predicted score is missing
                        if img_idx == 0:
                            print('WARNING: prediction score is missing. Setting fallback score={}'.format(MIN_SCORE))
                        scores[idx_gt, idx_gt] = MIN_SCORE
                    else:
                        scores[idx_gt, idx_gt] = point_pred["score"][0]
                    has_pred[idx_gt, idx_gt] = True

        if len(pred_frames_all[img_idx]["annorect"]) and len(gt_frames_all[img_idx]["annorect"]):
            # predictions and GT are present
            # iterate over GT poses
            for idx_gt in range(len(gt_frames_all[img_idx]["annorect"])):
                # GT pose
                rect_gt = gt_frames_all[img_idx]["annorect"][idx_gt]
                # compute reference distance as head size
                head_size = getHeadSize(rect_gt["x1"][0], rect_gt["y1"][0],
                                       rect_gt["x2"][0], rect_gt["y2"][0])
                points_gt = []
                if len(rect_gt["annopoints"]) > 0:
                    points_gt = rect_gt["annopoints"][0]["point"]
                # iterate over predicted poses
                for idx_pred in range(len(pred_frames_all[img_idx]["annorect"])):
                    # predicted pose
                    rect_pred = pred_frames_all[img_idx]["annorect"][idx_pred]
                    points_pred = rect_pred["annopoints"][0]["point"]

                    # iterate over all possible body joints
                    for id in range(num_joint):
                        # GT joint
                        point_gt = getPointGTbyID(points_gt, id)
                        # predicted joint
                        point_pred = getPointGTbyID(points_pred, id)
                        # compute distance between predicted and GT joint locations
                        if has_pred[idx_gt, id] and has_gt[idx_gt, id]:
                            point_gt = [point_gt["x"][0], point_gt["y"][0]]
                            point_pr = [point_pred["x"][0], point_pred["y"][0]]
                            dist[idx_pred, idx_gt, id] = np.linalg.norm(np.subtract(point_gt, point_pr)) / head_size

            dist = np.array(dist)
            has_gt = np.array(has_gt)

            # number of annotated joints
            num_gt_points = np.sum(has_gt, axis=1)
            matches = dist <= dish_thresh
            PCK = 1.0 * np.sum(matches, axis=2)
            for idx_pred in range(has_pred.shape[0]):
                for idx_gt in range(has_gt.shape[0]):
                    if num_gt_points[idx_gt] > 0:
                        PCK[idx_pred, idx_gt] = PCK[idx_pred, idx_gt] / num_gt_points[idx_gt]

            # preserve best GT match only
            idx = np.argmax(PCK, axis=1)
            for idx_gt in range(PCK.shape[0]):
                for idx_pred in range(PCK.shape[1]):
                    if idx_gt != idx[idx_pred]:
                        PCK[idx_pred, idx_gt] = 0
            pred_to_gt = np.argmax(PCK, axis=0)
            val = np.max(PCK, axis=0)
            pred_to_gt[val == 0] = -1

            # info to compute MOT metrics
            MOT = {}
            for idx_gt in range(num_joint):
                MOT[idx_gt] = {}

            for idx_gt in range(num_joint):
                idx_gt = np.argwhere(has_gt[:, idx_gt] == True);
                idx_gt = idx_gt.flatten().tolist()
                idx_gt = np.argwhere(has_pred[:, idx_gt] == True);
                idx_gt = idx_gt.flatten().tolist()

                MOT[idx_gt]["trackidxGT"] = [track_idx_gt[idx] for idx in idx_gt]
                MOT[idx_gt]["trackidxPr"] = [track_idx_pr[idx] for idx in idx_gt]
                MOT[idx_gt]["ridxsGT"] = np.array(idx_gt)
                MOT[idx_gt]["ridxsPr"] = np.array(idx_gt)
                MOT[idx_gt]["dist"] = np.full((len(idx_gt), len(idx_gt)), np.nan)
                for iPr in range(len(idx_gt)):
                    for iGT in range(len(idx_gt)):
                        if matches[idx_gt[iPr], idx_gt[iGT], idx_gt]:
                            MOT[idx_gt]["dist"][iGT, iPr] = dist[idx_gt[iPr], idx_gt[iGT], idx_gt]

            # assign predicted poses to GT poses
            # loop over poses in a frame
            for idx_pred in range(has_pred.shape[0]):
                if idx_pred in pred_to_gt:  # pose matches to GT
                    # GT pose that matches the predicted pose
                    idx_gt = np.argwhere(pred_to_gt == idx_pred)
                    assert (idx_gt.size == 1)
                    idx_gt = idx_gt[0, 0]
                    score = scores[idx_pred, :]
                    match= np.squeeze(matches[idx_pred, idx_gt, :])
                    hp = has_pred[idx_pred, :]
                    for i in range(len(hp)):
                        if hp[i]:
                            scores_all[i][imgidx] = np.append(scores_all[i][imgidx], score[i])
                            labels_all[i][imgidx] = np.append(labels_all[i][imgidx], match[i])

                else:  # no matching to GT
                    score = scores[idx_pred, :]
                    match = np.zeros([matches.shape[2], 1], dtype=bool)
                    hp = has_pred[idx_pred, :]
                    for idx_pred in range(len(hp)):
                        if hp[idx_pred]:
                            scores_all[idx_pred][img_idx] = np.append(scores_all[idx_pred][img_idx], score[idx_pred])
                            labels_all[idx_pred][img_idx] = np.append(labels_all[idx_pred][img_idx], match[idx_pred])
        else:
            if not len(gt_frames_all[img_idx]["annorect"]):
                # loop over poses in a frame
                # No GT available. All predictions are false positives
                for id in range(has_pred.shape[0]):
                    score = scores[id, :]
                    match = np.zeros([num_joint, 1], dtype=bool)
                    hp = has_pred[id, :]
                    for id in range(len(hp)):
                        if hp[id]:
                            scores_all[id][img_idx] = np.append(scores_all[id][img_idx], score[id])
                            labels_all[id][img_idx] = np.append(labels_all[id][img_idx], match[id])
            MOT = {}
            for id in range(num_joint):
                MOT[id] = {}
            for id in range(num_joint):
                idx_gt = [0]
                id = [0]
                MOT[id]["trackidxGT"] = [0]
                MOT[id]["trackidxPr"] = [0]
                MOT[id]["ridxsGT"] = np.array(idx_gt)
                MOT[id]["ridxsPr"] = np.array(id)
                MOT[id]["dist"] = np.full((len(idx_gt), len(id)), np.nan)

        # save number of GT joints
        for idx_gt in range(has_gt.shape[0]):
            hg = has_gt[idx_gt, :]
            for idx_gt in range(len(hg)):
                num_GT_all[idx_gt, img_idx] += hg[idx_gt]

        MOT_all[img_idx] = MOT

    return scores_all, labels_all, num_GT_all, MOT_all
