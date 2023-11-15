from utils import performances_compute
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


verbose = True

#load prediction_scores from pred.txt
# aaflfhiktb.mp4, 0.0
# abbefwjpxv.mp4, 0.0
# acfambqsgr.mp4, 0.0
# acgqjweafr.mp4, 0.0006
# acpizqtkmg.mp4, 0.0
# adhuztpsff.mp4, 0.0
#reading the json file

def init(file,json_file): # Public Label.json or Private Label.json
    names = np.loadtxt(file, delimiter=",", usecols = 0, dtype=str)
    prediction_scores = np.loadtxt(file, delimiter=",", usecols = 1 , dtype=float)
    with open("/ceph/hpc/data/st2207-pgp-users/DataBase/DeepFake/DFGC-2022/Detection Dataset/"+json_file, "r") as file:
        gt_labels = json.load(file)
    gt_labels=[gt_labels.get(name) for name in names]
    return gt_labels,prediction_scores


def printAUC(gt_labels,prediction_scores):
#    names = np.loadtxt("preds.txt", delimiter=", ", usecols = 0, dtype=str)
#    prediction_scores = np.loadtxt("preds.txt", delimiter=", ", usecols = 1 , dtype=float)

    #get gt_labels from Public Label.json
    # {
    #     "glmjpqleyb.mp4": 0,
    #     "ffqvztgejx.mp4": 0,
    #     "tuvkdcksco.mp4": 0,
    #     "zubpmgsdfp.mp4": 0,
    #     "xbzwuisdwn.mp4": 0
    # }


#    with open("/media/borutb/disk11/DataBase/DeepFake/DFGC-2022/Detection Dataset/Public Label.json", "r") as file:
#        gt_labels = json.load(file)
#    gt_labels=[gt_labels.get(name) for name in names]
    _, eer_value, _ = performances_compute(prediction_scores, gt_labels, verbose = verbose)
    #print(f'EER value: {eer_value*100}')

    auc=roc_auc_score(gt_labels,prediction_scores)
    #print(f'AUC value: {auc*100}')
    return auc

def drawROC(gt_labels,prediction_scores):
    fpr, tpr, _ = roc_curve(gt_labels,prediction_scores)
    auc=roc_auc_score(gt_labels,prediction_scores)
    #create ROC curve
    plt.plot(fpr,tpr,label="AUC="+str("{:.3f}".format(auc*100)))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()

if __name__ == '__main__':
    #gt_labels,prediction_scores = init("preds.txt","Public Label.json")
    #gt_labels,prediction_scores = init("preds.txt","Private Label.json")
    gt_labels,prediction_scores = init("./save_result/txt/pred_convnext7_0_public_20.txt","Public Label.json")
    score = printAUC(gt_labels,prediction_scores)
    print('AUC score for model %s is %f' % ("model", score))
    #drawROC(gt_labels,prediction_scores)