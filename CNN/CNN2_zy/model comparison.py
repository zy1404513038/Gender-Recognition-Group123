import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from prettytable import PrettyTable

# 输出标量指标对比表格及P-R曲线
def compareTabel(testSet, modelname, *predictions):
    colors = ['crimson',
              'orange',
              'gold',
              'mediumseagreen',
              'steelblue',
              'mediumpurple',
              'pink'
              ]
    table = PrettyTable(["模型名称", "错误率", "精度", "查准率P", "查全率R", "F1"])
    table.padding_width = 1  # 填充宽度
    y_true = np.array([y[1] for y in testSet])
    precision = []
    recall = []
    for i, prediction in enumerate(predictions):
        acc = 0
        TP = 0
        FP = 0
        FN = 0
        for vec in prediction:
            if vec[1] == testSet[vec[0]][1]:
                acc += 1
            if vec[1] == 1 and testSet[vec[0]][1] == 1:
                TP += 1
            if vec[1] == -1 and testSet[vec[0]][1] == 1:
                FN += 1
            if vec[1] == 1 and testSet[vec[0]][1] == -1:
                FP += 1

        accRate = round(acc / len(testSet), 2)
        errerRate = round(1 - accRate, 2)
        P = round(TP/(TP + FP), 2)
        R = round(TP/(TP + FN), 2)
        F1 = round(2*P*R/(P+R), 2)
        table.add_row([modelname[i],errerRate, accRate, P, R, F1])

        y_score = np.array([y[1] for y in prediction])
        precisiontmp, recalltmp, thresholds = precision_recall_curve(y_true, y_score)
        precision.append(precisiontmp)
        recall.append(recalltmp)

    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 用来正常显示中文标签
    plt.title('P-R曲线')
    lw = 2
    plt.subplot(1, 1, 1)
    for i in range(len(modelname)):
        plt.plot(precision[i], recall[i], color=colors[i],
                 lw=lw, label=modelname[i])
    plt.legend(loc="upper right")
    plt.show()
    print(table)


# 输出ROC曲线
def ROC(testSet, modelname, *predictions):
    colors = ['crimson',
              'orange',
              'gold',
              'mediumseagreen',
              'steelblue',
              'mediumpurple',
              'pink'
              ]
    y_true = np.array([y[1] for y in testSet])
    fpr = []
    tpr = []
    roc_auc = []
    for i, prediction in enumerate(predictions):
        y_score = np.array([y[1] for y in prediction])
        fp, tp, threshold = roc_curve(y_true, y_score)
        fpr.append(fp)
        tpr.append(tp)
        roc_auctmp = auc(fpr[i], tpr[i])  # 准确率代表所有正确的占所有数据的比值
        roc_auc.append(roc_auctmp)
        # print('roc_auc:', roc_auc)

    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 用来正常显示中文标签
    plt.title('Validation ROC')
    lw = 2
    plt.subplot(1, 1, 1)
    for i in range(len(modelname)):
        plt.plot(fpr[i], tpr[i], color=colors[i],
                 lw=lw, label=modelname[i] + '(AUC = %0.2f)' % roc_auc[i])  # 假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()



if __name__ == '__main__':

    # 各个模型名称
    modelName = ["CNN1——skb", 'CNN2——zy']

    # 读入训练数据集
    testSet = pd.read_csv('../data/train.csv').values[:,:].tolist()
    # 读取csv结果文件（id ，label）
    CNN_pre_train_skb = pd.read_csv('./results/CNN_pre_train_skb.csv').values[:,:].tolist() # CNN1结果
    CNN_pre_train_zy = pd.read_csv('./results/CNN_pre_train_zy.csv').values[:,:].tolist() # CNN2结果
    # 输出标量指标对比表格及P-R曲线
    compareTabel(testSet, modelName, CNN_pre_train_skb, CNN_pre_train_zy)
    # 输出ROC曲线图
    ROC(testSet, modelName, CNN_pre_train_skb, CNN_pre_train_zy)