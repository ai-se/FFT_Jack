import matplotlib.pyplot as plt
from helpers import subtotal, get_recall

PRE, REC, SPEC, FPR, NPV, ACC, F1 = 7, 6, 5, 4, 3, 2, 1
COLORS = ["#800000", "#6B8E23", "#0000CD", "#FFFF00", "#8A2BE2",  "#00FF00", "#00FFFF", "#FF00FF"]
MARKERS = ['v', 2, ',', 'h',  ">", 's', '*', 'p', '8']

"Plot ROC"
def plotROC(fft, soa, img_path="~/tmp"):
    fig, ax = plt.subplots()
    ax.set_title('ROC: FFT splits on ' + fft.criteria + '  |  Data ' + fft.data_name)
    ax.set_xlabel("False Alarm Rates")
    ax.set_ylabel("Recall")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    # plot diagonal
    x, y = [0.001 * i for i in range(1000)], [0.001 * i for i in range(1000)]
    ax.scatter(x, y, s=4)

    # plot fft peformances
    fft.roc = [None] * fft.tree_cnt
    roc = fft.roc
    tmp = {"Accuracy": 0, "Dist2Heaven": 1, "Gini": 2, "InfoGain": 3}
    k = tmp[fft.criteria]
    s_id = fft.best
    for i in range(fft.tree_cnt):
        metric = fft.performance_on_test[i]
        roc[i] = [metric[-FPR], metric[-REC]]
        if i == s_id:
            continue
        ax.scatter(roc[i][0], roc[i][1], c=COLORS[k], s=100)
        ax.annotate(i, (roc[i][0], roc[i][1]))
    t = 0 if s_id != 0 else 1
    ax.scatter(roc[t][0], roc[t][1], c=COLORS[k], s=100, label="FFT")

    # plot the best fft in red
    ax.scatter(roc[s_id][0], roc[s_id][1], c='r', marker=MARKERS[0], s=100, label="Best_FFT")
    ax.annotate("B_FFT", (roc[s_id][0], roc[s_id][1]))

    # plot state of the art performance
    for i in range(4):
        ax.scatter(soa.performances[i][-FPR], soa.performances[i][-REC], s=120, \
                   c=COLORS[i + 4], marker=MARKERS[i + 4], label=soa.names[i])
        ax.annotate(soa.names[i], (soa.performances[i][-FPR], soa.performances[i][-REC]))

    legend = ax.legend(loc='lower right', shadow=True, fontsize='small')
    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('#CEE5DD')
    # plt.show()
    plt.savefig(img_path)


"Plot ROC"
def plotLOC(data, learners, names, img_path="~/tmp"):
    fig, ax = plt.subplots()
    ax.set_title('LOC: Data ' + learners[0].data_name)
    ax.set_xlabel("%code churn")
    ax.set_ylabel("%bug detection")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    # plot diagonal
    x, y = [0.001 * i for i in range(1000)], [0.001 * i for i in range(1000)]
    ax.scatter(x, y, s=4)

    data.sort(columns=["bug", "loc"], ascending=[0, 1], inplace=True)
    x_sum = float(sum(data['loc']))
    x = data['loc'].apply(lambda t: t / x_sum)
    xx = subtotal(x)

    # plot optimal
    k = len(learners)
    y = data['bug'].values
    yy = get_recall(y, y)
    ax.plot(xx, yy, markersize=10, color=COLORS[-k], marker=MARKERS[-k], label="Optimal")

    # plot worst
    xx = subtotal(x[::-1])
    y = data['bug'][::-1].values
    yy = get_recall(y, y)
    ax.plot(xx, yy, markersize=10, color=COLORS[-k-1], marker=MARKERS[-k-1], label="Worst")

    # plot state of the art performance
    for i, clf in enumerate(learners):
        if names[i].startswith("FFT"):
            y = clf.predict(data)
        else:
            y = clf.predict(data.iloc[:, :-2]).tolist()
        data['prediction'] = y
        data.sort(columns=["prediction", "loc"], ascending=[0, 1], inplace=True)
        x = data['loc'].apply(lambda t: t / x_sum)
        xx = subtotal(x)
        yy = get_recall(y, data['bug'].values)
        ax.plot(xx, yy, markersize=10, color=COLORS[-i], marker=MARKERS[-i], label=names[i])

    legend = ax.legend(loc='lower right', shadow=True, fontsize='small')
    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('#CEE5DD')
    # plt.show()
    plt.savefig(img_path)



def plot_compare(fft1, fft2, img_path="~/tmp"):
    # plot ROC
    fig, ax = plt.subplots()
    ax.set_title('FFT Comparison |  Data: ' + fft1.data_name)
    ax.set_xlabel("False Alarm Rates")
    ax.set_ylabel("Recall")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    # plot diagonal
    x, y = [0.001 * i for i in range(1000)], [0.001 * i for i in range(1000)]
    ax.scatter(x, y, s=4)

    # plot fft peformances
    tmp = {"Accuracy": 0, "Dist2Heaven": 1, "Gini": 2, "InfoGain": 3}
    markers = ['*', 'o']
    colors = ['#800000', '#6B8E23', '#65ff00', '#ff0000']
    for fft in [fft1, fft2]:
        roc = fft.roc
        s_id = fft.best
        k = tmp[fft.criteria]
        for i in range(fft.tree_cnt):
            if i == s_id:
                continue
            ax.scatter(roc[i][0], roc[i][1], marker=markers[k], c=colors[k], s=400-k*300)
            ax.annotate(i, (roc[i][0], roc[i][1]))
        t = 0 if s_id != 0 else 1
        ax.scatter(roc[t][0], roc[t][1], c=colors[k], marker=markers[k], s=400-k*300, label="FFT(" + fft.criteria + ")")

        # plot the best fft
        ax.scatter(roc[s_id][0], roc[s_id][1], c=colors[-k-1], \
               marker=markers[k], s=400-k*300, label="Best_FFT(" + fft.criteria + ")")
        ax.annotate("B_FFT(" + fft.criteria[0] + ")", (roc[s_id][0], roc[s_id][1]))

    legend = ax.legend(loc='lower right', shadow=True, fontsize='small')
    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('#CEE5DD')
    # plt.show()
    plt.savefig(img_path)


