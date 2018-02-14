import os
import plotly
# plotly.tools.set_credentials_file(username='dichen001', api_key='czrCH0mQHmX5HLXSHBqS')
plotly.tools.set_credentials_file(username='ustcjackchen', api_key='A0ILo156LBSqQfIOaxpX')
import plotly.plotly as py
import plotly.graph_objs as go
import cPickle


cwd = os.getcwd()
data_path = os.path.join(cwd, "data", "issue_close_time")
details_path = os.path.join(data_path, 'issue_close_time_details_5x10_mdlp_30.pkl')
details = cPickle.load(open(details_path, 'rb'))

# folder = "1 day"
# folder = "7 days"
# folder = "14 days"
folder = "30 days"
# folder = "90 days"
# folder = "180 days"
# folder = "365 days"
details = details[folder]
titles = details.keys()
classifiers = ["DT", "RF", "LR", "kNN", "FFT-Dist2Heaven"]
colors = ["#AED6F1", "#5DADE2", "#2874A6", "#1B4F72", "#FF5722", "#E53935"]

l = len(details[titles[0]][classifiers[0]]['dist2heaven'])
x = []
for t1 in titles:
    x.extend([t1] * l)
data = []

for i, clf in enumerate(classifiers):
    y = []
    for n1 in titles:
        y.extend(sorted(details[n1][clf]['dist2heaven']))
    tmp_bar = go.Box(
        y=y,
        x=x,
        name=clf,
        marker=dict(
            color=colors[i]
        )
    )
    data.append(tmp_bar)

layout = go.Layout(
    title=folder + ' 5x10 cross-val MDLP',
    yaxis=dict(
        title='Distance to Heaven',
        zeroline=False
    ),
    xaxis=dict(
        title='Issue Close Time Dataset',
        zeroline=False
    ),
    boxmode='group'
)
fig = go.Figure(data=data, layout=layout)
py.plot(fig, filename=folder + " - 5x10 CV - MDLP")
