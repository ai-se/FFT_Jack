import os
import pandas as pd
from helpers import load_obj

cwd = os.getcwd()
data_path = os.path.join(cwd, "data")
data = {"@ivy":     ["ivy-1.1.csv", "ivy-1.4.csv", "ivy-2.0.csv"],\
        "@lucene":  ["lucene-2.0.csv", "lucene-2.2.csv", "lucene-2.4.csv"],\
        "@poi":     ["poi-1.5.csv", "poi-2.0.csv", "poi-2.5.csv", "poi-3.0.csv"],\
        "@synapse": ["synapse-1.0.csv", "synapse-1.1.csv", "synapse-1.2.csv"],\
        "@velocity":["velocity-1.4.csv", "velocity-1.5.csv", "velocity-1.6.csv"], \
        "@camel": ["camel-1.0.csv", "camel-1.2.csv", "camel-1.4.csv", "camel-1.6.csv"], \
        "@jedit": ["jedit-3.2.csv", "jedit-4.0.csv", "jedit-4.1.csv", "jedit-4.2.csv", "jedit-4.3.csv"], \
        "@log4j": ["log4j-1.0.csv", "log4j-1.1.csv", "log4j-1.2.csv"], \
        "@xalan": ["xalan-2.4.csv", "xalan-2.5.csv", "xalan-2.6.csv", "xalan-2.7.csv"], \
        "@xerces": ["xerces-1.2.csv", "xerces-1.3.csv", "xerces-1.4.csv"]
        }
# criterias = ["Accuracy", "Dist2Heaven", "Gini", "InfoGain"]
criterias = ["Accuracy", "Dist2Heaven", "LOC_AUC"] # "Gini", "InfoGain"]
soa_names = ['SL', 'NB', 'EM', 'SMO']

all_data_filepath = os.path.join(data_path, "NewData.pkl")


if os.path.exists(all_data_filepath):
    all_data = load_obj(all_data_filepath)
else:
    all_data = {}

heaven_csv = []
fft_stat = []
for i in range(len(criterias)):
    tmp = {"Learners": "FFT_" + criterias[i]}
    stat = {"Learners": "FFT_" + criterias[i]}
    for name in data.iterkeys():
        fft = all_data[name]["FFT"][i]
        tmp.update({name: abs(fft.dist2heavens[fft.best])})
        stat.update({name: [abs(x) for x in fft.tree_scores]})
    heaven_csv.append(tmp)
    fft_stat.append(stat)

for i in range(len(soa_names)):
    soa_name = soa_names[i]
    tmp = {"Learners": soa_name}
    for name in data.iterkeys():
        soa = all_data[name]["SOA"]
        tmp.update({name: soa.dist2heavens[i][0]})
    heaven_csv.append(tmp)

# for name, files in data.iteritems():
#     # save scores to csv
#     heaven_csv.append({"Learners": "FFT_" + criterias[i]})
#
#     for i in range(4):
#         fft = all_data[name]["FFT"][i]
#         heaven_csv.append({"Data": name, \
#                            "Learners": "FFT_" + criterias[i], \
#                            "Dist2Heaven": fft.tree_scores[fft.best]})
#         fft_stat.append({"Data": name, \
#                          "Criteria": criterias[i], \
#                          "Dist2Heavens": fft.tree_scores})
#     for i in range(4):
#         soa = all_data[name]["SOA"]
#         heaven_csv.append({"Data": name, \
#                            "Learners": soa.names[i], \
#                            "Dist2Heaven": soa.dist2heavens[i]})

heaven_df = pd.DataFrame(heaven_csv)
fft_stat_df = pd.DataFrame(fft_stat)
heaven_df_path = os.path.join(data_path, "_dist2heavens.csv")
fft_stat_path = os.path.join(data_path, "_fft_stats.csv")
heaven_df.to_csv(heaven_df_path)
fft_stat_df.to_csv(fft_stat_path)

print "Statistics save in: " + heaven_df_path
