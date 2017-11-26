import os
import pandas as pd

from helpers import load_obj, save_obj
from new_fft import FFT
from SOA import SOA
from plot import plot, plot_compare


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

all_data_filepath = os.path.join(data_path, "NewData.pkl")
if os.path.exists(all_data_filepath):
    all_data = load_obj(all_data_filepath)
else:
    all_data = {}

for name, files in data.iteritems():
    if name not in all_data:
        print "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        print name
        paths = [os.path.join(data_path, file_name) for file_name in files]
        train_df = pd.concat([pd.read_csv(path) for path in paths[:-1]])
        test_df = pd.read_csv(paths[-1])
        train_df['bug'] = train_df['bug'].apply(lambda x: 0 if x == 0 else 1)
        test_df['bug'] = test_df['bug'].apply(lambda x: 0 if x == 0 else 1)
        print "training on: " + ', '.join(files[:-1])
        print "testing on: " + files[-1]
        all_data[name] = {}
        soa = SOA(train=train_df, test=test_df)
        soa.get_performances()
        all_data[name]["SOA"] = soa
        all_data[name]["FFT"] = []
        criterias = ["Accuracy", "Dist2Heaven", "Gini", "InfoGain"]
        # criterias = ["Dist2Heaven"] #, "Accuracy", "Gini", "InfoGain"]
        for criteria in criterias:
            print "  ...................... " + criteria + " ......................"
            fft = FFT(4)
            fft.criteria = criteria
            fft.data_name = name
            fft.train, fft.test = train_df, test_df
            fft.build_trees()
            fft.eval_trees()
            fft.find_best_tree()
            soa.print_soa()
            img_path0 = os.path.join(data_path, fft.criteria + "_" + name + ".png")
            plot(fft, soa, img_path=img_path0, type="ROC")
            all_data[name]["FFT"] += [fft]
    img_path1 = os.path.join(data_path, "FFT_Compare_" + name + ".png")
    plot_compare(all_data[name]["FFT"][0], all_data[name]["FFT"][1], img_path=img_path1)
    # plot_compare(name, all_data[name]["Accuracy"], all_data[name]["Dist2Heaven"])
    # plot_effort(name, all_data[name]["Dist2Heaven"])

if not os.path.exists(all_data_filepath):
    save_obj(all_data, all_data_filepath)