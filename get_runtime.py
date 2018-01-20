import os
import timeit
import pandas as pd
from new_fft import FFT


cwd = os.getcwd()
data_path = os.path.join(cwd, "data")
data = {"@camel": ["camel-1.0.csv", "camel-1.2.csv", "camel-1.4.csv", "camel-1.6.csv"], \
        "@ivy":     ["ivy-1.1.csv", "ivy-1.4.csv", "ivy-2.0.csv"], \
        "@jedit": ["jedit-3.2.csv", "jedit-4.0.csv", "jedit-4.1.csv", "jedit-4.2.csv", "jedit-4.3.csv"], \
        "@log4j": ["log4j-1.0.csv", "log4j-1.1.csv", "log4j-1.2.csv"], \
        "@lucene":  ["lucene-2.0.csv", "lucene-2.2.csv", "lucene-2.4.csv"], \
        "@poi":     ["poi-1.5.csv", "poi-2.0.csv", "poi-2.5.csv", "poi-3.0.csv"], \
        "@synapse": ["synapse-1.0.csv", "synapse-1.1.csv", "synapse-1.2.csv"], \
        "@velocity":["velocity-1.4.csv", "velocity-1.5.csv", "velocity-1.6.csv"], \
        "@xalan": ["xalan-2.4.csv", "xalan-2.5.csv", "xalan-2.6.csv", "xalan-2.7.csv"], \
        "@xerces": ["xerces-1.2.csv", "xerces-1.3.csv", "xerces-1.4.csv"]
        }
criterias = ["Accuracy", "Dist2Heaven", "LOC_AUC"] # "Gini", "InfoGain"]

def extract(s):
    return s[:-4].split('-')[-1]

# get training, testing, % defective
stats = []
for name, files in data.iteritems():
    stat = {"data set": name[1:]}

    stat["training versions"] = ', '.join(map(extract, files[:-1]))
    stat["testing versions"] = extract(files[-1])

    print "training on: " + stat["training versions"]
    print "testing on: " + stat["testing versions"]


run_times = []
for name, files in data.iteritems():
    print "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print name
    paths = [os.path.join(data_path, file_name) for file_name in files]
    train_df = pd.concat([pd.read_csv(path) for path in paths[:-1]], ignore_index=True)
    test_df = pd.read_csv(paths[-1])
    train_df, test_df = train_df.iloc[:, 3:], test_df.iloc[:, 3:]
    train_df['bug'] = train_df['bug'].apply(lambda x: 0 if x == 0 else 1)
    test_df['bug'] = test_df['bug'].apply(lambda x: 0 if x == 0 else 1)
    print "training on: " + ', '.join(files[:-1])
    print "testing on: " + files[-1]
    run_time = {"data": name}
    for c, criteria in enumerate(criterias):
        print "  ...................... " + criteria + " ......................"
        fft = FFT(5)
        fft.criteria = criteria
        fft.data_name = name
        fft.train, fft.test = train_df, test_df
        t1 = timeit.default_timer()
        fft.build_trees()
        t2 = timeit.default_timer()
        fft.eval_trees()
        t3 = timeit.default_timer()
        run_time[criteria + "_build"] = round(t2 - t1, 2)
        run_time[criteria + "_eval"] = round(t3 - t2, 2)
    run_times += [run_time]

df = pd.DataFrame(run_times)
df_path = os.path.join(data_path, "_runtimes.csv")
df.to_csv(df_path)
print 'Runtimes are saved in: ' + df_path