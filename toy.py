import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

cwd = os.getcwd()
data_path = os.path.join(cwd, "data")
path = os.path.join(data_path, '_0guoke.csv')

df = pd.read_csv(path)
headers = list(df)

g = sns.pairplot(df, vars=headers[:-1], hue=headers[-1], palette="husl", markers=["o", "s"])
# Generate a custom diverging colormap
for i, j in zip(*np.triu_indices_from(g.axes, 1)):
    g.axes[i, j].set_visible(False)

save = os.path.join(data_path, '_guoke_0.png')
g.savefig(save)






corr = df.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(240, 10, n=9, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
g = sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, square=True, annot=True, fmt='.3f', linewidths=.5, cbar_kws={"shrink": .5})
g.set_xticklabels(g.get_xticklabels(), rotation=45, fontsize=10)
g.set_yticklabels(g.get_yticklabels(), rotation=45, fontsize=10)
save = os.path.join(data_path, '_guoke_corr.png')
plt.savefig(save)

# plt.show()

print corr




