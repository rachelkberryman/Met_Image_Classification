import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./data/CSVs/all_data.csv', lineterminator='\n', index_col=0)

data.shape

images_data = data[data['primaryImageSmall']!='']

# grouping by culture, in the whole dataset.
classification_df = images_data.groupby('classification').count().sort_values(by='objectID', ascending=False)

paintings_df = images_data[images_data['classification']=='Paintings']
paintings_numbs = paintings_df.groupby('culture').count().sort_values(by='objectID', ascending=False).iloc[:15,1]
# %%
# settingfont settings for the plots
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

fig, ax = plt.subplots(1, 1, figsize=(30, 10))
plot = paintings_numbs.plot(ax=ax, color='dimgrey', kind='bar')

# Add some text for labels, title and custom x-axis tick labels, etc.
plot.set_ylabel('No. of Occurances')
plot.set_xlabel('Culture')
plot.yaxis.labelpad = 25
plot.xaxis.labelpad = 25
ax.set_title('Number of Paintings by Culture')
# ax.get_legend().remove()
ax.set_xticklabels(paintings_numbs.index, rotation=45, ha='right')
fig.savefig(f'./data/plots/painting_culture_counts.png', bbox_inches='tight')
plt.close(fig)