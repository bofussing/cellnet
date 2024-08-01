# %% 
import json, pandas as pd, seaborn as sns, matplotlib.pyplot as pyplot

review = pd.read_csv('review.csv', sep='\t', index_col=0)
m = json.load(open('bad-random-0.05.json'))

human_inacc = list(review.query('accurate != "x"').index)
human_all = set(review.index)


human_inacc = [m['i2pi'][str(i)] for i in human_inacc]
human_all = [m['i2pi'][str(i)] for i in human_all]

# %%

part = '5%' # '5%'

data = pd.DataFrame(columns=['pi', 'Model Loss', 'Human Review'],
                     data=[[int(pi), L, 'inaccurate' if (pi in human_inacc) else 'accurate' if (pi in human_all) else 'unlabeled'] for pi, L in enumerate(m['pi2L']) if part=='all' or pi in human_all]
                     ).sort_values('Model Loss', ascending=False).reset_index(drop=True)

data # type: ignore

# %% 

fig, ax = pyplot.subplots(figsize=(15,5))
g = sns.barplot(data=data, x=data.index, y='Model Loss', hue='Human Review', palette={'inaccurate':'tab:red', 'accurate':'tab:blue', 'unlabeled':'tab:grey'}, gap=0, dodge=False)
g.set_title('Model and Human Assessment of Annotation Quality')
g.set(xticklabels=[])
g.tick_params(bottom=False)  # remove the tick
g.set(xlabel='Annotation')

## add a vertical line at the 5% mark
cut = (0.5 if part == '5%' else 0.05)*len(data)
g.axvline(cut, color='black', linestyle='--')
ax.text(cut +.5, 107, 'Model', ha='left')
ax.text(cut -.5, 107, 'Review', ha='right')
ax.text(cut + 1, 100, 'inaccurate', ha='left')
ax.text(cut - 1, 100, 'accurate', ha='right')

