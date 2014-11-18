import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

df = pd.read_csv("data/train.csv", header=0)

# This command will plot the Number of people who survived by sex and Pclass
df.groupby(['Sex', 'Pclass']).Survived.sum().plot(kind='bar')

# Survivabl: Plotting by PfClass and Sex
death_counts= pd.crosstab([df.Pclass, df.Sex], df.Survived.astype(bool)).apply(lambda r: r, axis=1)
death_counts.plot(kind='bar',stacked=True,color=['black','gold'], grid=False)

death_counts.div(death_counts.sum(1).astype(float), axis = 0).plot(kind='barh', stacked=True, color=['black','gold'])
pdb.set_trace()

plt.show()

print "Done"