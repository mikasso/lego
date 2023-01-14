import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("clustering_2/per_label/2.csv")
df = df.set_index("k").drop(columns=df.columns[0], axis=1)
print(df)
df[["inertia"]].diff(1).plot(title="inertia diff")
df[["silhouette_score"]].plot()
df[["biggest_cluster_size"]].plot()
plt.show()
