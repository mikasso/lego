import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("clustering_2/per_label/0.csv")
df = df.set_index("k").drop(columns=df.columns[0], axis=1)
print(df)
df[["inertia", "adj_inertia"]].plot()
df[["silhouette_score", "adj_silhouette_score"]].plot()
plt.show()
