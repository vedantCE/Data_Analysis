import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from pandas.plotting import parallel_coordinates

df = pd.read_csv("winequality-red.csv", sep=";")

print("Shape:", df.shape)
print("\nMissing values:", df.isnull().sum().sum())
print("\nQuality distribution:\n", df["quality"].value_counts().sort_index())

features = df.drop("quality", axis=1)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

selected = ["fixed acidity", "volatile acidity", "citric acid",
            "alcohol", "pH", "quality"]

sns.pairplot(df[selected], hue="quality", palette="coolwarm",
             plot_kws={"alpha": 0.5, "s": 20})
plt.suptitle("Scatter Plot Matrix - Wine Quality", y=1.02, fontsize=14)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlGn", center=0,
            linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
plt.title("Correlation Heatmap - Wine Features", fontsize=14, pad=15)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 7))
bubble_size = (df["quality"] ** 2) * 10
scatter = plt.scatter(
    df["alcohol"],
    df["volatile acidity"],
    s=bubble_size,
    c=df["quality"],
    cmap="RdYlGn",
    alpha=0.6,
    edgecolors="grey",
    linewidth=0.3
)
plt.colorbar(scatter, label="Quality Score")
plt.xlabel("Alcohol (%)", fontsize=12)
plt.ylabel("Volatile Acidity (g/dm³)", fontsize=12)
plt.title("Bubble Plot: Alcohol vs Volatile Acidity\n(Bubble size = Quality)", fontsize=13)
plt.tight_layout()
plt.show()

para_cols = ["fixed acidity", "volatile acidity", "citric acid",
             "alcohol", "sulphates", "pH", "quality"]
df_para = df[para_cols].copy()

for col in para_cols[:-1]:
    df_para[col] = (df_para[col] - df_para[col].min()) / \
                   (df_para[col].max() - df_para[col].min())

df_para["quality"] = df_para["quality"].astype(str)

plt.figure(figsize=(14, 6))
parallel_coordinates(df_para, class_column="quality",
                     colormap="RdYlGn", alpha=0.3, linewidth=0.8)
plt.title("Parallel Coordinate Plot - Multi-feature Comparison by Quality", fontsize=13)
plt.xlabel("Features", fontsize=11)
plt.ylabel("Normalized Value", fontsize=11)
plt.xticks(rotation=20)
plt.legend(title="Quality", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.show()

print("\nRunning t-SNE (this may take 30-60 seconds)...")

tsne = TSNE(n_components=2, random_state=42, perplexity=30,
            max_iter=1000, learning_rate="auto", init="pca")
tsne_result = tsne.fit_transform(features_scaled)

tsne_df = pd.DataFrame(tsne_result, columns=["TSNE-1", "TSNE-2"])
tsne_df["quality"] = df["quality"].values

plt.figure(figsize=(11, 8))
palette = sns.color_palette("RdYlGn", n_colors=df["quality"].nunique())
quality_levels = sorted(df["quality"].unique())

for i, q in enumerate(quality_levels):
    subset = tsne_df[tsne_df["quality"] == q]
    plt.scatter(subset["TSNE-1"], subset["TSNE-2"],
                label=f"Quality {q}",
                color=palette[i],
                alpha=0.7, s=25, edgecolors="none")

plt.title("t-SNE: 2D Projection of Wine Features\nColored by Quality", fontsize=14)
plt.xlabel("t-SNE Dimension 1", fontsize=12)
plt.ylabel("t-SNE Dimension 2", fontsize=12)
plt.legend(title="Quality", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.show()
