import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    "data/feature_engineered_data.csv"
)

print("=" * 60)
print("STUDENT ENGAGEMENT CLUSTERING")
print("=" * 60)

print("Dataset shape:", df.shape)


# ============================================================
# 2. SELECT ENGAGEMENT FEATURES
# ============================================================

# We deliberately DO NOT use G3 or support_category.
# These are target-related variables.

cluster_features = [
    "studytime",
    "failures",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "absences",
    "activities",
    "schoolsup",
    "famsup"
]

cluster_df = df[cluster_features].copy()


# ============================================================
# 3. CONVERT CATEGORICAL FEATURES TO NUMERIC
# ============================================================

cluster_df = pd.get_dummies(
    cluster_df,
    drop_first=False
)


print("\nClustering features:")
print(cluster_df.columns.tolist())


# ============================================================
# 4. SCALE FEATURES
# ============================================================

scaler = StandardScaler()

X_cluster = scaler.fit_transform(
    cluster_df
)


# ============================================================
# 5. K-MEANS
# ============================================================

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(
    X_cluster
)


# ============================================================
# 6. ADD CLUSTER LABELS
# ============================================================

df["cluster"] = clusters


print("\nCluster distribution:")

print(
    df["cluster"].value_counts().sort_index()
)


# ============================================================
# 7. CLUSTER PROFILE
# ============================================================

profile = df.groupby("cluster")[
    [
        "studytime",
        "failures",
        "freetime",
        "goout",
        "Dalc",
        "Walc",
        "absences"
    ]
].mean()


print("\n" + "=" * 60)
print("CLUSTER PROFILES")
print("=" * 60)

print(
    profile.round(2)
)


# ============================================================
# 8. PCA FOR VISUALIZATION
# ============================================================

pca = PCA(
    n_components=2,
    random_state=42
)

X_pca = pca.fit_transform(
    X_cluster
)


# ============================================================
# 9. CREATE GRAPH FOLDER
# ============================================================

os.makedirs(
    "eda_graphs",
    exist_ok=True
)


# ============================================================
# 10. CLUSTER VISUALIZATION
# ============================================================

plt.figure(
    figsize=(8, 6)
)

sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=clusters,
    palette="deep",
    s=60
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title(
    "Student Engagement Clusters"
)

plt.legend(
    title="Cluster"
)

plt.tight_layout()

plt.savefig(
    "eda_graphs/19_student_clusters.png"
)

plt.close()


# ============================================================
# 11. SAVE CLUSTERED DATA
# ============================================================

df.to_csv(
    "data/student_clusters.csv",
    index=False
)

profile.to_csv(
    "data/cluster_profiles.csv"
)


# ============================================================
# 12. SAVE SCALER AND MODEL
# ============================================================

import joblib

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    scaler,
    "models/clustering_scaler.pkl"
)

joblib.dump(
    kmeans,
    "models/kmeans.pkl"
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("CLUSTERING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nClustered dataset:")
print("data/student_clusters.csv")

print("\nCluster profiles:")
print("data/cluster_profiles.csv")

print("\nCluster graph:")
print("eda_graphs/19_student_clusters.png")