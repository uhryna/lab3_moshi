import random
from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

def distance(a, b):
    return euclidean(a, b)


def kmeans(data, k, max_iterations=100):
    centers = random.sample(data, k)
    for i in range(max_iterations):
        clusters = [[] for _ in range(k)]
        for point in data:
            distances = [distance(point, center) for center in centers]
            cluster_index = distances.index(min(distances))
            clusters[cluster_index].append(point)
        new_centers = []
        for j in range(k):
            if clusters[j]:
                new_center = (
                    sum(point[0] for point in clusters[j]) / len(clusters[j]),
                    sum(point[1] for point in clusters[j]) / len(clusters[j])
                )
                new_centers.append(new_center)
            else:
                new_centers.append(centers[j])
        if new_centers == centers:
            break
        centers = new_centers
    return clusters, centers


def hierarchical_clustering(data, k):
    distances = pdist(data)
    K = linkage(distances, method='ward')
    clusters = fcluster(K, k, criterion='maxclust')
    return clusters


# послідовність з N=1000 парами дійсних чисел на одиничному квадраті
data = [(random.uniform(0, 1), random.uniform(0, 1)) for _ in range(1000)]
k = 4
kmeans_clusters, kmeans_centers = kmeans(data, k)
hierarchical_clusters = hierarchical_clustering(data, k)

# порівняння результатів кластеризації
print("K-means:")
for i, cluster in enumerate(kmeans_clusters):
    print("Cluster {}: {} points, center at {}".format(i + 1, len(cluster), kmeans_centers[i]))
print("Hierarchical:")
for i in range(1, k + 1):
    cluster = [data[j] for j in range(len(hierarchical_clusters)) if hierarchical_clusters[j] == i]
    center = (
        sum(point[0] for point in cluster) / len(cluster),
        sum(point[1] for point in cluster) / len(cluster)
    )
    print("Cluster {}: {} points, center at {}".format(i, len(cluster), center))

# відобразити точки даних та центри кластерів для кластеризації K-середніх
plt.subplot(1, 2, 1)
for i, cluster in enumerate(kmeans_clusters):
    x = [point[0] for point in cluster]
    y = [point[1] for point in cluster]
    plt.scatter(x, y, label="Cluster {}".format(i + 1))
for center in kmeans_centers:
    plt.scatter(center[0], center[1], s=100, marker='x', color='black', linewidths=2)
plt.title("K-means")

# відобразити точки даних для кластеризації ієрархічним методом
plt.subplot(1, 2, 2)
for i in range(1, k + 1):
    cluster = [data[j] for j in range(len(hierarchical_clusters)) if hierarchical_clusters[j] == i]
    x = [point[0] for point in cluster]
    y = [point[1] for point in cluster]
    plt.scatter(x, y, label="Cluster {}".format(i))
plt.title("Hierarchical")

plt.show()
