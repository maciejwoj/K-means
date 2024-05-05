import math


class KMeans:
    def __init__(self, k):
        self.k = k
        self.centroids = []

    def load_data(self, filename):
        data = []
        with open(filename, 'r') as file:
            next(file)
            for line in file:
                line = line.strip().split(',')
                features = [float(x) for x in line[:-1]]
                data.append(features)
        return data

    def initialize_centroids(self, data):
        self.centroids = data[:self.k]

    def euclidean_distance(self, a, b):
        distance = 0.0
        for i in range(len(a)):
            distance += (a[i] - b[i]) ** 2
        return math.sqrt(distance)

    def assign_clusters(self, data):
        clusters = [[] for _ in range(self.k)]
        for point in data:
            distances = [self.euclidean_distance(point, centroid) for centroid in self.centroids]
            cluster_index = distances.index(min(distances))
            clusters[cluster_index].append(point)
        return clusters

    def update_centroids(self, clusters):
        for i in range(self.k):
            cluster_points = clusters[i]
            if cluster_points:
                centroid = [sum(x) / len(cluster_points) for x in zip(*cluster_points)]
                self.centroids[i] = centroid

    def fit(self, data, max_iterations=100):
        self.initialize_centroids(data)
        for _ in range(max_iterations):
            clusters = self.assign_clusters(data)
            old_centroids = self.centroids[:]
            self.update_centroids(clusters)
            if old_centroids == self.centroids:
                break

    def predict(self, point):
        distances = [self.euclidean_distance(point, centroid) for centroid in self.centroids]
        cluster_index = distances.index(min(distances))
        return cluster_index


# Testowanie klasyfikatora KMeans
if __name__ == "__main__":
    k = int(input("Podaj liczbę klastrów: "))
    kmeans = KMeans(k)
    data = kmeans.load_data("iris.txt")
    kmeans.fit(data)

    while True:
        new_vector = input("Podaj nowy wektor (sepal length, sepal width, petal length, petal width): ")
        new_vector = [float(x) for x in new_vector.split(',')]
        cluster = kmeans.predict(new_vector)
        print("Wektor został przypisany do klastra:", cluster)
