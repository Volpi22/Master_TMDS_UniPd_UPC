
from pyspark import SparkContext, SparkConf

from collections import defaultdict
from pyspark.mllib.clustering import KMeans

import sys
import os

# Input formatting
def formatInputFile(input_file):
	output = input_file.map(lambda line: (tuple([float(x) for x in line.split(',')[:-1]]), line.split(',')[-1]))
	return output

# 0. Find centroids
def LloydsAlgorithm(inputPoints, num_cluster_K, num_iteration_M):
    points = inputPoints.map(lambda x: x[0])
    clusters = KMeans.train(points, num_cluster_K, maxIterations=num_iteration_M, initializationMode="parallel")
    return clusters.centers

# 1. K-means
def MRComputeStandardObjective(inputPoints, centroids):
	points = inputPoints.map(lambda x: x[0])
	def error(point):
		closest_centroid = min(centroids, key=lambda center: sum((point - center) ** 2))
		return sum((point - closest_centroid) ** 2)
	
	# N_count = inputPoints.map(lambda x: x[1]).countByKey()
	# NA = N_count["A"]
	# NB = N_count["B"]
	N_count = inputPoints.map(lambda x: (x[1], 1)).reduceByKey(lambda x, y: x + y).collectAsMap()
	NA = N_count.get("A", 0)
	NB = N_count.get("B", 0)

	return 1/(NA+NB) * points.map(lambda point: error(point)).reduce(lambda x, y: x + y)
	

# 2. Fairness K-means
def MRComputeFairObjective(inputPoints, centroids):
	# points_A = inputPoints.filter(lambda x: x[1] == "A").map(lambda x: x[0])
	# points_B = inputPoints.filter(lambda x: x[1] == "B").map(lambda x: x[0])
	# NA,NB = points_A.count(), points_B.count()
	NA = inputPoints.filter(lambda x: x[1] == "A").map(lambda x: ("A", 1)).reduceByKey(lambda x, y: x + y).collectAsMap().get("A", 0)
	NB = inputPoints.filter(lambda x: x[1] == "B").map(lambda x: ("B", 1)).reduceByKey(lambda x, y: x + y).collectAsMap().get("B", 0)

	def error(point):
		closest_centroid = min(centroids, key=lambda center: sum((point - center) ** 2))
		return sum((point - closest_centroid) ** 2)

	# dist_A = points_A.map(lambda point: error(point)).reduce(lambda x, y: x + y)
	# dist_B = points_B.map(lambda point: error(point)).reduce(lambda x, y: x + y)

	dist_A = inputPoints.filter(lambda x: x[1] == "A").map(lambda x: error(x[0])).reduce(lambda x, y: x + y)
	dist_B = inputPoints.filter(lambda x: x[1] == "B").map(lambda x: error(x[0])).reduce(lambda x, y: x + y)

	return max(1/NA * dist_A, 1/NB * dist_B)

# 3. Print Statistics
def MRPrintStatistics(inputPoints, centroids):
	def find_closest_centroid(point):
		point_x = point[0]
		point_label = point[1]
		distances = [sum((p1 - p2) ** 2 for p1, p2 in zip(point_x, centroid))
					 for centroid in centroids]

		distances = list(zip(distances, centroids, point_label * len(centroids)))

		closest_centroid = min(distances)
		centroid_label = (tuple(closest_centroid[1]), closest_centroid[2])

		return centroid_label

	#output = inputPoints.map(find_closest_centroid).countByValue()
	# Other option by using map and reduce instead of the built in function, faster and more safe
	# This function has a problem with empty classes so we can use defaultdict
	output = inputPoints.map(lambda point: (find_closest_centroid(point), 1)).reduceByKey(lambda x, y: x + y).collectAsMap()
	output =  defaultdict(int, output)
	return output

def main():

	# CHECKING NUMBER OF CMD LINE PARAMETERS
	assert len(sys.argv) == 5, "Usage: python G24HW1.py <input_file> <L> <K> <M>"

	# SPARK SETUP
	conf = SparkConf().setAppName('G24HW1')
	sc = SparkContext(conf=conf)

	# 4.1. Prints the command-line arguments and stores L,K,M into suitable variables. 
	input_file_path = sys.argv[1]  # Input file
	num_partition_L = sys.argv[2]  # Number of partitions
	num_cluster_K = sys.argv[3]  # Number of clusters
	num_iteration_M = sys.argv[4]  # Number of iterations

	assert num_partition_L.isdigit(), "L must be an integer"
	assert num_cluster_K.isdigit(), "K must be an integer"
	assert num_iteration_M.isdigit(), "M must be an integer"

	num_partition_L = int(num_partition_L)
	num_cluster_K = int(num_cluster_K)
	num_iteration_M = int(num_iteration_M)


	# 4.2 Reads the input points into an RDD of (point,group) pairs called inputPoints,
	# subdivided into L partitions.

	assert os.path.isfile(input_file_path), "File or folder not found"
	input_file = sc.textFile(input_file_path, num_partition_L) # This one or the one below
	#input_file = sc.textFile(input_file_path).repartition(numPartitions=num_partition_L).cache()
    # need to adjust basd ont given input data
	inputPoints = formatInputFile(input_file)

	# 4.3 Prints the number N of points, the number NA of points of group A,
	# and the number NB of points of group B
	# N_count = inputPoints.map(lambda x: x[1]).countByKey()
	# NA = N_count["A"]
	# NB = N_count["B"]
	# N = NA + NB

	N_count = inputPoints.map(lambda x: (x[1], 1)).reduceByKey(lambda x, y: x + y).collectAsMap()
	NA = N_count.get("A", 0)
	NB = N_count.get("B", 0)
	N = NA + NB

	# 4.4 Computes a set C of K centroids by using the Spark implementation of the standard Lloyd's algorithm for the input points,
	# disregarding the points' demographic groups, and using M as number of iterations. 
	centroids = LloydsAlgorithm(inputPoints, num_cluster_K, num_iteration_M)

	# 4.5 Computes and prints the standard objective function Δ(U,C) and the fairness objective function Φ(A,B,C)
	standard = MRComputeStandardObjective(inputPoints, centroids)
	fair = MRComputeFairObjective(inputPoints, centroids)

	# 4.6 Prints the statistics of the clusters, i.e., the number of points in each cluster and the number of points of group A and B in each cluster.
	stastistic_centroids = MRPrintStatistics(inputPoints, centroids)

	print(f"Input file = {input_file_path}, L = {num_partition_L}, K = {num_cluster_K}, M = {num_iteration_M}")
	print(f"N = {N}, NA = {NA}, NB = {NB}")
	print("Delta(U,C) = {:6f}".format(standard))
	print("Phi(A,B,C) = {:6f}".format(fair))

	for i in range(len(centroids)):
		centroid = tuple(centroids[i])
		NA = stastistic_centroids[(centroid, "A")]
		NB = stastistic_centroids[(centroid, "B")]
		output_text = "i = " + str(i) + " center = ("
		for coordinate in centroid:
			coordinate = '{:.6f}'.format(coordinate)
			output_text += coordinate + ","
		output_text = output_text[:-1] + "),"
		output_text += " NA"+str(i)+" = "+str(NA)+", NB"+str(i)+" = "+str(NB)
		print(output_text)

if __name__ == "__main__":
	main()


