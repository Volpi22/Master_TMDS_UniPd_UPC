from pyspark import SparkContext, SparkConf
from pyspark.mllib.clustering import KMeans

import numpy as np
import time
import sys
import os

# Input formatting
def formatInputFile(input_file):
	output = input_file.map(lambda line: (tuple([float(x) for x in line.split(',')[:-1]]), line.split(',')[-1]))
	return output

# 0. Functions from the previous assignment
# 0.1 Find centroids with Lloyd's algorithm
def TimedLloydsAlgorithm(inputPoints, num_cluster_K, num_iteration_M):

	points = inputPoints.map(lambda x: x[0])
	
	# the time is calculated just for the KMeans.train part, as per the assignment
	start_time = time.time()
	clusters = KMeans.train(points, num_cluster_K, maxIterations=num_iteration_M, initializationMode="parallel", seed=None)
	end_time = time.time()
	
	RUNNING_TIME_ON["centroids_by_standard_lloyd"] = (end_time - start_time) * 1000

	centroids = tuple([tuple(centroid) for centroid in clusters.centers])
	return centroids

def LloydsAlgorithm(inputPoints, num_cluster_K, num_iteration_M):

	points = inputPoints.map(lambda x: x[0])
	
	# the time is calculated just for the KMeans.train part, as per the assignment
	clusters = KMeans.train(points, num_cluster_K, maxIterations=num_iteration_M, initializationMode="parallel", seed=None)

	centroids = tuple([tuple(centroid) for centroid in clusters.centers])
	return centroids

# 1. Write a method/function MRFairLloyd which implements the above Fair K-Means Clustering algorithm. Specifically, MRFairLloyd takes in input an RDD representing a set U
# of points, with demographic group labels, and two parameters K,M,(integers), and does the following:
def MRFairLloyd(inputPoints, num_cluster_K, num_iteration_M):

	D = len(inputPoints.first()[0])  # Dimension of the points
	
	# the time is calculated just for the KMeans.train part, as per the assignment
	# initial_centroids = KMeans.train(points, num_cluster_K, maxIterations=0, initializationMode="parallel", seed=None)

	def calculateEuclideanDistance(point1, point2):
		# return np.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))
		point1 = np.array(point1)
		point2 = np.array(point2)
		return np.sqrt(np.sum((point1 - point2) ** 2))


	def find_closest_centroid(point, centroids):
		# Finds the closest centroid for each point in the inputPoints RDD.
		# def distanceBetween(point, centroid):
		# 	point = np.array(point)
		# 	centroid = np.array(centroid)
		# 	return np.sum((point - centroid) ** 2)
		
		point_x = point[0]
		point_label = point[1]

		centroids_x = np.array(centroids)

		distances = np.linalg.norm(centroids_x - point_x, ord = 1, axis=1)
		closest_centroid_idx = np.argmin(distances)

		# distances = [calculateEuclideanDistance(point_x, centroid) for centroid in centroids]
		# closest_centroid_idx = distances.index(min(distances))
		# closest_centroid = centroids[closest_centroid_idx]
		closest_centroid_idx_str = 'c'+str(closest_centroid_idx)

		return tuple([tuple([closest_centroid_idx_str, point_label]), tuple([point_x, 1])])

	def summaryCentroidLabelPartition(x, y):
		coordinate_sum = np.array(x[0]) + np.array(y[0])
		count_sum = x[1] + y[1]

		output = [tuple(coordinate_sum), count_sum]
		return tuple(output)

	# Calculated centroid alpha, beta, miu_A, miu_B and l return in dictionary
	def getParameterByCentroid(partition_summary, num_cluster_K,D):
		# Convert summary output into dictionary also add 0 if not there for robust
		def selectCentroidPartition(partition_summary, centroid_idx):
			centroid_str = 'c' + str(centroid_idx)
			output_dict = {
				'A': {'coord': np.zeros(D), 'count': 0},
				'B': {'coord': np.zeros(D), 'count': 0}
			}
			for label in ['A', "B"]:
				key = (centroid_str, label)
				if key in partition_summary:
					output_dict[label]['coord'] = np.array(partition_summary[key][0])
					output_dict[label]['count'] = partition_summary[key][1]
			
			return output_dict

		# Computer parameters by centroid
		# alpha, beta, miu_A, miu_B and l are calculated here
		def computeParametes(p_i):

			output_dict = dict()

			output_dict['alpha'] = p_i['A']['count'] / NA
			output_dict['beta'] = p_i['B']['count'] / NB

			# output_dict['miu_A'] = (np.array(p_i['A']['coord'])/p_i['A']['count']) if p_i['A']['count'] != 0 else np.zeros(D)
			# output_dict['miu_B'] = (np.array(p_i['B']['coord'])/p_i['B']['count']) if p_i['B']['count'] != 0 else np.zeros(D)

			output_dict['miu_A'] = (np.array(p_i['A']['coord'])/p_i['A']['count']) if p_i['A']['count'] != 0 else (np.array(p_i['A']['coord'])/p_i['B']['count'])
			output_dict['miu_B'] = (np.array(p_i['B']['coord'])/p_i['B']['count']) if p_i['B']['count'] != 0 else (np.array(p_i['B']['coord'])/p_i['A']['count'])

			output_dict['l'] = calculateEuclideanDistance(output_dict['miu_A'], output_dict['miu_B'])

			return output_dict

		output_dict = dict()

		for idx in range(num_cluster_K):
			centroid_str = 'c' + str(idx)
			p_i = selectCentroidPartition(partition_summary, idx)

			output_dict[centroid_str] = computeParametes(p_i)

		return output_dict

	def computeDistanceToMiu(point, centroid_parameters):
		idx_c = point[0][0]
		idx_label = point[0][1]

		point_x = np.array(point[1][0])
		miu = centroid_parameters[idx_c]['miu_' + idx_label]
		delta_i = np.sum((point_x - miu) ** 2)

		return (idx_label, delta_i)



	# Executes M iterations of the above repeat-until loop.
	# Returns the final set C of centroids.
	# The set C must be represented as an array of Vector in Java and an array of tuples in Python.

	# Calculates the parameters within the partitioned data.
	# Computes the parameters alpha_i, beta_i, miu_A_i, miu_B_i, l_i, fixedA, fixedB for each partition U𝑖 of the input set U.

	def CentroidsSelection(alpha_lst, beta_lst, miu_A_lst, miu_B_lst, l_lst, fixed_A, fixed_B, num_cluster_K):
		# Updates the centroids based on the partitioned data.
		new_centroids = []

		# Given function Computer VectorX from calculated parameters
		def computeVectorX(fixed_a, fixed_b, alpha, beta, ell, k):
			alpha = np.array(alpha, dtype=float)
			beta = np.array(beta, dtype=float)
			ell = np.array(ell, dtype=float)

			gamma = 0.5
			x_dist = np.zeros(k, dtype=float)
			power = 0.5
			t_max = 10

			for _ in range(t_max):
				f_a = fixed_a
				f_b = fixed_b
				power /= 2

				denominator = gamma * alpha + (1 - gamma) * beta
				numerator = (1 - gamma) * beta * ell

				temp = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

				x_dist[:] = temp

				f_a += np.sum(alpha * temp * temp)

				temp = ell - temp

				f_b += np.sum(beta * temp * temp)

				if np.isclose(f_a, f_b):
					break

				if f_a > f_b:
					gamma += power
				else:
					gamma -= power

			return x_dist.tolist()

		# Computes the vector X for each partition.
		vector_x = computeVectorX(fixed_A, fixed_B, alpha_lst, beta_lst, l_lst, num_cluster_K)
		
		def calculateNewCentroid(miu_A_lst, miu_B_lst, l_lst, vector_x):
			miu_A = np.array(miu_A_lst)
			miu_B = np.array(miu_B_lst)
			l = np.array(l_lst).reshape(-1,1)
			x = np.array(vector_x).reshape(-1,1)

			num = (l - x) * miu_A + x * miu_B
			den =  np.where(l ==0,1.0,l)

			calcuated_centroids = num / den

			final_centroids = np.where(l == 0, miu_A, calcuated_centroids)

			return tuple(map(tuple, final_centroids))

		new_centroids = calculateNewCentroid(miu_A_lst, miu_B_lst, l_lst, vector_x)

		return new_centroids

	# Iteration
	# Initializes a set C of K centroids using kmeans++ (this can be achieved by running the Spark implementation of LLody's algorithm with 0 iterations).
	initial_centroids = LloydsAlgorithm(inputPoints, num_cluster_K, 0)
	bc_initial_centroids = sc.broadcast(initial_centroids)

	start_time = time.time()

	for _ in range(num_iteration_M):
		# Find the closet centroid for point
		point_set = inputPoints.map(lambda p: find_closest_centroid(p, bc_initial_centroids.value)).cache()
		
		# Calculate parameters especiall miu for the calculation of delta in following
		centroid_summary =	point_set.reduceByKey(summaryCentroidLabelPartition).collectAsMap()
		centroid_parameters = getParameterByCentroid(centroid_summary, num_cluster_K,D)

		# Broadcast the centroid parameters to all nodes
		bc_params = sc.broadcast(centroid_parameters)

		# Calculate delta
		delta = point_set.map(lambda p: computeDistanceToMiu(p, bc_params.value))\
			.reduceByKey(lambda x, y: x + y)\
			.collectAsMap()

		new_centroids = CentroidsSelection(
			alpha_lst = [x['alpha'] for x in centroid_parameters.values()],
			beta_lst = [x['beta'] for x in centroid_parameters.values()],
			miu_A_lst = [x['miu_A'] for x in centroid_parameters.values()],
			miu_B_lst = [x['miu_B'] for x in centroid_parameters.values()],
			l_lst = [x['l'] for x in centroid_parameters.values()],
			fixed_A = delta['A']/NA,
			fixed_B = delta['B']/NB,
			num_cluster_K = num_cluster_K
		)
		bc_initial_centroids = sc.broadcast(new_centroids)

	end_time = time.time()
	RUNNING_TIME_ON["centroids_by_fair_lloyd"] = (end_time - start_time) * 1000
	# After M iterations, the final centroids are returned.
	centroids = new_centroids

	return centroids

# 2. Include method/function MRComputeFairObjective from HW1, which takes in input the set U = A union B and a set C of centroids,
# and returns the value of the objective function phi(A,B,C) described above. Make sure to correct bugs, if any.
def MRComputeFairObjective(inputPoints, centroids):
	# Computes the fairness objective function phi(A,B,C) for the input points and the centroids.
	# Returns the value of the fairness objective function.
	# NA = inputPoints.filter(lambda x: x[1] == "A").map(lambda x: ("A", 1)).reduceByKey(lambda x, y: x + y).collectAsMap().get("A", 0)
	# NB = inputPoints.filter(lambda x: x[1] == "B").map(lambda x: ("B", 1)).reduceByKey(lambda x, y: x + y).collectAsMap().get("B", 0)

	centroids = np.array(centroids)

	def error(point):
		point_arr = np.array(point)
		closest_centroid = np.sum((centroids - point_arr) ** 2, axis=1)
		return np.min(closest_centroid)

	distances = inputPoints.map(lambda x: (x[1], error(x[0]))).reduceByKey(lambda x, y: x + y).collectAsMap()
	dist_A = distances['A']
	dist_B = distances['B']

	return max(1/NA * dist_A, 1/NB * dist_B)

# 3. Write a program GxxHW2.java (for Java users) or GxxHW2.py (for Python users), where xx is your 2-digit group number,
# which receives in input, as command-line arguments, a path to the file storing the input points, and 3 integers L,K,M, and does the following:

def main():

	# CHECKING NUMBER OF CMD LINE PARAMETERS
	assert len(sys.argv) == 5, "Usage: python G24HW2.py <input_file> <L> <K> <M>"

	# SPARK SETUP
	conf = SparkConf().setAppName('G24HW2')
	global sc
	sc = SparkContext(conf=conf)

	# 3.1. Prints the command-line arguments and stores L,K,M into suitable variables. 
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


	# 3.2 Reads the input points into an RDD of (point,group) pairs called inputPoints,
	# subdivided into L partitions.

	# Remove the assertation to enable to run hdfs file on the cluster
	# assert os.path.isfile(input_file_path), "File or folder not found"
	input_file = sc.textFile(input_file_path, num_partition_L) # This one or the one below
	inputPoints = formatInputFile(input_file).cache()

	# 3.3 Prints the number N of points, the number NA of points of group A, and the number NB of points of group B
	global NA, NB

	N_count = inputPoints.map(lambda x: (x[1], 1)).reduceByKey(lambda x, y: x + y).collectAsMap()
	NA = N_count.get("A", 0)
	NB = N_count.get("B", 0)
	N = NA + NB

	# 3.4 Prepare time variables for calculating the running time
	global RUNNING_TIME_ON

	RUNNING_TIME_ON = dict(
		centroids_by_standard_lloyd=0,
		centroids_by_fair_lloyd=0,
		fairness_by_standard_lloyd=0,
		fairness_by_fair_lloyd=0
	)

	# 3.5 Computes a set Cstand of K centroids for the input points, by running the Spark implementation of the standard Lloyd's algorithm,
	# with M iterations, disregarding the demographic groups.
	centroids_by_standard_lloyd = TimedLloydsAlgorithm(inputPoints, num_cluster_K, num_iteration_M)

	# 3.6 Computes a set Cfair of K centroids by running MRFairLloyd(inputPoints,K,M).
	centroids_by_fair_lloyd = MRFairLloyd(inputPoints, num_cluster_K, num_iteration_M)

	# 3.7 Computes and prints phi(A, B,Cstand) and phi(A, B,Cfair).
	# 3.7.1 Computes the fairness of the centroids by standard Lloyd's algorithm
	start_time = time.time()
	standard_fairness = MRComputeFairObjective(inputPoints, centroids_by_standard_lloyd)
	end_time = time.time()
	RUNNING_TIME_ON["fairness_by_standard_lloyd"] = (end_time - start_time) * 1000 # convert to milliseconds
	# 3.7.2 Computes the fairness of the centroids by Fair Lloyd's algorithm
	start_time = time.time()
	fair_fairness = MRComputeFairObjective(inputPoints, centroids_by_fair_lloyd)
	end_time = time.time()
	RUNNING_TIME_ON["fairness_by_fair_lloyd"] = (end_time - start_time) * 1000 # convert to milliseconds

	# Prints separately the times, in seconds, spent to compute : Cstand , Cfair , phi(A, B,Cstand) and phi(A, B,Cfair).	

	print(f"Input file = {input_file_path}, L = {num_partition_L}, K = {num_cluster_K}, M = {num_iteration_M}")
	print(f"N = {N}, NA = {NA}, NB = {NB}")

	print(f"Fair Objective with Standard Centers = {standard_fairness:.4f}")
	print(f"Fair Objective with Fair Centers = {fair_fairness:.4f}")

	print(f"Time to compute standard centers = {RUNNING_TIME_ON['centroids_by_standard_lloyd']:.0f} ms")
	print(f"Time to compute fair centers = {RUNNING_TIME_ON['centroids_by_fair_lloyd']:.0f} ms")
	print(f"Time to compute objective with standard centers = {RUNNING_TIME_ON['fairness_by_standard_lloyd']:.0f} ms")
	print(f"Time to compute objective with fair centers = {RUNNING_TIME_ON['fairness_by_fair_lloyd']:.0f} ms")	

if __name__ == "__main__":
	main()
