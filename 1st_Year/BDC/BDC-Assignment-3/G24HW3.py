from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark import StorageLevel
import numpy as np
import threading
import sys

# After how many items should we stop?
THRESHOLD = -1 # To be set via command line

# 0. Generate hash functions
def generate_hash_func(num_D, num_C):
    """Generates a hash function.

        Args:
            num_D (int): The number of hash functions to generate.
            num_C (int): The number of buckets in the hash table.

        Returns:
            function: A hash function that takes an integer as input and returns an integer.
    """

    np.random.seed(0)  # For reproducibility
    num_P = 8191
    vec_a = np.random.randint(1, num_P-1, size=num_D)
    vec_b = np.random.randint(0, num_P-1, size=num_D)
    def hash_func(x):
        num_P = 8191
        intermediate_val = (vec_a * x + vec_b) % num_P
        hash_vector = intermediate_val % num_C
        return hash_vector
    return hash_func


def update_CM_metrix(key, value):
    """Updates the Count-Min sketch matrix with a new key-value pair.

        This function takes a key and a value, hashes the key using the Count-Min hash function,
        and updates the Count-Min sketch matrix by adding the value to the corresponding cells
        determined by the hash values.

        Args:
            key: The key to be added to the Count-Min sketch.
            value: The value to be added to the Count-Min sketch for the given key.
        """
    global CM_metrix, h_hash_func, num_D, num_W
    hash_vector = h_hash_func(key)
    metrix = np.zeros((num_D, num_W), dtype=int)
    x = np.arange(len(hash_vector))
    y = hash_vector
    metrix[x, y] = value

    # original_value = CM_metrix[x, y]
    # min_value = np.min(original_value)
    # x_to_update = np.where(original_value == min_value)
    # y_to_update = hash_vector[x_to_update]
    # metrix[x_to_update, y_to_update] = 1
    CM_metrix += metrix

def update_CS_metrix(key, value):
    """Updates the Count-Sketch matrix with a given key and value.

        This function uses the Count-Sketch algorithm to update the matrix.
        It calculates the hash vector of the key, creates a matrix of zeros,
        and then updates the matrix with the given value multiplied by the
        second hash function applied to the hash vector. Finally, it adds
        the updated matrix to the global Count-Sketch matrix.

        Args:
            key: The key to be inserted into the Count-Sketch matrix.
            value: The value to be associated with the key.
        """
    global CS_metrix, h_hash_func, g_hash_func, num_D, num_W

    hash_vector = h_hash_func(key)
    metrix = np.zeros((num_D, num_W), dtype=int)
    x = np.arange(len(hash_vector))
    y = hash_vector
    metrix[x, y] = value * g_hash_func(key)
    CS_metrix += metrix


# Operations to perform after receiving an RDD 'batch' at time 'time'
def process_batch(time, batch):
    # We are working on the batch at time `time`.
    global streamLength, histogram
    batch_size = batch.count()
    # If we already have enough points (> THRESHOLD), skip this batch.
    if streamLength[0]>=THRESHOLD:
        return
    streamLength[0] += batch_size
    # Extract the distinct items from the batch
    batch_items = batch.map(lambda s: (int(s), 1)).reduceByKey(lambda i1, i2: i1+i2).collectAsMap()

    # Update the streaming state
    for key in batch_items:
        update_CM_metrix(key, batch_items[key])
        update_CS_metrix(key, batch_items[key])
        histogram[key] = histogram.get(key, 0) + batch_items[key]
            
    # If we wanted, here we could run some additional code on the global histogram
    if batch_size > 0:
        print("Batch size at time [{0}] is: {1}".format(time, batch_size))

    if streamLength[0] >= THRESHOLD:
        stopping_condition.set()


# 4. calculate the average relative error of top-K heavy hitters
def calculateTopKAverageRelativeError(histogram, sketch_metrix, sketch_type, num_K):
    """Calculates the average relative error for the top-K items between a true histogram and a sketch matrix.
        Args:
            histogram (dict): A dictionary representing the true histogram, where keys are items and values are their counts.
            sketch_metrix (numpy.ndarray): The sketch matrix used for estimation.
            sketch_type (str): The type of sketch being used, either "CM" (Count-Min) or "CS" (Count-Sketch).
            num_K (int): The number of top items to consider for the average relative error calculation.
        Returns:
            dict: A dictionary containing the average relative error, the top-K items from the true histogram, and their estimated counts from the sketch.
                 The dictionary has the following keys:
                    - 'avg_relative_error' (float): The average relative error for the top-K items.
                    - 'true_dict' (list): A list of tuples representing the top-K items from the true histogram and their counts.
                    - 'sketch_dict' (dict): A dictionary of the top-K items and their estimated counts from the sketch.
        Raises:
            ValueError: If an unknown sketch_type is provided.
        """
    def query_sketch(key, sketch_type):
        """Queries the sketch matrix to estimate the frequency of a given key.
            Args:
                key (str): The key to query.
                sketch_type (str): The type of sketch to use ("CM" for Count-Min, "CS" for Count-Sketch).
            Returns:
                float: The estimated frequency of the key.
            Raises:
                ValueError: If an unknown sketch type is specified.
            """
        # Query the sketch matrix using the appropriate hash function
        def query_CM_sketch(key):
            global h_hash_func
            hash_vector = h_hash_func(key)
            x = np.arange(len(hash_vector))
            y = hash_vector
            item_vector_extracted = sketch_metrix[x, y]
            output = np.min(item_vector_extracted)
            return output
        
        def query_CS_sketch(key):
            global h_hash_func, g_hash_func
            hash_vector = h_hash_func(key)
            x = np.arange(num_D)
            y = hash_vector
            item_vector_extracted = sketch_metrix[x, y] * g_hash_func(key)
            output = np.median(item_vector_extracted)
            return output

        if sketch_type == "CM":
            return query_CM_sketch(key)
        elif sketch_type == "CS":
            return query_CS_sketch(key)
        else:
            raise ValueError("Unknown sketch type")

    # Get the top-K items from the true histogram
    # But here it asks for top k heavy hitters, so need modification
    sorted_true_items = sorted(histogram.items(), key=lambda x: x[1], reverse=True)

    value_on_K = sorted_true_items[num_K-1][1]
    top_K_true_items = []
    i = 0
    # while sorted_true_items[i][1] >= value_on_K:
    while i < len(sorted_true_items) and sorted_true_items[i][1] >= value_on_K:

        top_K_true_items.append(sorted_true_items[i])
        i += 1

    top_K_true_items.sort(key=lambda x: x[0])  # Sort by item ID for consistency

    top_K_sketch_items = {item: query_sketch(item, sketch_type) for item, _ in top_K_true_items}

    total_error = 0.0
    for item, count in top_K_true_items:
        estimate = top_K_sketch_items.get(item, 0)
        if estimate > 0:  # Avoid division by zero
            relative_error = abs(count - estimate) / count
            total_error += relative_error
    
    output = dict(
        avg_relative_error=total_error / len(top_K_true_items),
        true_dict=top_K_true_items,
        sketch_dict=top_K_sketch_items,
    )
    return output


if __name__ == '__main__':
    # assert len(sys.argv) == 3, "USAGE: port, threshold"
    # portExp: The port number to connect to on algo.dei.unipd.it.
    # T: The target number of items to process.
    # D: The number of rows for each sketch.
    # W: The number of columns for each sketch.
    # K: The number of top frequent items of interest.

    # 0. Exam input parameters:
    assert len(sys.argv) == 6, "USAGE: python G24HW3.py <portExp> <T> <D> <W> <K>"


    # IMPORTANT: when running locally, it is *fundamental* that the
    # `master` setting is "local[*]" or "local[n]" with n > 1, otherwise
    # there will be no processor running the streaming computation and your
    # code will crash with an out of memory (because the input keeps accumulating).
    conf = SparkConf().setMaster("local[*]").setAppName("G24HW3")
    # If you get an OutOfMemory error in the heap consider to increase the
    # executor and drivers heap space with the following lines:
    # conf = conf.set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
    
    
    # Here, with the duration you can control how large to make your batches.
    # Beware that the data generator we are using is very fast, so the suggestion
    # is to use batches of less than a second, otherwise you might exhaust the memory.
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 0.01)  # Batch duration of 0.01 seconds
    ssc.sparkContext.setLogLevel("ERROR")
    
    # TECHNICAL DETAIL:
    # The streaming spark context and our code and the tasks that are spawned all
    # work concurrently. To ensure a clean shut down we use this semaphore.
    # The main thread will first acquire the only permit available and then try
    # to acquire another one right after spinning up the streaming computation.
    # The second tentative at acquiring the semaphore will make the main thread
    # wait on the call. Then, in the `foreachRDD` call, when the stopping condition
    # is met we release the semaphore, basically giving "green light" to the main
    # thread to shut down the computation.
    # We cannot call `ssc.stop()` directly in `foreachRDD` because it might lead
    # to deadlocks.
    stopping_condition = threading.Event()
    
    
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # INPUT READING
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    portExp = int(sys.argv[1])
    print("Receiving data from port =", portExp)
    
    THRESHOLD = int(sys.argv[2])
    print("Threshold = ", THRESHOLD)

    num_D = int(sys.argv[3])
    num_W = int(sys.argv[4])
    num_K = int(sys.argv[5])

    assert portExp in [8886, 8887, 8888, 8889], "Port must be one of 8886, 8887, 8888, 8889"
    assert THRESHOLD > 0, "Threshold must be a positive integer"
    assert num_D > 0, "D must be a positive integer"
    assert num_W > 0, "W must be a positive integer"
    assert num_K > 0, "K must be a positive integer"
        
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # DEFINING THE REQUIRED DATA STRUCTURES TO MAINTAIN THE STATE OF THE STREAM
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    # Variable streamLength below is used to maintain the number of processed stream items.
    # It must be defined as a 1-element array so that the value stored into the array can be
    # changed within the lambda used in foreachRDD. Using a simple external counter streamLength of type
    # long would not work since the lambda would not be allowed to update it.
    
    streamLength = [0]
    histogram = {} # Hash Table for the distinct elements

    CM_metrix = np.zeros((num_D, num_W), dtype=int)
    CS_metrix = np.zeros((num_D, num_W), dtype=int)

    h_hash_func = generate_hash_func(num_D, num_W)
    # CS_hash_func = generate_hash_func(num_D, num_W)

    g_hash_func_01 = generate_hash_func(num_D, 2)
    def g_hash_func (x):
        return np.where(g_hash_func_01(x) % 2 == 0, -1, 1)

    # CODE TO PROCESS AN UNBOUNDED STREAM OF DATA IN BATCHES
    stream = ssc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevel.MEMORY_AND_DISK)
    # For each batch, to the following.
    # BEWARE: the `foreachRDD` method has "at least once semantics", meaning
    # that the same data might be processed multiple times in case of failure.
    stream.foreachRDD(lambda time, batch: process_batch(time, batch))
    
    # MANAGING STREAMING SPARK CONTEXT
    print("Starting streaming engine")
    ssc.start()
    print("Waiting for shutdown condition")
    stopping_condition.wait()
    print("Stopping the streaming engine")
    
    # The following command stops the execution of the stream. The first boolean, if true, also
    # stops the SparkContext, while the second boolean, if true, stops gracefully by waiting for
    # the processing of all received data to be completed. You might get some error messages when the
    # program ends, but they will not affect the correctness.
    
    ssc.stop(False, False)
    print("Streaming engine stopped")



    # COMPUTE AND PRINT FINAL STATISTICS
    # print("Number of items processed =", streamLength[0])
    # print("Number of distinct items =", len(histogram))
    # largest_item = max(histogram.keys())
    # print("Largest item =", largest_item)
    # print(CM_metrix)
    # print(CS_metrix)

    CM_average_relative_error = calculateTopKAverageRelativeError(histogram, CM_metrix, "CM", num_K)
    CS_average_relative_error = calculateTopKAverageRelativeError(histogram, CS_metrix, "CS", num_K)
    

    # ====== OUTPUT FORMAT ====
    print(f'portExp={portExp}, T={THRESHOLD}, D={num_D}, W={num_W}, K={num_K}')
    # The number of distinct items in the stream
    print("Number of processed items =", streamLength[0])
    print("Number of distinct items =", len(histogram))
    print("Number of Top-K Heavy Hitters =", len(CM_average_relative_error['true_dict']))
    # The average relative error of CM for the top-K heavy hitters
    print("Avg Relative Error for Top-K Heavy Hitters with CM =", CM_average_relative_error['avg_relative_error'])
    # The average relative error of CS for the top-K heavy hitters
    print("Avg Relative Error for Top-K Heavy Hitters with CS =", CS_average_relative_error['avg_relative_error'])
    # Only if K <= 10 The true and estimated (using CM) frequencies of the top-K heavy hitters
    if num_K <= 10:
        print("Top-K Heavy Hitters:")
        for item, true_count in CM_average_relative_error['true_dict']:
            estimate_cm = CM_average_relative_error['sketch_dict'].get(item, 0)
            print(f"Item {item} True Frequency = {true_count} Estimated Frequency with CM = {int(round(estimate_cm))}")
        # print(CM_average_relative_error['true_dict'])
        # print(CM_average_relative_error['sketch_dict'])

        # print("Top-K heavy hitters (CS):")
        # print(CS_average_relative_error['true_dict'])
        # print(CS_average_relative_error['sketch_dict'])
