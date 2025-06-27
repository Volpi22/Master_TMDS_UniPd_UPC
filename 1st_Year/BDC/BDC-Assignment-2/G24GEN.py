import numpy as np
import pandas as pd
import sys

def points_generation(N: int, K: int) -> pd.DataFrame:
    """
    Generates a 2D dataset with N points and K main clusters.

    Args:
        N (int): Total number of points to generate.
        K (int): Number of main clusters.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame with columns ['x', 'y', 'label'] containing the generated points.
                      'label' can be 'A' or 'B'.
    """

    assert N >= 10, "N must be greater or equal to 10."
    assert K >= 1, "K must be greater or equal to 1."
    assert K < N , "K must be less than N."
    assert (N // K)*0.1 >= 1, "Each cluster must have at least 1 point for each label. Increase N or decrease K." 

    all_points_x = []
    all_points_y = []
    all_labels = []

    # We want to ensures the subclusters distance is small if compared to the distance of the main clusters

    # Base standard deviation for the sub-clusters
    subcluster_std_dev = 1

    # Separating A (right) and B (left) subclusters horizontally
    x_separation = 10.0  
    x_offset_subclusters = x_separation * subcluster_std_dev

    # Separating main clusters vertically
    y_separation = 100.0 
    y_offset_main_clusters = y_separation * subcluster_std_dev

    # Distribute N points among K clusters
    base_points_per_main_cluster = N // K
    remainder_points = N % K

    for i in range(K):
        current_total_points_in_cluster = base_points_per_main_cluster
        if i < remainder_points:
            current_total_points_in_cluster += 1

        if current_total_points_in_cluster == 0:
            continue

        # Number of A and B points for the cluster (10% A, 90% B)
        num_A_in_cluster = int((0.1 * current_total_points_in_cluster) + 0.5)
        num_B_in_cluster = current_total_points_in_cluster - num_A_in_cluster

        # Center for this main cluster
        y_center_main = i * y_offset_main_clusters
        
        # Spreadness for B clusters
        std_dev_B = subcluster_std_dev / 3.0
        # Spreadness for A clusters
        std_dev_A = subcluster_std_dev / 3.0

        # X-coordinate for B sub-cluster
        center_B_x = -x_offset_subclusters / 2.0
        # X-coordinate for A sub-cluster
        center_A_x = x_offset_subclusters / 2.0
        
        if num_B_in_cluster > 0:
            points_B_x = np.random.normal(loc=center_B_x, scale=std_dev_B, size=num_B_in_cluster)
            points_B_y = np.random.normal(loc=y_center_main, scale=std_dev_B, size=num_B_in_cluster)
            all_points_x.extend(points_B_x)
            all_points_y.extend(points_B_y)
            all_labels.extend(['B'] * num_B_in_cluster)

        if num_A_in_cluster > 0:
            points_A_x = np.random.normal(loc=center_A_x, scale=std_dev_A, size=num_A_in_cluster)
            points_A_y = np.random.normal(loc=y_center_main, scale=std_dev_A, size=num_A_in_cluster)
            all_points_x.extend(points_A_x)
            all_points_y.extend(points_A_y)
            all_labels.extend(['A'] * num_A_in_cluster)

    df = pd.DataFrame({
        'x': all_points_x,
        'y': all_points_y,
        'label': all_labels
    })

    return df

def main():
    if len(sys.argv) != 3:
        print("Usage: python G24GEN.py <N> <K>")
        sys.exit(1)

    N = int(sys.argv[1])
    K = int(sys.argv[2])

    data = points_generation(N, K)
    return data

if __name__ == "__main__":
    main()