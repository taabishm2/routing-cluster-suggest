import csv
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint

df = pd.read_csv('source_pincodes_bkp.csv')
coords = df.as_matrix(columns=['lat', 'lon'])

kms_per_radian = 6371.0088
epsilon = 1.5 / kms_per_radian
db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
cluster_labels = db.labels_
num_clusters = len(set(cluster_labels))
clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])
print('Number of clusters: {}'.format(num_clusters))

def suggest():
    dest_pincode_file = open("destination_pincodes.csv", "r")
    dest_pincode_reader = csv.reader(dest_pincode_file)
    dest_pincode_set = set([p[1] for p in dest_pincode_reader][1:])

    src_pincode_file = open("source_pincodes_bkp.csv", "r")
    src_pincode_reader = csv.reader(src_pincode_file)
    src_pincode_set = set([p[1] for p in src_pincode_reader][1:])



if __name__ == "__main__":
    suggest()
