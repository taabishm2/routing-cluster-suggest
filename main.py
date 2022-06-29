import csv
import random
from collections import defaultdict

import geopy.distance
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

sns.set()


# Test function to generate source pin-codes randomly
def generate_random_sources(count):
    dataset = open('data/geo_dataset.csv', 'r')
    dataset_reader = csv.reader(dataset)

    rows = [(e[9], e[10]) for e in dataset_reader][1:]
    picked_locs = random.sample(rows, count)

    output_sources = open('data/source_pincodes.csv', 'w')
    data_writer = csv.writer(output_sources)

    data_writer.writerow(["wh", "lat", "lon"])
    for idx, val in enumerate(picked_locs):
        data_writer.writerow(["WH" + str(idx), val[0], val[1]])
    output_sources.close()


# Test function to generate destination pin-codes randomly
def generate_random_destinations(count):
    dataset = open('data/geo_dataset.csv', 'r')
    dataset_reader = csv.reader(dataset)

    rows = [(e[1], e[9], e[10]) for e in dataset_reader][1:]
    picked_locs = random.sample(rows, count)

    output_sources = open('data/destination_pincodes.csv', 'w')
    data_writer = csv.writer(output_sources)

    data_writer.writerow(["postalcode", "lat", "lon"])
    for idx, val in enumerate(picked_locs):
        data_writer.writerow([val[0], val[1], val[2]])
    output_sources.close()


# Add locations in /uploads/sources.csv with format (warehouse_name, pincode)
def generate_source_coords(pincode_coord_map):
    upload_file = open('uploads/sources.csv')
    uplaod_reader = csv.reader(upload_file)

    output_sources = open('data/source_pincodes.csv', 'w')
    data_writer = csv.writer(output_sources)

    data_writer.writerow(["wh", "lat", "lon"])
    next(uplaod_reader, None)
    for row in uplaod_reader:
        data_writer.writerow([row[0], pincode_coord_map[row[1]][0], pincode_coord_map[row[1]][1]])
    output_sources.close()


# Add serviceable pincodes in /uploads/serviceable.csv with format (pincode)
def generate_destn_coords(pincode_coord_map):
    upload_file = open('uploads/serviceable.csv')
    uplaod_reader = csv.reader(upload_file)

    output_sources = open('data/destination_pincodes.csv', 'w')
    data_writer = csv.writer(output_sources)

    data_writer.writerow(["postalcode", "lat", "lon"])
    next(uplaod_reader, None)
    for row in uplaod_reader:
        data_writer.writerow([row[0], pincode_coord_map[row[0]][0], pincode_coord_map[row[0]][1]])
    output_sources.close()


def cluster(num_of_clusters, show_clusters):
    df = pd.read_csv('data/source_pincodes.csv')
    X = df.loc[:, ['wh', 'lat', 'lon']]

    kmeans = KMeans(n_clusters=num_of_clusters, init='k-means++')
    kmeans.fit(X[X.columns[1:3]])
    X['cluster_label'] = kmeans.fit_predict(X[X.columns[1:3]])
    centers = kmeans.cluster_centers_
    labels = kmeans.predict(X[X.columns[1:3]])

    X.plot.scatter(x='lon', y='lat', c=labels, s=5, cmap='gist_rainbow')
    plt.scatter(centers[:, 1], centers[:, 0], c='black', s=20, alpha=0.5)
    if show_clusters: plt.show()
    return X, centers


def get_area_cluster_mappings(cluster_data, centers):
    cluster_centers = dict()
    for i in range(len(centers)):
        cluster_centers[i] = (centers[i][0], centers[i][1])

    cluster_groups = defaultdict(list)
    for index, row in cluster_data.iterrows():
        cluster_groups[row["cluster_label"]].append((row['lat'], row['lon']))

    dest_pincodes = open("data/destination_pincodes.csv", "r")
    dest_pincodes_reader = csv.reader(dest_pincodes)

    pincode_cluster_map = defaultdict(list)

    next(dest_pincodes_reader, None)
    pincode_cluster_avg_distance_map = defaultdict(list)
    for destination in dest_pincodes_reader:
        destination_pincode = destination[0]
        destination_coords = (destination[1], destination[2])

        distance_matrix = defaultdict(dict)
        for source_cluster in cluster_groups:
            min_distance = geopy.distance.geodesic(cluster_centers[source_cluster], destination_coords).km
            distance_matrix[destination_pincode][source_cluster] = min_distance
            pincode_cluster_avg_distance_map[destination_pincode].append(
                (source_cluster, distance_matrix[destination_pincode][source_cluster]))

    for x in pincode_cluster_avg_distance_map:
        pincode_cluster_avg_distance_map[x] = sorted(pincode_cluster_avg_distance_map[x], key=lambda x: x[1])

    for x in pincode_cluster_avg_distance_map:
        pincode_cluster_map[x] = [c[0] for c in pincode_cluster_avg_distance_map[x]][:5]
    return pincode_cluster_map


def get_pincode_coord_map():
    dataset = open('data/geo_dataset.csv', 'r')
    dataset_reader = csv.reader(dataset)

    next(dataset_reader, None)
    coord_map = {e[1]: (e[9], e[10]) for e in dataset_reader}
    return coord_map


def main():
    is_test_run = True
    source_count, destination_count = 2000, 2000

    cluster_count = 10
    show_clusters = False

    if is_test_run:
        generate_random_sources(source_count)
        generate_random_destinations(destination_count)
    else:
        pincode_coord_map = get_pincode_coord_map()
        generate_source_coords(pincode_coord_map)
        generate_destn_coords(pincode_coord_map)

    print("Clustering ...")
    cluster_data, centers = cluster(cluster_count, show_clusters)
    print("Generating Mappings ...")
    mappings = get_area_cluster_mappings(cluster_data, centers)

    for k in mappings: print(k, mappings[k])
    print(len(mappings), "mappings generated")


main()
