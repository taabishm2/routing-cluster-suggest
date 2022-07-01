import csv
import random
import time
import geopy.distance
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import codecs
import os
import zipfile
import io

from collections import defaultdict
from tqdm import tqdm
from sklearn.cluster import KMeans
from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
matplotlib.pyplot.switch_backend('Agg')


@app.post("/suggest-clusters", status_code=201)
async def root(api_key: str, cluster_count: int, sources: UploadFile = File(...), serviceable: UploadFile = File(...)):
    if api_key != "vk4d56r2enfh":
        raise HTTPException(status_code=401, detail="Request Unauthorized")

    try:
        main(cluster_count, sources, serviceable)
        return get_zipped_files()
    except Exception as e:
        print("Errors Encountered!")
        f = open("routing-suggest-error.csv", "w")
        f.write(str(e))
        f.close()


def get_zipped_files():
    filenames = ["output/clustered.png", "output/clusters.csv", "output/cluster_definition.csv", "output/area_cluster_mapping.csv"]
    output_zip_name = "suggested_clusters.zip"

    io_bytes = io.BytesIO()
    zip_file = zipfile.ZipFile(io_bytes, "w")

    for fpath in filenames:
        file_dir, file_name = os.path.split(fpath)
        zip_file.write(fpath, file_name)
    zip_file.close()
    response_zip = Response(io_bytes.getvalue(), media_type="application/x-zip-compressed", headers={
        'Content-Disposition': f'attachment;filename={output_zip_name}'
    })

    return response_zip


# Test function to generate source pin-codes randomly
def generate_random_sources(count):
    geo_dataset = open('data/geo_dataset.csv', 'r')
    dataset_reader = csv.reader(geo_dataset)

    source_rows = [(row[9], row[10]) for row in dataset_reader][1:]
    picked_locations = random.sample(source_rows, count)

    output_sources = open('data/source_pincodes.csv', 'w')
    output_sources_writer = csv.writer(output_sources)

    output_sources_writer.writerow(["wh", "lat", "lon"])
    for idx, val in enumerate(picked_locations):
        output_sources_writer.writerow(["WH" + str(idx), val[0], val[1]])
    output_sources.close()


# Test function to generate destination pin-codes randomly
def generate_random_destinations(count):
    geo_dataset = open('data/geo_dataset.csv', 'r')
    dataset_reader = csv.reader(geo_dataset)

    picked_locations = [(row[1], row[9], row[10]) for row in dataset_reader][1:]
    picked_locations = random.sample(picked_locations, count)

    output_sources = open('data/destination_pincodes.csv', 'w')
    output_sources_writer = csv.writer(output_sources)

    output_sources_writer.writerow(["postalcode", "lat", "lon"])
    for idx, val in enumerate(picked_locations):
        output_sources_writer.writerow([val[0], val[1], val[2]])
    output_sources.close()


# Add locations in /uploads/sources.csv with format (warehouse_name, pincode)
def generate_source_coordinates(pincode_coord_map, uploaded_sources_file):
    sources_file_reader = csv.reader(codecs.iterdecode(uploaded_sources_file.file, 'utf-8'))

    output_sources_file = open('data/source_pincodes.csv', 'w')
    output_file_writer = csv.writer(output_sources_file)

    output_file_writer.writerow(["wh", "lat", "lon"])
    next(sources_file_reader, None)
    for row in sources_file_reader:
        output_file_writer.writerow([row[0], pincode_coord_map[row[1]][0], pincode_coord_map[row[1]][1]])
    output_sources_file.close()


# Add serviceable pincodes in /uploads/serviceable.csv with format (pincode)
def generate_destination_coords(pincode_coord_map, uploaded_destinations_file):
    destinations_file_reader = csv.reader(codecs.iterdecode(uploaded_destinations_file.file, 'utf-8'))

    output_destinations_file = open('data/destination_pincodes.csv', 'w')
    destinations_file_writer = csv.writer(output_destinations_file)

    destinations_file_writer.writerow(["postalcode", "lat", "lon"])
    next(destinations_file_reader, None)
    for row in destinations_file_reader:
        destinations_file_writer.writerow([row[0], pincode_coord_map[row[0]][0], pincode_coord_map[row[0]][1]])
    output_destinations_file.close()


def cluster(cluster_count, show_cluster_graph):
    df = pd.read_csv('data/source_pincodes.csv')
    source_pincode_df = df.loc[:, ['wh', 'lat', 'lon']]

    # Performing k-means clustering based on direct distance between two coordinates
    kmeans = KMeans(n_clusters=cluster_count, init='k-means++')
    kmeans.fit(source_pincode_df[source_pincode_df.columns[1:3]])
    source_pincode_df['cluster_label'] = kmeans.fit_predict(source_pincode_df[source_pincode_df.columns[1:3]])

    cluster_centers = kmeans.cluster_centers_
    cluster_labels = kmeans.predict(source_pincode_df[source_pincode_df.columns[1:3]])

    # Generating csvs to be used for uploading in Assure
    save_clusters_to_csv(cluster_labels)
    save_cluster_definitions_to_csv(source_pincode_df)

    plot_cluster_graph(cluster_centers, cluster_labels, show_cluster_graph, source_pincode_df)
    return source_pincode_df, cluster_centers


def plot_cluster_graph(cluster_centers, cluster_labels, show_clusters, source_pincode_df):
    source_pincode_df.plot.scatter(x='lon', y='lat', c=cluster_labels, s=5, cmap='gist_rainbow')
    plt.scatter(cluster_centers[:, 1], cluster_centers[:, 0], c='black', s=20, alpha=0.5)
    plt.title("Warehouse-cluster distribution", fontsize=15)
    plt.xlabel("latitude")
    plt.ylabel("longitude")

    # Annotating cluster centers with their names
    for i, c in enumerate(cluster_centers):
        plt.annotate(str(i), (c[1], c[0]))

    if show_clusters: plt.savefig("output/clustered.png", dpi=1000)


def save_cluster_definitions_to_csv(source_pincode_df):
    cluster_definition_output = open("output/cluster_definition.csv", "w")
    cluster_definition_output_writer = csv.writer(cluster_definition_output)
    cluster_definition_output_writer.writerow(["clusterName", "locationName"])
    for index, row in source_pincode_df.iterrows():
        cluster_definition_output_writer.writerow([row["cluster_label"], row['wh']])
    cluster_definition_output.close()


def save_clusters_to_csv(labels):
    cluster_output_file = open("output/clusters.csv", "w")
    cluster_output_writer = csv.writer(cluster_output_file)
    cluster_output_writer.writerow(["clusterName"])
    for c in set(labels):
        cluster_output_writer.writerow([c])
    cluster_output_file.close()


def get_area_cluster_mappings(cluster_data, centers):
    cluster_centers = dict()
    for i in range(len(centers)):
        cluster_centers[i] = (centers[i][0], centers[i][1])

    cluster_groups = defaultdict(list)
    for index, row in cluster_data.iterrows():
        cluster_groups[row["cluster_label"]].append((row['lat'], row['lon']))

    destination_pincode_file = open("data/destination_pincodes.csv", "r")
    destination_pincode_reader = csv.reader(destination_pincode_file)

    pincode_cluster_map = defaultdict(list)

    next(destination_pincode_reader, None)
    pincode_cluster_avg_distance_map = defaultdict(list)
    for destination in tqdm(destination_pincode_reader, desc='Processing:'):
        destination_pincode = destination[0]
        destination_coordinates = (destination[1], destination[2])

        get_distance_matrix(cluster_centers, cluster_groups, destination_coordinates, destination_pincode,
                            pincode_cluster_avg_distance_map)

    for x in pincode_cluster_avg_distance_map:
        pincode_cluster_avg_distance_map[x] = sorted(pincode_cluster_avg_distance_map[x], key=lambda e: e[1])
        pincode_cluster_map[x] = [c[0] for c in pincode_cluster_avg_distance_map[x]][:5]

    save_area_cluster_mapping_to_csv(pincode_cluster_map)

    return pincode_cluster_map


def save_area_cluster_mapping_to_csv(pincode_cluster_map):
    mapping_output = open("output/area_cluster_mapping.csv", "w")
    mapping_output_writer = csv.writer(mapping_output)
    mapping_output_writer.writerow(["areaCodePrefix", "cluster1", "cluster2", "cluster3", "cluster4", "cluster5"])
    for pin in pincode_cluster_map:
        mapping_output_writer.writerow([pin] + pincode_cluster_map[pin])
    mapping_output.close()


def get_distance_matrix(cluster_centers, cluster_groups, destination_coords, destination_pincode,
                        pincode_cluster_avg_distance_map):
    distance_matrix = defaultdict(dict)
    for source_cluster in cluster_groups:
        min_distance = geopy.distance.geodesic(cluster_centers[source_cluster], destination_coords).km
        distance_matrix[destination_pincode][source_cluster] = min_distance
        pincode_cluster_avg_distance_map[destination_pincode].append(
            (source_cluster, distance_matrix[destination_pincode][source_cluster]))


def get_pincode_coord_map():
    dataset = open('data/geo_dataset.csv', 'r')
    dataset_reader = csv.reader(dataset)

    next(dataset_reader, None)
    coord_map = {e[1]: (e[9], e[10]) for e in dataset_reader}
    return coord_map


def main(cluster_count, sources, serviceable):
    start_time = time.time()

    is_test_run = False
    source_count, destination_count = 10, 10

    if is_test_run:
        generate_random_sources(source_count)
        generate_random_destinations(destination_count)
    else:
        pincode_coord_map = get_pincode_coord_map()
        generate_source_coordinates(pincode_coord_map, sources)
        generate_destination_coords(pincode_coord_map, serviceable)

    print("Clustering ...")
    cluster_data, centers = cluster(cluster_count, True)
    print("Generating Mappings ...")
    mappings = get_area_cluster_mappings(cluster_data, centers)

    print(len(mappings), "mappings generated in", time.time() - start_time, "sec")
