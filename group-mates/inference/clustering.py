# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datasets
import logging
import random

import numpy as np
import yaml
from clustering.clustering import compute_centroids
from clustering.sort_clusters import assign_and_sort_clusters

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

confg_file = "clustering/configs/group-mates.yaml"
## -- Load kmeans clustering parameters from configs file
with open(confg_file, "r") as y_file:
    params = yaml.load(y_file, Loader=yaml.FullLoader)

SEED = params["seed"]
random.seed(SEED)
emb_memory_loc = params["emb_memory_loc"]
paths_memory_loc = params["paths_memory_loc"]
emb_size = params["emb_size"]
path_str_type = params["path_str_type"]

out_dir = emb_memory_loc.split("/emb.npy")[0]
emb_memory_list = []
for i in range(8):
    dataset = datasets.load_from_disk(f"{out_dir}/{i}")
    emb_memory = np.array(dataset["reps"])
    print(i, emb_memory.shape)
    emb_memory_list.append(emb_memory)
emb_memory = np.concatenate(emb_memory_list, axis=0)
print(emb_memory.shape)
emb_array = np.memmap(
    emb_memory_loc,
    dtype="float32",
    mode="w+",
    shape=emb_memory.shape,
)
emb_array[:] = emb_memory[:]
emb_array.flush()

emb_memory = np.memmap(
    emb_memory_loc,
    dtype="float32",
    mode="r",
)
emb_memory = emb_memory.reshape(-1, emb_size)
dataset_size = len(emb_memory)
print(dataset_size)

path = [f"{i}" for i in range(dataset_size)]
paths_array = np.memmap(
    paths_memory_loc,
    dtype=path_str_type,
    mode="w+",
    shape=(dataset_size,),
)
paths_array[:] = path[:]

compute_centroids(
    data=emb_memory,
    ncentroids=params["ncentroids"],
    niter=params["niter"],
    seed=params["seed"],
    Kmeans_with_cos_dist=params["Kmeans_with_cos_dist"],
    save_folder=params["save_folder"],
    logger=logger,
    verbose=True,
)

paths_memory = np.memmap(
    paths_memory_loc,
    dtype=path_str_type,
    mode="r",
    shape=(dataset_size,),
)

assign_and_sort_clusters(
    data=emb_memory,
    paths_list=paths_memory,
    sim_metric=params["sim_metric"],
    keep_hard=params["keep_hard"],
    kmeans_with_cos_dist=params["Kmeans_with_cos_dist"],
    save_folder=params["save_folder"],
    sorted_clusters_file_loc=params["sorted_clusters_file_loc"],
    cluster_ids=range(0, params["ncentroids"]),
    logger=logger,
)
