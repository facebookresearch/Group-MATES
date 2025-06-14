# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import os

from tqdm import tqdm
import numpy as np
import datasets
import yaml


def group_select(out_dir, num_clusters, params, cluster_chose_ratio):
    dataset = datasets.concatenate_datasets([datasets.load_from_disk(f"{out_dir}/{i}") for i in range(8)])
    prediction = np.array(dataset["prediction"]).reshape(-1)

    temp = 1.0 # TODO: should be assigned from the config file
    alpha = 1.0 # TODO: should be assigned from the config file
    emb_memory = np.memmap(
        params["emb_memory_loc"],
        dtype="float32",
        mode="r",
    )
    emb_memory = emb_memory.reshape(-1, params["emb_size"])
    print(">> Reps shape:", emb_memory.shape)

    selected_indices = []
    for cluster_id in tqdm(range(num_clusters)):
        cluster_i = np.load(
            os.path.join(
                params["sorted_clusters_file_loc"],
                f"cluster_{cluster_id}.npy",
            )
        )
        indices = cluster_i[:, 0].astype("int32")
        selection_size = int(cluster_chose_ratio[cluster_id] * len(indices))
        metrics = prediction[indices]
        reps = emb_memory[indices]
        avg_rel = np.zeros(reps.shape[0])
        tmp_indices = []
        for i in range(selection_size):
            scores = avg_rel * metrics if i > 0 else metrics
            scores[tmp_indices] = float("inf")
            selected_index = np.argmin(scores)
            selected_indices.append(indices[selected_index])
            tmp_indices.append(selected_index)
            cur_rel = alpha * (1 - np.matmul(reps, reps[selected_index].transpose()) / temp)
            avg_rel = (i * avg_rel + cur_rel) / (i + 1)
    return selected_indices


if __name__ == "__main__":
    confg_file = "clustering/configs/group-mates.yaml"
    with open(confg_file, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)

    SEED = params["seed"]
    random.seed(SEED)
    num_clusters = params["ncentroids"]
    selection_ratio = params["selection_ratio"]
    out_dir = params["emb_memory_loc"].split("/emb.npy")[0]

    cluster_chose_ratio = np.ones(num_clusters) * selection_ratio
    selected_indices = group_select(out_dir, num_clusters, params, cluster_chose_ratio)
    print(">> Selected indices shape:", len(selected_indices))
    np.save(
        f"{out_dir}/selected_indices_{selection_ratio}.npy",
        selected_indices,
    )
