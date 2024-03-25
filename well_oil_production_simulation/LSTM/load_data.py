import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np


def load_and_process_data(data_folder="./origen_data/"):
    Inpu_nms = {
        "DEPTH",
        "TRANY",
        "PORV",
        "TRANX",
        "PERMZ",
        "TRANZ",
        "PORO",
        "SATNUM",
        "PERMY",
        "PERMX",
    }
    data = {}

    def np_load(input_nm, file_nm, rp):
        try:
            d = np.load(rp, allow_pickle=True)
            return d  # Always load data
        except ValueError as e:
            print(input_nm, file_nm, e)
            raise e

    for root, dirs, files in os.walk(data_folder, topdown=False):
        root_parts = root.split("/")
        assert len(root_parts) == 3, "too many parts in %s" % root
        input_name = root_parts[-1]

        # Process only if the input_name is in the specified set
        if input_name in Inpu_nms:
            for file in filter(".DS_Store".__ne__, files):
                file_parts = file.split(".")
                assert len(file_parts) == 2, "unexpected filename pattern: %s" % file
                dataset_name = file_parts[0]
                realpath = os.path.realpath(os.path.join(root, file))
                # Load data and store it in the dictionary
                data[input_name, dataset_name] = np_load(input_name, file, realpath)

    target_data = {}
    target_names = {"BHP", "WOPR"}

    for root, dirs, files in os.walk(data_folder, topdown=False):
        root_parts = root.split("/")
        assert len(root_parts) == 3, "too many parts in %s" % root
        input_name = root_parts[-1]

        # Print "processing input" only for specified input_nm
        if input_name in target_names:
            for file in filter(".DS_Store".__ne__, files):
                file_parts = file.split(".")
                assert len(file_parts) == 2, "unexpected filename pattern: %s" % file
                dataset_name = file_parts[0]
                realpath = os.path.realpath(os.path.join(root, file))
                # Print "loading" only for specified input_nm
                # print("loading %s file %s" % (input_name, dataset_name))
                target_data[input_name, dataset_name] = np_load(
                    input_name, file, realpath
                )

    def read_df(input_nm: str):
        try:
            return pd.DataFrame(
                {
                    "%s_%s" % (input_nm.lower(), k.split("_")[0]): data[input_nm, k]
                    for k in sorted(
                        [k[1] for k in data.keys() if k[0] == input_nm],
                        key=lambda x: int(x.split("_")[0][3:]),
                    )
                }
            )
        except ValueError as e:
            print(input_nm, e)
            return None

    dfs = {input_nm: read_df(input_nm) for input_nm in set([k[0] for k in data.keys()])}

    Inpu_nms = dfs.keys()
    df = pd.concat(
        [
            pd.concat(
                [
                    pd.DataFrame(
                        {
                            "name": [
                                k,
                            ]
                            * v.shape[0]
                        },
                        index=v.index,
                    ),
                    v.rename(columns={c: c.split("_")[-1] for c in v.columns}),
                ],
                axis=1,
            )
            for k, v in dfs.items()
            if k in Inpu_nms
        ]
    )

    df_result = pd.concat([df for df in dfs.values()], axis=0, ignore_index=True)

    dfs = pd.concat(
        [
            pd.concat(
                [
                    pd.DataFrame(
                        {
                            "name": [
                                k,
                            ]
                            * v.shape[0]
                        },
                        index=v.index,
                    ),
                    v.rename(columns={c: c.split("_")[-1] for c in v.columns}),
                ],
                axis=1,
            )
            for k, v in dfs.items()
            if k in Inpu_nms
        ]
    )

    # Assuming 'name' column is categorical
    label_encoder = LabelEncoder()
    dfs["name_encoded"] = label_encoder.fit_transform(dfs["name"]) + 2

    target_data = {}
    target_names = {"BHP", "WOPR"}

    for root, dirs, files in os.walk(data_folder, topdown=False):
        root_parts = root.split("/")
        assert len(root_parts) == 3, "too many parts in %s" % root
        input_name = root_parts[-1]

        # Print "processing input" only for specified input_nm
        if input_name in target_names:
            for file in filter(".DS_Store".__ne__, files):
                file_parts = file.split(".")
                assert len(file_parts) == 2, "unexpected filename pattern: %s" % file
                dataset_name = file_parts[0]
                realpath = os.path.realpath(os.path.join(root, file))
                # Print "loading" only for specified input_nm
                # print("loading %s file %s" % (input_name, dataset_name))
                target_data[input_name, dataset_name] = np_load(
                    input_name, file, realpath
                )

    df_BHP = pd.DataFrame(
        {
            "%s_%s" % ("BHP".lower(), k.split("_")[0]): target_data["BHP", k].flatten()
            for k in sorted(
                [k[1] for k in target_data.keys() if k[0] == "BHP"],
                key=lambda x: int(x.split("_")[0][3:]),
            )
        }
    )

    df_WOPR = pd.DataFrame(
        {
            "%s_%s"
            % ("WOPR".lower(), k.split("_")[0]): target_data["WOPR", k].flatten()
            for k in sorted(
                [k[1] for k in target_data.keys() if k[0] == "WOPR"],
                key=lambda x: int(x.split("_")[0][3:]),
            )
        }
    )

    target_df = pd.concat(
        [
            pd.concat(
                [
                    pd.DataFrame(
                        {
                            "name": [
                                k,
                            ]
                            * v.shape[0]
                        },
                        index=v.index,
                    ),
                    v.rename(columns={c: c.split("_")[-1] for c in v.columns}),
                ],
                axis=1,
            )
            for k, v in {"WOPR": df_WOPR}.items()
        ]
    )

    label_encoder = LabelEncoder()
    target_df["name_encoded"] = label_encoder.fit_transform(target_df["name"])

    dfs_encoded = dfs.drop(columns=["name"])
    target_dfs_encoded = target_df.drop(columns=["name"])

    df_t = dfs_encoded.transpose()
    df_t = pd.concat([df_t, df_BHP], axis=1)

    target_df_t = target_dfs_encoded.transpose()

    df_t = dfs_encoded.transpose().iloc[:, :2000]  # Reduce size to 1000
    target_df_t = target_dfs_encoded.transpose().iloc[:, :600]  # Reduce size to 500
    # print(df_t.shape)
    # print(target_df_t.shape)

    return df_t, target_df_t
