import os
import numpy as np


def load_data():
    directory = os.path.realpath("../../origen_interview_data/")
    input_names = {
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
    target_data = {}
    BHP_data = {}
    target_names = {"WOPR"}  # Specify your target names here
    bhp = {"BHP"}

    def np_load(input_nm, file_nm, rp):
        try:
            d = np.load(rp, allow_pickle=True)
            if input_nm in input_names:
                d = d.reshape(15, 25, 24)
            return d
        except ValueError as e:
            print(input_nm, file_nm, e)
            raise e

    for root, dirs, files in os.walk(directory, topdown=False):
        root_parts = root.split("/")[-3:]
        assert len(root_parts) == 3, "too many parts in %s" % root
        input_name = root_parts[-1]

        if input_name in input_names:
            for file in filter(".DS_Store".__ne__, files):
                file_parts = file.split(".")
                assert len(file_parts) == 2, "unexpected filename pattern: %s" % file
                dataset_name = file_parts[0]
                realpath = os.path.realpath(os.path.join(root, file))
                data[input_name, dataset_name] = np_load(input_name, file, realpath)

        if input_name in target_names:
            for file in filter(".DS_Store".__ne__, files):
                file_parts = file.split(".")
                assert len(file_parts) == 2, "unexpected filename pattern: %s" % file
                dataset_name = file_parts[0]
                realpath = os.path.realpath(os.path.join(root, file))
                target_data[input_name, dataset_name] = np_load(
                    input_name, file, realpath
                )

        if input_name in bhp:
            for file in filter(".DS_Store".__ne__, files):
                file_parts = file.split(".")
                assert len(file_parts) == 2, "unexpected filename pattern: %s" % file
                dataset_name = file_parts[0]
                realpath = os.path.realpath(os.path.join(root, file))
                BHP_data[input_name, dataset_name] = np_load(input_name, file, realpath)

    assert len(data) > 0, "failed to load data"
    assert len(target_data) > 0, "failed to load data and target data"

    return data, target_data, BHP_data
