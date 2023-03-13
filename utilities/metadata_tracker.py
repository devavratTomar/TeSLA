
import torch
import pandas as pd
import os

from .distributed import concat_all_gather


class MetadataTracker:
    def __init__(self):
        self.metadata = {}
        self.metadata_agg = {}
        self.is_aggregated = False

    def update_metadata(self, metadata_dict):
        for k, v in metadata_dict.items():
            if k in self.metadata:
                self.metadata[k].append(v)
            else:
                self.metadata[k] = [v]

    def aggregate(self):
        if not self.is_aggregated:
            agg_dict = {}
            for k, v in self.metadata.items():
                agg_dict[k] = torch.cat(v)
                agg_dict[k] = concat_all_gather(agg_dict[k].cuda()).cpu()

            self.metadata_agg = agg_dict
        self.is_aggregated = True

    def to_pkl(self, keys, path):
        if self.is_aggregated:
            dict_to_pkl = {k : v for k,v in self.metadata_agg.items() if k in keys}
            for k, v in dict_to_pkl.items():
                save_file = os.path.join(path,"{:s}.pth".format(k))
                torch.save(v, save_file)

    def to_csv(self, keys, path):
        if self.is_aggregated:
            dict_to_csv = {k : v.numpy() for k,v in self.metadata_agg.items() if k in keys}
            df = pd.DataFrame.from_dict(dict_to_csv)
            df.to_csv(os.path.join(path, "metadata.csv"))

    def reset(self):
        self.metadata={}
        self.metadata_agg={}
        self.is_aggregated = False

    def __getitem__(self, key):
        if self.is_aggregated:
            return self.metadata_agg[key]
        return torch.cat(self.metadata[key])
        