from data.glue_dataset import GlueDataset
from data.decanlp_dataset import DecaNlpDataset


DATASET_MAP = {
    'glue': GlueDataset,
    'decanlp': DecaNlpDataset
}

