import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import numpy as np

from .txd_parser import TXDParser, TXDData
from .graph import GraphBuilder
from typing import Dict, Any

class OCRDataset(Dataset):
    def __init__(self,
                 txd_dir: str,
                 image_dir: str,
                 labels_file: str,
                 transform=None,
                 max_strokes: int = 100,
                 max_points_per_stroke: int = 50) -> None:
        self.txd_dir = txd_dir
        self.image_dir = image_dir
        self.label_file = labels_file
        self.transform = transform

        self.max_strokes = max_strokes
        self.max_points = max_points_per_stroke

        self.parser = TXDParser()
        self.graph_builder = GraphBuilder()

        self.labels = self._load_labels(labels_file)
        self.file_list = self._get_file_list()

    def _load_labels(self, labels_file: str) -> Dict[str, str]:
        labels = {}
        with open(labels_file, 'r', encoding='windows-1251') as f:
            for line in f:
                if ' ' in line:
                    filename, text = line.strip().split(' ', 1)
                    labels[filename] = text
        return labels

    def _get_file_list(self):
        txd_files = [f for f in os.listdir(self.txd_dir) if f.endswith('.txd')]
        valid_files = []

        for txd_file in txd_files:
            base_name = txd_file.replace('.txd', '')
            image_file = base_name + '.bmp'
            if os.path.exists(os.path.join(self.image_dir, image_file)) and base_name in self.labels:
                valid_files.append(base_name)
        return valid_files

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base_name = self.file_list[idx]
        txd_path = os.path.join(self.txd_dir, base_name + '.txd')
        try:
            txd_data = self.parser.parse_file(txd_path)
        except Exception as e:
            print(f"Error parsing {txd_path}: {e}")
            txd_data = TXDData(filename=base_name, strokes=[], axis_y=0, total_strokes=0)

        graph = self.graph_builder.build_graph(txd_data)
        image_path = os.path.join(self.image_dir, base_name + '.bmp')
        image = Image.open(image_path).convert('L')
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0) / 255.0
        text = self.labels[base_name]
        return {
            'image': image,
            'graph': graph,
            'text': text,
            'filename': base_name
        }