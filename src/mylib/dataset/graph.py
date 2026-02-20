import torch
from typing import Dict
from torch_geometric.data import Data
from .txd_parser import TXDData

class GraphBuilder:
    def __init__(self, distance_threshold: float=10.0):
        self.distance_threshold = distance_threshold
        self.stroke_types = {'Chain': 0, 'Ring': 1, 'StickVert': 2, 'StickHor': 3}
        self.positions = {'начало слова': 0, 'середина слова': 1, 'конец слова': 2}

    def build_graph(self, txd_data: TXDData) -> Data:
        num_strokes = min(len(txd_data.strokes), 100)
        num_type = len(self.stroke_types)
        num_pos = len(self.positions)

        feature_dim = num_type + num_pos + 4
        node_features = torch.zeros((num_strokes, feature_dim))

        edge_index = []
        centers = []

        for i in range(num_strokes):
            stroke = txd_data.strokes[i]
            points = [(point.x, point.y) for point in stroke.points]
            cx = sum(point[0] for point in points) / max(stroke.num_points, 1)
            cy = sum(point[1] for point in points) / max(stroke.num_points, 1)
            centers.append((cx, cy))

            type_id = self.stroke_types.get(stroke.type, 0)
            pos_id = self.positions.get(stroke.position, 1)
            node_features[i, type_id] = 1.0
            node_features[i, num_type + pos_id] = 1.0
            node_features[i, num_type + num_pos] = stroke.num_points
            if stroke.num_points > 1:
                length = sum(
                    torch.dist(torch.tensor(points[j]), torch.tensor(points[j + 1])) for j in
                    range(stroke.num_points - 1))
                node_features[i, num_type + num_pos + 1] = length.item()
            node_features[i, num_type + num_pos + 2] = cx
            node_features[i, num_type + num_pos + 3] = cy

        for i in range(num_strokes):
            for j in range(i + 1, num_strokes):
                dist = torch.dist(torch.tensor(centers[i]), torch.tensor(centers[j])).item()
                if dist < self.distance_threshold:
                    edge_index.append([i, j])
                    edge_index.append([j, i])
        edge_index = torch.tensor(edge_index).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)
        return Data(
            x=node_features.float(),
            edge_index=edge_index
        )