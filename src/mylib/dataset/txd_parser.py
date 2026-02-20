import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


@dataclass
class StrokePoint:
    x: float
    y: float
    param: float


@dataclass
class Stroke:
    id: int
    type: str
    num_points: int
    position: str
    points: List[StrokePoint]

    @property
    def coordinates(self) -> List[Tuple[float, float]]:
        return [(p.x, p.y) for p in self.points]


@dataclass
class TXDData:
    filename: str
    axis_y: float
    total_strokes: int
    strokes: List[Stroke]

    def get_stroke_by_id(self, stroke_id: int) -> Optional[Stroke]:
        for stroke in self.strokes:
            if stroke.id == stroke_id:
                return stroke
        return None

    def get_strokes_by_type(self, stroke_type: str) -> List[Stroke]:
        return [s for s in self.strokes if s.type == stroke_type]

    def get_stroke_by_position(self, position: str) -> List[Stroke]:
        return [s for s in self.strokes if s.position == position]


class TXDParser:
    def __init__(self):
        self.patterns = {
            'header': re.compile(r'^(.+\.bmp)\s+(\d+)\s+штрихов$'),
            'axis': re.compile(r'.*[Оо]сь строки\s*Y=(\d+)', re.IGNORECASE),
            'stroke_header': re.compile(
                r'^\s*[Шш]трих\s+(\d+)\s+(\w+)\s+точек\s+(\d+)\s+строка\s+\d+\s+уровень\s+базовый\s+(.+)$'
            ),
            'point': re.compile(r'^\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$')
        }

    def parse_file(self, file_path: str) -> TXDData:
        with open(file_path, 'r', encoding='windows-1251') as f:
            lines = f.readlines()
        return self._parse_lines(lines)

    def parse_content(self, content: str) -> TXDData:
        lines = content.split('\n')
        return self._parse_lines(lines)

    def _parse_lines(self, lines: List[str]) -> TXDData:
        filename, total_strokes = self._parse_header(lines[0])
        axis_y = self._parse_axis(lines[1])

        strokes = []
        i = 2
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            stroke_match = self.patterns['stroke_header'].match(line)
            if stroke_match:
                stroke_id = int(stroke_match.group(1))
                stroke_type = stroke_match.group(2)
                num_points = int(stroke_match.group(3))
                position = stroke_match.group(4)

                points = []
                for j in range(1, num_points + 1):
                    if i + j < len(lines):
                        point_line = lines[i + j].strip()
                        point_match = self.patterns['point'].match(point_line)
                        if point_match:
                            x = float(point_match.group(1))
                            y = float(point_match.group(2))
                            param = float(point_match.group(3))
                            points.append(StrokePoint(x, y, param))
                stroke = Stroke(
                    id=stroke_id,
                    type=stroke_type,
                    num_points=num_points,
                    position=position,
                    points=points
                )
                strokes.append(stroke)
                i += num_points + 1
            else:
                print('Отвергли строку:', stroke)
                i += 1
        return TXDData(
            filename=filename,
            axis_y=axis_y,
            total_strokes=total_strokes,
            strokes=strokes
        )

    def _parse_header(self, line: str) -> Tuple[str, int]:
        match = self.patterns['header'].match(line.strip())
        if match:
            return match.group(1), int(match.group(2))
        raise ValueError(f'Неверный формат заголовка: {line.strip()}')

    def _parse_axis(self, line: str) -> float:
        match = self.patterns['axis'].match(line.strip())
        if match:
            return float(match.group(1))
        raise ValueError(f'Неверный формат оси: {line.strip()}')