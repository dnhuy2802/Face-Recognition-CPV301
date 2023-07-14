import os

HAAR_CASCADE_PATH = os.path.join(
    os.path.dirname(__file__), 'assets', 'haarcascade.xml')


class DetectionBox:
    def __init__(self):
        self.box_default_points_offset = 40
        self.cordinates = tuple()

    def get_box_default_points(self, shape: tuple[int, int]) -> tuple[int, int, int, int]:
        self.box_default_points_offset = 20
        height_pt, width_pt = shape
        center_pt = (height_pt // 2, width_pt // 2)
        split_pt = min(height_pt, width_pt) // 2
        self.cordinates = (
            center_pt[1] - (split_pt - self.box_default_points_offset),
            center_pt[0] - (split_pt - self.box_default_points_offset),
            split_pt * 2 - self.box_default_points_offset * 2,
            split_pt * 2 - self.box_default_points_offset * 2
        )

    def set_cordinates(self, cordinates: tuple[int, int, int, int]):
        self.cordinates = cordinates
