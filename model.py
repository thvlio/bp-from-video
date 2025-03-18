import enum


class ModelType(enum.StrEnum):
    FACE_DETECTOR = enum.auto()
    FACE_LANDMARKER = enum.auto()
    HAND_LANDMARKER = enum.auto()
    PERSON_SEGMENTER = enum.auto()
