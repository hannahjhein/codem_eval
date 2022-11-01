from dataclasses import dataclass

@dataclass
class Params:
    lo_radius: float
    hi_radius: float
    lo_angle: float
    hi_angle: float
    lo_trans: int
    hi_trans: int
    reps: int

@dataclass
class Files:
    csv: str
    input_comp_data: str
    input_found_data: str
    output_prefix: str
    output_suffix: str

@dataclass
class Paths:
    input_data_path: str

@dataclass
class CodemHydraConfig:
    params: Params
    files: Files
    paths: Paths