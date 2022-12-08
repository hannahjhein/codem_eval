from dataclasses import dataclass

@dataclass
class Options:
    reps: int
    resample_aoi: bool
    crop_aoi: bool
    transform_aoi: bool
    register_aoi: bool
    remove_files: bool

@dataclass
class Params:
    lo_radius: float
    hi_radius: float
    lo_angle_x: float
    hi_angle_x: float
    lo_angle_y: float
    hi_angle_y: float
    lo_angle_z: float
    hi_angle_z: float
    lo_trans_x: int
    hi_trans_x: int
    lo_trans_y: int
    hi_trans_y: int
    lo_trans_z: int
    hi_trans_z: int
    min_len: int

@dataclass
class Files:
    csv: str
    input_comp_data: str
    input_found_data: str
    output_prefix: str

@dataclass
class Paths:
    input_data_path: str

@dataclass
class CodemEvalConfig:
    options: Options
    params: Params
    files: Files
    paths: Paths