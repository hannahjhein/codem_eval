# main.py #
# run: python main.py hydra.job.chdir=True

#######################
# Prepare environment #
#######################
import codem
import csv
import dataclasses
import datetime
import hydra
import numpy as np
import numpy.lib.recfunctions as rfn
import os
import pdal

from collections import namedtuple
from config import CodemEvalConfig
from hydra.core.config_store import ConfigStore
from scipy.spatial.transform import Rotation as R

# set printing options for float values
np.set_printoptions(precision=6, suppress=True)

# define fontana score metric, as per Fontana et al. (2022)
def fontana_score(Bin, Ain):
    A = rfn.structured_to_unstructured(Ain[['X','Y','Z']])
    B = rfn.structured_to_unstructured(Bin[['X','Y','Z']])
    centroid = np.mean(A, axis=0)
    weights = np.linalg.norm(A - centroid, 2, axis=1)
    distances = np.linalg.norm(A - B, 2, axis=1)/len(weights)
    return np.sum(distances/weights), np.sqrt(np.mean(np.linalg.norm(A - B, 2, axis=1)**2))
print("Environment prepared")

######################
# Link Configuration #
######################
cs = ConfigStore.instance()
cs.store(name="codem_eval_config", node=CodemEvalConfig)

# establish configuration and confirm working directory is new hydra date/time directory
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: CodemEvalConfig):
    time = datetime.datetime.now()
    datetime_tag = str(f"{datetime.date.today()}_{time.hour}-{time.minute}-{time.second}")
    working_dir = os.getcwd()
    print(f"Working directory is {working_dir}")
    
    #############################
    # Write new .csv for output #
    #############################
    perm_csv = (cfg.files.csv + datetime_tag + ".csv")
    with open (perm_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ['resample', 'crop', 'transform', 'register', 
        'codem_success', 'error_type', 'crop_origin_x', 'crop_origin_y', 'aoi_x_side', 'aoi_y_side', 'aoi_area', 'found_area', 'overlap_area', 'overlap_per',
        'radius', 'yaw', 'pitch', 'roll', 
        'translation_x', 'translation_y', 'translation_z', 
        'input_centroid_x', 'input_centroid_y', 'input_centroid_z', 
        'prereg_sre', 'prereg_rmse', 'postreg_sre', 'postreg_rmse', 'runtime', 
        'registered_centroid_x', 'registered_centroid_y', 'registered_centroid_z', 'registered_centroid_dx', 'registered_centroid_dy', 'registered_centroid_dz',
        'dsm_omega', 'dsm_phi', 'dsm_kappa', 'dsm_trans_x', 'dsm_trans_y', 'dsm_trans_z', 'dsm_scale', 'dsm_n_pairs', 'dsm_rmse_x', 'dsm_rsme_y', 'dsm_rmse_z', 'dsm_rmse_3d', 
        'icp_omega', 'icp_phi', 'icp_kappa', 'icp_trans_x', 'icp_trans_y', 'icp_trans_z', 'icp_scale', 'icp_n_pairs', 'icp_rmse_x', 'icp_rmse_y', 'icp_rmse_z', 'icp_rmse_3d', 
        'min_threshold', 'dsm_akaze_threshold', 'dsm_lowes_ratio', 'dsm_ransac_max_iter', 'dsm_ransac_threshold', 'dsm_solve_scale', 'dsm_strong_filter', 'dsm_weak_filter',
        'icp_angle_threshold', 'icp_distance_threshold', 'icp_max_iter', 'icp_rmse_threshold', 'icp_robust', 'icp_solve_scale']
        writer.writerow(header)
        print(f"{perm_csv} created")

        # main function
        # set reps in config.yaml
        for i in range(cfg.options.reps):


            ####################
            # Set up variables #
            ####################

            print(f" \n########################### \nProcessing iteration {i} \n########################### \n")

            ### DEFINE TRANSFORMATION VALUES ###
            if cfg.options.transform_aoi == True:
                # randomly select rotation angles in x, y, and z based on range of values in config file
                roll = float(np.random.uniform(cfg.params.lo_angle_x, cfg.params.hi_angle_x, 1))
                pitch = float(np.random.uniform(cfg.params.lo_angle_y, cfg.params.hi_angle_y, 1))
                yaw = float(np.random.uniform(cfg.params.lo_angle_z, cfg.params.hi_angle_z, 1))
                angles = (yaw, pitch, roll)
                # randomly select translations in x, y, and z based on range of values in config file (in z, y, x order)
                trans_x = float(np.random.uniform(cfg.params.lo_trans_x, cfg.params.hi_trans_x, 1))
                trans_y = float(np.random.uniform(cfg.params.lo_trans_y, cfg.params.hi_trans_y, 1))
                trans_z = float(np.random.uniform(cfg.params.lo_trans_z, cfg.params.hi_trans_z, 1))
                translations = (trans_x, trans_y, trans_z)
                print(f"Transformation generated: \nRotation (Z,Y,X) = {angles}, \nTranslation (X,Y,Z) = {translations}")
            else:
                rotation = [0, 0, 0]
                translations = [0, 0, 0]
                print("No transformation parameters generated")

            ### DEFINE DATA VARIABLES ###
            # input data to be transformed, must be point cloud
            input_comp_data = (os.path.normpath(cfg.paths.input_data_path + "\\" + cfg.files.input_comp_data))
            root, ext = os.path.splitext(input_comp_data)

            if cfg.options.resample_aoi == True:
                # randomly select sampling radius based on range of values in config file
                radius = np.random.uniform(cfg.params.lo_radius, cfg.params.hi_radius)
                # define name for output file after resampling with random sampling radius
                output_truth = (cfg.files.output_prefix + "{:.2f}m_truth".format(radius) + ext)
                # define name for output file after resampling and cropping
                output_crop = (cfg.files.output_prefix + "{:.2f}m_crop".format(radius) + ext)
                # define name for output file after resampling, cropping, and transformation
                output_perturb = (cfg.files.output_prefix + "{:.2f}m_perturb".format(radius) + ext)
                # define registered output using the sample radius in the output name
                output_reg = (cfg.files.output_prefix + "{:.2f}m_registered".format(radius) + ext)
                print(f"Resampling radius generated: {radius}m")
            else:
                radius = "Null"
                # define name for output file
                output_truth = (cfg.files.output_prefix + "truth" + ext)
                # define name for output file with cropping
                output_crop = (cfg.files.output_prefix + "crop" + ext)
                # define name for output file after cropping and transformation
                output_perturb = (cfg.files.output_prefix + "perturb" + ext)
                # define registered output
                output_reg = (cfg.files.output_prefix + "registered" + ext)
            


            ################
            # Resample AOI #
            ################

            if cfg.options.resample_aoi == True:
                # read original input file
                truth = pdal.Reader(input_comp_data).pipeline()
                # sample aoi to randomly selected radius
                truth |= pdal.Filter.sample(radius = radius)
                # write the sampled file to the current working directory using output_truth file name
                truth |= pdal.Writer.las(output_truth, forward = "all")
                truth.execute()
                print(f"Resampling complete, truth file = {output_truth}")
            else:
                truth = pdal.Reader(input_comp_data).pipeline()
                truth |= pdal.Writer.las(output_truth, forward = "all")
                truth.execute()
                print(f"No resampling executed, continuing to cropping")



            ############
            # Crop AOI #
            ############

            if cfg.options.crop_aoi == True:
                # store max and min xy values and ranges of input complement file
                minx = truth.quickinfo['readers.las']['bounds']['minx']
                maxx = truth.quickinfo['readers.las']['bounds']['maxx']
                miny = truth.quickinfo['readers.las']['bounds']['miny']
                maxy = truth.quickinfo['readers.las']['bounds']['maxy']
                xrange = maxx-minx
                yrange = maxy-miny

                # define side length of new bbox
                if xrange > yrange:
                    bbox_side = np.random.randint(cfg.params.min_len, int(yrange), 1)
                    print(f"Y value used, side length = {bbox_side}")
                else:
                    bbox_side = np.random.randint(cfg.params.min_len, int(xrange), 1)
                    print(f"X value used, side length = {bbox_side}")
                aoi_x_side = int(bbox_side)
                aoi_y_side = int(bbox_side)

                # select random coordinates for new bbox origin (note buffer of length bbox_side applied)
                origin_x = int(np.random.randint(minx, (maxx-bbox_side), 1))
                origin_y = int(np.random.randint(miny, (maxy-bbox_side), 1))
                print(f"Crop bounding box origin = {origin_x, origin_y}")

                # build new bbox
                bbox_minx = float(origin_x)
                bbox_maxx = float(origin_x + bbox_side)
                bbox_miny = float(origin_y)
                bbox_maxy = float(origin_y + bbox_side)

                bbox = ([bbox_minx, bbox_maxx], [bbox_miny, bbox_maxy])
                print(f"Crop bounding box min/max x and y: {str(bbox)}")

                # apply bbox to crop
                truth = pdal.Reader(output_truth).pipeline()
                truth |= pdal.Filter.crop(bounds = str(bbox))
                truth |= pdal.Writer.las(output_crop, forward = "all")
                truth.execute()
                print(f"Cropping complete, cropped file = {output_crop}")
            else:
                origin_x = "Null"
                origin_y = "Null"
                print("No cropping executed, continuing to transformation")



            ########################
            # Apply transformation #
            ########################

            if cfg.options.transform_aoi == True:
                # define centroid
                input_centroid = np.mean(rfn.structured_to_unstructured(truth.arrays[0][['X','Y','Z']]), axis=0)

                # generate transformation matrix
                A = np.eye(4)
                A[0:3, 3] = -input_centroid

                rot = R.from_euler('zyx', angles)
                B = np.eye(4)
                B[0:3, 0:3] = rot.as_matrix()

                C = np.eye(4)
                C[0:3, 3] = translations

                D = np.eye(4)
                D[0:3, 3] = input_centroid

                T = D @ C @ B @ A

                # define angles used
                rotation = [(rot.as_euler('zyx', degrees=True)[0]), (rot.as_euler('zyx', degrees=True)[1]), (rot.as_euler('zyx', degrees=True)[2])]

                # read resampled truth file
                if cfg.options.crop_aoi == True:
                    perturb = pdal.Reader(output_crop).pipeline()
                    # apply the transformation matrix
                    perturb |= pdal.Filter.transformation(matrix=np.array_str(T.flatten(), max_line_width=999)[1:-1])
                    # write the transformed file to the current working directory using the output_perturb file name
                    perturb |= pdal.Writer.las(output_perturb, forward="all")
                    perturb.execute()
                    print(f"Transformation complete, perturbed file = {output_perturb}")
                else:
                    perturb = pdal.Reader(output_truth).pipeline()
                    # apply the transformation matrix
                    perturb |= pdal.Filter.transformation(matrix=np.array_str(T.flatten(), max_line_width=999)[1:-1])
                    # write the transformed file to the current working directory using the output_perturb file name
                    perturb |= pdal.Writer.las(output_perturb, forward="all")
                    perturb.execute()
                    print(f"Transformation complete, perturbed file = {output_perturb}")
            
            else:
                input_centroid = np.mean(rfn.structured_to_unstructured(truth.arrays[0][['X','Y','Z']]), axis=0)

                print("No transformation executed, continuing to registration")


            
            ###########################
            # Registration with CODEM #
            ###########################

            if cfg.options.register_aoi == True:
                ### DEFINE VARIABLES AND CALCULATE AREA ###
                # data set that needs to be registered
                if cfg.options.transform_aoi == True:
                    comp = (output_perturb)
                else:
                    if cfg.options.crop_aoi == True:
                        comp = (output_crop)
                    else:
                        comp = (output_truth)
                # foundational data set, bring into workspace using data path
                found = (os.path.normpath(cfg.paths.input_data_path + "\\" + cfg.files.input_found_data))

                # calculate foundation area (use gdal if tif, use pdal if pc)
                if found.endswith('.tif'):
                    found_pipeline = pdal.Reader(found).pipeline()
                    found_minx = found_pipeline.quickinfo['readers.gdal']['bounds']['minx']
                    found_maxx = found_pipeline.quickinfo['readers.gdal']['bounds']['maxx']
                    found_miny = found_pipeline.quickinfo['readers.gdal']['bounds']['maxy']
                    found_maxy = found_pipeline.quickinfo['readers.gdal']['bounds']['miny']
                else:
                    found_pipeline = pdal.Reader(found).pipeline()
                    found_minx = found_pipeline.quickinfo['readers.las']['bounds']['minx']
                    found_maxx = found_pipeline.quickinfo['readers.las']['bounds']['maxx']
                    found_miny = found_pipeline.quickinfo['readers.las']['bounds']['maxy']
                    found_maxy = found_pipeline.quickinfo['readers.las']['bounds']['miny']
                found_x_side = np.abs(found_maxx - found_minx)
                found_y_side = np.abs(found_maxy - found_miny)
                found_area = (found_x_side*found_y_side)

                if cfg.options.crop_aoi == True:
                    # calculate crop area using side length of crop bbox
                    aoi_area = float(np.square(bbox_side))
                    # calculate area of overlap between cropped complement and foundation
                    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
                    rfound = Rectangle(found_minx, found_miny, found_maxx, found_maxy)
                    rcrop = Rectangle(bbox_minx, bbox_miny, bbox_maxx, bbox_maxy)

                    def area(a, b):  # returns None if rectangles don't intersect
                        dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
                        dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
                        if (dx>=0) and (dy>=0):
                            return dx*dy

                    overlap_area = area(rfound, rcrop)
                    if isinstance(overlap_area, float):
                        overlap_per = (overlap_area/found_area)*100 
                    else:
                        overlap_area = "None"
                        overlap_per = "None"
                    
                else:
                    # calculate aoi area using input file bounds
                    aoi_pipeline = pdal.Reader(comp).pipeline()
                    aoi_minx = aoi_pipeline.quickinfo['readers.las']['bounds']['minx']
                    aoi_maxx = aoi_pipeline.quickinfo['readers.las']['bounds']['maxx']
                    aoi_miny = aoi_pipeline.quickinfo['readers.las']['bounds']['maxy']
                    aoi_maxy = aoi_pipeline.quickinfo['readers.las']['bounds']['miny']
                    aoi_x_side = np.abs(aoi_maxx - aoi_minx)
                    aoi_y_side = np.abs(aoi_maxy - aoi_miny)
                    aoi_area = (aoi_x_side*aoi_y_side)

                    # calculate area of overlap between cropped complement and foundation
                    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
                    rfound = Rectangle(found_minx, found_miny, found_maxx, found_maxy)
                    raoi = Rectangle(aoi_minx, aoi_miny, aoi_maxx, aoi_maxy)

                    def area(a, b):  # returns None if rectangles don't intersect
                        dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
                        dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
                        if (dx>=0) and (dy>=0):
                            return dx*dy

                    overlap_area = area(rfound, raoi)
                    if isinstance(overlap_area, float):
                        overlap_per = (overlap_area/found_area)*100 
                    else:
                        overlap_area = "None"
                        overlap_per = "None"
                    
                ### START TIME ###
                st = datetime.datetime.now()
                print(" \nCodem pre-processing start")

                ### CODEM PRE-PROCESSING ###
                # define the configuration as (foundation file, complement file)
                config = dataclasses.asdict(codem.CodemRunConfig(found,comp))
                # run preprocess tool on config object
                fnd_obj, aoi_obj = codem.preprocess(config)
                # prep data for registration
                fnd_obj.prep()
                aoi_obj.prep()
                print("Codem pre-processing complete \n ")

                ### CODEM COARSE REGISTRATION ###
                try:
                    print("Codem coarse registration start")
                    dsm_reg = codem.coarse_registration(fnd_obj, aoi_obj, config)
                    print("Codem coarse registration complete \n ")
                    codem_success = "Y"
                    error_type = "None"
                except RuntimeError:
                    codem_success = "N"
                    error_type = "RuntimeError"
                    data = [cfg.options.resample_aoi, cfg.options.crop_aoi, cfg.options.transform_aoi, cfg.options.register_aoi, codem_success, error_type, origin_x, origin_y, aoi_x_side, aoi_y_side, aoi_area, found_area, overlap_area, overlap_per, radius, rotation[0], rotation[1], rotation[2], translations[0], translations[1], translations[2], input_centroid[0], input_centroid[1], input_centroid[2]]
                    writer.writerow(data)
                    if cfg.options.remove_files == True:
                        os.remove(output_truth)
                    else:
                        pass
                    if cfg.options.remove_files == True and cfg.options.crop_aoi == True:
                        os.remove(output_crop)
                    else:
                        pass
                    if cfg.options.remove_files == True and cfg.options.transform_aoi == True:
                        os.remove(output_perturb)
                    else:
                        pass
                    print(f"Codem coarse registration failed ({error_type}) \nResults for failed test written to {perm_csv} \nOnto next iteration \n####################################### \n ")
                    continue
                except AssertionError:
                    codem_success = "N"
                    error_type = "AssertionError"
                    data = [cfg.options.resample_aoi, cfg.options.crop_aoi, cfg.options.transform_aoi, cfg.options.register_aoi, codem_success, error_type, origin_x, origin_y, aoi_x_side, aoi_y_side, aoi_area, found_area, overlap_area, overlap_per, radius, rotation[0], rotation[1], rotation[2], translations[0], translations[1], translations[2], input_centroid[0], input_centroid[1], input_centroid[2]]
                    writer.writerow(data)
                    if cfg.options.remove_files == True:
                        os.remove(output_truth)
                    else:
                        pass
                    if cfg.options.remove_files == True and cfg.options.crop_aoi == True:
                        os.remove(output_crop)
                    else:
                        pass
                    if cfg.options.remove_files == True and cfg.options.transform_aoi == True:
                        os.remove(output_perturb)
                    else:
                        pass
                    print(f"Codem coarse registration failed ({error_type}) \nResults for failed test written to {perm_csv} \nOnto next iteration \n####################################### \n ")
                    continue
                except ValueError:
                    codem_success = "N"
                    error_type = "ValueError"
                    data = [cfg.options.resample_aoi, cfg.options.crop_aoi, cfg.options.transform_aoi, cfg.options.register_aoi, codem_success, error_type, origin_x, origin_y, aoi_x_side, aoi_y_side, aoi_area, found_area, overlap_area, overlap_per, radius, rotation[0], rotation[1], rotation[2], translations[0], translations[1], translations[2], input_centroid[0], input_centroid[1], input_centroid[2]]
                    writer.writerow(data)
                    if cfg.options.remove_files == True:
                        os.remove(output_truth)
                    else:
                        pass
                    if cfg.options.remove_files == True and cfg.options.crop_aoi == True:
                        os.remove(output_crop)
                    else:
                        pass
                    if cfg.options.remove_files == True and cfg.options.transform_aoi == True:
                        os.remove(output_perturb)
                    else:
                        pass
                    print(f"Codem coarse registration failed ({error_type}) \nResults for failed test written to {perm_csv} \nOnto next iteration \n####################################### \n ")
                    continue

                ### CODEM FINE REGISTRATION ###
                print("Codem fine registration start")
                icp_reg = codem.fine_registration(fnd_obj, aoi_obj, dsm_reg, config)
                print("Codem fine registration complete \n ")

                ### CODEM APPLY REGISTRATION ###
                print("Codem apply registration start")
                reg = codem.apply_registration(fnd_obj, aoi_obj, icp_reg, config)
                print("Codem apply registration complete \n ")

                ### END TIME ###
                et = datetime.datetime.now()
                runtime = et - st
                
                ### RECORD CENTROIDS ###
                # DSM
                """ dsm = pdal.Reader(dsm_reg).pipeline()
                dsm.execute()
                dsm_centroid = np.mean(rfn.structured_to_unstructured(dsm.arrays[0][['X','Y','Z']]), axis=0) """

                # ICP
                """ icp = pdal.Reader(icp_reg).pipeline()
                icp.execute()
                icp_centroid = np.mean(rfn.structured_to_unstructured(icp.arrays[0][['X','Y','Z']]), axis=0) """

                # Registered (also write file to hydra folder for error calc later)
                registered = pdal.Reader(reg).pipeline()
                registered |= pdal.Writer.las(output_reg)
                registered.execute()
                registered_centroid = np.mean(rfn.structured_to_unstructured(registered.arrays[0][['X','Y','Z']]), axis=0)
                print(f"Registered file = {output_reg}")
                

                #####################
                # Calculate results #
                #####################

                # calculate pre-registration fontana score and rmse
                if cfg.options.transform_aoi == True:
                    pre_fontana, pre_rmse = fontana_score(perturb.arrays[0], truth.arrays[0])
                else:
                    pre_fontana = "Null"
                    pre_rmse = "Null"

                # calculate post-registration fontana score and rmse
                post_fontana, post_rmse = fontana_score(registered.arrays[0], truth.arrays[0])

                # call dsm registration results
                dsm_omega = dsm_reg.registration_parameters.get("omega")
                dsm_phi = dsm_reg.registration_parameters.get("phi")
                dsm_kappa = dsm_reg.registration_parameters.get("kappa")
                dsm_trans_x = dsm_reg.registration_parameters.get("trans_x")
                dsm_trans_y = dsm_reg.registration_parameters.get("trans_y")
                dsm_trans_z = dsm_reg.registration_parameters.get("trans_z")
                dsm_scale = dsm_reg.registration_parameters.get("scale")
                dsm_n_pairs = dsm_reg.registration_parameters.get("n_pairs")
                dsm_rmse_x = dsm_reg.registration_parameters.get("rmse_x")
                dsm_rmse_y = dsm_reg.registration_parameters.get("rmse_y")
                dsm_rmse_z = dsm_reg.registration_parameters.get("rmse_z")
                dsm_rmse_3d = dsm_reg.registration_parameters.get("rmse_3d")

                # call icp registration results
                icp_omega = icp_reg.registration_parameters.get("omega")
                icp_phi = icp_reg.registration_parameters.get("phi")
                icp_kappa = icp_reg.registration_parameters.get("kappa")
                icp_trans_x = icp_reg.registration_parameters.get("trans_x")
                icp_trans_y = icp_reg.registration_parameters.get("trans_y")
                icp_trans_z = icp_reg.registration_parameters.get("trans_z")
                icp_scale = icp_reg.registration_parameters.get("scale")
                icp_n_pairs = icp_reg.registration_parameters.get("n_pairs")
                icp_rmse_x = icp_reg.registration_parameters.get("rmse_x")
                icp_rmse_y = icp_reg.registration_parameters.get("rmse_y")
                icp_rmse_z = icp_reg.registration_parameters.get("rmse_z")
                icp_rmse_3d = icp_reg.registration_parameters.get("rmse_3d")

                # call codem config params
                min_resolution = config.get("MIN_RESOLUTION")
                dsm_akaze_threshold = config.get("DSM_AKAZE_THRESHOLD")
                dsm_lowes_ratio = config.get("DSM_LOWES_RATIO")
                dsm_ransac_max_iter = config.get("DSM_RANSAC_MAX_ITER")
                dsm_ransac_threshold = config.get("DSM_RANSAC_THRESHOLD")
                dsm_solve_scale = config.get("DSM_SOLVE_SCALE")
                dsm_strong_filter = config.get("DSM_STRONG_FILTER")
                dsm_weak_filter = config.get("DSM_WEAK_FILTER")
                icp_angle_threshold = config.get("ICP_ANGLE_THRESHOLD")
                icp_distance_threshold = config.get("ICP_DISTANCE_THRESHOLD")
                icp_max_iter = config.get("ICP_MAX_ITER")
                icp_rmse_threshold = config.get("ICP_RMSE_THRESHOLD")
                icp_robust = config.get("ICP_ROBUST")
                icp_solve_scale = config.get("ICP_SOLVE_SCALE")

                print("Results calculated")

                #################
                # Write results #
                #################

                # write transformation and error metrics to .csv
                data = [cfg.options.resample_aoi, cfg.options.crop_aoi, cfg.options.transform_aoi, cfg.options.register_aoi, 
                    codem_success, error_type, origin_x, origin_y, aoi_x_side, aoi_y_side, aoi_area, found_area, overlap_area, overlap_per, 
                    radius, rotation[0], rotation[1], rotation[2],
                    translations[0], translations[1], translations[2], 
                    input_centroid[0], input_centroid[1], input_centroid[2],
                    pre_fontana, pre_rmse, post_fontana, post_rmse, runtime,
                    registered_centroid[0], registered_centroid[1], registered_centroid[2], (input_centroid[0] - registered_centroid[0]), (input_centroid[1] - registered_centroid[1]), (input_centroid[2] - registered_centroid[2]),
                    dsm_omega, dsm_phi, dsm_kappa, dsm_trans_x, dsm_trans_y, dsm_trans_z, dsm_scale, dsm_n_pairs, dsm_rmse_x, dsm_rmse_y, dsm_rmse_z, dsm_rmse_3d, 
                    icp_omega, icp_phi, icp_kappa, icp_trans_x, icp_trans_y, icp_trans_z, icp_scale, icp_n_pairs, icp_rmse_x, icp_rmse_y, icp_rmse_z, icp_rmse_3d,                    
                    min_resolution, dsm_akaze_threshold, dsm_lowes_ratio, dsm_ransac_max_iter, dsm_ransac_threshold, dsm_solve_scale, dsm_strong_filter, dsm_weak_filter,
                    icp_angle_threshold, icp_distance_threshold, icp_max_iter, icp_rmse_threshold, icp_robust, icp_solve_scale
                ]
                writer.writerow(data)
                print("Results written \n#######################################")
            
            else:
                # full null values for unused variables and parameters
                codem_success = "Null"
                error_type = "Null"

                aoi_pipeline = pdal.Reader(output_perturb).pipeline()
                aoi_minx = aoi_pipeline.quickinfo['readers.las']['bounds']['minx']
                aoi_maxx = aoi_pipeline.quickinfo['readers.las']['bounds']['maxx']
                aoi_miny = aoi_pipeline.quickinfo['readers.las']['bounds']['maxy']
                aoi_maxy = aoi_pipeline.quickinfo['readers.las']['bounds']['miny']
                aoi_x_side = np.abs(aoi_maxx - aoi_minx)
                aoi_y_side = np.abs(aoi_maxy - aoi_miny)
                aoi_area = (aoi_x_side*aoi_y_side)

                found_area = "Null"

                overlap_area = "Null"
                overlap_per = "Null"

                # write transformation and error metrics to .csv
                data = [cfg.options.resample_aoi, cfg.options.crop_aoi, cfg.options.transform_aoi, cfg.options.register_aoi, 
                    codem_success, error_type, origin_x, origin_y, aoi_x_side, aoi_y_side, aoi_area, found_area, overlap_area, overlap_per, 
                    radius, rotation[0], rotation[1], rotation[2], 
                    translations[0], translations[1], translations[2], 
                    input_centroid[0], input_centroid[1], input_centroid[2]]
                writer.writerow(data)
                print("Results written \n#######################################")
            
            ##########################
            # Remove generated files #
            ##########################
            if cfg.options.remove_files == True:
                os.remove(output_truth)
            else:
                pass
            if cfg.options.remove_files == True and cfg.options.crop_aoi == True:
                os.remove(output_crop)
            else:
                pass
            if cfg.options.remove_files == True and cfg.options.transform_aoi == True:
                os.remove(output_perturb)
            else:
                pass
            if cfg.options.remove_files == True and cfg.options.register_aoi == True:
                os.remove(output_reg)
            else:
                pass

if __name__ == "__main__":
    main()
