# main.py #
# run: python main.py hydra.job.chdir=True

# Prepare environment
import hydra
import csv
import numpy as np
import numpy.lib.recfunctions as rfn
import pdal
import os
import codem
import dataclasses

from scipy.spatial.transform import Rotation as R
from hydra.core.config_store import ConfigStore
from config import CodemHydraConfig

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

# link configuration node
cs = ConfigStore.instance()
cs.store(name="codem_hydra_config", node=CodemHydraConfig)

# establish configuration and confirm working directory is new hydra date/time directory
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: CodemHydraConfig):
    working_dir = os.getcwd()
    print(f"Working directory is {working_dir}")
    
    # Write new .csv with header for 
    perm_csv = (cfg.files.csv)
    with open (perm_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ['radius', 'yaw', 'pitch', 'roll', 'dx', 'dy', 'dz', 'cx', 'cy', 'cz', 
                  'prereg_sre', 'prereg_rmse', 'postreg_sre', 'postreg_rmse', 'runtime', 
                  'dsm_omega', 'dsm_phi', 'dsm_kappa', 'dsm_trans_x', 'dsm_trans_y', 'dsm_trans_z', 'dsm_scale', 'dsm_n_pairs', 'dsm_rmse_x', 'dsm_rsme_y', 'dsm_rmse_z', 'dsm_rmse_3d', 
                  'icp_omega', 'icp_phi', 'icp_kappa', 'icp_trans_x', 'icp_trans_y', 'icp_trans_z', 'icp_scale', 'icp_n_pairs', 'icp_rmse_x', 'icp_rmse_y', 'icp_rmse_z', 'icp_rmse_3d']]
        writer.writerow(header)

        # main function
        # set reps in config.yaml
        for i in range(cfg.params.reps):

            ##################
            # Transform data #
            ##################

            ### DEFINE TRANSFORMATION VALUES ###
            # randomly select sampling radius based on range of values in config file
            radius = np.random.uniform(cfg.params.lo_radius, cfg.params.hi_radius)
            # randomly select rotation angles in x, y, and z based on range of values in config file
            roll = float(np.random.uniform(cfg.params.lo_angle_x, cfg.params.hi_angle_x, 1))
            pitch = float(np.random.uniform(cfg.params.lo_angle_y, cfg.params.hi_angle_y, 1))
            yaw = float(np.random.uniform(cfg.params.lo_angle_z, cfg.params.hi_angle_z, 1))
            angles = (yaw, pitch, roll)
            # randomly select translations in x, y, and z based on range of values in config file
            trans_x = float(np.random.uniform(cfg.params.lo_trans_x, cfg.params.hi_trans_x, 1))
            trans_y = float(np.random.uniform(cfg.params.lo_trans_y, cfg.params.hi_trans_y, 1))
            trans_z = float(np.random.uniform(cfg.params.lo_trans_z, cfg.params.hi_trans_z, 1))
            translations = (trans_z, trans_y, trans_x)
            print(f"Radius = {radius}, \nRotation (X,Y,Z) = {angles}, \nTranslation (Z,Y,X) = {translations}")


            ### DEFINE DATA VARIABLES ###
            # input data to be transformed, must be point cloud
            input_comp_data = (os.path.normpath(cfg.paths.input_data_path + "\\" + cfg.files.input_comp_data))
            # define name for output file after resampling with random sampling radius
            output_truth = (cfg.files.output_prefix + "{:.2f}m_truth".format(radius) + cfg.files.output_suffix)
            # define name for transformed file after resampling with random sampling radius
            output_perturb = (cfg.files.output_prefix + "{:.2f}m_perturb".format(radius) + cfg.files.output_suffix)
            

            ### RESAMPLE AOI ###
            # read original input file
            p = pdal.Reader(input_comp_data).pipeline()
            # sample aoi to randomly selected radius
            p |= pdal.Filter.sample(radius=radius)
            # write the sampled file to the current working directory using output_truth file name
            p |= pdal.Writer.las(output_truth, forward="all")
            p.execute()
            print(f"Truth file = {output_truth}")


            ### APPLY TRANSFORMATION ###
            # define centroid
            centroid = np.mean(rfn.structured_to_unstructured(p.arrays[0][['X','Y','Z']]), axis=0)

            # generate transformation matrix
            A = np.eye(4)
            A[0:3, 3] = -centroid

            rot = R.from_euler('zyx', angles)
            B = np.eye(4)
            B[0:3, 0:3] = rot.as_matrix()

            C = np.eye(4)
            C[0:3, 3] = translations

            D = np.eye(4)
            D[0:3, 3] = centroid

            T = D @ C @ B @ A

            # read resampled truth file
            q = pdal.Reader(output_truth).pipeline()
            # apply the transformation matrix
            q |= pdal.Filter.transformation(matrix=np.array_str(T.flatten(), max_line_width=999)[1:-1])
            # write the transformed file to the current working directory using the output_perturb file name
            q |= pdal.Writer.las(output_perturb, forward="all")
            q.execute()
            print(f"Perturbed file = {output_perturb}")
            

            ###########################
            # Registration with CODEM #
            ###########################

            ### DEFINE VARIABLES ###
            # data set that needs to be registered
            comp = (output_perturb)
            # foundational data set, bring into workspace using data path
            found = (os.path.normpath(cfg.paths.input_data_path + "\\" + cfg.files.input_found_data))
            # define registered output using the sample radius in the output name
            output_reg = (cfg.files.output_prefix + "{:.2f}m_registered".format(radius) + cfg.files.output_suffix)

            ### START TIME ###
            st = datetime.datetime.now()
            
            ### CODEM PRE-PROCESSING ###
            # define the configuration as (foundation file, complement file)
            config = dataclasses.asdict(codem.CodemRunConfig(found,comp))
            # run preprocess tool on config object
            fnd_obj, aoi_obj = codem.preprocess(config)
            # prep data for registration
            fnd_obj.prep()
            aoi_obj.prep()

            ### CODEM COARSE REGISTRATION ###
            dsm_reg = codem.coarse_registration(fnd_obj, aoi_obj, config)

            ### CODEM FINE REGISTRATION ###
            icp_reg = codem.fine_registration(fnd_obj, aoi_obj, dsm_reg, config)

            ### CODEM APPLY REGISTRATION ###
            reg = codem.apply_registration(fnd_obj, aoi_obj, icp_reg, config)
            print(f"Applied registration = {reg}")
            
            ### END TIME ###
            et = datetime.datetime.now()
            runtime = et - st
            
            # write registered file to hydra folder
            r = pdal.Reader(reg)
            r |= pdal.Writer.las(output_reg)
            r.execute()
            print(f"Registered file = {output_reg}")
            

           #####################
           # Calculate results #
           #####################

            # calculate pre-registration fontana score and rmse
            pre_fontana, pre_rmse = fontana_score(q.arrays[0], p.arrays[0])

            # calculate post-registration fontana score and rmse
            post_fontana, post_rmse = fontana_score(r.arrays[0], p.arrays[0])


            #################
            # Write results #
            #################

            # print transformation and error metrics
            print("{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.4f} {:.2f} {:.4f} {:.2f}".format(
                radius,
                rot.as_euler('zyx', degrees=True)[0], rot.as_euler('zyx', degrees=True)[1], rot.as_euler('zyx', degrees=True)[2],
                translations[0], translations[1], translations[2],
                centroid[0], centroid[1], centroid[2],
                pre_fontana, pre_rmse,
                post_fontana, post_rmse))
            
            # write transformation and error metrics to .csv
            data = [radius,
                rot.as_euler('zyx', degrees=True)[0], rot.as_euler('zyx', degrees=True)[1], rot.as_euler('zyx', degrees=True)[2],
                translations[0], translations[1], translations[2],
                centroid[0], centroid[1], centroid[2],
                pre_fontana, pre_rmse,
                post_fontana, post_rmse,
                runtime    
            ]
            writer.writerow(data)

if __name__ == "__main__":
    main()
