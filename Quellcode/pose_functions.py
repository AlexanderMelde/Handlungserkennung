import os
import cv2
import glob
import numpy as np
from tpe.tf_pose.estimator import TfPoseEstimator
from tpe.tf_pose.networks import get_graph_path


def initializeEstimator():
    """
    Funktion, die das Modell für den Pose Estimator initialisiert/in den Cache lädt und zurückgibt.

    :return: Modell für Pose Estimator
    """
    return TfPoseEstimator(get_graph_path("mobilenet_thin"), target_size=(432, 368))


def convert_to_pose_images(image, estimator):
    """
    Funktion, die Körperhaltungen in einem Bild erkennt und in das Bild sowie auf schwarzen Hintergrund einzeichnet

    :param image: Zu untersuchendes Bild (opencv image, bereits eingelesen)
    :param estimator: Modell für Pose Estimator
    :return: Bild mit Skeletten im Originalbild, Bild mit Skeletten vor schwarzem Hintergrund und Datenstruktur, die Skelette beschreibt.
    """
    humans = estimator.inference(image, resize_to_default=True, upsample_size=4.0)
    pose_image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    image_black = np.zeros((pose_image.shape[0], pose_image.shape[1], 3), np.uint8)
    pose_image_black = TfPoseEstimator.draw_humans(image_black, humans, imgcopy=False)
    return pose_image, pose_image_black, humans


def generate_poses(folder_img_original, folder_img_pose_rgb, folder_img_pose_black, folder_train_data_pose, estimator, should_save_empty_poses):
    """
    Funktion zum massenhaften Generieren von Skelett-Bildern für alle Einzelbilder eines Ordners

    :param folder_img_original: Ordner der Original-Einzelbilder enthält
    :param folder_img_pose_rgb: Ordner in den die Skelette im Originalbild eingezeichnet abgespeichert werden
    :param folder_img_pose_black: Ordner in den die Skelette vor schwarzem Hintergrund eingezeichnet abgespeichert werden
    :param folder_train_data_pose: Ordner in den die Skelett-Datenstrukturen abgespeichert werden
    :param estimator: Modell für Pose Estimator
    :param should_save_empty_poses: Auf `False` setzen, um bei der Erkennung von Skeletten keine Bilder abzuspeichern, wenn keine Skelette im Bild erkannt wurden. Auf `True` setzen, wenn auch leere / schwarze Bilder gespeichert werden sollen.
    :return: None, speichert Dateien ab in die angegebenen Ordner
    """
    os.chdir(os.path.join(os.path.dirname(__file__), folder_img_original))

    video_folder_subfolders = next(os.walk('.'))[1]

    for subfolder in video_folder_subfolders:
        os.chdir(os.path.join(os.path.dirname(__file__), folder_img_original, subfolder))
        for file in glob.glob("*.jpg"):
            filename = os.path.basename(file)
            directory_pose_rgb = os.path.join(os.path.dirname(__file__), folder_img_pose_rgb, subfolder)
            directory_pose_black = os.path.join(os.path.dirname(__file__), folder_img_pose_black, subfolder)
            directory_pose_data = os.path.join(os.path.dirname(__file__), folder_train_data_pose, subfolder)
            path_pose_rgb = os.path.join(directory_pose_rgb, filename)
            path_pose_black = os.path.join(directory_pose_black, filename)
            path_pose_data = os.path.join(directory_pose_data, filename+".txt")

            # If file has been marked as empty or any of the files that should be generated do not exist
            if not os.path.isfile(file+".nopose") and (not os.path.isfile(path_pose_black) or not os.path.isfile(path_pose_rgb)or not os.path.isfile(path_pose_data)):
                image = cv2.imread(file)
                pose_rgb, pose_black, humans = convert_to_pose_images(image, estimator)    # Generate Pose Images

                if should_save_empty_poses or humans:
                    # Generate Subdirs if they dont exist yet
                    if not os.path.exists(directory_pose_rgb):
                        os.makedirs(directory_pose_rgb)
                    if not os.path.exists(directory_pose_black):
                        os.makedirs(directory_pose_black)
                    if not os.path.exists(directory_pose_data):
                        os.makedirs(directory_pose_data)

                    # Write Pose Images to their folders
                    cv2.imwrite(path_pose_rgb, pose_rgb)
                    cv2.imwrite(path_pose_black, pose_black)
                    with open(path_pose_data, "w") as text_file:
                        print(humans, file=text_file)

                    print("[POSEGEN] Saved poses for file " + subfolder + "/" + file)
                else:
                    print("[POSEGEN] Not Saving pose, No poses detected for file " + subfolder + "/" + file)
                    open(file + ".nopose", 'a').close()  # create an empty file to mark as empty pose
            else:
                print("[POSEGEN] Skipping already generated pose for file " + subfolder + "/" + file)
