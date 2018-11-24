import os
import cv2
import glob
import numpy as np
from pose_functions import convert_to_pose_images

def convertVideosToOneImage(video_folder, video_as_img_folder, convert_each_nth_frame, estimator, max_frames_per_vid, split_videos, do_not_save_too_short_videos):
    """
    Funktion zum Generieren und Überlagern von Skeletten in Videos.
    Betrifft Prototyp 2.

    :param video_folder: Ordner, in dem die Original-Videos sind
    :param video_as_img_folder: Ordner, in den die Überlagerungs-Bilder (.jpg) gespeichert werden sollen
    :param convert_each_nth_frame: Bestimmt die Anzahl der Einzelbilder, die bei der Verarbeitung von Videos übersprungen werden sollen (+1). Setze auf 1 um jedes Einzelbild zu berücksichtigen, auf 3 um jedes dritte, ...
    :param estimator: Modell für Pose Estimator
    :param max_frames_per_vid: Bestimmt, wie viele Skelette in ein Bild überlagert werden. Kann mit `convert_each_nth_frame` kombiniert werden, wobei übersprungene Einzelbilder nicht hinzugezählt werden.
    :param split_videos: Auf `False` setzen, um nach Abschluss einer Überlagerung zum nächsten Video zu springen. Auf `True` setzen, wenn für längere Video mehrere Überlagerungen erstellt werden dürfen.
    :param do_not_save_too_short_videos: Auf `True` setzen, um Überlagerungen, die nicht aus genau `--max_frames_per_vid` Einzelbildern bestehen nicht abzuspeichern. Auf `False` setzen, wenn auch Überlagerungen aus weniger Einzelbilder erstellt werden dürfen.
    :return:
    """
    print("Now converting Vid2Img")
    os.chdir(os.path.join(os.path.dirname(__file__), video_folder))

    video_folder_subfolders = next(os.walk('.'))[1]

    for subfolder in video_folder_subfolders:
        os.chdir(os.path.join(os.path.dirname(__file__), video_folder, subfolder))
        for file in glob.glob("*.avi"):
            if not os.path.isfile(file + ".jpg"):
                print("[VID2IMG] Now Converting " + subfolder + "/" + file)
                filename = os.path.basename(file)
                directory = os.path.join(os.path.dirname(__file__), video_as_img_folder, subfolder)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                filename_img = os.path.join(directory, filename + ".jpg")

                vidcap = cv2.VideoCapture(file)
                success, image = vidcap.read()
                frame = 0
                count = 0

                pose_rgb, pose_black, humans = convert_to_pose_images(image, estimator)    # Generate Pose Images

                while success:
                    if (frame % convert_each_nth_frame) == 0:  # Nur jeden n-ten Frame abspeichern, da sonst viel zu viele Bilder entstehen.

                        if count > max_frames_per_vid:
                            if split_videos:
                                filename_img = os.path.join(directory, filename + "_" + str(frame) + ".jpg")
                                pose_black = np.clip(pose_black, 0, 255)
                                if cv2.countNonZero(cv2.cvtColor(pose_black, cv2.COLOR_BGR2GRAY)) > 0:
                                    cv2.imwrite(filename_img, pose_black)
                                else:
                                    print("[VID2IMG] Skipping empty split video")
                                count = 0
                                pose_black = np.zeros((pose_black.shape[0], pose_black.shape[1], 3), np.uint8)
                            else:
                                break

                        pose_rgb_i, pose_black_i, humans_i = convert_to_pose_images(image, estimator)  # Generate Pose Images
                        pose_black += pose_black_i
                        success, image = vidcap.read()
                        # do not count empty frames
                        if humans_i:
                            count += 1
                        # print('Read a new frame: ', success)
                    frame += 1

                if not do_not_save_too_short_videos or count == 30:
                    pose_black = np.clip(pose_black,0,255)
                    if cv2.countNonZero(cv2.cvtColor(pose_black, cv2.COLOR_BGR2GRAY)) > 0:
                        cv2.imwrite(filename_img, pose_black)
                    else:
                        print("[VID2IMG] Skipping empty split video")
                # open(file + ".converted", 'a').close()  # create an empty file to mark as converted
            else:
                print("[VID2IMG] Skipping already converted file " + subfolder + "/" + file)
