#!/usr/bin/env python
"""
    File name: main.py
    Author: Alexander Melde
    Date created: 03.07.2018
    Date last modified: 27.08.2018
    Python Version: 3.6

    Start-Skript für die Prototypen der Bachelorarbeit
    "Erkennung menschlicher Handlungen durch Auswertung der Körperhaltungen von Personen in einem Video mithilfe von Machine Learning und neuronalen Netzen"
    von Alexander Melde

    Eine Beschreibung der Prototypen befindet sich in Kapitel 4 der Bachelorarbeit.

    Hinweise zur Benutzung können mit `python main.py -h` angezeigt werden.


"""

import cv2
import os
import glob
import time
import argparse
import numpy as np
from classify_pose import classify_pose
from pose_functions import initializeEstimator, convert_to_pose_images, generate_poses
from classifier_functions import retrain, analyze_classifier, generate_pose_and_classify
from util_functions import convert_videos_to_images, clean, helper_deleteblackimages
from vid2img_functions import convertVideosToOneImage
from signal_functions import convertVideoToSignals, retrain_with_signals


def main():
    parser = argparse.ArgumentParser(description='Mit diesem Skript lassen sich alle Prototypen der Bachelorarbeit "Erkennung menschlicher Handlungen durch Auswertung der Körperhaltungen von Personen in einem Video mithilfe von Machine Learning und neuronalen Netzen" von Alexander Melde starten. Eine Beschreibung der Prototypen befindet sich in Kapitel 4 der Bachelorarbeit.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Argumente zur Konfiguration der Unterordner
    parser.add_argument('--folder_train_vid_original', type=str, default="training/input_videos", help='Ordner, in dem die Original-Trainings-Videos enthalten sind, benötigt für Training der Prototypen 1, 2 und 3.')
    parser.add_argument('--folder_train_img_original', type=str, default="training/input_videos_frames", help='Ordner, in den die Einzelbilder der Original-Trainings-Videos extrahiert werden sollen/sind, benötigt für Prototyp 1.')
    parser.add_argument('--folder_train_img_pose_rgb', type=str, default="training/input_videos_frames_pose_rgb", help='Ordner, in den die Einzelbidlder mit eingezeichneten Skeletten generiert werden sollen/sind, benötigt für Prototyp 1.')
    parser.add_argument('--folder_train_img_pose_black', type=str, default="training/input_videos_frames_pose_black", help='Ordner, in den die schwarzen Skelett-Einzelbilder generiert werden sollen/sind, benötigt für Prototyp 1.')
    parser.add_argument('--folder_train_data_pose', type=str, default="training/input_videos_frames_pose_data", help='Ordner, in den die Skelett-Daten der Einzelbilder gespeichert werden sollen, benötigt für Prototyp 1.')

    parser.add_argument('--folder_train_vid_as_img', type=str, default="training/input_videos_as_superimposed_frames", help='Ordner, in den die anhand von Videos überlagerten Skelett-Bilder gespeichert werden sollen/sind, benötigt für Prototyp 2.')
    parser.add_argument('--folder_train_vid_as_sig', type=str, default="training/input_videos_as_signal", help='Ordner, in den die anhand von Videos generierten Signale von Gelenk-Positionen gespeichert werden sollen/sind, benötigt für Prototyp 3.')

    parser.add_argument('--folder_classification_files', type=str, default="training/classification_files", help='Ordner, in den die Dateien des Klassifikators gespeichert werden sollen/sind, benötigt für Prototyp 1 und 2.')

    # Argumente zur Steuerung der Funktionalität: Was soll ausgeführt werden?
    parser.add_argument('--clean', type=bool, default=False, help='Funktion zum Löschen aller konvertierter Daten und temporären Dateien, die zum Speichern des Konvertierungs-Fortschritts genutzt werden. Auf `True` setzen zum Aktivieren. Betrifft Prototyp 1.')
    parser.add_argument('--delete_black_images', type=bool, default=False, help='[Veraltet!] Funktion zum Löschen aller Bilder im Ordner der überlagerten Skelett-Bilder, die ausschließlich schwarze Pixel enthalten. Verwendung nicht mehr notwendig, da solche Bilder gar nicht mehr gespeichert werden. Betrifft Prototyp 2.')

    parser.add_argument('--convert_videos_to_images', type=bool, default=False, help='Funktion zum Konvertieren aller (.avi) Videos im Ordner `--folder_train_vid_original` zu Einzelbildern (.jpg) im Ordner `--folder_train_img_original`. Auf `True` setzen zum Aktivieren. Betrifft Prototyp 1.')
    parser.add_argument('--convert_each_nth_frame', type=int, default=10, help='Parameter für die Funktionen `--convert_videos_to_images` und `--convertVideosToOneImage`. Bestimmt die Anzahl der Einzelbilder, die bei der Verarbeitung von Videos übersprungen werden sollen (+1). Setze auf 1 um jedes Einzelbild zu berücksichtigen, auf 3 um jedes dritte, ... Betrifft Prototyp 1 und 2.')

    parser.add_argument('--generate_poses', type=bool, default=False, help='Funktion zum Generieren von Skelett-Bildern (.jpg) in die Ordner `--input_videos_frames_pose_rgb`, `--input_videos_frames_pose_black` und Skelett-Daten (.txt) in den Ordner `--input_videos_frames_pose_data` zu allen (.jpg) Bildern im Ordner `--folder_train_img_original`. Auf `True` setzen zum Aktivieren. Betrifft Prototyp 1.')
    parser.add_argument('--save_empty_poses', type=bool, default=False, help='Parameter für die Funktion `--generate_poses`. Auf `False` setzen, um bei der Erkennung von Skeletten keine Bilder abzuspeichern, wenn keine Skelette im Bild erkannt wurden. Auf `True` setzen, wenn auch leere / schwarze Bilder gespeichert werden sollen. Betrifft Prototyp 1.')

    parser.add_argument('--convertVideosToOneImage', type=bool, default=False, help='Funktion zum Überlagern von Skeletten, die aus den Einzelbildern der Videos im Ordner `--folder_train_vid_original` generiert wurden. Die Überlagerungen (.jpg) werden im Ordner `--folder_train_vid_as_img` gespeichert. Auf `True` setzen zum Aktivieren. Betrifft Prototyp 2.')
    parser.add_argument('--max_frames_per_vid', type=int, default=30, help='Parameter für die Funktion `--convertVideosToOneImage`. Bestimmt, wie viele Skelette in ein Bild überlagert werden. Kann mit `--convert_each_nth_frame` kombiniert werden, wobei übersprungene Einzelbilder nicht hinzugezählt werden. Betrifft Prototyp 2.')
    parser.add_argument('--split_videos', type=bool, default=True, help='Parameter für die Funktion `--convertVideosToOneImage`. Auf `False` setzen, um nach Abschluss einer Überlagerung zum nächsten Video zu springen. Auf `True` setzen, wenn für längere Video mehrere Überlagerungen erstellt werden dürfen. Betrifft Prototyp 2.')
    parser.add_argument('--do_not_save_too_short_videos', type=bool, default=False, help='Parameter für die Funktion `--convertVideosToOneImage`. Auf `True` setzen, um Überlagerungen, die nicht aus genau `--max_frames_per_vid` Einzelbildern bestehen nicht abzuspeichern. Auf `False` setzen, wenn auch Überlagerungen aus weniger Einzelbilder erstellt werden dürfen. Betrifft Prototyp 2.')

    parser.add_argument('--convertVideosToSignals', type=bool, default=False, help='Funktion zum Generieren von Signalen, die die Körperhaltungen in den Einzelbildern der Videos im Ordner `--folder_train_vid_original` beschreiben. Die Signale (.txt, .pdf) werden im Ordner `--folder_train_vid_as_sig` gespeichert. Auf `True` setzen zum Aktivieren. Betrifft Prototyp 3.')

    parser.add_argument('--retrain', type=bool, default=False, help='Funktion zum Trainieren des Bildklassifikators. Welche Bilder zum trainieren verwendet werden, kann mithilfe der Parameter `--train_with_black_poses` und `--train_with_vid_as_img` festgelegt werden. Die trainierten Modelle werden im Ordner `--folder_classification_files` gespeichert. Auf `True` setzen zum Aktivieren. Hinweis: In einem Teil der Versuchsläufe kam es vor, dass das Skript bei Verwendung dieses Parameters das erste mal eine Fehlermeldung produziert, nachdem der Download des vortrainierten Modells abgeschlossen ist. Bei erneutem Aufruf des Skripts nach dem Download sollte die Fehlermeldung nicht mehr erscheinen. Betrifft Prototyp 1 und 2.')
    parser.add_argument('--training_steps', type=int, default=5000, help='Parameter für die Funktion `--retrain`. Bestimmt die Anzahl der Trainingsschritte (Epochen). Betrifft Prototyp 1 und 2.')
    parser.add_argument('--train_with_black_poses', type=bool, default=True, help='Parameter für die Funktion `--retrain`. Auf `True` setzen, um die Skelette-Bilder vor schwarzem Hintergrund (aus dem Ordner `--folder_train_img_pose_black`) für das Training zu verwenden. Auf `False` setzen, um die in das Original-Bild eingezeichneten Skelett-Bilder (aus dem Ordner `--folder_train_img_pose_rgb`) zu verwenden. Betrifft Prototyp 1.')
    parser.add_argument('--train_with_vid_as_img', type=bool, default=False, help='Parameter für die Funktion `--retrain`. Auf `True` setzen, um die Skelett-Überlagerungen aus dem Ordners `--folder_train_vid_as_img` für das Training zu verwenden (Wichtig: Erfordert `--train_with_black_poses=True`). Auf `False` setzen, um keine Änderung des Ordners vorzunehmen. Betrifft Prototyp 2.')

    parser.add_argument('--retrain_with_signals', type=bool, default=False, help='Funktion zum Trainieren des Signal-Klassifikators. Verwendet die Signale im Ordner `--folder_train_vid_as_sig`. Das generierte Modell wird nicht gespeichert, zum Abschluss des Trainings findet aber ein Test mit Berechnung der Genauigkeit und Wahrheitsmatrix statt. Auf `True` setzen zum Aktivieren. Betrifft den dritten Prototyp.')

    parser.add_argument('--analyze_classifier', type=bool, default=False, help='Funktion zum Start von "TensorBoard", einem Analyse-Tool zur Untersuchung des mittels `--retrain` erstellten Klassifikators. Auf `True` setzen zum Aktivieren. Betrifft Prototyp 1 und 2.')

    # WHAT TO DO: Classifying Input
    parser.add_argument('--classify_image', type=bool, default=False, help='Funktion zum Klassifizieren eines Einzelbilds anhand des Modells im Ordner `--folder_classification_files`. Das zu klassifizierende Bild kann mit dem Parameter `--file_test_in` bestimmt werden. Auf `True` setzen zum Aktivieren. Betrifft Prototyp 1 und 2.')
    parser.add_argument('--file_test_in', type=str, default="test_images/example.jpg", help="Parameter für die Funktion `--classify_image`. Bestimmt das zu klassifizierende Bild. Erwartet einen relativen oder absoluten Dateipfad. Betrifft Prototyp 1 und 2.")

    parser.add_argument('--classify_folder', type=bool, default=False, help='Funktion zum Klassifizieren aller Bilder (.jpg) des Ordners `--folder_test_in` anhand des Modells im Ordner `--folder_classification_files`. Auf `True` setzen zum Aktivieren. Betrifft Prototyp 1 und 2.')
    parser.add_argument('--folder_test_in', type=str, default="test_images", help="Parameter für die Funktion `--classify_folder`. Bestimmt den Ordner, in dem die zu klassifizierenden Bilder liegen. Erwartet einen relativen oder absoluten Dateipfad. Betrifft Prototyp 1 und 2.")

    parser.add_argument('--classify_webcam', type=bool, default=False, help='Funktion zum Klassifizieren aller Bilder einer Webcam anhand des Modells im Ordner `--folder_classification_files`. Auf `True` setzen zum Aktivieren. Betrifft Prototyp 1.')
    parser.add_argument('--webcam_video_source', type=str, default="0", help='Parameter für die Funktionen `--classify_webcam` und `--classify_webcam_asImgs`. Bestimmt die Nummer der zu verwendenden, an den PC angeschlossenen, Webcam. Um einen Internet-Livestream (z.B. im RTSP-Format) zu verwenden, kann statt der Nummer eine URL angegeben werden. Betrifft Prototyp 1.')

    parser.add_argument('--classify_webcam_asImgs', type=bool, default=False, help='Funktion zum Klassifizieren aller überlagerten Bilder (berücksichtigt `--max_frames_per_vid`) einer Webcam anhand des Modells im Ordner `--folder_classification_files`. Auf `True` setzen zum Aktivieren (Wichtig: Erfordert `--classify_webcam=False`). Betrifft Prototyp 2.')

    args = parser.parse_args()


    # Funktionen anhand der Argumente aufrufen:

    if args.clean:
        clean(args.folder_train_vid_original, args.folder_train_img_original, args.folder_train_img_pose_rgb, args.folder_train_img_pose_black, args.folder_train_data_pose, args.folder_classification_files)

    if args.delete_black_images:
        helper_deleteblackimages(args.folder_train_vid_as_img)

    # Modell zum Berechnen von Körperhaltungen laden (wenn benötigt)
    pose_estimator = initializeEstimator() if args.generate_poses or args.classify_image or args.classify_folder or args.classify_webcam or args.classify_webcam_asImgs or args.convertVideosToOneImage or args.convertVideosToSignals else None

    if args.convert_videos_to_images:
        convert_videos_to_images(args.folder_train_vid_original, args.folder_train_img_original, args.convert_each_nth_frame)

    if args.generate_poses:
        generate_poses(args.folder_train_img_original, args.folder_train_img_pose_rgb, args.folder_train_img_pose_black, args.folder_train_data_pose, pose_estimator, args.save_empty_poses)

    if args.convertVideosToOneImage:
        convertVideosToOneImage(args.folder_train_vid_original, args.folder_train_vid_as_img, args.convert_each_nth_frame, pose_estimator, args.max_frames_per_vid, args.split_videos, args.do_not_save_too_short_videos)

    if args.convertVideosToSignals:
        convertVideoToSignals(args.folder_train_vid_original, args.folder_train_vid_as_sig, pose_estimator)

    if args.retrain:
        retrain(args.folder_train_vid_as_img if args.train_with_vid_as_img else args.folder_train_img_pose_black if args.train_with_black_poses else args.folder_train_img_pose_rgb, args.folder_classification_files, args.training_steps)

    if args.analyze_classifier:
        analyze_classifier(args.folder_classification_files)

    if args.retrain_with_signals:
        retrain_with_signals(args.folder_train_vid_as_sig)

    if args.classify_image:
        img = cv2.imread(args.file_test_in)
        print(generate_pose_and_classify(img, pose_estimator, args.folder_classification_files, args.train_with_black_poses)[0])

    if args.classify_folder:
        os.chdir(args.folder_test_in)
        for file in glob.glob("*.jpg"):
            print(file)
            img = cv2.imread(file)
            print(generate_pose_and_classify(img, pose_estimator, args.folder_classification_files, args.train_with_black_poses)[0])

    # Sowohl lokale als auch Internet-Webcam erlauben: Wenn im String nur eine Zahl übergeben wurde, dann als Integer speichern:
    try:
        webcam_source = int(args.webcam_video_source)
    except ValueError:
        webcam_source = args.webcam_video_source

    if args.classify_webcam:
        fps_time = 0
        cam = cv2.VideoCapture(webcam_source)
        while True:
            ret_val, image = cam.read()

            if image is None:
                print("Error: Webcam not connected.")
                break

            image = cv2.resize(image, (432, 368))

            labels, img_pose_rgb, img_pose_black = generate_pose_and_classify(image, pose_estimator, args.folder_classification_files, args.train_with_black_poses)

            cv2.putText(img_pose_rgb,
                        "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

            cv2.putText(img_pose_rgb,
                        "Action: "+labels[0][0],
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

            cv2.putText(img_pose_rgb,
                        "Accuracy: %f" % labels[0][1],
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

            cv2.imshow('tf-pose-estimation result', img_pose_rgb)
            fps_time = time.time()
            if cv2.waitKey(1) == 27:
                break

        cv2.destroyAllWindows()

    if args.classify_webcam_asImgs:
        fps_time = 0
        cam = cv2.VideoCapture(webcam_source)
        frame = 0
        count = 0

        ret_val, image = cam.read()
        image = cv2.resize(image, (432, 368))
        pose_rgb, pose_black, humans = convert_to_pose_images(image, pose_estimator)
        last_classification = np.zeros((pose_black.shape[0], pose_black.shape[1], 3), np.uint8)

        count += 1

        while True:
            ret_val, image = cam.read()

            if image is None:
                print("Error: Webcam not connected.")
                break

            image = cv2.resize(image, (432, 368))

            # Generate Pose Images and Add Pose to VidImg
            pose_rgb_i, pose_black_i, humans_i = convert_to_pose_images(image, pose_estimator)  # Generate Pose Images
            pose_black += pose_black_i

            # Classify Pose Images when Max_frames is reached
            if count > args.max_frames_per_vid:
                labels = classify_pose(pose_black, args.folder_classification_files)

                cv2.putText(pose_black,
                            "Action: " + labels[0][0],
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)

                cv2.putText(pose_black,
                            "Accuracy: %f" % labels[0][1],
                            (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)

                # cv2.imshow('Zur Klassifikation genutzte Ueberlagerung', pose_black)
                last_classification = pose_black.copy()
                pose_black = np.zeros((pose_black.shape[0], pose_black.shape[1], 3), np.uint8)
                count = 0

            cv2.putText(pose_rgb_i,
                        "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

            # cv2.imshow('Aktuelle Ueberlagerung', pose_black)
            # cv2.imshow('Kamerabild mit Pose', pose_rgb_i)

            cv2.imshow("Handlungserkennung. Links Aktuelle Ueberlagerung, Mitte Kamerabild mit Pose, Rechts die letzte zur Klassifikation genutzte Ueberlagerung", np.hstack((pose_black, pose_rgb_i, last_classification)))

            fps_time = time.time()
            frame += 1
            count += 1
            if cv2.waitKey(1) == 27:
                break


if __name__ == '__main__':
    main()