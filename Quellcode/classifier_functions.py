import os
import subprocess
from classify_pose import classify_pose
from pose_functions import convert_to_pose_images


def retrain(training_folder, folder_classification_files, training_steps):
    """
    Funktion zum Trainieren des Bildklassifikators.
    Ruft einen Subprozess auf, damit die Parameter des Skripts `retrain_pose_classifier.py` verwendet werden können.

    :param training_folder: Welche Bilder zum trainieren verwendet werden, kann mithilfe der Parameter `--train_with_black_poses` und `--train_with_vid_as_img` festgelegt werden.
    :param folder_classification_files:  Die trainierten Modelle werden im Ordner `--folder_classification_files` gespeichert.
    :param training_steps: Bestimmt die Anzahl der Trainingsschritte (Epochen).
    :return: Referenz zum aufgerufenen Subprozess
    """
    # Step 2: Train Model on Poses
    dir_path = os.path.dirname(__file__)
    os.chdir(dir_path)
    cmd = "venv/Scripts/python retrain_pose_classifier.py --bottleneck_dir="+dir_path+"/"+folder_classification_files+"/bottlenecks --how_many_training_steps="+str(training_steps)+" --model_dir="+dir_path+"/"+folder_classification_files+"/inception --output_graph="+dir_path+"/"+folder_classification_files+"/retrained_graph.pb --output_labels="+dir_path+"/"+folder_classification_files+"/retrained_labels.txt --image_dir="+dir_path+"/"+training_folder+" --summaries_dir="+dir_path+"/"+folder_classification_files+"/summaries" #+" --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/1"
    #p = subprocess.Popen(cmd) #--flip_left_right
    #out, err = p.communicate()
    #result = out.split('\n')
    #for lin in result:
    #    if not lin.startswith('#'):
    #        print(lin)
    return subprocess.call(cmd)


def analyze_classifier(folder_classification_files):
    """
    Funktion zum Start von "TensorBoard", einem Analyse-Tool zur Untersuchung des mittels `--retrain` erstellten Klassifikators.
    Ruft einen Subprozess auf, da TensorBoard eine eigenständige Binärdatei ist.

    :param folder_classification_files: Ordner `--folder_classification_files`, der die trainierten Modelle enthält.
    :return: Referenz zum aufgerufenen Subprozess
    """
    # Use TensorBoard to analyze classifier
    dir_path = os.path.dirname(__file__)
    os.chdir(dir_path)
    cmd = "venv/Scripts/tensorboard --logdir="+dir_path+"/"+folder_classification_files+"/summaries"
    return subprocess.call(cmd)


def generate_pose_and_classify(img, estimator, folder_classification_files, use_black_poses):
    """
    Funktion, die Körperhaltungen in einem Bild erkennt und anhand dieser eine Klasssifikation durchführt.

    :param img: bereits eingelesenes OpenCV-Bild
    :param estimator: Modell für Pose Estimation
    :param folder_classification_files: Modell für Handlungsklassifikator
    :param use_black_poses: Auf `True` setzen, um das Skelette-Bilder vor schwarzem Hintergrund für die Klassifikation zu verwenden. Auf `False` setzen, um die in das Original-Bild eingezeichneten Skelett-Bilder zu verwenden.
    :return: Vorhergesagte Klasse als String sowie die Bilder mit den eingezeichneten Skeletten (Original und Schwarz)
    """
    img_pose, img_pose_black, humans = convert_to_pose_images(img, estimator)
    labels = classify_pose(img_pose_black if use_black_poses else img_pose, folder_classification_files)
    return labels, img_pose, img_pose_black