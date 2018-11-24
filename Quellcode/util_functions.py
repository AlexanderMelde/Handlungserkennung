import os
import cv2
import glob

def convert_videos_to_images(video_folder, image_folder, convert_each_nth_frame):
    """
    Funktion zum Aufteilen von Videos in Einzelbilder

    :param video_folder: Ordner, der die Videos enthält
    :param image_folder: Ordner, in den die Einzelbilder gespeichert werden sollen
    :param convert_each_nth_frame: Bestimmt die Anzahl der Einzelbilder, die bei der Verarbeitung von Videos übersprungen werden sollen (+1). Setze auf 1 um jedes Einzelbild zu berücksichtigen, auf 3 um jedes dritte, ...
    :return: None, speichert Einzelbilder in Ordner ab
    """
    print("Now converting")
    os.chdir(os.path.join(os.path.dirname(__file__), video_folder))

    video_folder_subfolders = next(os.walk('.'))[1]

    for subfolder in video_folder_subfolders:
        os.chdir(os.path.join(os.path.dirname(__file__), video_folder, subfolder))
        for file in glob.glob("*.avi"):
            if not os.path.isfile(file+".converted"):
                print("[CONVERTER] Now Converting "+subfolder+"/"+file)
                filename = os.path.basename(file)
                vidcap = cv2.VideoCapture(file)
                success, image = vidcap.read()
                count = 0
                while success:
                    if (count % convert_each_nth_frame) == 0: #Nur jeden n-ten Frame abspeichern, da sonst viel zu viele Bilder entstehen.
                        directory = os.path.join(os.path.dirname(__file__), image_folder, subfolder)
                        if not os.path.exists(directory):
                            os.makedirs(directory)
                        path = os.path.join(directory, filename+".frame%d.jpg" % count)

                        #print("[CONVERTER] Now Converting " + subfolder+"/"+file + "(frame #"+str(count+1)+")")
                        cv2.imwrite(path, image)  # save frame as JPEG file
                        success, image = vidcap.read()
                        # print('Read a new frame: ', success)
                    count += 1
                open(file+".converted", 'a').close() # create an empty file to mark as converted
            else:
                print("[CONVERTER] Skipping already converted file "+subfolder+"/"+file)



def clean(folder_train_vid_original, folder_train_img_original, folder_train_img_pose_rgb, folder_train_img_pose_black, folder_train_data_pose, folder_classification_files): # Remove all generated content
    """
    Funktion zum Löschen aller konvertierter Daten und temporären Dateien, die zum Speichern des Konvertierungs-Fortschritts genutzt werden.
    Betrifft nur Prototyp 1.

    :param folder_train_vid_original: Ordner, in dem alle *.converted Dateien gelöscht werden sollen
    :param folder_train_img_original: Ordner, in dem alle *.nopose Dateien gelöscht werden sollen
    :param folder_train_img_pose_rgb: Ordner, der komplett gelöscht werden soll
    :param folder_train_img_pose_black: Ordner, der komplett gelöscht werden soll
    :param folder_train_data_pose: Ordner, der komplett gelöscht werden soll
    :param folder_classification_files: Ordner, der komplett gelöscht werden soll
    :return: None, löscht Dateien und Ordner im Dateisystem
    """
    print("[CLEAN] This will clean some of the generated files, do you really want to continue? The following files will be deleted:")
    print("[CLEAN] - *.converted files in ", folder_train_vid_original, "(will not delete original videos)")
    print("[CLEAN] - *.nopose files in ", folder_train_img_original, "(will not delete original images)")
    print("[CLEAN] - the folder ", folder_train_img_pose_rgb, "(deletes poses on rgb pictures)")
    print("[CLEAN] - the folder ", folder_train_img_pose_black, "(deletes poses on black pictures)")
    print("[CLEAN] - the folder ", folder_train_data_pose, "(deletes poses data files)")
    print("[CLEAN] - the folder ", folder_classification_files, "(deletes image classification files)")
    confirm = ""
    while confirm != 'y' and confirm != 'n':
        confirm = input("[CLEAN] [y]es, Delete now. [n]o, Cancel. Type y or n.")
        if confirm != 'y' and confirm != 'n':
            print("[CLEAN] Invalid Option. Please Enter a Valid Option.")
    if confirm == 'y':
        import shutil
        os.chdir(os.path.dirname(__file__))
        for root, dirs, files in os.walk(folder_train_vid_original):
            for currentFile in files:
                if currentFile.lower().endswith(".converted"):
                    os.remove(os.path.join(root, currentFile))
        for root, dirs, files in os.walk(folder_train_img_original):
            for currentFile in files:
                if currentFile.lower().endswith(".nopose"):
                    os.remove(os.path.join(root, currentFile))
        shutil.rmtree(folder_train_img_pose_rgb, ignore_errors=True) #ignore errors means it deletes when dir is not empty
        shutil.rmtree(folder_train_img_pose_black, ignore_errors=True)
        shutil.rmtree(folder_train_data_pose, ignore_errors=True)
        shutil.rmtree(folder_classification_files, ignore_errors=True)
        print("[CLEAN] Successfully deleted all progress markers and the generated content listed above.")


def helper_deleteblackimages(folder):
    """
    [Veraltet!]
    Funktion zum Löschen aller Bilder im Ordner der überlagerten Skelett-Bilder, die ausschließlich schwarze Pixel enthalten.
    Verwendung nicht mehr notwendig, da solche Bilder gar nicht mehr gespeichert werden.
    Betrifft Prototyp 2.

    :param folder: Ordner, in dem alle schwarzen Bilder gelöscht werden sollen
    :return: None, löscht Dateien in Dateisystem
    """
    return False # Warning: Currently this code deletes all images in folder!
    os.chdir(os.path.dirname(__file__))
    for root, dirs, files in os.walk(folder):
        for currentFile in files:
            if currentFile.lower().endswith(".jpg"):
                image = cv2.imread(currentFile, 0)
                nonzeroes = cv2.countNonZero(image) #TODO: currently always returns 0 (for all kind of images)
                if nonzeroes == 0:
                    os.remove(os.path.join(root, currentFile))
                    print("[BLACKIMG] Deleted all-black image with", nonzeroes, "non-black pixels", currentFile)