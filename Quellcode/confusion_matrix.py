import matplotlib.pyplot as plt
import numpy as np
import itertools


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Funktion zur Vorbereitung der grafischen Darstellung einer Wahrheitsmatrix.

    :param cm: Wahrheitsmatrix als zweidimensionales Array
    :param classes: Labels / Beschriftungen der Klassen in richtiger Reihenfolge (Alphabetisch bei unserem Klassifikator)
    :param normalize: Normalisierung einschalten (Prozentangaben statt absolute Werte)
    :param title: Titel der Visualisierung
    :param cmap: Farbschema
    :return: None, es werden Funktionen der plotting-Bibliothek aufgerufen, die den Zustand der Bibliothek verändern und wie im Beispiel gezeigt weiterverwendet werden können.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Tatsächliche Klasse (true label)')
    plt.xlabel('Vorhergesagte Klasse (predicated label)')


def run_example():
    """
    Funktion zum Generieren einer im Code definierten Confusion Matrix

    :return: None, es werden Dateien abgespeichert und Konsolen-Ausgaben gemacht (siehe Quelltext)
    """
    LABELS = ['boxing', 'handclapping', 'handwaving', 'walking']
    # LABELS = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
    # LABELS = ['1p boxing', '1p handclapping', '1p handwaving', '1p jogging', '1p jumping', '1p pointing', '1p running', '1p walking', '2p highfive', '2p hugging', '2p kicking', '2p punching', '2p pushing', '2p shakinghands']
    n_classes = len(LABELS)

    print("Confusion Matrix:")
    """

                        confusion_matrix = [[ 24,  2,  0,  0],
                        [  2, 14,  0,  0],
                        [  2,  0, 20,  0],
                        [  2,  0,  0, 76]]
    """
    confusion_matrix = [[24, 2, 0, 0, 0, 0],
                        [2, 18, 0, 0, 0, 0],
                        [0, 0, 26, 0, 0, 0],
                        [0, 0, 0, 16, 2, 6],
                        [0, 0, 0, 2, 12, 0],
                        [4, 0, 0, 2, 2, 22]]
    """
    confusion_matrix =  [[7496, 186, 128, 236, 180,  16, 438, 278,   0,   0,   0,   0,   0,   0],
                         [ 194,6810,1172,  18, 264,   6,  74,  22,   0,   0,   0,   0,   0,   0],
                         [ 584,2814,6496,  30, 540,   4,  60, 126,   0,   0,   0,   0,   0,   0],
                         [ 250,  30,  18,1138, 106,   6,1084, 858,   0,   0,   0,   0,   0,   0],
                         [   2,   6,   0,   2, 276,   0,  10,  12,   0,   0,   0,   2,   0,   0],
                         [   0,   0,   0,   0,   0, 414,   0,   2,   0,   0,   0,   0,   0,   0],
                         [ 140,  24,  12, 326, 144,  18,1324, 384,   0,   0,   0,   0,   0,   0],
                         [ 438,  70,  54, 784, 394,  38,1060,3040,   0,   0,   0,   0,   0,   0],
                         [   0,   0,   0,   0,   6,   4,   0,   0, 612, 112, 144,  64,  82, 168],
                         [   0,   0,   0,   0,  14,  50,   0,   4, 366,1384, 290, 180, 188, 284],
                         [   0,   0,   0,   0,   0,   2,   0,   0,  88,  24, 640, 132,  56,  28],
                         [   0,   0,   0,   0,   4,   4,   0,   0, 176,  50, 300, 370, 154,  22],
                         [   0,   0,   0,   0,   2,   0,   0,   0,  96,  86, 214, 146, 566,  26],
                         [   0,   0,   0,   0,   0,   6,   2,   0, 194,  96, 122,  14,  38, 696]]

    """
    confusion_matrix = [[284, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [12, 236, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [4, 20, 266, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 70, 0, 0, 14, 8, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 8, 2, 0, 44, 2, 0, 0, 0, 0, 0, 0],
                        [2, 0, 0, 8, 0, 0, 6, 116, 0, 2, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 10, 6, 0, 6, 0, 8],
                        [0, 0, 0, 0, 0, 0, 0, 0, 10, 46, 6, 2, 4, 6],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 2, 0, 2],
                        [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 10, 12, 6, 2],
                        [0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 0, 10, 12, 0],
                        [0, 0, 0, 0, 0, 2, 0, 0, 4, 4, 6, 0, 0, 20]]

    confusion_matrix = [[7480, 226, 100, 378, 226, 332],
                        [214, 7538, 676, 66, 54, 22],
                        [530, 3400, 6086, 58, 26, 130],
                        [226, 38, 8, 1956, 604, 780],
                        [114, 34, 14, 670, 994, 336],
                        [344, 78, 48, 1452, 550, 3172]]

    confusion_matrix = [[7808, 178, 150, 606],
                        [210, 6782, 1504, 74],
                        [424, 2056, 7656, 94],
                        [854, 108, 152, 10322]]

    print(np.sum(confusion_matrix))
    # Compute confusion matrix
    confusion_matrix = np.asanyarray(confusion_matrix)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    f2 = plt.figure(figsize=(8, 6))  # few labels
    f2 = plt.figure(figsize=(6, 4))  # very few labels
    # f2 = plt.figure(figsize=(12,8)) #many labels
    plot_confusion_matrix(confusion_matrix, classes=LABELS,
                          title='Wahrheitsmatrix (confusion matrix)')

    f2.savefig("confusion_matrix.pdf", bbox_inches='tight')
