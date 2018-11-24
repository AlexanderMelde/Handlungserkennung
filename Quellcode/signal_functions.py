# this tries to detect poses in videos and uses these poses to generate signals for each joint

#[BodyPart:0-(0.30, 0.35) score=0.82 BodyPart:1-(0.31, 0.41) score=0.74 BodyPart:2-(0.28, 0.41) score=0.67 BodyPart:3-(0.23, 0.40) score=0.56 BodyPart:4-(0.19, 0.39) score=0.46 BodyPart:5-(0.34, 0.41) score=0.63 BodyPart:6-(0.38, 0.40) score=0.13 BodyPart:8-(0.29, 0.52) score=0.54 BodyPart:9-(0.28, 0.62) score=0.41 BodyPart:10-(0.28, 0.73) score=0.23 BodyPart:11-(0.33, 0.52) score=0.51 BodyPart:12-(0.33, 0.62) score=0.50 BodyPart:13-(0.33, 0.73) score=0.22 BodyPart:14-(0.29, 0.35) score=0.71 BodyPart:15-(0.31, 0.35) score=0.75 BodyPart:16-(0.28, 0.36) score=0.54 BodyPart:17-(0.32, 0.35) score=0.67]


import os
import cv2
import sys
import ast
import glob
import numpy as np
import tensorflow as tf
from tpe.tf_pose import common
from matplotlib import pyplot as plt
from signal_classifier_lstm import one_hot, LSTM_RNN, extract_batch_size
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from confusion_matrix import plot_confusion_matrix


def convertVideoToSignals(video_folder, video_as_signal_folder, estimator):
    """
    Funktion zum Generieren von Signalen, die die Körperhaltungen in Videos beschreiben.

    :param video_folder: Der Ordner, der die Original-Videos enthält
    :param video_as_signal_folder: Der Ordner, in dem die Signale (.txt, .pdf) gespeichert werden
    :param estimator: Modell für Pose Estimator
    :return: None, die Funktion speichert in Ordner
    """

    print("Now converting Vid2Signal")
    os.chdir(os.path.join(os.path.dirname(__file__), video_folder))

    video_folder_subfolders = next(os.walk('.'))[1]

    for subfolder in video_folder_subfolders:
        os.chdir(os.path.join(os.path.dirname(__file__), video_folder, subfolder))
        for file in glob.glob("*.avi"):
            filename = os.path.basename(file)
            directory = os.path.join(os.path.dirname(__file__), video_as_signal_folder, subfolder)
            if not os.path.exists(directory):
                os.makedirs(directory)
            filename_signal = os.path.join(directory, filename + ".signal.txt")
            filename_signal_acc = os.path.join(directory, filename + ".signal_acc.txt")
            filename_signal_acc_pdf = os.path.join(directory, filename + ".signal_acc.pdf")

            if not os.path.isfile(filename_signal) or not os.path.isfile(filename_signal_acc):
                print("[VID2SIGNAL] Now Converting " + subfolder + "/" + file)

                signal = [] # array of form [framenr][bodypartnr] (x,y) absolute pos

                vidcap = cv2.VideoCapture(file)
                success, image = vidcap.read()
                length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
                MAX_FRAMENR = length
                MAX_BODYPARTNR = 18
                signal = np.empty((MAX_FRAMENR,MAX_BODYPARTNR,2))                # initialize with None
                signal_acceleration = np.empty((MAX_FRAMENR,MAX_BODYPARTNR,2))   # initialize with None (was .zeros)
                signal[::] = np.nan
                signal_acceleration[::] = np.nan

                # print(signal)

                frame = 0

                while success:
                    humans = estimator.inference(image, resize_to_default=True, upsample_size=4.0)  # Generate Poses

                    image_h, image_w = image.shape[:2]
                    for human in humans:
                        for i in range(common.CocoPart.Background.value):
                            if i not in human.body_parts.keys():
                                continue

                            body_part = human.body_parts[i]
                            center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
                            signal[frame][i] = center

                            if frame > 0:
                                signal_acceleration[frame][i] = np.subtract(center, signal[frame - 1][i])
                            else:
                                signal_acceleration[frame][i] = (0, 0)

                        break  # only one human per frame atm

                    success, image = vidcap.read()
                    frame += 1

                with open(filename_signal, "w") as text_file:
                    print(signal.tolist(), file=text_file)

                with open(filename_signal_acc, "w") as text_file:
                    print(signal_acceleration.tolist(), file=text_file)

                #print("[VID2SIGNAL] Validating Conversion:")
                #body_part_0 = [] #list of x,y tuples
                #body_part_1 = []
                #for j in range(MAX_FRAMENR):
                #    # body_part_0_x.append((j, signal[j][0]))
                #    body_part_0.append(signal_acceleration[j][0])
                #    body_part_1.append(signal_acceleration[j][1])
                #    print("[VID2SIGNAL] Frame " + str(j) + " Bodypart0", signal[j][0], signal_acceleration[j][0])


                f = plt.figure(figsize=(8, 40))
                #f.subplots_adjust(top=1.1)
                plt.suptitle(
                    "video " + subfolder + "/" + file + '\nx (blue) and y (orange) acceleration in each frame for different body parts',
                    y=1)
                for p in range(18):
                    signal_by_bodypart = []
                    for j in range(MAX_FRAMENR):
                        signal_by_bodypart.append(signal_acceleration[j][p])
                    plt.subplot(18, 1, p+1)
                    plt.ylim(np.nanmin(signal_acceleration), np.nanmax(signal_acceleration))
                    plt.ylabel("body_part "+str(p))
                    plt.plot(signal_by_bodypart)

                plt.xlabel("frame nr.")
                # plt.show()
                f.savefig(filename_signal_acc_pdf, bbox_inches='tight')

                #plt.subplot(body_part_1)
                #plt.title("body_part 1 x (blue) and y (orange) acceleration in each frame")
                #plt.show()

            else:
                print("[VID2SIGNAL] Skipping already converted file " + subfolder + "/" + file)


def read_saved_signals(video_as_signal_folder):
    """
    Funktion zum Einlesen der gespeicherten Signale.

    :param video_as_signal_folder: Ordner, in dem die Signale gespeichert sind
    :return: Datenstruktur, die Signale für alle Videos des Datensatzes enthält; Array, das die verfügbaren Labels enthält und Array, das Signale und Labels zuordnet.
    """
    maxfilesperfolder = 400000 #limit for debugging, set to high number (9999999) to disable or small  number (3) to test
    labels = [] #eg. ["walking", "running", "jumping"]
    dataset = [] #contains a row for every video. Structure: value=dataset[videoindex][frameindex][joint_x_or_y_signal]
    dataset_ground_truth = [] #contains id of label in that {videoindex] of the database, id is index of labels list
    print("Now reading saved signals")
    os.chdir(os.path.join(os.path.dirname(__file__), video_as_signal_folder))

    signal_folder_subfolders = next(os.walk('.'))[1]

    for subfolder in signal_folder_subfolders:
        os.chdir(os.path.join(os.path.dirname(__file__), video_as_signal_folder, subfolder))

        try:
            labelindex = labels.index(subfolder)
        except ValueError: #if subfolder not in labels:
            labelindex = len(labels)
            labels.append(subfolder)

        cnt = 0
        for file in glob.glob("*.signal_acc.txt"):
            if cnt < maxfilesperfolder:
                filename = os.path.basename(file)
                with open(filename, 'r') as f:
                    signal_acc_str = f.read().replace('\n', '').replace('nan', 'None') #converting str to list fails with NaN using None instead
                    signal_acc = ast.literal_eval(signal_acc_str)
                    #print(filename, signal_acc)
                    # convert back: None to NaN
                    for frame_index, frame_value in enumerate(signal_acc):
                        for body_part_index, body_part_value in enumerate(frame_value):
                            for pos_index, pos_value in enumerate(body_part_value):
                                if pos_value is None:
                                    #signal_acc[frame_index][body_part_index][pos_index] = np.nan #TODO Hotfix for NAN problems
                                    signal_acc[frame_index][body_part_index][pos_index] = 0
                    # convert x and y to seperate signals by extracting all y values from this array into signal_acc_y and all x to signal_axx_x
                    signal_acc_x = [] #enthält nr_of_frames viele arrays die jeweils für jedes gelenk einen x-wert haben
                    signal_acc_y = []
                    for frame_index, frame_value in enumerate(signal_acc):
                        if frame_index < 100: #TODO This is the hotfix for  maximum video length see "Preparation" below
                            x_values = []
                            y_values = []
                            for body_part_index, body_part_value in enumerate(frame_value):
                                x_values.append(body_part_value[0])
                                y_values.append(body_part_value[1])
                            signal_acc_x.append(x_values)
                            signal_acc_y.append(y_values)
                    #print(signal_acc_y)
                    new_signal_acc = signal_acc_x + signal_acc_y # old: (x,y) = [frame][bodypartnr], new: x = [frame][bodypartnr] with bodypartnr < nr_of_bodyparts and y = [frame][bodypartnr] with bodypartnr >= nr_of_bodyparts
                    #print(filename, new_signal_acc)
                    dataset.append(np.asarray(new_signal_acc))
                    dataset_ground_truth.append(labelindex + 1) #TODO This +1 lets us use onehot encodinglater
                cnt += 1

    return labels, dataset, dataset_ground_truth


def retrain_with_signals(video_as_signal_folder):
    """
    Funktion zum Einlesen und Klassifizieren von Signalen

    :param video_as_signal_folder: Ordner, der die zu klassifizierenden Ergebnisse enthält.
    :return: None, gibt Trainignsergebnisse (genauigkeit und wahrheitsmatrix) in Konsole aus und speichert diese in den Ordner ./output/
    """
    # Config
    validation_split = 0.3 #Percentage of Dataset to use for valdiation

    # Read Dataset
    labels, dataset, dataset_ground_truth = read_saved_signals(video_as_signal_folder)
    print("LABELS", labels)
    print("DATASET", len(dataset), len(dataset_ground_truth))#, dataset)

    # Split Dataset and its groundtruth values into Validation and Training Data
    train_dataset                   = np.asarray(dataset[:int(len(dataset)*(1-validation_split))])                 #equal to X_train in original source code
    train_dataset_ground_truth_o    = np.asarray(dataset_ground_truth[:int(len(dataset)*(1-validation_split))])
    validation_dataset              = np.asarray(dataset[int(len(dataset)*(1-validation_split)):])                 #equal to X_test  in original source code
    validation_dataset_ground_truth_o= np.asarray(dataset_ground_truth[int(len(dataset)*(1-validation_split)):])


    # Train Neural Network

    # -----------------------------------
    # Step 1: Preparing Data (TODO: This is not ideal)
    # -----------------------------------
    #currently the net only accepts equal-length-signals. All longer signals are therefore cut off.
    # we need to find the shortest video and use its amount of frames to cut off.
    # minimum_signal_length =
    # as a hot fix, we cut off while reading above #TODO

    # The Network needs the ground truth in the form [[4] [4] [4] ... [1] [1] [1]] currently it has the form [4 4 4 ... 1 1 1]
    train_dataset_ground_truth = np.zeros((len(train_dataset_ground_truth_o), 1), dtype=int)                       #equal to y_train in original source code
    validation_dataset_ground_truth = np.zeros((len(validation_dataset_ground_truth_o), 1), dtype=int)             #equal to y_test  in original source code

    for index, value in np.ndenumerate(train_dataset_ground_truth_o):
        train_dataset_ground_truth[index][0] = value
    for index, value in np.ndenumerate(validation_dataset_ground_truth_o):
        validation_dataset_ground_truth[index][0] = value


    print("TRAIN_DATASET", len(train_dataset), len(train_dataset_ground_truth))#, train_dataset)
    print("VALIDATION_DATASET", len(validation_dataset), len(validation_dataset_ground_truth))#, validation_dataset)

    X_train = train_dataset
    X_test= validation_dataset
    y_train = train_dataset_ground_truth
    y_test = validation_dataset_ground_truth - 1 # fix for labels starting at 1 instead of 0

    # -----------------------------------
    # Step 2: define parameters for model
    # -----------------------------------

    # Input Data

    training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
    test_data_count = len(X_test)  # 2947 testing series
    n_steps = len(X_train[0])  # 128 timesteps per series
    n_input = len(X_train[0][0])  # 9 input parameters per timestep

    # LSTM Neural Network's internal structure

    n_hidden = 32  # Hidden layer num of features
    n_classes = len(labels)  # Total classes (should go up, or should go down)

    # Training

    learning_rate = 0.0025
    lambda_loss_amount = 0.0015
    training_iters = training_data_count * 300  # Loop 300 times on the dataset
    batch_size = 1500 #1500
    display_iter = 30000  # To show test set accuracy during training

    # Some debugging info
    print(y_test)
    print("Some useful info to get an insight on dataset's shape and normalisation:")
    print("(X shape, y shape, every X's mean, every X's standard deviation)")
    print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))
    print("The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.")

   # print(validation_dataset_ground_truth)
   # print("Some useful info to get an insight on dataset's shape and normalisation:")
   # print("features shape, labels shape, each features mean, each features standard deviation")
   # print(validation_dataset.shape, validation_dataset_ground_truth.shape,
   #       np.mean(validation_dataset), np.nanstd(validation_dataset))
   # print("the dataset is therefore properly normalised, as expected.")

    # ------------------------------------------------------
    # Step 3: Let's get serious and build the neural network
    # ------------------------------------------------------

    # Graph input/output
    x = tf.placeholder(tf.float32, [None, n_steps, n_input])  #Shape (?, 20, 18)
    y = tf.placeholder(tf.float32, [None, n_classes])         #Shape (?, 4)

    # Graph weights
    weights = {
        'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])),  # Hidden layer weights
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([n_hidden])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    pred = LSTM_RNN(x, weights, biases, n_input, n_steps, n_hidden)

    # Loss, optimizer and evaluation
    l2 = lambda_loss_amount * sum(
        tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
    )  # L2 loss prevents this overkill neural network to overfit the data
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) + l2  # Softmax loss
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Adam Optimizer

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # now train
    # To keep track of training's performance
    test_losses = []
    test_accuracies = []
    train_losses = []
    train_accuracies = []

    # Launch the graph
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    init = tf.global_variables_initializer()
    sess.run(init)

    # Perform Training steps with "batch_size" amount of example data at each loop
    step = 1
    while step * batch_size <= training_iters:
        batch_xs = extract_batch_size(X_train, step, batch_size)
        batch_ys = one_hot(extract_batch_size(y_train -1, step, batch_size))
        print("x", batch_xs.shape)
        print("y",batch_ys.shape)

        # Fit training using batch data
        _, loss, acc = sess.run(
            [optimizer, cost, accuracy],
            feed_dict={
                x: batch_xs,
                y: batch_ys
            }
        )
        train_losses.append(loss)
        train_accuracies.append(acc)

        # Evaluate network only at some steps for faster training:
        if (step * batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters):
            # To not spam console, show training accuracy/loss in this "if"
            print("Training iter #" + str(step * batch_size) + \
                  ":   Batch Loss = " + "{:.6f}".format(loss) + \
                  ", Accuracy = {}".format(acc))

            # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
            print("x_itereval", X_test.shape)
            #y_test = y_test - 1 #fix for labels starting at 1 instead of 0
            print("y_itereval", y_test.shape, y_test)
            print("y_itereval_onehot", one_hot(y_test).shape, one_hot(y_test))
            loss, acc = sess.run(
                [cost, accuracy],
                feed_dict={
                    x: X_test,
                    y: one_hot(y_test)
                }
            )
            test_losses.append(loss)
            test_accuracies.append(acc)
            print("PERFORMANCE ON TEST SET: " + \
                  "Batch Loss = {}".format(loss) + \
                  ", Accuracy = {}".format(acc))

        step += 1

    print("Optimization Finished!")

    # Accuracy for test data

    one_hot_predictions, accuracy, final_loss = sess.run(
        [pred, accuracy, cost],
        feed_dict={
            x: X_test,
            y: one_hot(y_test)
        }
    )

    test_losses.append(final_loss)
    test_accuracies.append(accuracy)

    print("FINAL RESULT: " + \
          "Batch Loss = {}".format(final_loss) + \
          ", Accuracy = {}".format(accuracy))


    #
    #   graphical plot
    #

    # (Inline plots: )
    # %matplotlib inline


    width = 8
    height = 5
    f1 = plt.figure(figsize=(width, height))

    indep_train_axis = np.array(range(batch_size, (len(train_losses)+1)*batch_size, batch_size))
    plt.plot(indep_train_axis, np.array(train_losses),     "b--", label="Fehler beim Training (Train losses)")
    plt.plot(indep_train_axis, np.array(train_accuracies), "g--", label="Genauigkeit beim Training (Train accuracies)")

    indep_test_axis = np.append(
        np.array(range(batch_size, len(test_losses)*display_iter, display_iter)[:-1]),
        [training_iters]
    )
    plt.plot(indep_test_axis, np.array(test_losses),     "b-", label="Fehler beim Test (Test losses)")
    plt.plot(indep_test_axis, np.array(test_accuracies), "g-", label="Genauigkeit beim Test (Test accuracies)")

    #plt.title("Fortschritt der Trainingseinheit über die Iterationen")
    plt.legend(loc='upper right') #, shadow=True)
    plt.ylabel('Fortschritt des Trainings (Fehler- und Genauigkeits-Werte)')
    plt.xlabel('Trainings-Iteration')

    #plt.show()
    os.chdir(os.path.dirname(sys.argv[0]))
    f1.savefig("output/signal_accuracy.pdf", bbox_inches='tight')

    #
    #   confusion matrix
    #

    # Results

    predictions = one_hot_predictions.argmax(1)

    print("Testing Accuracy: {}%".format(100 * accuracy))

    print("")
    print("Precision: {}%".format(100 * precision_score(y_test, predictions, average="weighted")))
    print("Recall: {}%".format(100 * recall_score(y_test, predictions, average="weighted")))
    print("f1_score: {}%".format(100 * f1_score(y_test, predictions, average="weighted")))

    print("")
    print("Confusion Matrix:")
    confmatrix = confusion_matrix(y_test, predictions)
    print(confmatrix)

    # Plot Results:

    # Compute confusion matrix
    confmatrix = np.asanyarray(confmatrix)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    f2 = plt.figure(figsize=(8, 6))  # few labels
    f2 = plt.figure(figsize=(6, 4))  # very few labels
    # f2 = plt.figure(figsize=(12,8)) #many labels
    plot_confusion_matrix(confmatrix, classes=labels,
                          title='Wahrheitsmatrix (confusion matrix)')

    #plt.show()
    f2.savefig("output/signal_confusion_matrix.pdf", bbox_inches='tight')
