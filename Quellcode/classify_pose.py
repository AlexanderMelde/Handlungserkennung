import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # disable TF warning about cpu instructions
import tensorflow as tf
import numpy as np
import os

#this function classifies an opencv image
def classify_pose(opencv_image, folder_classification_files):
    """
    Funktion zum Klassifizieren eines Einzelbilds mithilfe des Bild-Klassifikators. (Anwendung des Klassifikators)

    :param opencv_image: Bereits eingelesenes JPG-Bild.
    :param folder_classification_files: Ordner, in dem das Modell des Bildklassifikators liegt
    :return:
    """
    with tf.device('/device:GPU:0'):                        # use gpu instead of cpu ("/cpu:0")
        image_data = np.array(opencv_image)[:, :, 0:3]      # convert opencv image to a numpy array that can be processed by the DecodeJpeg tensor
        tf.reset_default_graph()                            # Reset Graph
        label_lines = [line.rstrip() for line               # Loads label file, strips off carriage return
                       in tf.gfile.GFile(os.path.join(os.path.dirname(__file__),folder_classification_files,"retrained_labels.txt"))]
        graph_def = tf.GraphDef()                           # Unpersists graph from file
        graph_def.ParseFromString(tf.gfile.FastGFile(os.path.join(os.path.dirname(__file__),folder_classification_files,"retrained_graph.pb"), 'rb').read())
        _ = tf.import_graph_def(graph_def, name='')

        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = tf.Session().graph.get_tensor_by_name('final_result:0')
        predictions = tf.Session().run(softmax_tensor, {'DecodeJpeg:0': image_data})
        # predictions = tf.Session().run(softmax_tensor, {'DecodeJPGInput:0': image_data})


        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        results = []
        results_readable = []
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            results_readable.append('%s (score = %.5f)' % (human_string, score))
            results.append((human_string,score))

        return results
