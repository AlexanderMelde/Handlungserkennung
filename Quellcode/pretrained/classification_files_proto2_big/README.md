Dieser Ordner enth�lt ein vortrainiertes Modell, dass f�r den Handlungsklassifikator eingesetzt werden kann.

Beim Training dieses Modells wurden die Videos des f�r diese Arbeit zusammengesetzten Datensatzes �berlagert (Prototyp 2).

Zum Start der Webcam-Demo mit diesem vortrainierten Modell kann der folgende Befehl genutzt werden:

``python main.py --folder_classification_files="pretrained/classification_files_proto2_big" --classify_webcam_asImgs=True``