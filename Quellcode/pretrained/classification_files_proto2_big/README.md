Dieser Ordner enthält ein vortrainiertes Modell, dass für den Handlungsklassifikator eingesetzt werden kann.

Beim Training dieses Modells wurden die Videos des für diese Arbeit zusammengesetzten Datensatzes überlagert (Prototyp 2).

Zum Start der Webcam-Demo mit diesem vortrainierten Modell kann der folgende Befehl genutzt werden:

``python main.py --folder_classification_files="pretrained/classification_files_proto2_big" --classify_webcam_asImgs=True``