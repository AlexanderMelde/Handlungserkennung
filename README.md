# Erkennung menschlicher Handlungen durch Auswertung der Körperhaltungen von Personen in einem Video mithilfe von Machine Learning und neuronalen Netzen
_Bachelorarbeit_
<table>
<tr><th>Autor</th><td>Alexander Melde (7939560)</td></tr>
<tr><th>Betreuer</th><td>Dr. M. Sieck, EnBW AG<br/>Prof. Dr. M. Babilon, DHBW Stuttgart</td></tr>
<tr><th>Studiengang/Kurs</th><td>B. Sc. Angewandte Informatik – Kommunikationsinformatik TINF15K</td></tr>
<tr><th>Titel der Arbeit</th><td>Erkennung menschlicher Handlungen durch Auswertung der Körperhaltungen von Personen in einem Video mithilfe von Machine Learning und neuronalen Netzen</td></tr>
<tr><th>Anlass</th><td>Bachelorarbeit, 3. Studienjahr</td></tr>
<tr><th>Bearbeitungszeitraum</th><td>11.06.2018 - 31.08.2018</td></tr>
<tr><th>Abgabedatum</th><td>03.09.2018</td></tr>
</table>

## Kurzbeschreibung
> Dank bedeutender Forschungsergebnisse in den Bereichen der künstlichen Intelligenz und digitalen Bildverarbeitung ist es Computern mithilfe von künstlichen neuronalen Netze möglich, Personen in Videos zu detektieren und deren Körperhaltungen abzuschätzen.
> 
> In dieser Arbeit wird geprüft, ob durch Auswertung dieser Körperhaltungen menschliche Handlungen erkannt werden können.
> 
> Die selbstständige Klassifikation von Videos oder gar Handlungen in Videos durch einen Computer ist ein noch nicht gelöstes Problem, zu dem noch viel geforscht wird. Um die komplexen Zusammenhänge, Lösungsvorschläge und Implementierungen zu verstehen, werden in dieser Arbeit zunächst einige Grundlagen aus den Bereichen Videoüberwachung, digitale Bildverarbeitung und künstliche Intelligenz erarbeitet. Anschließend werden zahlreiche Ansätze zur Handlungsklassifikation in Videos miteinander verglichen und bewertet.
> 
> Damit die künstliche Intelligenz angelernt werden kann, werden darüber hinaus zahlreiche Datensätze mit beschrifteten Handlungen aufgezeigt.
> 
> Ausgehend von den Ergebnissen dieser Untersuchung werden anschließend mehrere im Rahmen dieser Arbeit entwickelter Prototypen zur Handlungserkennung vorgestellt. Für einen produktiven Einsatz werden abschließend Erweiterungs- und Optimierungs-Möglichkeiten gezeigt.
> 
> In jedem Schritt wurden darüber hinaus Optimierungen hinsichtlich des Anwendungsfall „Überwachung von öffentlichen Plätzen“ geprüft. Durch die Haltungserkennung sollen Gewaltsituationen in Videoüberwachungs-Streams erkannt und Aktionen wie Schläge oder Tritte von normalen Alltagshandlungen unterschieden werden können.


## Veröffentlichung
Nach Abschluss des Prüfungsverfahrens soll unter dieser Adresse der im Rahmen der Bachelorarbeit geschriebene Quelltext zur Handlungserkennung veröffentlicht werden.


## Installation

1) Python und Pip installieren
    1) Python 3.6 [herunterladen](https://www.python.org/downloads/) und installieren (beinhaltet pip)
2) Virtual Environment erstellen mit allen in der Datei ``requirements.txt`` genannten Modulen
    1) Das Modul "Virtual Environment" installieren mit ``pip install virtualenv``
    2) Neue virtuelle Umgebung im Projektordner erstellen mit ``virtualenv venv``
    3) Virtuelle Umgebung betreten, indem im Ordner ``venv/Scripts/`` ausgeführt wird: 
        1) Unix (Bash): ``.\activate``
        2) Windows (PowerShell): ``PowerShell.exe -ExecutionPolicy UNRESTRICTED`` und ``.\Activate.ps1``
    4) Module in virtuelle Umgebung installieren mit ``pip install -r requirements.txt``
3) Die Bibliothek ```pafprocess``` kompilieren
    1) Compiler SWIG installieren
        1) Unix: ``sudo apt install swig``
        2) Windows:
            1) SWIG herunterladen und Installieren [Download](http://www.swig.org/download.html)
            2) Visual C++ Build Tools von Microsoft installieren via [Direktlink](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=15) oder [Download-Website besuchen](https://visualstudio.microsoft.com/downloads), "Tools für Visual Studio 2017" auswählen und bei "Build Tools für Visual Studio 2017" auf "Herunterladen" drücken
    1) Im Ordner ``tpe/tf_pose/pafprocess`` die Befehle ``swig -python -c++ pafprocess.i`` und ``python setup.py build_ext --inplace`` ausführen.


## Schnellstart

Nach der Installation kann der zweite Prototyp in zwei Schritten mit einer Webcam getestet werden:

1) Nach Handlungen sortierte Videos in Unterordner von ```training/input_videos``` legen, jeweils mindestens 20 Videos im .avi Format z.B. 30 Videos  im Ordner ``training/input_videos/boxing`` und 25 Videos im Ordner ``training/input_videos/walking``.
    1) z.B. [Download des KTH-Datensatzes](http://www.nada.kth.se/cvap/actions/)
    2) z.B. [Download des Weizmann-Datensatzes](http://www.wisdom.weizmann.ac.il/~vision/SpaceTimeActions.html)
2) Prototyp 2 testen mit verkettetem Ablauf ``python main.py --convertVideosToOneImage=True --retrain=True --train_with_vid_as_img=True --classify_webcam_asImgs=True`` oder einzeln:
    1) Generierung von Überlagerungen starten mit ``python main.py --convertVideosToOneImage=True``
    2) Das Training des zweiten Prototypen starten mit ``python main.py --retrain=True --train_with_vid_as_img=True``
    3) Anwendung auf Webcam starten mit ``python main.py --classify_webcam_asImgs=True``

Weitere Aufrufe siehe Bachelorarbeit und Hilfe via ``python main.py -h``


## Kontakt
E-Mail: alexander@melde.net
