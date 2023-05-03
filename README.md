# Vorhersage von Flusspegelständen mithilfe von hydrologischen Rohdaten und einem Recurrent Neural Network


## Einleitung

Spätestens seit der Hochwasserkatastrophe 2021 in Deutschland und der generellen Zunahme durch den Klimawandel induzierter Extremwetterereignisse wie z.B. Starkregen ist klar, dass eine genaue Vorhersage von
Flusspegelständen lebensrettend ist (BMUV 2022).

In Nordrhein-Westfalen (NRW) werden hydrologische Rohdaten, wie zum Beispiel Pegelstände, Wassertemperatur und Niederschlag, durch das Landesamt für Natur, Umwelt und Verbraucherschutz (LANUV) auf der
Webseite HYGON (Hydrologische Rohdaten Online) dargestellt und zusätzlich eine Auswahl an Rohdaten über
OpenGeodata.NRW zum Download bereitgestellt. Grundsätzlich erfolgt die Darstellung aller Rohdaten in Form
von Diagrammen, die beispielsweise einen gemessenen Pegelstand als 7-Tage-Ganglinie abbilden. Zudem
stehen interaktive Karten zur Verfügung, welche einen Überblick über alle Messstationen bieten.

Neben den beschriebenen Funktionalitäten werden jedoch keine Vorhersagen über zukünftige Werte der bereitgestellten hydrologischen Rohdaten angeboten, obwohl die Prognose zukünftiger Flusspegelstände auf Basis
der bereitgestellten Daten erheblich zur Hochwasservorsorge sowie -schutz beitragen könnte. Durch ein Gutachten zur Verbesserung der Hochwasservorhersage für mittlere und kleine Fließgewässer in NRW, aufgegeben durch den Landtag NRW im Kontext der Hochwasserkatastrophe 2021, wird ein Bedarf jenes Angebot
bestätigt und unter anderem neuronale Netze als Lösungsansatz ermittelt (Mudersbach et al. 2022).

Die vorliegende Projektarbeit füllt somit eine bestehende Lücke, indem sie ein Modell zur Vorhersage zukünftiger Flusspegelstände der Erft an der Messstation Neubrück entwickelt und präsentiert. Das entwickelte Modell
basiert auf einem Recurrent Neural Network (RNN) und kann theoretisch auf andere Messstationen angewendet werden. Für die Prognose werden in einem multivariaten Verfahren verschiedene Messwerte aus dem Einzugsgebiet der Erft verwendet, um dem RNN viele mögliche Einflussfaktoren des Pegelstandes am Flussende
in Neubrück bereitzustellen.
Im weiteren Untersuchungsprozess stellen sich folgende Forschungsfragen:
* In welcher Genauigkeit können Prognosen für den Pegelstand in einem Zeitraum von 24 Stunden getroffen
werden?
* Nimmt die Genauigkeit der Prognose mit Zunahme des Vorhersagezeitraums ab?

---

## Methodisches Vorgehen

Für die Datenverarbeitung werden hauptsächlich die Bibliotheken
„pandas“ und „numpy“ verwendet, das neuronale Netz wird mit der Bibliothek „Keras“ von TensorFlow erstellt.

### Download & Import der Rohdaten
Die täglich aktualisierten hydrologischen Rohdaten, bestehend aus Niederschlagswerten, Wassertemperaturen sowie Pegelständen, werden von OpenGeodata.NRW zum Download angeboten. Diese tagesaktuellen
Dateien werden bei Ausführung des Skripts heruntergeladen und entpackt, um so als Data-Frames eingelesen
werden zu können.

### Data-Preprocessing
Am Anfang des Skriptes wird die Messstation des Pegelstandes, welcher vorhergesagt werden soll, sowie alle
Messstationen, die für die Prognose benötigt werden, definiert. Die eingelesenen Daten erfordern ein Preprocessing, um von dem Recurrent Neural Network verarbeitet werden zu können.

Da Messwerte sowohl in stündlichen als auch in viertelstündlichen Intervallen vorliegen, müssen die Daten über
ein Resampling auf eine gleiche Intervallgröße gebracht werden. Über eine Verknüpfung der Tabellen werden
nun die eingangs definierten Messwerte selektiert und in eine neue Tabelle übertragen. Da neuronale Netze
nicht mit fehlenden Messwerten arbeiten können, müssen mögliche Lücken mit Daten gefüllt werden. Dafür
wird eine Interpolier-Methode angewendet, die alle Datenlücken bereinigt.

Ein RNN benötigt eine Skalierung der Daten, um effektiv trainiert werden zu können. Die Daten werden skaliert,
um sicherzustellen, dass ihre Merkmale in einem bestimmten Wertebereich liegen, der von der Aktivierungsfunktion des RNN unterstützt wird (zumeist im Wertebereich -1 bis 1).
Da das RNN mit sequenziellen Daten arbeitet, werden die erforderlichen Sequenzen folgend aus der Wertetabelle erstellt. Dabei werden als Eingabe sowohl X- als auch Y-Sequenzen benötigt (Graves 2014: 2-4). Eine
Sequenz an X-Daten besteht dabei aus 96 Zeitschritten (d.h. in diesem Fall 96 Stunden) der Messwerte aller
Messstationen, die Y-Daten aus den darauffolgenden 24 Zeitschritten der zu vorhersagenden Pegelstände.

### Kompilierung & Training des RNNs
Um die Leistung eines RNN-Modells zu evaluieren, zu optimieren und sicherzustellen, dass es in der Lage
ist, Muster in neuen Daten zu erkennen, werden Trainings- und Testdaten benötigt. Das Trainingsset wird verwendet, um das Modell anzupassen, während der Testdatensatz die Leistung des Modells auf neuen Daten
evaluiert und Überanpassung („Overfitting“) vermeidet (Nami 2020). In diesem Fall werden 80% der Daten als
Trainingsdaten und 20% als Testdaten verwendet.

Bevor das Modell trainiert wird, erfordert es eine Kompilierung des Models. Dabei werden die Input-, Hidden-
und Outputlayer erstellt und dem Modell hinzugefügt. Außerdem wird ein Optimizer festgelegt, welcher während
des Trainings die Gewichtung der Neuronen anpasst, um somit die Leistung des Modells zu verbessern. Das
Training umfasst dabei mehrere Epochen, in welchen der Trainingsprozess mit allen Trainingsdaten durchgeführt wird. Ziel des Trainingsprozesses ist es, den Fehler zwischen den Vorhersagen des Modells und den
tatsächlichen Werten zu minimieren. Dies kann mithilfe von einer mathematischen Funktion (Loss-Funktion)
überprüft werden (Salehinejad et al. 2018: 3).

### Vorhersage der Daten & Evaluation
Das trainierte RNN kann nun Vorhersagen für den gewünschten Pegelstand auch aus neuen, unbekannten
Daten treffen. Wichtig ist, dass die im vorherigen durchgeführte Skalierung wieder rückgängig gemacht werden
muss. Das Modell gibt für jede Zeitsequenz jetzt eine Liste von den vorhergesagten Pegelständen der folgenden 24 Zeitschritte aus. Abbildung 5 zeigt dabei beispielhaft eine Vorhersage des Pegelstandes der nächsten
24 Stunden.

---

## Recurrent Neural Network (RNN)

> Ein Recurrent Neural Network (RNN) ist eine Art neuronaler Netze, das verschiedene Arten von sequentiellen
Daten wie z.B. Zeitreihendaten oder natürliche Sprache verarbeiten kann. Es verfügt über ein „Gedächtnis“
(memory), mit dem es Informationen aus früheren Zeitschritten behalten und weitergeben kann. Diese Funktionalität ermöglicht es, Vorhersagen oder Entscheidungen auf der Grundlage des Kontexts der gesamten Sequenz und nicht nur des aktuellen Zeitschritts zu treffen (Amidi & Amidi o.J., Mwiti 2022).

> Das „Gedächtnis“ wird durch einen Zustandsvektor (hidden state) repräsentiert, der bei jeder neuen Eingabe
aktualisiert wird. Dadurch, dass das RNN über Sequenzen von Vektoren operieren kann, lassen sich verschiedene Eingabe-Ausgabe-Szenarien modellieren, wie zum Beispiel many-to-one, one-to-many und many-to-many. Dies ist ein großer Vorteil gegenüber anderen Arten von neuronalen Netzen, wie beispielsweise dem „Con-
volutional Neural Network“, welche eine feste Größe des Input- und Outputvektor definieren (Karpathy 2015).

> Aus der beschriebenen Funktionsweise eines RNN ergeben sich jedoch auch einige Einschränkungen und
Probleme. Darunter gehört z.B. das Problem der vanishing oder exploading gradients. Dadurch, dass die Backpropagation für jeden Zeitschritt angewendet wird, kann es bei größeren RNN dazu kommen, dass die Gradienten exponentiell sehr klein bis hin zu null (vanishing) oder sehr groß bis hin zu unendlich (exploading) werden
können. Dies wirkt sich somit auf die Gewichte aus und führt dazu, dass das neuronale Netz nicht mehr lernt
(vanishing) oder überlastet wird (exploading) (McCullum o.J.). Um diesen Problemen entgegenzuwirken, wurden verschiedene Methoden entwickelt, unter anderem die auf einem Gating-Mechanismus basierenden Gated
Recurrent Unit (GRU) und Long Short-Term Memory (LSTM) (Mwiti 2022, PHI 2018).

---

## Kofiguration & Evaluierung des RNN

Ein Recurrent Neural Network ist mit vielen Einstellungsmöglichkeiten konfigurierbar. In der Projektarbeit wurden in einem iterativen Prozess verschiedene Einstellungen getestet, um so einen möglichst geringen Fehlerwert zu erhalten. Das beste Ergebnis erzielte in diesem Fall ein neuronales Netz mit einem LSTM Input-Layer
mit 128 Neuronen, zwei LSTM Hidden-Layern mit 64 Neuronen und einem Dense Output Layer. Außerdem wurden diesem Prozess verschiedene Aktivierungsfunktionen (tanh, ReLU, Sigmoid), Optimizer (SGD, Adam, RMSprop) sowie unterschiedliche Batchsizes und Epochenwiederholungen getestet (Salehinejad et al. 2018: 3–8). Schlussendlich wurde die tanh-Aktivierungsfunktion auf die LSTM-Layer angewendet
und das Modell mit dem Adam-Optimizer kompiliert. Das Training wurde in 27 Epochen mit einer Batchsize von
8 durchgeführt und so ein Mean Squared Error (MSE) von 0,016 erreicht. In Abbildung 2 werden Loss-Funktionen (Mean Squared Error) für die Trainingsprozesse drei verschiedener RNN-Konfigurationen verglichen.
Dabei lässt sich erkennen, dass der Fehlerwert mit der Anzahl der Neuronen der Layer abnimmt.

Außerdem wird für die Evaluierung der vorhergesagten Messwerte ein Vergleich zwischen den vorhergesagten
und realen Messwerten vorgenommen. In Abbildung 4 wird die Genauigkeit des ersten (t+1) und letzten Vorhersagewerts (t+24) verglichen („first step“ und „last step“). So lässt sich beurteilen, wie die Vorhersagegenauigkeit
mit zunehmender Zeit abnimmt.

Für die weitere Beurteilung werden der Mean Absolute Percentage Error (MAPE) und der Mean Squared Error
hinzugezogen. Der MAPE gibt die durchschnittliche prozentuale Abweichung des vorhergesagten Wertes aus
(Khair et al. 2017). Der erste Schritt hat dabei einen MAPE von 1,9%, während der letzte Schritt einen MAPE
von 2,4% aufweist. Da der MAPE mögliche Extremwerte nicht ausreichend abbilden kann, wird zusätzlich die
quadratische Abweichung (MSE) berechnet (Wallach & Goffinet 1989). Dabei liegt der MSE des ersten Schrittes
bei 3,3 und der MSE des zweiten Schrittes bei 3,8. Da das Modell mit aktuellen Daten arbeitet und der Trainingsprozess bei jeder Ausführung des Skriptes variiert, können sich die Genauigkeit und Fehlerwerte mit jeder
Ausführung ebenfalls verändern.

---

## Fazit

Durch die Erstellung und Training eines auf einem Recurrent Neural Network basierenden Modells zur Prognose von zukünftigen Pegelständen der Erft an der Messstation Neubrück können die im Vorhinein aufgestellten
Forschungsfragen wie folgt beantwortet werden:

**In welcher Genauigkeit können Prognosen für den Pegelstand in einem Zeitraum von 24 Stunden getroffen werden?**

> Der Pegelstand der Erft an dieser Messstelle liegt etwa im Bereich von 70-110 cm, sodass die Vorhersagegenauigkeit auch nach 24 Stunden mit einer mittleren Abweichung von 2,4% im Bereich weniger Zentimeter
liegt. Dabei zeigt ein geringer MSE von 3,8, dass das Auftreten möglicher falscher Extremwerte unwahrscheinlich ist.
Leider werden die Messwerte nur für einen begrenzten Zeitraum (ca. 8 Wochen) online bereitgestellt, wodurch kein Zugriff auf ältere Datenbestände besteht. Wenn der Zugriff auf größere Datenmengen möglich
wäre, könnten die Trainingsdaten vergrößert und die Genauigkeit des Modells weiter verbessert werden.

**Nimmt die Genauigkeit der Prognose mit Zunahme des Vorhersagezeitraums ab?**

> Die Ergebnisse zeigen, dass die Vorhersage mit zunehmenden Zeitschritten ungenauer wird. Klar ist, dass
ein Pegelstand in einem Zeitschritt (eine Stunde) nur begrenzt von dem vorherigen Messwert abweichen
kann, sodass das RNN so eine höhere Genauigkeit erzielt. Jedoch ist der Vorhersagewert nach 24 Stunden
mit einer mittleren Abweichung von 2,4% nur geringfügig ungenauer als der Wert nach einer Stunde (1,9%).

Die durchgeführte Untersuchung stellt dabei vielversprechende Potenziale der Nutzung von neuronalen Netzen im Hochwasserschutz dar und bestätigt das in der Einleitung erwähnte Gutachten von Mudersbach et al.
(2022). Zugleich zeigt die Untersuchung weitere mögliche Forschungsbereiche auf. So ist das trainierte Modell
auf den Einsatz für eine kurzzeitige Prognose (bis 24 Stunden) optimiert, wobei das Verhalten bei längeren
Prognosen (z. B. 3-7 Tage) unerforscht bleibt. RNN könnten so die Bevölkerung noch früher vor drohenden Gefahren warnen. Außerdem könnte durch den Einsatz weiterer Variablen, wie z. B. Bodenfeuchte, Evaporation
oder Grundwasserstand, die Genauigkeit der Vorhersage verbessert werden.

---

## Literaturverzeichnis

* Amidi, A.; Amidi, S. (Hg.) (o.J.): Deep Learning. https://stanford.edu/~shervine/teaching/cs230/cheatsheet-recurrent-neural-networks [28.04.2023]
* Bundesministerium für Umwelt, Naturschutz, nukleare Sicherheit und Verbraucherschutz(BMUV)(Hg.)(2022): Hitze, Dürre, Starkregen: Über 80 Milliarden Euro Schäden durch Extremwetter in
Deutschland. https://www.bmuv.de/pressemitteilung/hitze-duerre-starkregen-ueber-80-milliarden-euro-schaeden-durch-extremwetter-in-deutschland [18.04.2023].
* Graves, A. (2014): Generating Sequences With Recurrent Neural Networks.
* Khair, U., Fahmi, H., Hakim, S., Rahim, R. (2017): Forecasting Error Calculation with Mean Absolute Deviation and Mean Absolute Percentage Error. In: Journal of Physics: Conference Series
930, S. 12002.
* Karpathy, A. (Hg.) (2015): The Unreasonable Effectiveness of Recurrent Neural Networks.
http://karpathy.github.io/2015/05/21/rnn-effectiveness/ [29.04.2023].
* McCullum, N. (Hg.) (o.J.): The Vanishing Gradient Problem in Recurrent Neural Networks.
https://www.nickmccullum.com/python-deep-learning/vanishing-gradient-problem/ [29.04.2023].
* Mudersbach, C., Leandro, J., Reggiani, P. (2022): Gutachten. Verbesserung der Hochwasservorhersage für mittlere und kleine Fließgewässer, insbesondere im Mittelgebirge von Nordrhein-
Westfalen. https://www.landtag.nrw.de/files/live/sites/landtag-r20/files/Internet/I.A.1/PUA/PUA_II/Gutachten%20Prof.%20Leandro%2C%20Prof.%20Mudersbach%20und%20Prof.%20Reggiani.
pdf [30.04.2023].
* Mwiti, D. (2022): Getting Started with Recurrent Neural Network (RNNs).
https://towardsdatascience.com/getting-started-with-recurrent-neural-network-rnnsad1791206412 [28.04.2023].
* Nami, Y. (2020): Addressing the difference between Keras’ validation_split and sklearn’s train_test_split. Towards better practice for training neural networks. https://towardsdatascience.com/
addressing-the-difference-between-keras-validation-split-and-sklearn-s-train-test-split-a3fb803b733 [30.04.2023].
* Phi, M. (2018): Illustrated Guide to LSTM’s and GRU’s: A step by step explanation.
https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-stepexplanation-44e9eb85bf21 [28.04.2023].
* Salehinejad, H.; Sankar, S.; Barfett, J.; Colak, E.; Valaee, S. (2018): Recent Advances in Recurrent Neural Networks.
* Wallach, D., Goffinet, B. (1989): Mean squared error of prediction as a criterion for evaluating and comparing system models. In: Ecological Modelling 44 (3-4), S. 299–306.