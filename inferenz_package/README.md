# DUET-Prob Inferenz-Paket

Dieses Paket enthält alles Notwendige, um Inferenzen mit einem vortrainierten `DUET-Prob`-Modell durchzuführen.

## Struktur

- `run_single_forecast.py`: Das Hauptskript zur Ausführung einer Vorhersage.
- `requirements.txt`: Eine Liste aller notwendigen Python-Pakete.
- `checkpoints/`: Enthält die trainierte Modelldatei (`best_model.pt`).
- `ts_benchmark/`: Enthält den Python-Quellcode, der die Modellarchitektur definiert.

## Benutzung

Folgen Sie diesen Schritten auf der Maschine, auf der Sie die Inferenz ausführen möchten.

### 1. Python-Umgebung einrichten

Es wird dringend empfohlen, eine virtuelle Umgebung zu verwenden, um Konflikte mit anderen Python-Projekten zu vermeiden.

```bash
# Eine neue virtuelle Umgebung erstellen
python3 -m venv venv

# Die Umgebung aktivieren
# Unter Linux/macOS:
source venv/bin/activate
# Unter Windows:
# venv\Scripts\activate
```

### 2. Abhängigkeiten installieren

Installieren Sie alle erforderlichen Pakete mit der bereitgestellten `requirements.txt`-Datei.

```bash
pip install -r requirements.txt
```

### 3. Inferenz ausführen

Führen Sie das Skript `run_single_forecast.py` aus. Sie müssen drei Argumente angeben:
- `--checkpoint`: Der Pfad zu Ihrer Modell-Checkpoint-Datei.
- `--data-file`: Der Pfad zu den neuen CSV-Daten, die Sie für die Vorhersage verwenden möchten.
- `--output-csv`: Der Pfad, unter dem die resultierende Vorhersage-CSV gespeichert werden soll.

**Beispiel:**

```bash
python run_single_forecast.py \
    --checkpoint checkpoints/best_model.pt \
    --data-file /pfad/zu/ihren/neuen_daten.csv \
    --output-csv /pfad/zu/ihrer/vorhersage_ausgabe.csv
```

Das Skript wird dann:
1. Das Modell und die Daten laden.
2. Die letzten `seq_len` Datenpunkte aus Ihrer CSV als Eingabe verwenden.
3. Die nächsten `horizon` Zeitschritte vorhersagen.
4. Die Quantil-Vorhersagen in die angegebene Ausgabe-CSV-Datei speichern.

