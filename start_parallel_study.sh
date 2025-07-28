#!/bin/bash

# ==============================================================================
#  Automatisierter Start-Skript für parallele Optuna-Trials
# ==============================================================================

# --- KONFIGURATION ---
# Wie viele parallele Python-Prozesse (Trials) sollen gestartet werden?
# - Bei GPU-Nutzung: Idealerweise 1 pro verfügbarer GPU.
# - Bei reiner CPU-Nutzung: Hängt von der Anzahl der CPU-Kerne ab.
#   Ein guter Startwert ist (Anzahl der CPU-Kerne / WORKERS_PER_TRIAL).
NUM_PARALLEL_TRIALS=4

# Wie viele CPU-Worker soll JEDER Trial für den DataLoader verwenden?
# Dies beschleunigt das Laden der Daten für jeden einzelnen Trial.
# Gute Faustregel: (Gesamt-CPU-Kerne / NUM_PARALLEL_TRIALS).
# Beispiel: 16 Kerne / 4 parallele Trials -> WORKERS_PER_TRIAL=4
# WICHTIG: Stellen Sie sicher, dass die Gesamtzahl der Worker
# (NUM_PARALLEL_TRIALS * WORKERS_PER_TRIAL) nicht die Anzahl
# Ihrer CPU-Kerne bei weitem übersteigt, um "Thrashing" zu vermeiden.
WORKERS_PER_TRIAL=4

# --- SKRIPT-LOGIK ---

# NEU: Führe das Vorbereitungsskript einmalig aus, um die Studie zu erstellen und Warm-Starts zu enqueuen.
echo "Führe das Vorbereitungsskript aus..."
python prepare_study.py

echo "=================================================="
echo "Starte Optuna-Studie mit $NUM_PARALLEL_TRIALS parallelen Trials."
echo "Jeder Trial wird mit $WORKERS_PER_TRIAL DataLoader-Workern konfiguriert."
echo "=================================================="

for i in $(seq 1 $NUM_PARALLEL_TRIALS)
do
   echo "Starte Trial-Worker #$i ..."
   # Starte den Python-Prozess im Hintergrund mit nohup
   # Das Logging wird jetzt vom Python-Skript selbst gehandhabt.
   TRIAL_WORKERS=$WORKERS_PER_TRIAL nohup python optuna_full_search.py &
done

echo -e "\nAlle $NUM_PARALLEL_TRIALS Worker wurden gestartet."
echo "Überwache die GPU-Auslastung mit: watch nvidia-smi"
echo "Überwache die Prozesse mit: htop"
echo "Die Log-Dateien werden im Verzeichnis 'logs/<study_name>/' gespeichert."