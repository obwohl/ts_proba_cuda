#!/bin/bash

# ==============================================================================
#                 UMFASSENDES PROJEKT-AUFRÄUM-SKRIPT
# ==============================================================================
#
# Dieses Skript führt mehrere Aufräumaktionen durch, um Speicherplatz
# freizugeben und das Projekt sauber zu halten.
#
# Es fragt vor jeder potenziell destruktiven Aktion nach Bestätigung.
#
# ==============================================================================

set -e # Beendet das Skript bei einem Fehler

echo "--- Starte umfassende Projekt-Bereinigung ---"

# Funktion für Ja/Nein-Abfragen
ask_yes_no() {
    while true; do
        read -p "$1 [y/n]: " yn
        case $yn in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Bitte mit 'y' oder 'n' antworten.";;
        esac
    done
}

# 1. Python-Bytecode-Cache löschen (sicher)
echo -e "\n[1/5] Lösche Python-Cache-Dateien (*.pyc, __pycache__)..."
python3 clean.py

# 2. Git-Repository optimieren (sicher)
echo -e "\n[2/5] Führe Git Garbage Collection durch..."
git gc --prune=now --aggressive

# 3. Optuna-Ergebnisordner löschen (potenziell destruktiv)
if [ -d "results" ]; then
    du -sh results
    if ask_yes_no "Soll der Ordner 'results' mit allen Trial-Ergebnissen gelöscht werden?"; then
        rm -rf results
        echo "-> 'results' wurde gelöscht."
    fi
fi

# 4. Optuna-Log-Ordner löschen (potenziell destruktiv)
if [ -d "logs" ]; then
    du -sh logs
    if ask_yes_no "Soll der Ordner 'logs' mit allen Worker-Logs gelöscht werden?"; then
        rm -rf logs
        echo "-> 'logs' wurde gelöscht."
    fi
fi

# 5. Optuna-Datenbank löschen (potenziell destruktiv)
if [ -f "optuna_study.db" ]; then
    du -sh optuna_study.db
    if ask_yes_no "Soll die Optuna-Datenbank 'optuna_study.db' gelöscht werden?"; then
        rm -f optuna_study.db
        echo "-> 'optuna_study.db' wurde gelöscht."
    fi
fi

echo -e "\n--- ✅ Bereinigung abgeschlossen. ---"