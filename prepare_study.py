import optuna
import json
import os

# --- KONFIGURATION (muss mit optuna_full_search.py übereinstimmen) ---
STUDY_NAME = "eisbach_airtemp_pressure_focus_auf_wassertemp"
STORAGE_NAME = "sqlite:///optuna_study.db"

# Konfiguration für den Warm-Start
WARM_START_FROM_OLD_STUDY = None
NUM_WARM_START_TRIALS_FROM_STUDY = 5
USE_WARM_START_FROM_JSON = True

def main():
    """
    Bereitet die Optuna-Studie vor: Erstellt sie, falls sie nicht existiert,
    und fügt einmalig die Warm-Start-Trials hinzu.
    """
    print("==================================================")
    print(f"Vorbereiten der Optuna-Studie '{STUDY_NAME}'...")
    print("==================================================")

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_NAME,
        direction="minimize",
        load_if_exists=True,
    )

    # Führe die Warm-Start-Logik nur aus, wenn die Studie brandneu ist (keine Trials).
    # Dies verhindert, dass bei jedem Neustart weitere Trials hinzugefügt werden.
    if len(study.get_trials(deepcopy=False)) == 0:
        print("Die Studie ist neu. Versuche, Warm-Start-Trials hinzuzufügen...")

        # Option 1 (bevorzugt): Automatisierter Warm-Start aus einer vorherigen Studie
        if WARM_START_FROM_OLD_STUDY:
            print(f"\nVersuche Warm-Start aus der vorherigen Studie '{WARM_START_FROM_OLD_STUDY}'...")
            try:
                old_study = optuna.load_study(study_name=WARM_START_FROM_OLD_STUDY, storage=STORAGE_NAME)
                completed_trials = sorted(
                    [t for t in old_study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None],
                    key=lambda t: t.value
                )
                if not completed_trials:
                    print("  -> WARNUNG: Alte Studie hat keine abgeschlossenen Trials.")
                else:
                    trials_to_enqueue = completed_trials[:NUM_WARM_START_TRIALS_FROM_STUDY]
                    print(f"Füge die besten {len(trials_to_enqueue)} Trials aus '{WARM_START_FROM_OLD_STUDY}' zur Warteschlange hinzu...")
                    for i, old_trial in enumerate(trials_to_enqueue):
                        study.enqueue_trial(old_trial.params, skip_if_exists=True)
                        print(f"  -> Konfiguration #{i+1} hinzugefügt (von altem Trial #{old_trial.number})")
            except Exception as e:
                print(f"  -> FEHLER beim Laden der alten Studie: {e}. Starte ohne Warm-Start.")

        # Option 2: Manuelles Einreihen von Trials aus einer JSON-Datei
        elif USE_WARM_START_FROM_JSON:
            warm_start_file = "optuna_warm_starts.json"
            print(f"\nVersuche Warm-Start aus '{warm_start_file}'...")
            try:
                with open(warm_start_file, 'r') as f:
                    warm_start_configs = json.load(f)
                print(f"Füge {len(warm_start_configs)} initiale Konfigurationen zur Warteschlange hinzu...")
                for i, params in enumerate(warm_start_configs):
                    study.enqueue_trial(params, skip_if_exists=True)
                    print(f"  -> Konfiguration #{i+1} hinzugefügt.")
            except Exception as e:
                print(f"  -> FEHLER beim Lesen der JSON-Datei: {e}. Starte ohne Warm-Start.")
    else:
        print("Studie existiert bereits und hat Trials. Überspringe das Hinzufügen von Warm-Start-Trials.")

    print("\nStudien-Vorbereitung abgeschlossen.")

if __name__ == "__main__":
    main()