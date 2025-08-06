import subprocess
import time
import optuna
import json
import os
import sys
import uuid
import signal

# -- Studien-Konfiguration --
STUDY_NAME = "preci_short"
STORAGE_NAME = "sqlite:///optuna_study.db"  # Fester DB-Name. Studien werden intern durch STUDY_NAME unterschieden.

# -- Parallelisierungs-Konfiguration --
# Wie viele parallele Python-Prozesse (Trials) sollen gestartet werden?
NUM_PARALLEL_TRIALS = 1
# Wie viele CPU-Worker soll JEDER Trial für den DataLoader verwenden?
WORKERS_PER_TRIAL = 0
# NEU: Eine Verzögerung zwischen dem Start der Worker, um DB-Race-Conditions zu entschärfen.
# Dies ist eine zusätzliche Sicherheitsmaßnahme zu den enqueued placeholder trials.
DELAY_BETWEEN_WORKERS_S = 10

# -- Warm-Start-Konfiguration --
# Setze auf den Namen der alten Studie, von der die besten Trials übernommen werden sollen.
# Auf `None` setzen, um diese Funktion zu deaktivieren.
WARM_START_FROM_OLD_STUDY = None
NUM_WARM_START_TRIALS_FROM_STUDY = 0
# Pfad zur JSON-Datei für manuelle Warm-Starts. Wird ignoriert, wenn WARM_START_FROM_OLD_STUDY gesetzt ist.
WARM_START_JSON_FILE = "optuna_warm_starts.json"

# --- 2. SKRIPT-LOGIK (Normalerweise nicht anzupassen) ---

def prepare_study(study_name, storage_name, num_parallel_workers, warm_start_study, num_warm_trials, warm_start_json):
    """Erstellt die Studie und fügt bei Bedarf einmalig Warm-Start-Trials hinzu."""
    print("==================================================")
    print(f"Vorbereiten der Optuna-Studie '{study_name}'...")
    print("==================================================")

    try:
        optuna.delete_study(study_name=study_name, storage=storage_name)
        print(f"INFO: Vorhandene Studie '{study_name}' wurde gelöscht.")
    except:
        pass # Study didn't exist, which is fine.

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        load_if_exists=False, # Should be False now
    )

    # --- ROBUSTER FIX GEGEN RACE CONDITIONS ---
    # Wir prüfen den Zustand der Studie, um zu entscheiden, was zu tun ist.
    all_trials = study.get_trials(deepcopy=False)
    waiting_trials = [t for t in all_trials if t.state == optuna.trial.TrialState.WAITING]
    running_trials = [t for t in all_trials if t.state == optuna.trial.TrialState.RUNNING]

    if running_trials:
        print(f"WARNUNG: {len(running_trials)} Trial(s) haben den Status 'RUNNING'.")
        print("Dies könnte auf einen anderen, noch aktiven Studienlauf hindeuten. Breche ab, um Konflikte zu vermeiden.")
        sys.exit(1)

    # --- VERBESSERUNG: Flexiblerer Warm-Start ---
    # Die Warm-Start-Logik wird nun immer ausgeführt, wenn konfiguriert (und keine
    # Studie aktiv läuft). Das `skip_if_exists=True` in `enqueue_trial` verhindert
    # das Hinzufügen von Duplikaten und macht das Skript robuster für das
    # Fortsetzen von Studien.
    
    # Zähle wartende Trials VOR dem Warm-Start, um den Effekt zu messen.
    waiting_trials_before = len(waiting_trials)

    try:
        # Option 1: Aus einer alten Studie
        if warm_start_study:
            print(f"\nVersuche Warm-Start aus der vorherigen Studie '{warm_start_study}'...")
            old_study = optuna.load_study(study_name=warm_start_study, storage=storage_name)
            completed_trials = sorted(
                [t for t in old_study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None],
                key=lambda t: t.value
            )
            if not completed_trials:
                print("  -> WARNUNG: Alte Studie hat keine abgeschlossenen Trials.")
            else:
                trials_to_enqueue = completed_trials[:num_warm_trials]
                print(f"Füge die besten {len(trials_to_enqueue)} Trials aus '{warm_start_study}' zur Warteschlange hinzu...")
                for i, old_trial in enumerate(trials_to_enqueue):
                    study.enqueue_trial(old_trial.params, skip_if_exists=True)
                print(f"  -> Versuch, {len(trials_to_enqueue)} Konfiguration(en) hinzuzufügen (Duplikate werden übersprungen).")

        # Option 2: Aus JSON-Datei
        elif os.path.exists(warm_start_json):
            print(f"\nVersuche Warm-Start aus '{warm_start_json}'...")
            with open(warm_start_json, 'r') as f:
                warm_start_configs = json.load(f)
            print(f"Füge {len(warm_start_configs)} initiale Konfigurationen zur Warteschlange hinzu...")
            for i, params in enumerate(warm_start_configs):
                study.enqueue_trial(params, skip_if_exists=True)
            print(f"  -> Versuch, {len(warm_start_configs)} Konfiguration(en) hinzuzufügen (Duplikate werden übersprungen).")
    except Exception as e:
        print(f"  -> FEHLER während des Warm-Starts: {e}. Überspringe Warm-Start.")

    # Nach dem Warm-Start die Anzahl der wartenden Trials neu abfragen, um den Effekt zu sehen.
    waiting_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.WAITING])
    waiting_trials_after = len(waiting_trials)
    if (num_added := waiting_trials_after - waiting_trials_before) > 0:
        print(f"  -> {num_added} neue Konfiguration(en) wurden durch den Warm-Start zur Warteschlange hinzugefügt.")

    # --- DEFINITIVER FIX GEGEN RACE CONDITIONS ---
    # Wenn nach dem Warm-Start nicht genügend Trials in der Warteschlange sind, um alle
    # Worker zu beschäftigen, füllen wir die Warteschlange mit leeren Platzhaltern auf.
    # Optuna wird diese Platzhalter an die Worker verteilen, die dann den Sampler
    # verwenden, um die Parameter zu generieren. Dies bricht die Symmetrie und verhindert,
    # dass mehrere Worker gleichzeitig den Sampler für den "ersten" Trial befragen.
    num_to_add = num_parallel_workers - waiting_trials_after
    if num_to_add > 0:
        print(f"\nDie Warteschlange hat {waiting_trials_after} Trial(s). Füge {num_to_add} hinzu, um alle {num_parallel_workers} Worker auszulasten.")
        for i in range(num_to_add):
            # KORREKTUR: Füge einen EINDEUTIGEN Platzhalter hinzu.
            # Ein leeres dict `{}` ist für Optuna nicht eindeutig und führt dazu, dass alle Worker
            # versuchen, den gleichen "ersten" Trial zu starten. Ein einzigartiger (aber für das
            # Modell irrelevanter) Parameter zwingt Optuna, separate Trial-Einträge in der DB
            # zu erstellen und die Race Condition wird so an der Wurzel verhindert.
            study.enqueue_trial({'_placeholder_id': str(uuid.uuid4())})
            print(f"  -> Eindeutiger Platzhalter-Trial #{i+1} zur Warteschlange hinzugefügt.")

def run_workers(study_name, storage_name, num_trials, workers_per_trial):
    """Startet die Worker-Prozesse im Hintergrund."""
    print("\n==================================================")
    print(f"Starte Optuna-Studie mit {num_trials} parallelen Workern.")
    print(f"Jeder Worker wird mit {workers_per_trial} DataLoader-Workern konfiguriert.")
    print("==================================================")

    processes = []
    for i in range(num_trials):
        print(f"Starte Worker #{i+1}...")
        env = os.environ.copy()
        env["TRIAL_WORKERS"] = str(workers_per_trial)

        # Der Worker-Prozess `optuna_full_search.py` kümmert sich selbst um sein Logging.
        process = subprocess.Popen(
            [
                sys.executable,  # Benutze den gleichen Python-Interpreter
                "optuna_full_search.py",
                "--study-name", study_name,
                "--storage-name", storage_name
            ],
            env=env,
            preexec_fn=os.setsid # WICHTIG: Startet den Worker in einer neuen Prozessgruppe
        )
        processes.append(process)

        # Füge eine Verzögerung hinzu, um den Workern Zeit zu geben, sich zu initialisieren
        # und einen Trial aus der DB zu holen, bevor der nächste Worker startet.
        if i < num_trials - 1:
            print(f"  -> Warte {DELAY_BETWEEN_WORKERS_S} Sekunden, um den Start zu staffeln und Race Conditions zu vermeiden...")
            time.sleep(DELAY_BETWEEN_WORKERS_S)

    print(f"\nAlle {len(processes)} Worker wurden gestartet.")
    print("Die Log-Dateien werden im Verzeichnis 'logs/<study_name>/' von den Workern selbst erstellt.")
    print("Überwache die GPU-Auslastung mit: watch nvidia-smi")
    print("Überwache die Prozesse mit: htop")
    print("\nDrücke Strg+C, um dieses Skript zu beenden und alle Worker-Prozesse zu terminieren.")

    try:
        for p in processes:
            p.wait()  # Warte, bis alle Prozesse beendet sind.
    except KeyboardInterrupt:
        print("\nStrg+C erkannt. Terminiere alle Worker-Prozesse...")
        for p in processes:
            try:
                # Sende SIGTERM an die gesamte Prozessgruppe, um auch Kindprozesse (DataLoader) zu beenden.
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except ProcessLookupError:
                # Der Prozess ist möglicherweise bereits beendet.
                pass
        print("Alle Worker-Prozesse wurden beendet.")

if __name__ == "__main__":
    prepare_study(STUDY_NAME, STORAGE_NAME, NUM_PARALLEL_TRIALS, WARM_START_FROM_OLD_STUDY, NUM_WARM_START_TRIALS_FROM_STUDY, WARM_START_JSON_FILE)
    run_workers(STUDY_NAME, STORAGE_NAME, NUM_PARALLEL_TRIALS, WORKERS_PER_TRIAL)
    print("\nAlle Worker-Prozesse sind beendet. Die Studie ist abgeschlossen.")