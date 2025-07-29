import optuna
import sys

# ==============================================================================
#                 RETTUNGSSKRIPT FÜR "ZOMBIE"-TRIALS
# ==============================================================================
#
# Problem: Wenn Optuna-Worker abrupt beendet werden (z.B. durch pkill oder
#          Schließen des Terminals), bleibt ihr Status in der Datenbank auf
#          'RUNNING'. Das Haupt-Skript 'run_study.py' verweigert dann den
#          Start, um Konflikte zu vermeiden.
#
# Lösung: Dieses Skript findet alle Trials im Status 'RUNNING' und setzt
#         sie manuell auf 'FAIL', damit die Studie fortgesetzt werden kann.
#
# ==============================================================================

# --- WICHTIG: Passen Sie diese Werte an Ihre Konfiguration an! ---
# Diese müssen mit den Werten in `run_study.py` übereinstimmen.
STUDY_NAME = "eisbach_grand_96"
STORAGE_NAME = "sqlite:///optuna_study.db"

def fix_stale_running_trials(study_name, storage_name):
    """Findet 'RUNNING' Trials und setzt sie auf 'FAIL'."""
    print(f"Lade Studie '{study_name}' aus '{storage_name}'...")
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_name)
    except KeyError:
        print(f"FEHLER: Studie '{study_name}' nicht in der Datenbank gefunden. Bitte Namen überprüfen.")
        sys.exit(1)

    running_trials = [t for t in study.get_trials(deepcopy=False) if t.state == optuna.trial.TrialState.RUNNING]

    if not running_trials:
        print("Keine 'RUNNING' Trials gefunden. Alles in Ordnung.")
        return

    print(f"\n{len(running_trials)} 'RUNNING' Trial(s) gefunden. Setze Status auf 'FAIL'...")
    for trial in running_trials:
        # KORREKTUR: study.tell() erwartet die Trial-Nummer (ein Integer),
        # nicht das ganze FrozenTrial-Objekt.
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
        print(f"  -> Trial #{trial.number} wurde auf FAIL gesetzt.")

    print("\nAufräumen abgeschlossen. Du kannst die Studie jetzt neu starten.")

if __name__ == "__main__":
    fix_stale_running_trials(STUDY_NAME, STORAGE_NAME)