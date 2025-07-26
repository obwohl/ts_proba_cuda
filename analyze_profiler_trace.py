import json
import argparse
from collections import defaultdict
import pandas as pd
import sys

def find_device_pids(trace_events):
    """
    Finds the Process IDs (PIDs) associated with device (CUDA/MPS) streams.
    These are identified by looking for 'process_name' metadata events.
    """
    device_pids = set()
    is_mps = False
    for event in trace_events:
        if event.get('ph') == 'M' and event.get('name') == 'process_name':
            process_name = event.get('args', {}).get('name', '').lower()
            # Common names for GPU/MPS streams in profiler traces
            if 'stream' in process_name or 'cuda' in process_name:
                device_pids.add(event['pid'])
            if 'mps' in process_name or 'metal' in process_name:
                is_mps = True
                # For MPS, device work often runs under the main PID
    return device_pids, is_mps

def analyze_trace(file_path):
    """
    Loads a PyTorch profiler JSON trace, analyzes it, and prints a summary.
    """
    print(f"--- Lade und parse die Trace-Datei: {file_path} ---")
    print("Dies kann bei großen Dateien einen Moment dauern...")
    try:
        with open(file_path, 'r') as f:
            trace_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Fehler: Die JSON-Datei konnte nicht geparst werden. Sie könnte beschädigt sein. Details: {e}", file=sys.stderr)
        return
    except FileNotFoundError:
        print(f"Fehler: Datei nicht gefunden unter {file_path}", file=sys.stderr)
        return

    trace_events = trace_data.get('traceEvents', [])
    if not trace_events:
        print("Fehler: Keine Trace-Events in der JSON-Datei gefunden.", file=sys.stderr)
        return

    print("--- Identifiziere Device-Streams (CUDA/MPS) ---")
    device_pids, is_mps = find_device_pids(trace_events)
    if device_pids:
        print(f"Gefundene Device-PIDs: {list(device_pids)}")
    elif is_mps:
        print("Info: MPS-Trace erkannt. GPU-Zeit wird unter der CPU-Zeit des aufrufenden Threads erfasst und kann nicht separat ausgewiesen werden.")
    else:
        print("Warnung: Konnte keine spezifischen Device-PIDs (z.B. für CUDA) identifizieren. GPU-Zeit könnte unvollständig sein.")

    # defaultdict zur einfachen Aggregation der Statistiken
    op_stats = defaultdict(lambda: {
        'self_cpu_us': 0,
        'self_gpu_us': 0,
        'calls': 0
    })
    
    dataloader_us = 0
    memory_events = 0

    print("--- Aggregiere Event-Daten ---")
    # Filtere nach 'Complete'-Events ('X'), da diese Zeitdauern haben
    for event in filter(lambda e: e.get('ph') == 'X', trace_events):
        name = event.get('name', 'Unknown')
        duration_us = event.get('dur', 0)
        pid = event.get('pid')
        category = event.get('cat', 'Unknown')

        # Aggregiere DataLoader-Zeit separat
        if 'DataLoader' in category:
            dataloader_us += duration_us
            continue
        
        if 'memory' in category.lower():
            memory_events += 1
            continue

        # Aggregiere Operator-Statistiken
        op_stats[name]['calls'] += 1
        if pid in device_pids or 'kernel' in category.lower():
            op_stats[name]['self_gpu_us'] += duration_us
        else:
            op_stats[name]['self_cpu_us'] += duration_us

    if not op_stats:
        print("Fehler: Es konnten keine Operator-Statistiken aggregiert werden. Der Trace ist möglicherweise leer oder hat ein unerwartetes Format.", file=sys.stderr)
        return

    # --- In ein Pandas DataFrame konvertieren für einfachere Analyse ---
    df = pd.DataFrame.from_dict(op_stats, orient='index')
    df['total_self_us'] = df['self_cpu_us'] + df['self_gpu_us']
    df['avg_self_us'] = df['total_self_us'] / df['calls']
    df['avg_cpu_us'] = df['self_cpu_us'] / df['calls']
    df['avg_gpu_us'] = df['self_gpu_us'] / df['calls']

    # Mikrosekunden in Millisekunden umrechnen für bessere Lesbarkeit
    for col in ['self_cpu_us', 'self_gpu_us', 'total_self_us', 'avg_self_us', 'avg_cpu_us', 'avg_gpu_us']:
        df[col.replace('_us', '_ms')] = df[col] / 1000
        df.drop(columns=[col], inplace=True)

    # --- Zusammenfassungen ausgeben ---
    total_cpu_time_ms = df['self_cpu_ms'].sum()
    total_gpu_time_ms = df['self_gpu_ms'].sum()
    total_dataloader_ms = dataloader_us / 1000
    
    print("\n\n" + "="*80)
    print(" " * 26 + "PROFILER-ANALYSE-ZUSAMMENFASSUNG")
    print("="*80)
    
    print(f"\n--- Allgemeine Zeitverteilung ---")
    print(f"Gesamte CPU-Zeit (Operatoren): {total_cpu_time_ms:,.2f} ms")
    print(f"Gesamte GPU/MPS-Zeit (Kernels): {total_gpu_time_ms:,.2f} ms")
    print(f"Gesamte DataLoader-Zeit:       {total_dataloader_ms:,.2f} ms")
    print(f"Anzahl der Speicher-Events:    {memory_events:,}")
    
    if total_dataloader_ms > (total_cpu_time_ms + total_gpu_time_ms) * 0.1:
        print("\n[!] WARNUNG: Die DataLoader-Zeit ist signifikant. Dies könnte auf einen I/O-Engpass hindeuten.")

    # --- Top-N-Berichte ---
    top_n = 20
    
    print(f"\n--- Top {top_n} Operatoren nach Gesamter Self-Zeit (CPU + GPU/MPS) ---")
    print("Dies ist oft die relevanteste Metrik, insbesondere auf Apple Silicon (MPS).")
    print(df.sort_values('total_self_ms', ascending=False).head(top_n)[['total_self_ms', 'self_cpu_ms', 'calls', 'avg_self_ms']].to_string(float_format="%.3f"))

    print(f"\n--- Top {top_n} Operatoren nach Self-CPU-Zeit ---")
    print(df.sort_values('self_cpu_ms', ascending=False).head(top_n)[['self_cpu_ms', 'calls', 'avg_cpu_ms']].to_string(float_format="%.3f"))

    if total_gpu_time_ms > 0:
        print(f"\n--- Top {top_n} Kernels nach Self-GPU/MPS-Zeit ---")
        print(df[df['self_gpu_ms'] > 0].sort_values('self_gpu_ms', ascending=False).head(top_n)[['self_gpu_ms', 'calls', 'avg_gpu_ms']].to_string(float_format="%.3f"))

    print(f"\n--- Top {top_n} am häufigsten aufgerufene Operatoren ---")
    print(df.sort_values('calls', ascending=False).head(top_n)[['calls', 'total_self_ms', 'avg_self_ms']].to_string(float_format="%.3f"))
    
    print("\n" + "="*80)
    print("Analyse abgeschlossen. Bitte kopiere den Text ab 'PROFILER-ANALYSE-ZUSAMMENFASSUNG' und stelle ihn zur Überprüfung bereit.")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analysiert eine PyTorch-Profiler-JSON-Trace-Datei und erstellt eine Zusammenfassung.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "trace_file",
        type=str,
        help="Pfad zur .pt.trace.json Datei, die vom PyTorch Profiler generiert wurde."
    )
    args = parser.parse_args()
    
    analyze_trace(args.trace_file)