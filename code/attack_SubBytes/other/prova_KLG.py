import numpy as np
from scipy.stats import pearsonr
import TRS_Reader

def hamming_weight(x):
    """Calcola il peso di Hamming per un array di valori interi."""
    return np.array([bin(val).count("1") for val in x])

def attack_subbytes_input_by_bits(traces, plaintexts, num_traces, start_sample=0, end_sample=100):
    """
    Attacco per recuperare la chiave attaccando l'input della SubBytes,
    scegliendo la chiave con la correlazione massima.

    Args:
        traces (np.array): Tracce di consumo energetico (tracce x campioni).
        plaintexts (np.array): Array dei plaintexts (tracce x byte).
        num_traces (int): Numero di tracce da utilizzare.
        start_sample (int): Campione iniziale della finestra temporale.
        end_sample (int): Campione finale della finestra temporale.

    Returns:
        np.array: Chiave recuperata (16 byte).
    """
    num_samples = end_sample - start_sample  # Numero di campioni nella finestra
    key_guess = np.zeros(16, dtype=int)  # Array per memorizzare la chiave recuperata

    for byte_index in range(16):  # Per ogni byte della chiave
        print(f"\nAttaccando il byte {byte_index}...")
        max_corr = -1  # Valore iniziale per la correlazione massima
        best_guess = 0  # Migliore ipotesi di chiave

        for kg in range(256):  # Itera su tutte le ipotesi di chiave (0-255)
            # Calcola l'input della SubBytes per l'ipotesi kg
            input_subbytes = plaintexts[:num_traces, byte_index] ^ kg
            leakage_model = hamming_weight(input_subbytes)  # Peso di Hamming

            # Calcola la correlazione per ogni campione nella finestra
            correlations = [
                pearsonr(leakage_model, traces[:num_traces, sample])[0]
                for sample in range(start_sample, end_sample)
            ]

            # Trova la correlazione massima per questa ipotesi di chiave
            current_max_corr = max(abs(corr) for corr in correlations)

            # Debug: Stampa la correlazione per questa ipotesi
            print(f"Ipotesi chiave {kg:02X}: Correlazione massima = {current_max_corr:.4f}")

            # Aggiorna la migliore ipotesi se la correlazione è maggiore
            if current_max_corr >= max_corr:
                max_corr = current_max_corr
                best_guess = kg

        # Memorizza l'ipotesi con la correlazione massima
        key_guess[byte_index] = best_guess
        print(f"Byte {byte_index}: Miglior ipotesi chiave = {best_guess:02X}, Correlazione massima = {max_corr:.4f}")

    return key_guess





# Leggi il file TRS        

trs_filename = "TinyAES_625Samples_FirstRoundSbox.trs"
trs_filename = "Projects/1 - Analysis of a simple AES dataset/code/base/TinyAES_625Samples_FirstRoundSbox.trs"

trs = TRS_Reader.TRS_Reader(trs_filename)
trs.read_header()
trs.read_traces(1000, 0, trs.number_of_samples)

# Attacco
recovered_key = attack_subbytes_input_by_bits(trs.traces, trs.plaintext, num_traces=1000)
print("Chiave recuperata:", " ".join(f"{byte:02X}" for byte in recovered_key))

