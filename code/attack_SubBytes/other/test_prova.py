import numpy as np
from scipy.stats import pearsonr
import TRS_Reader

class AESAttackInputSubBytes:
    def __init__(self, trs_filename):
        self.trs_filename = trs_filename

    def calculate_leakage(self, plaintext, key_guess):
        """
        Calcola l'input della SubBytes dato il plaintext e una ipotesi di chiave.
        """
        return plaintext ^ key_guess  # Input della SubBytes = plaintext XOR key_guess

    def hamming_weight(self, values):
        """
        Calcola il peso di Hamming di un array di valori.
        """
        return np.array([bin(v).count("1") for v in values])

    def attack_byte(self, plaintexts, traces, byte_index):
        """
        Attacca un singolo byte della chiave basandosi sull'input della SubBytes.
        """
        max_correlation = -1
        best_guess = -1

        for key_guess in range(256):  # Per ogni possibile ipotesi di chiave
            input_subbytes_guess = self.calculate_leakage(plaintexts[:, byte_index], key_guess)
            leakage_model = self.hamming_weight(input_subbytes_guess)

            # Calcola la correlazione tra leakage simulato e tracce misurate
            correlations = [pearsonr(leakage_model, traces[:, t])[0] for t in range(traces.shape[1])]
            max_corr = max(abs(c) for c in correlations)

            if max_corr > max_correlation:
                max_correlation = max_corr
                best_guess = key_guess

        return best_guess

    def attack(self, num_traces):
        """
        Esegue l'attacco su tutti i byte della chiave.
        """
        trs = TRS_Reader.TRS_Reader(self.trs_filename)
        trs.read_header()
        trs.read_traces(num_traces, 0, trs.number_of_samples)

        plaintexts = trs.plaintext
        traces = trs.traces

        key_guess = np.zeros(16, dtype=int)

        for byte_index in range(16):
            key_guess[byte_index] = self.attack_byte(plaintexts, traces, byte_index)
            print(f"Recovered byte {byte_index}: {key_guess[byte_index]:02X}")

        print(f"Recovered Key: {' '.join(f'{b:02X}' for b in key_guess)}")
        return key_guess



import matplotlib.pyplot as plt
import TRS_Reader

def plot_multiple_traces(traces, num_traces=100, sample_start=0, sample_end=None):
    """
    Visualizza le prime `num_traces` tracce.
    Args:
        traces (np.array): Matrice delle tracce (righe: tracce, colonne: campioni temporali).
        num_traces (int): Numero di tracce da visualizzare.
        sample_start (int): Campione iniziale da visualizzare.
        sample_end (int): Campione finale da visualizzare.
    """
    if sample_end is None:
        sample_end = traces.shape[1]

    plt.figure(figsize=(12, 6))
    for i in range(min(num_traces, traces.shape[0])):  # Evita di superare il numero effettivo di tracce
        plt.plot(range(sample_start, sample_end), traces[i, sample_start:sample_end], label=f"Trace {i+1}" if i < 5 else "")

    plt.title(f"Prime {num_traces} Tracce")
    plt.xlabel("Campioni Temporali")
    plt.ylabel("Consumo")
    plt.grid()
    plt.show()

# Leggi le tracce da un file TRS
trs_filename = "TinyAES_625Samples_FirstRoundSbox.trs"
trs = TRS_Reader.TRS_Reader(trs_filename)
trs.read_header()
trs.read_traces(100, 0, trs.number_of_samples)  # Leggi 100 tracce dal file

# Visualizza le prime 100 tracce
plot_multiple_traces(trs.traces, num_traces=100)


'''
if __name__ == "__main__":
    trs_filename = "TinyAES_625Samples_FirstRoundSbox.trs"
    num_traces = 1000

    aes_attack = AESAttackInputSubBytes(trs_filename)
    aes_attack.attack(num_traces)
'''