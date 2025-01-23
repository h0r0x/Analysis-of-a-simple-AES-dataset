import TRS_Reader  # Module to read the TRS file
import numpy as np  # For numerical calculations
from scipy.stats import pearsonr  # To calculate the Pearson correlation
import matplotlib.pyplot as plt  # For plotting results

from heapq import nlargest  # Importa per ottenere i primi 5 elementi

class AESAttackSingleBit:
    # Define the AES S-box (SubBytes lookup table)
    sbox = [
        0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5,
        0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
        0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0,
        0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
        0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC,
        0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
        0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A,
        0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
        0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0,
        0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
        0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B,
        0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
        0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85,
        0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
        0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5,
        0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
        0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17,
        0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
        0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88,
        0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
        0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C,
        0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
        0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9,
        0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
        0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6,
        0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
        0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E,
        0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
        0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94,
        0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
        0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68,
        0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
    ]

    # Define the AES encryption key (fixed for the attack simulation)
    key = [
        0x2B, 0x7E, 0x15, 0x16,
        0x28, 0xAE, 0xD2, 0xA6,
        0xAB, 0xF7, 0x15, 0x88,
        0x09, 0xCF, 0x4F, 0x3C,
    ]

    def Sbox(self, X):
        # Apply S-box substitution to each byte in X
        y = np.array([self.sbox[val] for val in X], dtype=int)
        return y

    def ADK(self, X, k):
        # Perform XOR between input X and key candidate k
        y = X ^ k
        return y

    def bitLeakage(self, X, bit_pos):
        # Extract the specified bit from each byte in X
        y = ((X >> bit_pos) & 1).astype(int)
        return y

    def maxCorr(self, bit_values, traces):
        # Find the maximum Pearson correlation between bit values and traces
        corr = np.corrcoef(bit_values, traces, rowvar=False)[0, 1:]
        maxcorr = np.max(np.abs(corr))
        return maxcorr, corr

    def diffOfMeans(self, bit_values, traces):
        # Split traces into two groups based on leakage
        group0 = traces[bit_values == 0, :]
        group1 = traces[bit_values == 1, :]

        # Check for empty groups and handle gracefully
        if group0.shape[0] == 0 or group1.shape[0] == 0:
            return 0, np.zeros(traces.shape[1])  # Return zero difference if any group is empty

        # Compute mean traces for each group
        mean0 = np.mean(group0, axis=0)
        mean1 = np.mean(group1, axis=0)

        # Compute variances for each group
        var0 = np.var(group0, axis=0, ddof=1)
        var1 = np.var(group1, axis=0, ddof=1)

        # Number of samples in each group
        n0 = group0.shape[0]
        n1 = group1.shape[0]

        # Compute standard error
        se = np.sqrt(var0 / n0 + var1 / n1)

        # Avoid division by zero
        se[se == 0] = np.inf

        # Compute t-statistics
        t_stats = (mean1 - mean0) / se

        # Take the maximum absolute t-statistic as the metric
        max_t = np.max(np.abs(t_stats))

        return max_t, t_stats




    def validate_leakage(self, trs, BitLeakage, byteno, Nm):
        """
        Verifica che il modello di perdita corrisponda ai Hamming Weight attesi.
        """
        hw_model = np.array([bin(val).count("1") for val in trs.plaintext[:, byteno]])
        bit_values = BitLeakage[byteno, :, :, 0:Nm]
        
        for bit_pos in range(8):
            hw_corr = np.corrcoef(hw_model, bit_values[:, bit_pos].flatten())[0, 1]
            print(f"Correlation for bit {bit_pos}: {hw_corr:.4f}")


    def find_interesting_points(self, hw_values, traces, num_points=50):
        """
        Trova i punti di massima correlazione con il modello Hamming Weight.
        
        Args:
            hw_values (np.array): Modello di perdita (es. Hamming Weight).
            traces (np.array): Matrice delle tracce di consumo.
            num_points (int): Numero di punti interessanti da selezionare.
        
        Returns:
            np.array: Indici dei punti di massimo interesse.
        """
        correlations = np.zeros(traces.shape[1])
        
        # Calcola la correlazione per ogni campione temporale
        for i in range(traces.shape[1]):
            correlations[i], _ = pearsonr(hw_values, traces[:, i])
        
        # Trova gli indici dei campioni con la massima correlazione
        interesting_points = np.argsort(np.abs(correlations))[-num_points:]
        return interesting_points


    def Initialise(self, N):
        print("Initializing AES attack with single-bit leakage model...")
        trs = TRS_Reader.TRS_Reader("TinyAES_625Samples_FirstRoundSbox.trs")
        trs.read_header()
        trs.read_traces(N, 0, trs.number_of_samples)

        # Prepare arrays to store bit predictions
        BitLeakage = np.zeros((16, 256, 8, N), dtype=int)

        # For each byte of the key (16 total bytes)
        for byteno in range(16):
            X = trs.plaintext[:, byteno]  # Get plaintext byte

            # For each possible key guess (0-255)
            for kg in range(256):
                Y = self.ADK(X, kg)  # Apply AddRoundKey
                _ = self.Sbox(Y)  # Apply S-box transformation

                # For each bit position (0-7)
                for bit_pos in range(8):
                    BitLeakage[byteno, kg, bit_pos] = self.bitLeakage(Y, bit_pos)

        print("Initialization complete.")
        return trs, BitLeakage

    def attackBit(self, trs, BitLeakage, byteno, Nm, distinguisher="correlation", sample_start=0, sample_end=100):
        """
        Attacca un singolo byte della chiave usando solo un intervallo di campioni temporali.
        Args:
            trs: Oggetto TRS contenente le tracce.
            BitLeakage: Modello di perdita dei bit.
            byteno: Indice del byte da attaccare.
            Nm: Numero di tracce da utilizzare.
            distinguisher: Metodo di distinzione ("correlation" o "difference_of_means").
            sample_start: Campione iniziale della fase di interesse.
            sample_end: Campione finale della fase di interesse.
        Returns:
            Recovered byte della chiave.
        """
        print(f"Attacking byte {byteno} using {Nm} traces with {distinguisher} distinguisher.")
        traces = trs.traces[0:Nm, sample_start:sample_end]  # Isola solo i campioni rilevanti
        max_metric = np.zeros(8)  # Metriche massime per ogni bit
        maxkg = np.zeros(8, dtype=int)  # Ipotesi di chiave migliori per ogni bit

        for kg in range(256):  # Itera su tutte le ipotesi di chiave
            for bit_pos in range(8):
                bit_values = BitLeakage[byteno, kg, bit_pos, 0:Nm]  # Valori dei bit per l'ipotesi
                if distinguisher == "correlation":
                    metric, _ = self.maxCorr(bit_values, traces)
                elif distinguisher == "difference_of_means":
                    metric, _ = self.diffOfMeans(bit_values, traces)
                else:
                    raise ValueError("Invalid distinguisher selected.")

                # Aggiorna la metrica massima se necessario
                if metric > max_metric[bit_pos]:
                    max_metric[bit_pos] = metric
                    maxkg[bit_pos] = kg

        # Seleziona il valore di chiave più frequente tra le ipotesi con la massima metrica
        most_frequent_key = np.bincount(maxkg).argmax()
        print(f"Recovered key byte {byteno}: 0x{most_frequent_key:02X}")
        return most_frequent_key


if __name__ == "__main__":
    # Parametri principali
    N_traces_total = 1000  # Numero totale di tracce
    aes_attack = AESAttackSingleBit()
    trs, BitLeakage = aes_attack.Initialise(N_traces_total)

    sample_start, sample_end = 0, 200  # Intervallo di campioni temporali
    distinguisher = "correlation"  # Metodo di distinzione
    recovered_key = np.zeros(16, dtype=int)

    for byteno in range(16):
        recovered_key[byteno] = aes_attack.attackBit(
            trs, BitLeakage, byteno, Nm=N_traces_total, distinguisher=distinguisher, 
            sample_start=sample_start, sample_end=sample_end
        )

    print("Recovered key:")
    print(" ".join(f"{byte:02X}" for byte in recovered_key))
