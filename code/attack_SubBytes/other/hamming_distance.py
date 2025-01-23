import TRS_Reader  # Module to read the TRS file
import numpy as np  # For numerical calculations
from scipy.stats import pearsonr  # To calculate the Pearson correlation
from heapq import nlargest  # Importa per ottenere i primi 5 elementi
import csv  # Per salvare i risultati in un file CSV
from sklearn.metrics import mutual_info_score


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

    def Sbox(self, X):
        y = np.array([self.sbox[val] for val in X], dtype=int)
        return y

    def ADK(self, X, k):
        y = X ^ k
        return y

    def bitLeakage(self, X, bit_pos):
        y = ((X >> bit_pos) & 1).astype(int)
        return y

    def maxCorr(self, bit_values, traces):
        corr = np.corrcoef(bit_values, traces, rowvar=False)[0, 1:]
        maxcorr = np.max(np.abs(corr))
        return maxcorr, corr

    def diffOfMeans(self, bit_values, traces):
        group0 = traces[bit_values == 0, :]
        group1 = traces[bit_values == 1, :]

        if group0.shape[0] == 0 or group1.shape[0] == 0:
            return 0, np.zeros(traces.shape[1])

        mean0 = np.mean(group0, axis=0)
        mean1 = np.mean(group1, axis=0)

        var0 = np.var(group0, axis=0, ddof=1)
        var1 = np.var(group1, axis=0, ddof=1)

        n0 = group0.shape[0]
        n1 = group1.shape[0]

        se = np.sqrt(var0 / n0 + var1 / n1)
        se[se == 0] = np.inf

        t_stats = (mean1 - mean0) / se
        max_t = np.max(np.abs(t_stats))

        return max_t, t_stats

    from sklearn.metrics import mutual_info_score

    def mutualInformation(self, bit_values, traces):
        mi_scores = []
        for i in range(traces.shape[1]):
            mi = mutual_info_score(bit_values, traces[:, i])
            mi_scores.append(mi)
        max_mi = max(mi_scores)
        return max_mi, mi_scores

    def hamming_distance(self, X1, X2):
        return np.array([bin(x1 ^ x2).count('1') for x1, x2 in zip(X1, X2)], dtype=int)



    def hamming_weight(self, X):
        # Calcola il Peso di Hamming (numero di bit '1') per ogni valore in X
        return np.array([bin(val).count('1') for val in X], dtype=int)


    def Initialise(self, N):
        print("Initializing AES attack with Hamming weight leakage model on input of SubBytes...")
        trs = TRS_Reader.TRS_Reader("Projects/1 - Analysis of a simple AES dataset/code/base/TinyAES_625Samples_FirstRoundSbox.trs")
        trs.read_header()
        trs.read_traces(N, 0, trs.number_of_samples)

        # Prepara un array per memorizzare il Peso di Hamming
        HammingLeakage = np.zeros((16, 256, N), dtype=int)

        for byteno in range(16):
            X = trs.plaintext[:, byteno]  # Ottieni il byte del plaintext

            for kg in range(256):
                Y = self.ADK(X, kg)  # Applica AddRoundKey
                # Calcola il Peso di Hamming dell'input di SubBytes
                HammingLeakage[byteno, kg] = self.hamming_distance(X, Y)

        print("Initialization complete.")
        return trs, HammingLeakage


    def attackHammingWeight(self, trs, HammingLeakage, byteno, Nm, distinguisher="correlation"):
        print(f"\n--- Attacking byte {byteno} ---")
        print(f"Using {Nm} traces with {distinguisher} distinguisher.")

        key_metrics = {}
        all_metrics = []

        for kg in range(256):
            leakage = HammingLeakage[byteno, kg, 0:Nm]  # Ottieni il Peso di Hamming

            if distinguisher == "correlation":
                # Calcola la correlazione tra il leakage e le tracce di potenza
                corr = np.array([pearsonr(leakage, trs.traces[0:Nm, i])[0] for i in range(trs.traces.shape[1])])
                max_corr = np.max(np.abs(corr))
                metric = max_corr
            elif distinguisher == "difference_of_means":
                # Dividi le tracce in gruppi basati sul valore di leakage
                unique_leakages = np.unique(leakage)
                metrics = []
                for ul in unique_leakages:
                    group = trs.traces[0:Nm][leakage == ul, :]
                    metrics.append(np.mean(group, axis=0))
                # Calcola la differenza massima tra le medie dei gruppi
                metric = np.max(np.abs(metrics[0] - metrics[1]))
            else:
                raise ValueError("Invalid distinguisher selected.")

            key_metrics[kg] = metric
            all_metrics.append((kg, metric))

        # Identifica l'ipotesi di chiave con la metrica più alta
        best_kg = max(key_metrics, key=key_metrics.get)
        print(f"Most likely key guess: 0x{best_kg:02X} with metric {key_metrics[best_kg]:.4f}")

        # Stampa le prime 5 ipotesi di chiave
        top_5 = sorted(all_metrics, key=lambda x: x[1], reverse=True)[:5]
        print(f"Top 5 key guesses for byte {byteno}:")
        for kg, metric in top_5:
            print(f"  Key 0x{kg:02X}: Metric = {metric:.4f}")

        return best_kg


if __name__ == "__main__":
    # Inizializza l'attacco
    N_traces_total = 1000  # Numero totale di tracce disponibili
    aes_attack = AESAttackSingleBit()
    trs, HammingLeakage = aes_attack.Initialise(N_traces_total)

    # Numero di tracce da utilizzare per l'attacco
    Nm_list = [1000]

    # Esegui l'attacco per entrambi i distinguisher
    for distinguisher in ["correlation", "difference_of_means"]:
        print(f"\nPerforming attack using {distinguisher} distinguisher...\n")
        recovered_key = np.zeros(16, dtype=int)

        for Nm in Nm_list:
            print(f"Using {Nm} traces...")
            for byteno in range(16):
                recovered_key[byteno] = aes_attack.attackHammingWeight(trs, HammingLeakage, byteno, Nm, distinguisher)

            print(f"\nRecovered key ({Nm} traces):")
            print(" ".join(f"{byte:02X}" for byte in recovered_key))
