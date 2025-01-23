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


    def Initialise(self, N):
        print("Initializing AES attack with single-bit leakage model on input of SubBytes...")
        trs = TRS_Reader.TRS_Reader("Projects/1 - Analysis of a simple AES dataset/code/base/TinyAES_625Samples_FirstRoundSbox.trs")
        trs.read_header()
        trs.read_traces(N, 0, trs.number_of_samples)

        # Prepare arrays to store bit predictions
        BitLeakage = np.zeros((16, 256, 8, N), dtype=int)

        # For each byte of the key (16 total bytes)
        for byteno in range(16):
            X = trs.plaintext[:, byteno]
            unique_plaintexts = np.unique(X)
            print(f"Byte {byteno}: Unique plaintext values = {len(unique_plaintexts)}")
            if len(unique_plaintexts) == 1:
                print(f"Warning: Plaintext byte {byteno} is constant across all traces.")



            # For each possible key guess (0-255)
            for kg in range(256):
                Y = self.ADK(X, kg)  # Apply AddRoundKey
                # Y = self.Sbox(Y)  # Do not apply S-box, since we attack input of SubBytes

                # For each bit position (0-7)
                for bit_pos in range(8):
                    BitLeakage[byteno, kg, bit_pos] = self.bitLeakage(Y, bit_pos)

        print("Initialization complete.")
        return trs, BitLeakage

    def attackBit(self, trs, BitLeakage, byteno, Nm, distinguisher="correlation"):
        print(f"\n--- Attacking byte {byteno} ---")
        print(f"Using {Nm} traces with {distinguisher} distinguisher.")

        key_bit_counts = {}
        all_metrics = [[] for _ in range(8)]  # Metrics for each bit

        # Open CSV file to save results
        csv_filename = f"attack_results_byte_{byteno}.csv"
        with open(csv_filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Key Guess", "Bit Position", "Metric"])

            # Iterate over all possible key guesses
            for kg in range(256):
                print(f"\nKey Guess 0x{kg:02X}:")
                total_metric = 0  # Sum of metrics for this key guess

                for bit_pos in range(8):
                    bit_values = BitLeakage[byteno, kg, bit_pos, 0:Nm]  # Get bit values

                    if distinguisher == "correlation":
                        metric, _ = self.maxCorr(bit_values, trs.traces[0:Nm, :])
                    elif distinguisher == "difference_of_means":
                        metric, _ = self.diffOfMeans(bit_values, trs.traces[0:Nm, :])
                    elif distinguisher == "mutual_information":
                        metric, _ = self.mutualInformation(bit_values, trs.traces[0:Nm, :])
                    else:
                        raise ValueError("Invalid distinguisher selected.")


                    total_metric += metric  # Accumulate metric for key guess

                    # Save metric for analysis
                    all_metrics[bit_pos].append((kg, metric))

                    # Write to CSV
                    writer.writerow([f"0x{kg:02X}", bit_pos, f"{metric:.4f}"])

                    # Print detailed metric
                    print(f"  Bit {bit_pos}: Metric = {metric:.4f}")

                # Update key guess counts
                key_bit_counts[kg] = key_bit_counts.get(kg, 0) + total_metric

        # Print the top 5 key guesses for each bit
        print(f"\nTop 5 key guesses for byte {byteno}:")
        for bit_pos in range(8):
            top_5 = nlargest(5, all_metrics[bit_pos], key=lambda x: x[1])
            print(f"  Bit {bit_pos}: {[(f'0x{k:02X}', f'{m:.4f}') for k, m in top_5]}")

        # Determine the most frequent key guess across all bits
        most_likely_key = max(key_bit_counts, key=key_bit_counts.get)
        print(f"\nKey guesses total metrics for byte {byteno}:")
        sorted_keys = sorted(key_bit_counts.items(), key=lambda x: x[1], reverse=True)
        for k, v in sorted_keys[:5]:  # Print top 5 keys
            print(f"  Key 0x{k:02X}: Total Metric = {v:.4f}")

        print(f"\nMost likely key guess: 0x{most_likely_key:02X}")
        print(f"Recovered key byte {byteno}: 0x{most_likely_key:02X}")

        return most_likely_key

if __name__ == "__main__":
    # Initialize the attack
    N_traces_total = 1000  # Total number of traces available
    aes_attack = AESAttackSingleBit()
    trs, BitLeakage = aes_attack.Initialise(N_traces_total)

    # Number of traces to use for the attack
    Nm_list = [1000]

    # Perform attack for both distinguishers
    for distinguisher in ["mutual_information",   
                          "correlation", 
                          "difference_of_means"]:
        
        print(f"\nPerforming attack using {distinguisher} distinguisher...\n")
        recovered_key = np.zeros(16, dtype=int)

        for Nm in Nm_list:
            print(f"Using {Nm} traces...")
            for byteno in range(16):
                recovered_key[byteno] = aes_attack.attackBit(trs, BitLeakage, byteno, Nm, distinguisher)

            print(f"\nRecovered key ({Nm} traces):")
            print(" ".join(f"{byte:02X}" for byte in recovered_key))
