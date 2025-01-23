import TRS_Reader  # Module to read the TRS file
import numpy as np  # For numerical calculations
from heapq import nlargest  # To get the top elements
from scipy.stats import pearsonr  # To calculate the Pearson correlation
from collections import Counter  # To count frequencies

class AESAttackBitwiseOutputMostFrequent:
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

    def ADK(self, X, k):
        # Perform XOR between input X and key candidate k
        return X ^ k

    def Sbox(self, X):
        # Apply S-box substitution to each byte in X
        return np.array([self.sbox[val] for val in X], dtype=int)

    def bit_value(self, X, bit_pos):
        # Extract the specified bit from each byte in X
        return ((X >> bit_pos) & 1).astype(int)

    def maxCorr(self, bit_values, traces):
        # Compute the Pearson correlation for each sample point
        correlations = np.array([pearsonr(bit_values, traces[:, i])[0] for i in range(traces.shape[1])])
        maxcorr = np.max(np.abs(correlations))
        return maxcorr, correlations

    def Initialise(self, N):
        print("Initializing AES attack with bit-wise leakage model targeting output of SubBytes...")
        trs = TRS_Reader.TRS_Reader("TinyAES_625Samples_FirstRoundSbox.trs")
        trs.read_header()
        trs.read_traces(N, 0, trs.number_of_samples)

        # Prepare arrays to store bit predictions
        BitLeakage = np.zeros((16, 256, 8, N), dtype=int)

        for byteno in range(16):  # Iterate over all key bytes
            X = trs.plaintext[:, byteno]  # Get plaintext byte

            # For each possible key guess (0-255)
            for kg in range(256):
                Y = self.ADK(X, kg)  # Compute input to SubBytes (AddRoundKey)
                # The leakage model is applied to Y here
                _ = self.Sbox(Y)     # Apply SubBytes

                # For each bit position (0-7)
                for bit_pos in range(8):
                    BitLeakage[byteno, kg, bit_pos] = self.bit_value(Y, bit_pos)  # Get bit value

        print("Initialization complete.")
        return trs, BitLeakage

    def attackBit(self, trs, BitLeakage, byteno, Nm):
        print(f"Attacking byte {byteno} using {Nm} traces...")
        top_key_guesses = []  # List to store the top key guess for each bit

        # For each bit position
        for bit_pos in range(8):
            maxkg = 0
            max_metric = 0
            all_metrics = []

            # Iterate over all possible key guesses
            for kg in range(256):
                bit_values = BitLeakage[byteno, kg, bit_pos, :Nm]  # Get bit values
                metric, _ = self.maxCorr(bit_values, trs.traces[:Nm, :])  # Calculate correlation
                all_metrics.append((kg, metric))

                if abs(metric) > max_metric:
                    maxkg = kg
                    max_metric = abs(metric)

            # Store the key guess with the highest correlation for this bit
            top_key_guesses.append(maxkg)

            # Print the top 5 key guesses for this bit
            top_5 = nlargest(5, all_metrics, key=lambda x: abs(x[1]))
            print(f"Bit {bit_pos}:")
            print(f"Top 5 key guesses: {[(hex(k), f'{abs(m):.4f}') for k, m in top_5]}")
            print(f"Top key guess for bit {bit_pos}: 0x{maxkg:02X}, max correlation: {max_metric:.4f}\n")

        # Count the frequency of each key guess
        key_guess_counts = Counter(top_key_guesses)

        # Find the key guess with the highest frequency
        most_common = key_guess_counts.most_common()
        max_count = most_common[0][1]
        candidates = [kg for kg, count in most_common if count == max_count]

        if len(candidates) == 1:
            recovered_key_byte = candidates[0]
        else:
            # If there's a tie, select the key with the highest total correlation
            total_correlations = {}
            for kg in candidates:
                total_correlation = sum(
                    abs(self.maxCorr(BitLeakage[byteno, kg, bit_pos, :Nm], trs.traces[:Nm, :])[0])
                    for bit_pos in range(8)
                )
                total_correlations[kg] = total_correlation
            # Select the key with the highest total correlation
            recovered_key_byte = max(total_correlations, key=total_correlations.get)

        print(f"Key guesses frequency for byte {byteno}: { {hex(k): v for k, v in key_guess_counts.items()} }")
        print(f"Recovered key byte {byteno}: 0x{recovered_key_byte:02X}\n")
        return recovered_key_byte

if __name__ == "__main__":
    # Initialize the attack
    N_traces_total = 1000  # Total number of traces available
    aes_attack = AESAttackBitwiseOutputMostFrequent()
    trs, BitLeakage = aes_attack.Initialise(N_traces_total)

    # Perform attack
    Nm = 1000  # Number of traces to use for the attack
    recovered_key = np.zeros(16, dtype=int)

    for byteno in range(16):
        recovered_key[byteno] = aes_attack.attackBit(trs, BitLeakage, byteno, Nm)

    print("Recovered key:")
    print(" ".join(f"{byte:02X}" for byte in recovered_key))
