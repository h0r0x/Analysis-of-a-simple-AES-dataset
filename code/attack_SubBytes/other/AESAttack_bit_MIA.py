import TRS_Reader  # Module to read the TRS file
import numpy as np  # For numerical calculations
from scipy.stats import pearsonr  # To calculate the Pearson correlation
import matplotlib.pyplot as plt  # For plotting results

import time

from scipy.stats import ks_2samp
from sklearn.feature_selection import mutual_info_regression
from sklearn.mixture import GaussianMixture

from sklearn.feature_selection import mutual_info_regression


class AESAttack(object):

    # Define the AES S-box (SubBytes lookup table)
    global sbox

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

    global key

    # Define the AES encryption key (fixed for the attack simulation)
    key = [
        0x2B, 0x7E, 0x15, 0x16,
        0x28, 0xAE, 0xD2, 0xA6,
        0xAB, 0xF7, 0x15, 0x88,
        0x09, 0xCF, 0x4F, 0x3C,
    ]


    def Sbox(self, X):
        # Apply S-box substitution to each byte in X
        print("Applying S-box transformation...")
        y = np.zeros(len(X)).astype(int)
        for i in range(len(X)):
            y[i] = sbox[X[i]]
        return y

    def ADK(self, X, k):
        # Perform XOR between input X and key candidate k
        print(f"Performing AddRoundKey with key candidate: {k}")
        y = np.zeros(len(X)).astype(int)
        for i in range(len(X)):
            y[i] = X[i] ^ k
        return y

    def HW(self, X):
        # Calculate Hamming weight (number of 1 bits) for each byte in X
        print("Calculating Hamming weights...")
        y = np.zeros(len(X)).astype(int)
        for i in range(len(X)):
            y[i] = bin(X[i]).count("1")
        return y
    
    def HW_leakage(self, X):
        """
        Calcola il Hamming Weight di un array di valori interi.
        """
        return np.array([bin(x).count("1") for x in X])

    
    def mutual_information(self, leakage_model, traces):
        """
        Computes mutual information between the leakage model and each sample point in the traces.
        Returns the maximum mutual information score and the array of scores.
        """
        mi_scores = np.zeros(trs.number_of_samples)
        for i in range(trs.number_of_samples):
            mi_scores[i] = mutual_info_regression(traces[:, i].reshape(-1, 1), leakage_model, discrete_features=True)
        max_mi = np.max(mi_scores)
        return max_mi, mi_scores



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

    def maxCorrBit(self, bit_values, traces):
        """
        Calcola la massima correlazione per un singolo bit.
        """
        maxcorr = 0
        corr = np.zeros(trs.number_of_samples)
        for i in range(trs.number_of_samples):
            corr[i], _ = pearsonr(bit_values, traces[:, i])
            if abs(corr[i]) > maxcorr:
                maxcorr = abs(corr[i])
        return maxcorr, corr

    
    def SB(self, X, bit_index):
        """
        Estrae il valore di un singolo bit (bit_index) dopo l'AddRoundKey.
        """
        print(f"Applying single-bit leakage model for bit {bit_index} (AddRoundKey)...")
        y = np.zeros(len(X)).astype(int)
        for i in range(len(X)):
            # No S-box transformation
            y[i] = (X[i] >> bit_index) & 1  # Ottieni il bit_index-esimo bit dal risultato di AddRoundKey
        return y



    def Initialise(self, N):
        print("Initializing AES attack...")

        # Load the TRS file with power traces
        trs = TRS_Reader.TRS_Reader("TinyAES_625Samples_FirstRoundSbox.trs")
        trs.read_header()
        trs.read_traces(N, 0, trs.number_of_samples)

        # Prepare an array to store leakage model values
        leakage_model = np.zeros((16, 256, N)).astype(int)  # Values per byte

        # For each byte of the key (16 total bytes)
        for byteno in range(16):
            X = trs.plaintext[:, byteno]  # Extract the plaintext byte
            for kg in range(256):
                Y = AESAttack().ADK(X, kg)  # Apply AddRoundKey (plaintext XOR kg)
                leakage_model[byteno, kg] = Y
        print("Initialization complete.")
        return [trs, leakage_model]



    


if __name__ == "__main__":
    
    # Numero di tracce utilizzate
    Nm = 1000
    
    # Initialize the attack
    print("Starting the MIA attack on AddRoundKey...")
    [trs, leakage_model] = AESAttack().Initialise(Nm)

    best_keys_mia = [0] * 16

    # Timer to measure total time
    start_time_total = time.time()
    times_per_byte = []

    for byteno in range(16):
        start_time_byte = time.time()

        mi_scores = np.zeros(256)
        for kg in range(256):
            leakage_values = leakage_model[byteno, kg][:Nm]
            max_mi, mi_values = AESAttack().mutual_information(leakage_values, trs.traces[:Nm, :])
            mi_scores[kg] = max_mi

        # Determine the best key candidate
        best_kg_mia = np.argmax(mi_scores)
        best_keys_mia[byteno] = best_kg_mia

        # Print the top 5 key candidates
        sorted_mi_scores = np.argsort(mi_scores)[::-1]
        print(f"\nTop 5 key candidates for byte {byteno} (MIA):")
        for rank, kg in enumerate(sorted_mi_scores[:5]):
            print(f"  #{rank+1}: Key = 0x{kg:02X}, MI Score = {mi_scores[kg]:.4f}")

        # Measure the time for the current byte
        elapsed_time_byte = time.time() - start_time_byte
        times_per_byte.append(elapsed_time_byte)
        print(f"Time for byte {byteno}: {elapsed_time_byte:.4f} seconds")

        # Optionally, plot the MI scores for the byte
        plt.figure(figsize=(12, 6))
        plt.bar(range(256), mi_scores, color='purple', edgecolor='black')
        plt.xlabel('Key Guess (0-255)')
        plt.ylabel('Mutual Information')
        plt.title(f'Mutual Information Scores for Byte {byteno}')
        plt.tight_layout()
        plt.savefig(f"images/byte_{byteno}_mia_scores.png", dpi=300)
        print(f"Saved MI scores plot for byte {byteno} as 'images/byte_{byteno}_mia_scores.png'.")

    # Calculate total and average time
    total_time = time.time() - start_time_total
    average_time_per_byte = total_time / 16
    print(f"\nTotal time: {total_time:.4f} seconds")
    print(f"Average time per byte: {average_time_per_byte:.4f} seconds")

    # Print the found key
    print("\nKey found (MIA):")
    print(" ".join([f"{byte:02X}" for byte in best_keys_mia]))

    # Print the real key for comparison
    print("\nReal key:")
    print(" ".join([f"{byte:02X}" for byte in key]))

    # Turn off interactive plotting
    plt.ioff()
    plt.show()
    print("MIA attack on AddRoundKey completed.")








