import TRS_Reader  # Module to read the TRS file
import numpy as np  # For numerical calculations
from scipy.stats import pearsonr  # To calculate the Pearson correlation
import matplotlib.pyplot as plt  # For plotting results

import time

from scipy.stats import ks_2samp
from sklearn.feature_selection import mutual_info_regression
from sklearn.mixture import GaussianMixture


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


    def extract_bit_group(self, X, start_bit, num_bits):
        """
        Estrae un gruppo di bit (num_bits) da un array di valori interi a partire da start_bit.
        """
        mask = (1 << num_bits) - 1  # Maschera per num_bits
        return (X >> start_bit) & mask



    def Initialise(self, N, group_size=4):
        print("Initializing AES attack...")

        # Load the TRS file with power traces
        trs = TRS_Reader.TRS_Reader("TinyAES_625Samples_FirstRoundSbox.trs")
        trs.read_header()
        trs.read_traces(N, 0, trs.number_of_samples)

        # Prepare an array to store leakage model values
        HWguess = np.zeros((16, 256, N)).astype(int)  # Hamming Weight per gruppo di bit

        for byteno in range(16):
            X = trs.plaintext[:, byteno]
            for kg in range(256):
                Y = AESAttack().ADK(X, kg)
                # Calcola il gruppo di bit estratti
                bit_group = AESAttack().extract_bit_group(Y, start_bit=0, num_bits=group_size)
                HWguess[byteno, kg] = AESAttack().HW_leakage(bit_group)

        print("Initialization complete.")
        return [trs, HWguess]


    def analyze_bit_group(self, HWguess, trs, byteno, group_size):
        """
        Analizza un gruppo di bit per un byte specifico.
        """
        scores = np.zeros(256)
        for kg in range(256):
            group_values = HWguess[byteno, kg]
            score, _ = AESAttack().diffOfMeans(group_values, trs.traces)
            scores[kg] = score

        # Identifica la chiave con il punteggio più alto
        best_key = np.argmax(scores)
        return best_key, scores



if __name__ == "__main__":
    # Initialize the attack
    print("Starting the CPA attack focused on single bits...")
    [trs, HWguess] = AESAttack().Initialise(1000)

    # Lista per accumulare i migliori key guess per ogni bit
    found_key_bits = np.zeros((16, 8), dtype=int)  # 16 byte, 8 bit ciascuno

    # Creazione figura per le tracce
    plt.figure(figsize=(15, 10))
    for i in range(10):  # Plotta 10 tracce rappresentative
        plt.plot(trs.traces[i] + i * 0.05, label=f"Trace {i+1}", alpha=0.8)  # Offset per separarli
    plt.title("Example Traces")
    plt.xlabel("Sample Index")
    plt.ylabel("Power Consumption (arbitrary units)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/example_traces_bits.png", dpi=300, bbox_inches="tight")
    print("Traces plot saved as 'example_traces_bits.png'.")

    # Figura principale per i risultati
    fig = plt.figure(figsize=(18, 12))
    ax = []

    # Creazione sottografici per ciascun byte e bit della chiave
    for byteno in range(16):
        for bit_index in range(8):
            ax.append(fig.add_subplot(16, 8, byteno * 8 + bit_index + 1))

    # Numero di tracce utilizzate
    Nm = 1000

    print(f"\nUsing {Nm} traces for single-bit analysis...\n")

    # Initialize arrays to accumulate scores and counts for each key candidate
    byte_key_scores = np.zeros(256)  # For weighted average approach
    key_guess_counts = np.zeros(256, dtype=int)  # For most frequent approach

    best_keys_weighted = [0] * 16
    best_keys_frequent = [0] * 16
    
    # Timer per misurare il tempo totale
    start_time_total = time.time()
    times_per_byte = []  # Per memorizzare il tempo per ogni byte

    
    for byteno in range(16):
        
        # Timer per misurare il tempo di ogni byte
        start_time_byte = time.time()
        
        # Initialize arrays to accumulate scores and counts for each key candidate
        byte_key_scores = np.zeros(256)  # For weighted average approach
        key_guess_counts = np.zeros(256, dtype=int)  # For most frequent approach

        for bit_index in range(8):
            print(f"Analyzing byte {byteno}, bit {bit_index} with diffOfMeans...")

            max_scores = []
            key_guesses = []

            # Compute the leakage for each possible key
            for kg in range(256):
                hw_values = HWguess[byteno, kg]  # Hamming Weight values
                # maxscore, score_values = AESAttack().diffOfMeans(bit_values[:Nm], trs.traces[:Nm, :])
                maxscore, score_values = AESAttack().maxCorrBit(hw_values[:Nm], trs.traces[:Nm, :])
                max_scores.append(maxscore)
                key_guesses.append((kg, maxscore))

            # Sort the results by score
            key_guesses.sort(key=lambda x: x[1], reverse=True)

            # Print the top 5 most probable candidates
            print(f"Top 5 key guesses for byte {byteno}, bit {bit_index}:")
            for rank, (kg, score) in enumerate(key_guesses[:5]):
                print(f"  #{rank+1}: Key = 0x{kg:02X}, Score = {score:.4f}")

            # Save the best candidate for this bit
            found_key_bits[byteno, bit_index] = key_guesses[0][0]

            # Accumulate scores for the weighted average approach
            for kg, score in key_guesses:
                byte_key_scores[kg] += score  # Sum the scores for each key candidate

            # Increment count for the most frequent approach
            best_kg = key_guesses[0][0]  # The key with the highest score for this bit
            key_guess_counts[best_kg] += 1  # Increment the count for this key

            

            # Configurazione sottografico
            ax[byteno * 8 + bit_index].set_xlim([0, trs.number_of_samples])
            ax[byteno * 8 + bit_index].set_ylim([-1, 1])
            ax[byteno * 8 + bit_index].set_title(f"Byte {byteno}, Bit {bit_index}")
            ax[byteno * 8 + bit_index].set_xlabel("Samples")
            ax[byteno * 8 + bit_index].set_ylabel(r"$\rho$")
            ax[byteno * 8 + bit_index].legend()

        # Determine the best key candidate using the weighted average approach
        best_kg_weighted = np.argmax(byte_key_scores)
        print(f"\nBest key candidate for byte {byteno} (weighted average): 0x{best_kg_weighted:02X}")

        # Print the top 5 key candidates for the weighted average approach
        sorted_key_scores = np.argsort(byte_key_scores)[::-1]
        print("Top 5 key candidates (weighted average):")
        for kg in sorted_key_scores[:5]:
            print(f"  Key 0x{kg:02X}, Total Score = {byte_key_scores[kg]:.4f}")

        # Determine the best key candidate using the most frequent approach
        best_kg_frequent = np.argmax(key_guess_counts)
        print(f"\nBest key candidate for byte {byteno} (most frequent): 0x{best_kg_frequent:02X}")

        # Print the top 5 key candidates for the most frequent approach
        sorted_key_counts = np.argsort(key_guess_counts)[::-1]
        print("Top 5 key candidates (most frequent):")
        for kg in sorted_key_counts[:5]:
            print(f"  Key 0x{kg:02X}, Count = {key_guess_counts[kg]}")

        # Save the best key candidates for each approach
        best_keys_weighted[byteno] = best_kg_weighted
        best_keys_frequent[byteno] = best_kg_frequent

        # Misurare il tempo per il byte corrente
        elapsed_time_byte = time.time() - start_time_byte
        times_per_byte.append(elapsed_time_byte)
        print(f"Time for byte {byteno}: {elapsed_time_byte:.4f} seconds")

        
        # Grafico dei punteggi per il byte corrente
        plt.figure(figsize=(12, 6))
        plt.bar(range(256), byte_key_scores, color='orange', edgecolor='black')
        plt.xlabel('Key Guess (0-255)')
        plt.ylabel('Score (Weighted Average)')
        plt.title(f'Weighted Average Scores for Byte {byteno}')
        plt.tight_layout()
        plt.savefig(f"images/byte_{byteno}_weighted_scores.png", dpi=300)
        print(f"Saved weighted scores plot for byte {byteno} as 'images/byte_{byteno}_weighted_scores.png'.")

        # Grafico della frequenza dei key guesses
        plt.figure(figsize=(12, 6))
        plt.bar(range(256), key_guess_counts, color='green', edgecolor='black')
        plt.xlabel('Key Guess (0-255)')
        plt.ylabel('Frequency')
        plt.title(f'Key Guess Frequency for Byte {byteno}')
        plt.tight_layout()
        plt.savefig(f"images/byte_{byteno}_key_frequency.png", dpi=300)
        print(f"Saved key frequency plot for byte {byteno} as 'images/byte_{byteno}_key_frequency.png'.")

            
            
    # Calcolare il tempo totale e medio
    total_time = time.time() - start_time_total
    average_time_per_byte = total_time / 16
    print(f"\nTotal time: {total_time:.4f} seconds")
    print(f"Average time per byte: {average_time_per_byte:.4f} seconds")
    
    
    print(f"Length of times_per_byte: {len(times_per_byte)}")
    
    # Grafico dei tempi per ogni byte
    plt.figure(figsize=(10, 6))
    plt.bar(range(16), times_per_byte, color='skyblue', edgecolor='black')
    plt.xlabel('Byte Index')
    plt.ylabel('Time (seconds)')
    plt.title('Time per Byte Processing')
    plt.xticks(range(16))
    plt.tight_layout()
    plt.savefig("images/time_per_byte.png", dpi=300)
    print("Saved time per byte plot as 'images/time_per_byte.png'.")

    # Grafico del tempo cumulativo
    cumulative_time = np.cumsum(times_per_byte)
    plt.figure(figsize=(10, 6))
    plt.plot(range(16), cumulative_time, marker='o', color='red', label='Cumulative Time')
    plt.xlabel('Byte Index')
    plt.ylabel('Cumulative Time (seconds)')
    plt.title('Cumulative Time for Byte Processing')
    plt.xticks(range(16))
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/cumulative_time.png", dpi=300)
    print("Saved cumulative time plot as 'images/cumulative_time.png'.")

    
    
    
    
    
    
    
    # Save the results as images
    plt.tight_layout()
    plt.savefig("images/single_bit_attack.png", dpi=300, bbox_inches="tight")
    print("Single-bit attack plot saved as 'single_bit_attack.png'.")
    
    # Stampa la chiave trovata bit per bit
    print("\nKey found (best_keys_weighted):")
    print(" ".join([f"{byte:02X}" for byte in best_keys_weighted]))


        
    # Stampa la chiave trovata bit per bit
    print("\nKey found (best_keys_frequent):")
    print(" ".join([f"{byte:02X}" for byte in best_keys_frequent]))



    # Stampa la chiave reale (come confronto)
    print("\nReal key:")
    print(" ".join([f"{byte:02X}" for byte in key]))


    # Disattiva il plotting interattivo
    plt.ioff()
    plt.show()
    print("Single-bit CPA attack completed.")       




######## END ########
# Bit attacck mean single bit + HW













