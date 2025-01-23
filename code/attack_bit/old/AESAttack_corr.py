import TRS_Reader  # Module to read the TRS file
import numpy as np  # For numerical calculations
from scipy.stats import pearsonr  # To calculate the Pearson correlation
import matplotlib.pyplot as plt  # For plotting results

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
        Estrae il valore di un singolo bit (bit_index) dopo l'S-box.
        """
        print(f"Applying single-bit leakage model for bit {bit_index}...")
        y = np.zeros(len(X)).astype(int)
        for i in range(len(X)):
            # S-box transformation
            sbox_value = sbox[X[i]]
            # Extract the target bit
            y[i] = (sbox_value >> bit_index) & 1  # Ottieni il bit_index-esimo bit
        return y


    def Initialise(self, N):
        print("Initializing AES attack...")

        # Load the TRS file with power traces
        trs = TRS_Reader.TRS_Reader("TinyAES_625Samples_FirstRoundSbox.trs")
        trs.read_header()
        trs.read_traces(N, 0, trs.number_of_samples)

        # Prepare an array to store Hamming weight guesses
        HWguess = np.zeros((16, 8, 256, N)).astype(int)  # 8 bit per byte

        # For each byte of the key (16 total bytes)
        for byteno in range(16):
            X = trs.plaintext[:, byteno]  # Estrai il byte di plaintext
            for kg in range(256):
                Y = AESAttack().ADK(X, kg)  # Applica AddRoundKey
                for bit_index in range(8):
                    HWguess[byteno, bit_index, kg] = AESAttack().SB(Y, bit_index)


        print("Initialization complete.")
        return [trs, HWguess]

    


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
    
    for byteno in range(16):
        # Initialize arrays to accumulate scores and counts for each key candidate
        byte_key_scores = np.zeros(256)  # For weighted average approach
        key_guess_counts = np.zeros(256, dtype=int)  # For most frequent approach

        for bit_index in range(8):
            print(f"Analyzing byte {byteno}, bit {bit_index} with max correlation...")

            max_scores = []
            key_guesses = []

            # Compute the leakage for each possible key
            for kg in range(256):
                bit_values = HWguess[byteno, bit_index, kg]  # Bit values
                maxscore, score_values = AESAttack().maxCorrBit(bit_values[:Nm], trs.traces[:Nm, :])
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
        
    # Save the results as images
    plt.tight_layout()
    plt.savefig("images/single_bit_attack.png", dpi=300, bbox_inches="tight")
    print("Single-bit attack plot saved as 'single_bit_attack.png'.")
    
    # Stampa la chiave trovata bit per bit
    print("\nKey found (best_keys_weighted):")
    for byteno in range(16):
        print(f"Byte {byteno}: ", " ".join([str(b) for b in best_keys_weighted[byteno]]))
        
    # Stampa la chiave trovata bit per bit
    print("\nKey found (best_keys_frequent):")
    for byteno in range(16):
        print(f"Byte {byteno}: ", " ".join([str(b) for b in best_keys_frequent[byteno]]))

    # Stampa la chiave reale (come confronto)
    print("\nReal key (bitwise):")
    for byteno in range(16):
        real_byte = key[byteno]
        real_bits = [((real_byte >> bit_index) & 1) for bit_index in range(8)]
        print(f"Byte {byteno}: ", " ".join(map(str, real_bits)))

    # Disattiva il plotting interattivo
    plt.ioff()
    plt.show()
    print("Single-bit CPA attack completed.")       


















