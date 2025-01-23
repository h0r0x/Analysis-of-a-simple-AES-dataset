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

    def diffOfMeans(self, bit_values, traces):
        """
        Calcola la differenza di media tra due gruppi (bit = 0, bit = 1).
        """
        group0 = traces[bit_values == 0, :]
        group1 = traces[bit_values == 1, :]
        mean_diff = np.abs(group1.mean(axis=0) - group0.mean(axis=0))
        max_diff = np.max(mean_diff)
        return max_diff, mean_diff

    
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

    def ARK_SB(self, X, k, bit_index):
        """
        Modello di leakage per il bit specifico dopo AddRoundKey.
        X: plaintext
        k: candidato chiave
        bit_index: bit target (0-7)
        """
        print(f"Applying AddRoundKey and extracting bit {bit_index}...")
        y = np.zeros(len(X)).astype(int)
        for i in range(len(X)):
            # Applica AddRoundKey (XOR) e prendi il bit_index-esimo bit
            ark_value = X[i] ^ k
            y[i] = (ark_value >> bit_index) & 1
        return y



    def Initialise(self, N):
        print("Initializing AES attack with AddRoundKey leakage model...")

        # Leggi il file di tracce
        trs = TRS_Reader.TRS_Reader("TinyAES_625Samples_FirstRoundSbox.trs")
        trs.read_header()
        trs.read_traces(N, 0, trs.number_of_samples)

        # Prepara un array per memorizzare i modelli di leakage per ogni bit
        HWguess = np.zeros((16, 8, 256, N)).astype(int)  # 16 byte, 8 bit, 256 chiavi, N tracce

        # Per ogni byte della chiave
        for byteno in range(16):
            X = trs.plaintext[:, byteno]  # Estrai il byte del plaintext
            print(f"Processing byte {byteno}...")

            # Per ogni candidato chiave (0-255)
            for kg in range(256):
                for bit_index in range(8):
                    HWguess[byteno, bit_index, kg] = AESAttack().ARK_SB(X, kg, bit_index)

        print("Initialization complete.")
        return [trs, HWguess]   



    def corrAttackBit(self, trs, ax, byteno, bit_index, Nm, distinguisher="diffOfMeans"):
        """
        Esegue un attacco su singoli bit dopo AddRoundKey.
        """
        print(f"Performing {distinguisher} attack on byte {byteno}, bit {bit_index}...")
        maxkg = 0
        maxscore_k = 0

        for kg in range(256):
            # Estrai i valori del bit dopo AddRoundKey
            bit_values = HWguess[byteno, bit_index, kg]

            # Calcola il distinguisher selezionato
            if distinguisher == "maxCorr":
                maxscore, score_values = AESAttack().maxCorrBit(bit_values[:Nm], trs.traces[:Nm, :])
            elif distinguisher == "diffOfMeans":
                maxscore, score_values = AESAttack().diffOfMeans(bit_values[:Nm], trs.traces[:Nm, :])

            if maxscore > maxscore_k:
                maxkg = kg
                maxscore_k = maxscore

            # Plotta i risultati
            ax.plot(score_values, label=f"Key 0x{kg:02X}", alpha=0.8)

        ax.set_title(f"Byte {byteno}, Bit {bit_index}: Key 0x{maxkg:02X}")
        ax.legend()
        print(f"Most likely key for byte {byteno}, bit {bit_index}: {maxkg:02X}")
        return maxkg




if __name__ == "__main__":
    # Inizializza l'attacco
    print("Starting the CPA attack focused on AddRoundKey...")
    [trs, HWguess] = AESAttack().Initialise(1000)

    # Lista per accumulare i migliori candidati chiave
    found_key_bits = np.zeros((16, 8), dtype=int)  # 16 byte, 8 bit ciascuno

    # Configura la figura per i grafici
    fig = plt.figure(figsize=(18, 12))
    ax = []

    for byteno in range(16):
        for bit_index in range(8):
            ax.append(fig.add_subplot(16, 8, byteno * 8 + bit_index + 1))

    Nm = 1000

    # Per ogni byte
    for byteno in range(16):
        for bit_index in range(8):
            print(f"Analyzing byte {byteno}, bit {bit_index}...")
            found_key_bits[byteno, bit_index] = AESAttack().corrAttackBit(
                trs, ax[byteno * 8 + bit_index], byteno, bit_index, Nm, distinguisher="diffOfMeans"
            )

    # Salva l'immagine
    plt.tight_layout()
    plt.savefig("images/addroundkey_bit_attack.png", dpi=300, bbox_inches="tight")
    print("AddRoundKey bit attack plot saved as 'addroundkey_bit_attack.png'.")

    # Stampa la chiave trovata bit per bit
    print("\nKey found (bitwise):")
    for byteno in range(16):
        print(f"Byte {byteno}: ", " ".join([str(b) for b in found_key_bits[byteno]]))

    # Stampa la chiave reale (come confronto)
    print("\nReal key (bitwise):")
    for byteno in range(16):
        real_byte = key[byteno]
        real_bits = [((real_byte >> bit_index) & 1) for bit_index in range(8)]
        print(f"Byte {byteno}: ", " ".join(map(str, real_bits)))

    plt.ioff()
    plt.show()
    print("AddRoundKey bit CPA attack completed.")



