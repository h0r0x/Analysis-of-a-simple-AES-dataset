import TRS_Reader  # Module to read the TRS file
import numpy as np  # For numerical calculations
from scipy.stats import pearsonr  # To calculate the Pearson correlation
import matplotlib.pyplot as plt  # For plotting results

from scipy.stats import ks_2samp
from sklearn.feature_selection import mutual_info_regression
from sklearn.mixture import GaussianMixture

from itertools import product

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

    def maxCorr(self, hw, traces):
        # Find the maximum Pearson correlation between Hamming weights and traces
        # print("Calculating Pearson correlation for all samples...")
        maxcorr = 0
        corr = np.zeros(trs.number_of_samples)
        for i in range(trs.number_of_samples):
            [corr[i], pv] = pearsonr(hw, traces[:, i])  # Correlation and p-value
            if abs(corr[i]) > maxcorr:
                maxcorr = abs(corr[i])  # Track the highest correlation
        return [maxcorr, corr]

    def Initialise(self, N):
        print("Initializing AES attack...")

        # Load the TRS file with power traces
        trs = TRS_Reader.TRS_Reader("TinyAES_625Samples_FirstRoundSbox.trs")
        # trs = TRS_Reader.TRS_Reader("Projects/1 - Analysis of a simple AES dataset/code/base/TinyAES_625Samples_FirstRoundSbox.trs")
        trs.read_header()
        trs.read_traces(N, 0, trs.number_of_samples)

        # Prepare an array to store Hamming weight guesses
        HWguess = np.zeros((16, 256, N)).astype(int)

        # For each byte of the key (16 total bytes)
        for byteno in range(16):
            X = trs.plaintext[:, byteno]  # Get plaintext byte
            print(f"Processing byte number: {byteno}")

            # For each possible key guess (0-255)
            for kg in range(256):
                print(f"  Trying key guess: {kg}")
                Y = AESAttack().ADK(X, kg)  # Apply AddRoundKey
                # Y = AESAttack().Sbox(Y)  # Apply S-box transformation
                HWguess[byteno, kg] = AESAttack().HW(Y)  # Compute Hamming weight

        print("Initialization complete.")
        return [trs, HWguess]

    
    def maxKS(self, hw, traces):
        """
        Utilizza il Kolmogorov-Smirnov Test come distinguisher.
        """
        maxks = 0
        ks_values = np.zeros(trs.number_of_samples)
        for i in range(trs.number_of_samples):
            # Divide le tracce in due gruppi in base al modello di Hamming weight
            group0 = traces[hw == 0, i] if np.any(hw == 0) else []
            group1 = traces[hw == 1, i] if np.any(hw == 1) else []

            # Calcola il valore di K-S se entrambi i gruppi sono validi
            if len(group0) > 0 and len(group1) > 0:
                ks_values[i] = ks_2samp(group0, group1).statistic

            # Memorizza il massimo valore del K-S Test
            if ks_values[i] > maxks:
                maxks = ks_values[i]
        return [maxks, ks_values]

    def maxMIA(self, hw, traces):
        """
        Utilizza Mutual Information Analysis come distinguisher.
        """
        maxmi = 0
        mi_values = np.zeros(trs.number_of_samples)
        for i in range(trs.number_of_samples):
            mi_values[i] = mutual_info_regression(traces[:, i].reshape(-1, 1), hw)[0]
            if mi_values[i] > maxmi:
                maxmi = mi_values[i]
        return [maxmi, mi_values]

    def maxDPA(self, hw, traces):
        """
        Utilizza Differential Power Analysis come distinguisher.
        """
        maxdpa = 0
        dpa_values = np.zeros(trs.number_of_samples)
        for i in range(trs.number_of_samples):
            group0 = traces[hw == 0, i] if np.any(hw == 0) else []
            group1 = traces[hw == 1, i] if np.any(hw == 1) else []
            if len(group0) > 0 and len(group1) > 0:
                dpa_values[i] = abs(np.mean(group1) - np.mean(group0))
            if dpa_values[i] > maxdpa:
                maxdpa = dpa_values[i]
        return [maxdpa, dpa_values]
    
    def maxLDA(self, hw, traces):
        """
        Utilizza Linear Discriminant Analysis come distinguisher.
        """
        lda = LinearDiscriminantAnalysis(n_components=1)
        lda.fit(traces, hw)
        lda_transformed = lda.transform(traces)
        return [np.max(lda_transformed), lda_transformed]
    
    def maxTemplate(self, hw, traces):
        """
        Utilizza Template Attack come distinguisher.
        """
        maxscore = 0
        score_values = np.zeros(trs.number_of_samples)
        gmm = GaussianMixture(n_components=2)
        for i in range(trs.number_of_samples):
            gmm.fit(traces[:, i].reshape(-1, 1))
            score_values[i] = gmm.score_samples(hw.reshape(-1, 1)).mean()
            if score_values[i] > maxscore:
                maxscore = score_values[i]
        return [maxscore, score_values]


    def difference_of_means(self, hw, traces):
        """
        Calcola la differenza di medie tra due gruppi basati sull'Hamming weight.
        """
        mean_diff = np.zeros(traces.shape[1])
        for i in range(traces.shape[1]):
            group0 = traces[hw == 0, i] if np.any(hw == 0) else np.array([])
            group1 = traces[hw == 1, i] if np.any(hw == 1) else np.array([])
            if len(group0) > 0 and len(group1) > 0:
                mean_diff[i] = abs(np.mean(group1) - np.mean(group0))
        max_diff = np.max(mean_diff)
        return max_diff, mean_diff


    def corrAttack(self, trs, ax, byteno, Nm, distinguisher="difference_of_means"):
        """
        Esegue un attacco basato sul distinguisher selezionato.
        """
        print(f"Performing {distinguisher} attack on byte: {byteno} with {Nm} traces.")
        ax.clear()
        maxkg = 0
        maxscore_k = 0

        for kg in range(256):
            hw = HWguess[byteno, kg]

            # Usa il distinguisher selezionato
            if distinguisher == "maxCorr":
                [maxscore, score_values] = AESAttack().maxCorr(hw[0:Nm], trs.traces[0:Nm, :])
            elif distinguisher == "maxKS":
                [maxscore, score_values] = AESAttack().maxKS(hw[0:Nm], trs.traces[0:Nm, :])
            elif distinguisher == "maxMIA":
                [maxscore, score_values] = AESAttack().maxMIA(hw[0:Nm], trs.traces[0:Nm, :])
            elif distinguisher == "maxDPA":
                [maxscore, score_values] = AESAttack().maxDPA(hw[0:Nm], trs.traces[0:Nm, :])
            elif distinguisher == "maxTemplate":
                [maxscore, score_values] = AESAttack().maxTemplate(hw[0:Nm], trs.traces[0:Nm, :])
            elif distinguisher == "maxLDA":
                [maxscore, score_values] = AESAttack().maxLDA(hw[0:Nm], trs.traces[0:Nm, :])
            elif distinguisher == "difference_of_means":
                [maxscore, score_values] = AESAttack().difference_of_means(hw[0:Nm], trs.traces[0:Nm, :])

            if maxscore > maxscore_k:
                maxkg = kg
                maxscore_k = maxscore

            # Traccia i risultati
            if kg == key[byteno]:
                ax.plot(score_values, "r-", alpha=1)
            else:
                ax.plot(score_values, color=(0.8, 0.8, 0.8), alpha=0.8)

        ax.set_xlim([1, trs.number_of_samples])
        ax.set_ylim([0, 1] if distinguisher in ["maxKS", "maxMIA"] else [-1, 1])
        ax.title.set_text(f"Byte {byteno}=0x{maxkg:2x}")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Score")

        print(f"Most likely key for byte {byteno}: {maxkg:2x} (Score: {maxscore_k})")
        return maxkg


if __name__ == "__main__":
    # Initialize the attack
    print("Starting the CPA attack focused on max correlation...")
    [trs, HWguess] = AESAttack().Initialise(1000)

    # Lista per accumulare i migliori key guess
    found_key = []

    # Creazione figura per le tracce
    plt.figure(figsize=(15, 10))
    for i in range(10):  # Plotta 10 tracce rappresentative
        plt.plot(trs.traces[i] + i * 0.05, label=f"Trace {i+1}", alpha=0.8)  # Offset per separarli
    plt.title("Example Traces")
    plt.xlabel("Sample Index")
    plt.ylabel("Power Consumption (arbitrary units)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/example_traces.png", dpi=300, bbox_inches="tight")
    print("Traces plot saved as 'example_traces.png'.")

    # Figura principale per i risultati
    fig = plt.figure(figsize=(18, 12))
    ax = []

    # Creazione sottografici per ciascun byte della chiave
    for byteno in range(16):
        ax.append(fig.add_subplot(4, 4, byteno + 1))

    # Numero di tracce utilizzate
    Nm = 1000

    print(f"\nUsing {Nm} traces...\n")

    # Per ogni byte
    for byteno in range(16):
        print(f"Analyzing byte {byteno} with max correlation...")
        max_corrs = []
        key_guesses = []

        # Calcolo della correlazione per ogni possibile chiave
        for kg in range(256):
            hw = HWguess[byteno, kg]
            maxcorr, corr = AESAttack().maxCorr(hw[:Nm], trs.traces[:Nm, :])
            max_corrs.append(maxcorr)
            key_guesses.append((kg, maxcorr))

        # Ordina i risultati per valore di correlazione
        key_guesses.sort(key=lambda x: x[1], reverse=True)

        # Stampa i 5 candidati più probabili
        print(f"Top 5 key guesses for byte {byteno}:")
        for rank, (kg, score) in enumerate(key_guesses[:5]):
            print(f"  #{rank+1}: Key = 0x{kg:02X}, Correlation = {score:.4f}")

        # Aggiunge il migliore candidato alla chiave trovata
        found_key.append(key_guesses[0][0])

        # Visualizza il grafico
        for kg, score in key_guesses[:5]:  # Colora i top 5 in modo distintivo
            hw = HWguess[byteno, kg]
            _, corr = AESAttack().maxCorr(hw[:Nm], trs.traces[:Nm, :])
            ax[byteno].plot(corr, label=f"Key 0x{kg:02X}", alpha=0.8)

        # Configurazione sottografico
        ax[byteno].set_xlim([0, trs.number_of_samples])
        ax[byteno].set_ylim([-1, 1])
        ax[byteno].set_title(f"Byte {byteno}: Key 0x{key_guesses[0][0]:02X}")
        ax[byteno].set_xlabel("Samples")
        ax[byteno].set_ylabel(r"$\rho$")
        ax[byteno].legend()

    # Salva l'immagine
    plt.tight_layout()
    plt.savefig("images/max_correlation_attack.png", dpi=300, bbox_inches="tight")
    print("Max correlation attack plot saved as 'max_correlation_attack.png'.")

    # Stampa la chiave trovata e quella reale
    print("\nKey found:")
    print(" ".join([f"0x{b:02X}" for b in found_key]))
    print("\nReal key:")
    print(" ".join([f"0x{b:02X}" for b in key]))

    # Disattiva il plotting interattivo
    plt.ioff()
    plt.show()
    print("CPA attack completed.")

