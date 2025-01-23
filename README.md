## Analysis of a Simple AES Dataset

This repository contains the supplementary materials for the project "Analysis of a Simple AES Dataset". The complete report, detailing the methodology and results, can be found in the `report_Task1_Tam_Gabriele.pdf` file.

**Project Overview**

This project investigates the application of Differential Power Analysis (DPA) attacks on a simple AES implementation. The provided scripts demonstrate how to implement and simulate these attacks.

**Key Research Questions**

* **Single-Bit Leakage Model:** The scripts can be modified to attack individual bits instead of entire bytes using a single-bit leakage model. This includes implementing both correlation and difference of means distinguishers to analyze the impact on attack results. (See Section 3 of the report)
* **SubBytes Input Attack:** The scripts can be extended to target the input of the SubBytes operation using a chosen distinguisher. The effectiveness of this approach is compared to attacking the SubBytes output. (See Section 4 of the report)
* **Trace Requirements:** Based on the most successful attack strategy, the minimum number of traces required for reliable key recovery with reasonable effort is estimated. (See Section 5 of the report)

**Additional Files**

For transparency and reproducibility purposes, additional files, including the scripts used in the project, are available in this GitHub repository.

**Directory Structure**
   ```
task1/
    report_Task1_Tam_Gabriele.pdf    # This file (delivered)
    code/                            # Scripts used for DPA attacks
        attack_bit/                  # Scripts for Single-Bit Leakage Model attack
            ...                      # (details in report)
        attack_SubBytes/             # Scripts for SubBytes input/output attacks
            ...                      # (details in report)
        best_attack/                 # Scripts for the most successful attack
            ...                      # (details in report)
        comparison/                  # Data and analysis for comparing attacks
            ...                      # (details in report)
   ```
**Getting Started**

1. **Clone the repository:**

   ```bash
   git clone [https://github.com/h0r0x/Analysis-of-a-simple-AES-dataset.git](https://github.com/h0r0x/Analysis-of-a-simple-AES-dataset.git)
      ```
2. **Review the report (recommended):**

    Report_Task1_Tam_Gabriele.pdf provides a detailed explanation of the project goals, methodology, and results.

3. **Explore the scripts (optional):**

    The code directory contains scripts used in the DPA attacks. Refer to the report for specific usage instructions and details on each script's functionality.
