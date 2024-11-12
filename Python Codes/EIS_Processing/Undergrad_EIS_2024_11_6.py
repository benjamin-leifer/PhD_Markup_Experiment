import pandas as pd
import matplotlib.pyplot as plt
import os
from impedance import preprocessing
import numpy as np
from impedance.models.circuits import CustomCircuit


def readMPTData(filename):
    """
    Reads a biologic .mpt data file.

    Parameters
    ----------
    filename : str
        Path to a biologic file to read.

    Returns
    -------
    frequencies : ndarray
        Array of frequencies from the EIS data.
    Z : ndarray
        Array of complex impedance values.
    """
    try:
        with open(filename, 'r', encoding='cp1252') as readfile:
            lines = readfile.readlines()

            # Find the header line
            header_line = None
            for i, line in enumerate(lines):
                # Check if line contains known headers (adjust as necessary)
                if "Re(Z)/Ohm" in line or "Z'" in line or "freq" in line.lower():
                    header_line = i
                    break

            if header_line is None:
                print(f"Failed to determine header row for {filename}")
                return None, None

        # Read the data starting from the detected header line
        data = pd.read_csv(filename, header=header_line-3, sep='\t', engine='python', encoding='cp1252')

        # Standardize column names (remove whitespace and handle case)
        data.columns = data.columns.str.strip()
        data.columns = data.columns.str.lower()  # Convert to lowercase for consistency
        #print(f"Detected columns in {filename}: {data.columns}")

        # Extract frequencies and impedance values
        freq_col = [col for col in data.columns if 'freq' in col]
        if freq_col:
            frequencies = data[freq_col[0]].values
        else:
            print(f"Frequency column not found in {filename}")
            return None, None

        re_z_col = [col for col in data.columns if 're(z)' in col]
        im_z_col = [col for col in data.columns if 'im(z)' in col]
        if re_z_col and im_z_col:
            Z = data[re_z_col[0]].values + 1j * data[im_z_col[0]].values
        else:
            print(f"Required impedance columns not found in {filename}")
            return None, None

        return frequencies, Z
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None, None

def plotNyquist_calcRohm_with_fit(frequencies, Z, circuit_model, label, offset=0):
    """
    Plots Nyquist data with calculated ohmic resistance R_ohm and adds circuit fit.

    Parameters
    ----------
    frequencies : ndarray
        Array of frequencies.
    Z : ndarray
        Array of complex impedance values.
    circuit_model : CustomCircuit
        Fitted circuit model.
    label : str
        Label for the plot.
    offset : float, optional
        y offset in Nyquist plot.
    """
    real = Z.real
    imag = Z.imag  # Ensure we use the negative imaginary part for Nyquist plots

    # Find minimum imaginary impedance for reference (if needed)
    min_idx = imag.argmin()
    print(f"Minimum for {label} is at {real[min_idx]}, {imag[min_idx]}")

    # Plot experimental data
    plt.plot(real, imag + offset, 'o', markersize=4, label=f'{label} (Data)')
    plt.plot(real[min_idx], imag[min_idx] + offset, 'ko', markersize=3)  # Highlight minimum point

    # Check fitted parameters and plot the fitted circuit model explicitly
    try:
        # Generate fitted data using the fitted model
        Z_fit = circuit_model.predict(frequencies)

        # Plot the fitted data
        plt.plot(Z_fit.real, -Z_fit.imag + offset, '-', label=f'{label} (Fit)', linewidth=2)
    except Exception as e:
        print(f"Error plotting fit for {label}: {e}")

    plt.xlabel('Re(Z) (Ohm)')
    plt.ylabel('-Im(Z) (Ohm)')  # Labeling for the negative imaginary part
    plt.title(f'Nyquist Plot with Fit for {label}')
    plt.grid(True)
    plt.legend()


def plotBode(frequencies, Z, label):
    """
    Plots Bode magnitude and phase plots for the given impedance data.

    Parameters
    ----------
    frequencies : ndarray
        Array of frequencies.
    Z : ndarray
        Array of complex impedance values.
    label : str
        Label for the plot.
    """
    # Calculate magnitude and phase
    magnitude = abs(Z)
    phase = np.angle(Z, deg=True)  # Phase in degrees

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()

    ax1.loglog(frequencies, magnitude, 'b-', label=f'{label} |Z|')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('|Z| (Ohm)', color='b')
    ax1.tick_params('y', colors='b')
    ax1.grid(which='both', axis='both', linestyle='--', linewidth=0.5)

    ax2.semilogx(frequencies, phase, 'g-', label=f'{label} Phase')
    ax2.set_ylabel('Phase (degrees)', color='g')
    ax2.tick_params('y', colors='g')

    fig.tight_layout()
    plt.title(f'Bode Plot for {label}')
    plt.show()

# Define the circuit correctly
circuit = 'R1-p(R2,CPE0)-p(R3,CPE1)-W0'

# Corrected initial guesses (9 parameters)
initial_guess = [5, 50, 1e-5, 0.9, 50, 1e-5, 0.9, 1]


# Create the CustomCircuit model
#circuit_model = CustomCircuit(circuit, initial_guess=initial_guess)


# List of files to process and their cell codes
os.chdir(r'C:\Users\leifer.be\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Biologic\Leifer\Low Temp Li Ion\2024\10')
print(os.getcwd())
files_with_labels = [
    ('BL-LL-CC01_EIS_t1_C04.mpt', 'CC01 - Li_NMC LP'),
    ('BL-LL-CC02_EIS_t2_extended_C04.mpt', 'CC02 - Li_NMC LP (Extended)'),
    ('BL-LL-CC04_EIS-t1_C05.mpt', 'CC04 - Li_NMC LP'),
    ('BL-LL-CD02_EIS-t1_C05.mpt', 'CD02 - Gr_NMC DTF14'),
    ('BL-LL-CE04_EIS_t1_C05.mpt', 'CE04 - Li_NMC DTF14 (Second)'),
    ('BL-LL-CF01_EIS-t1_C05.mpt', 'CF01 - Li_NMC DT14'),
    ('BL-LL-CE04_EIS-t1_C05.mpt', 'CE04 - Li_NMC DTF14'),
    ('BL-LL-CF03_EIS-t1_C05.mpt', 'CF03 - Li_NMC DT14'),
    ('BL-LL-CC02_EIS_t1_C03.mpt', 'CC02 (First) - Li_NMC LP'),
    ('BL-LL-CC02_EIS_t1_C04.mpt', 'CC02 (Second) - Li_NMC LP')
]

import numpy as np  # Make sure you import numpy

for file_path, label in files_with_labels:
    plt.figure(figsize=(8, 6))
    try:
        frequencies, Z = readMPTData(file_path)
        if frequencies is None or Z is None:
            print(f"Skipping {label} due to read error.")
            continue

        # Fit the circuit model
        circuit_model = CustomCircuit(circuit, initial_guess=initial_guess)
        circuit_model.fit(frequencies, Z, method = 'trf')

        # Print fitted parameters
        print(f"Fitted Parameters for {label}: {circuit_model.parameters_}")

        # Plot Nyquist data with the fitted model
        plotNyquist_calcRohm_with_fit(frequencies, Z, circuit_model, label, offset=0)

        # Plot the Bode plot
        #plotBode(frequencies, Z, label)

        plt.savefig(f'{label}_fitted.png', dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
