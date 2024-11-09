import pandas as pd
import matplotlib.pyplot as plt
import os

def readMPTData(filename):
    """
    Reads a biologic .mpt data file.

    Parameters
    ----------
    filename : str
        Path to a biologic file to read.

    Returns
    -------
    data : pd.DataFrame
        DataFrame of mpt data.
    """
    with open(filename, 'r', encoding='cp1252') as readfile:
        header_rows = readfile.readlines(18)
        header_rows = str(header_rows).split()
        df = pd.DataFrame(header_rows)
        header_num = df.iloc[7]
        header_num = int(header_num) - 4  # Adjust header line
    data = pd.read_csv(filename, header=header_num, sep='\t', engine='python', encoding='cp1252')
    return data

def plotNyquist_calcRohm(data, label, offset=0):
    """
    Plots Nyquist data with calculated ohmic resistance R_ohm.

    Parameters
    ----------
    data : pd.DataFrame
        Data containing impedance measurements.
    label : str
        Label for the plot.
    offset : float, optional
        y offset in Nyquist plot.
    """
    df = data.loc[data['-Im(Z)/Ohm'] >= 0]
    mValueIndex = df[['-Im(Z)/Ohm']].idxmin().values[0]
    m_mum = data.iloc[mValueIndex]

    xf = float(m_mum['Re(Z)/Ohm'])
    yf = float(m_mum['-Im(Z)/Ohm']) + offset
    print(f'Minimum for {label} is at {xf}, {yf}')

    Real = df['Re(Z)/Ohm']
    Imag = df['-Im(Z)/Ohm'] + offset

    plt.plot(Real, Imag, '-o', markersize=4, label=label)
    plt.plot(xf, yf, 'ko', markersize=3)  # Highlight minimum point

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

#os.chdir(r'C:\Users\leifer.be\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Biologic\Leifer\Low Temp Li Ion\2024\10')

for file_path, label in files_with_labels:
    plt.figure(figsize=(8, 6))
    try:
        data = readMPTData(file_path)
        print(data.head())
        plotNyquist_calcRohm(data, label, offset=0)  # No offset for individual plots
        plt.xlabel('Re(Z) (Ohm)')
        plt.ylabel('-Im(Z) (Ohm)')
        plt.title(f'Nyquist Plot for {label}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{label}.png', dpi=300, bbox_inches='tight')
        #plt.show()
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
