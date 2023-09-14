import numpy as np
import pandas as pd
from math import isnan as isnan

from numpy import pi as pi
from scipy.constants import mu_0 as mu0
from scipy.constants import Planck as h
from scipy.constants import Boltzmann as kB
from scipy.constants import elementary_charge as e
from scipy.constants import electron_mass as m_e

from numpy import tan as tan
from numpy import cos as cos
from numpy import sin as sin
from numpy import exp as exp
from numpy import sqrt as sqrt
from numpy import arctan as Atan
from numpy import log10 as log10
#from mpmath import coth
#from mpmath import csc as cosec
from numpy import sinh as sinh
from numpy import cosh as cosh
from scipy.special import iv as I0
from scipy.special import kv as K0

from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.misc import derivative

from scipy.special import iv as I0
from scipy.special import kv as K0

from scipy.optimize import curve_fit
from scipy.optimize import leastsq

# from lmfit import Model

from matplotlib import pyplot as plt

import os
import pickle
from tempfile import TemporaryFile

'''
#############################
New read in and plot function
##############################
'''


def get_x_ax(a, N=1000):
    return np.linspace(min(a), max(a), N)


def get_QR_QC_and_plot(f_ar, S21_dB, f0, axis):
    '''
    f_ar needs to be in Hz
    '''
    
    # f_ar = f_ar*1e6 # convert MHZ to HZ
    
    
    spline = UnivariateSpline(f_ar, S21_dB + 3.0, s=0)
    
    three_dB_BW = abs(spline.roots()[1] - spline.roots()[0])
    
    
    # f0 = f_ar[S21_dB.argmin()]
    
    # QR = f0*1e6/three_dB_BW
    QR = f0/three_dB_BW
    
    dip_depth = min(S21_dB)
    
    QC = QR/(1-dip_depth)
    
    # axis.plot([spline.roots()[0]/1e6, spline.roots()[1]/1e6], [-3, -3], color="red", marker="o", linewidth=0.2, markersize=1, label="3dBBW={:.2g}MHz".format(three_dB_BW/1e6))
    axis.plot([spline.roots()[0]/1e9, spline.roots()[1]/1e9], [-3, -3], color="red", marker="o", linewidth=0.2, markersize=1, label="3dBBW={:.1f}KHz\n3dbBW_QR={:.0f}".format(three_dB_BW/1e3, QR))
    
    

    return QR, QC
    

temp_f0 = 0.0
temp_qr = 0.0



def S21mag_fit(freq, f0, qr, qc):
    
    #freq = freq*1e6
    #w = 2*pi*freq
    #print(freq)
    
    xr = freq/f0 - 1

    s21 = 1 - ( (qr/qc) * (1 / (1 + 2j*qr*xr)) ) 
    
    return np.abs(s21)

def S21mag_fit_with_caps(freq, cc):
    
    #freq = freq*1e6
    #w = 2*pi*freq
    #print(freq)
    
    f0 = temp_f0
    qr = temp_qr
    
    xr = freq/f0 - 1
    
    Ctot = temp_Ctot
    
    try:
        Z0_interp = interp1d([1.8e9, 3e9, 4.7e9], [65.2218, 65.2216, 65.2213])
        Z0 = float(Z0_interp(f0))
    except:
        Z0 = 65.2215
    
    qc = (2*(Ctot-cc))/(2*pi*f0*Z0*(cc**2))

    s21 = 1 - ( (qr/qc) * (1 / (1 + 2j*qr*xr)) ) 
    
    return np.abs(s21)


def volts2db(s21_volts):  # Transforming power (voltage**2) into decibells
    return 20 * np.log10(s21_volts)


def read_in_single_file_data_and_plot(live_file, title, print_xylab=(True,True), fit=True):
    
    exists = os.path.exists(live_file)
    
    if title=="":
        title=live_file
    
    fig=plt.figure(title, figsize=(12,6))
    rows = 1
    cols = 1
    grid = plt.GridSpec(rows, cols, top=0.95, bottom=0.092, left=0.05, right=0.95, hspace=0.2, wspace=0.2)
    ax_dict = {}
    for c in range(cols):
        for r in range(rows):
            ax_dict["ax{0}".format((c+1)+r*cols)] = plt.subplot(grid[r,c])
    
    axis = ax_dict["ax1"]
    if not exists:
        print(f'Error on file -> "{live_file}"')
        return 0.0, 0.0, 0.0
        
    else:
        
        skip_row=2
        start_NaN=True
        while start_NaN:
            current_data = pd.read_csv(live_file, names=["Frequency (GHz)", "RE[S11]", "IM[S11]", "RE[S12]", "IM[S12]", "RE[S21]", "IM[S21]", "RE[S22]", "IM[S22]"], skiprows=skip_row )
            if type(current_data["Frequency (GHz)"][0]) == str:
                skip_row+=1
            else:
                start_NaN=False
        
        # print(skip_row)
        Freq = np.array(current_data["Frequency (GHz)"]*1e9) # convert GHZ to HZ
        
        S21mag = np.abs(current_data['RE[S21]'] + 1j * current_data['IM[S21]'])

        
        peaks = find_peaks(-volts2db(S21mag), height=5, distance=100)
        
        try:
            freq_ar_around_peak = np.array(Freq[(peaks[0][0]-200):(peaks[0][0]+200)]) # freq array around the first peak
            S21_mag_around_peak = np.array(S21mag[(peaks[0][0]-200):(peaks[0][0]+200)])
        except IndexError:
            print("Multiple or no peaks found")
            return
            
        
        x0 = freq_ar_around_peak[S21_mag_around_peak.argmin()]
        
        # L_tot = L_tot_interp(x0)
        
        res_freq = x0    
        
        QR_from_3dB, _ = get_QR_QC_and_plot(freq_ar_around_peak, volts2db(S21_mag_around_peak), x0, axis) #Getting the QR and QC Values
        
        axis.scatter(Freq/1e9, volts2db(S21mag), s=0.5, label="f0={:.4g}GHz".format(res_freq/1e9))
        axis.plot(Freq/1e9, volts2db(S21mag), linewidth=0.2, alpha=0.3)
        axis.axvline(x0/1e9, linestyle="--", linewidth=0.8, color="red", alpha=0.3)
        
        if fit:
            
            QR_guess = 10e3
            QC_guess = 2*QR_guess
            init_guess=np.array([x0, QR_guess, QC_guess])
            popt, pcov = curve_fit(S21mag_fit, freq_ar_around_peak, S21_mag_around_peak, p0=init_guess)#, bounds=((0, 0, 0), (6000, 1000e3, 1000e3)))
            print("Live File = " + live_file)
            print("f0 = {:.6g}".format(popt[0]))
            print("QR = {:.6g}".format(popt[1]))
            print("QC = {:.6g}".format(popt[2]))
            QI = 1/((1/popt[1])-(1/popt[2]))
            print("Qi = {:.6g}".format(QI))
            print("\n")
            axis.plot(get_x_ax(freq_ar_around_peak, int(1e5))/1e9, volts2db(S21mag_fit(get_x_ax(freq_ar_around_peak, int(1e5)), *popt)), color="orange", linewidth=1.2, alpha=0.8, label="Fit Values:\nQR={:.0f}\nQC={:.0f}\nQI={:.0f}".format(popt[1], popt[2], QI))
            
        
        axis.legend(loc="best",fontsize=8)
        axis.grid(alpha=0.3)
        if print_xylab[0]:
            axis.set_xlabel("Frequency     (GHz)")
        if print_xylab[1]:
            axis.set_ylabel("S21")
        axis.set_title(title, font={'size':8}, pad=-20)
        axis.set_xlim((freq_ar_around_peak[0]/1e9, freq_ar_around_peak[-1]/1e9))
        axis.set_ylim((-5, 0.5))
        
        return QR_from_3dB, popt[1], x0
    return QR_from_3dB, 0, x0




# plt.close("all")




# data = np.zeros((8,4))
# col_names = ["f0"       , "Arms", "IDC", "CC"]
# data[0] = [1.97407e+09,      1-4,  1975, 1120]
# data[1] = [ 2.0844e+09,      1-4,  1100, 900]
# data[2] = [2.24149e+09,      5-8,  1100, 750]
# data[3] = [2.44326e+09,     9-12,  1100, 625]
# data[4] = [2.70781e+09,    13-16,  1100, 510]
# data[5] = [ 3.0693e+09,    17-20,  1100, 390]
# data[6] = [3.57511e+09,    25-28,  1975, 276]
# data[7] = [4.18219e+09,    25-28,  1100, 190]

# f0s = data[:,0]

# IDCLs = data[:,2]
# CCLs = data[:,3]

# fig=plt.figure("fitting f0 ccl", figsize=(12,6))
# rows = 1
# cols = 2
# grid = plt.GridSpec(rows, cols, top=0.95, bottom=0.092, left=0.05, right=0.95, hspace=0.2, wspace=0.2)
# ax_dict = {}
# for c in range(cols):
#     for r in range(rows):
#         ax_dict["ax{0}".format((c)+r*cols)] = plt.subplot(grid[r,c])

# ax0 = ax_dict["ax0"]
# ax1 = ax_dict["ax1"]

# # Axes 0 f0 against CCL
# ax0.scatter(f0s, CCLs)

# ax0.set_title("f0 vs CCL")
# ax0.set_xlabel("f0     (Hz)")
# ax0.set_ylabel("CCL     (um)")
# ax0.grid(alpha=0.3)

# # Axes 1 f0 against IDCL
# ax1.scatter(f0s, CCLs)

# ax1.set_title("f0 vs CCL")
# ax1.set_xlabel("f0     (Hz)")
# ax1.set_ylabel("CCL     (um)")
# ax1.grid(alpha=0.3)

# fig.show()




'''
###############################################################################
Longer coupler Lower Q,  Shorter coupler Higher Q
###############################################################################
'''




'''
###########
Arms 1 to 4
###########
'''
# Arms 1 to 4 at 1975
read_in_single_file_data_and_plot("sonnet_sims\\V8_(1-4)IDC1975_CC_1030.csv", "") #QR = 50214    f0 = 1.97012e+09          #Done 1975
read_in_single_file_data_and_plot("sonnet_sims\\V8_(1-4)IDC1975_CC_1045.csv", "") #QR = 48870    f0 = 1.97003e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(1-4)IDC1975_CC_1060.csv", "") #QR = 47989    f0 = 1.96994e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(1-4)IDC1975_CC_1070.csv", "") #QR = 47172    f0 = 1.96988e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(1-4)IDC1975_CC_1080.csv", "") #QR = 46586    f0 = 1.96983e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(1-4)IDC1975_CC_1120.csv", "") #QR = 43925    f0 = 1.9696e+09

# Arms 1 to 4 at 1800
read_in_single_file_data_and_plot("sonnet_sims\\V8_(1-4)IDC1800_CC_985.csv", "") #QR = 49403    f0 = 1.99284e+09          #Done 1800
read_in_single_file_data_and_plot("sonnet_sims\\V8_(1-4)IDC1800_CC_988.csv", "") #QR = 49112    f0 = 1.99282e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(1-4)IDC1800_CC_990.csv", "") #QR = 49101    f0 = 1.99281e+09

# Arms 1 to 4 at 1625
read_in_single_file_data_and_plot("sonnet_sims\\V8_(1-4)IDC1625_CC_920.csv", "") #QR = 51068    f0 = 2.01473e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(1-4)IDC1625_CC_930.csv", "") #QR = 50214    f0 = 2.01467e+09          #Done 1625

# Arms 1 to 4 at 1450
read_in_single_file_data_and_plot("sonnet_sims\\V8_(1-4)IDC1450_CC_890.csv", "") #QR = 50094    f0 = 2.03624e+09          #Done 1450


# Arms 1 to 4 at 1275
read_in_single_file_data_and_plot("sonnet_sims\\V8_(1-4)IDC1275_CC_850.csv", "") #QR = 50220    f0 = 2.05862e+09          #Done 1275
read_in_single_file_data_and_plot("sonnet_sims\\V8_(1-4)IDC1275_CC_870.csv", "") #QR = 48531    f0 = 2.05849e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(1-4)IDC1275_CC_890.csv", "") #QR = 46892    f0 = 2.05836e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(1-4)IDC1275_CC_900.csv", "") #QR = 46049    f0 = 2.05829e+09

# Arms 1 to 4 at 1100
read_in_single_file_data_and_plot("sonnet_sims\\V8_(1-4)IDC1100_CC_825.csv", "") #QR = 51816    f0 = 2.08081e+09          #REDONE 2
read_in_single_file_data_and_plot("sonnet_sims\\V8_(1-4)IDC1100_CC_830.csv", "") #QR = 49360    f0 = 2.08077e+09          #REDONE 1    #Done 1100
read_in_single_file_data_and_plot("sonnet_sims\\V8_(1-4)IDC1100_CC_870.csv", "") #QR = 51749    f0 = 2.08052e+09          #REDO_WRONG
read_in_single_file_data_and_plot("sonnet_sims\\V8_(1-4)IDC1100_CC_872.csv", "") #QR = 48219    f0 = 2.08051e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(1-4)IDC1100_CC_875.csv", "") #QR = 46784    f0 = 2.08049e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(1-4)IDC1100_CC_880.csv", "") #QR = 44238    f0 = 2.08052e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(1-4)IDC1100_CC_900.csv", "") #QR = 43525    f0 = 2.08033e+09



Arms_1_4 = {}
Arms_1_4["IDCL"] = [       1975,         1800,         1625,         1450,         1275,         1100]
Arms_1_4["CCL"] =  [       1030,          985,          930,          890,          850,          830]
Arms_1_4["f0"] =   [1.97012e+09,  1.99284e+09,  2.01467e+09,  2.03624e+09,  2.05862e+09,  2.08077e+09]
Arms_1_4["QR"] =   [      50214,        49403,        50214,        50094,        50220,        49360]

'''
###########
Arms 5 to 8
###########
'''
# Arms 5 to 8 at 1975
# Arms 5 to 8 at 1800
read_in_single_file_data_and_plot("sonnet_sims\\V8_(5-8)IDC1800_CC_775.csv", "") #QR = 50181    f0 = 2.11348e+09          #Done 1800
read_in_single_file_data_and_plot("sonnet_sims\\V8_(5-8)IDC1800_CC_800.csv", "") #QR = 47761    f0 = 2.11332e+09

# Arms 5 to 8 at 1625
read_in_single_file_data_and_plot("sonnet_sims\\V8_(5-8)IDC1625_CC_740.csv", "") #QR = 50730    f0 = 2.14434e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(5-8)IDC1625_CC_745.csv", "") #QR = 50063    f0 = 2.14431e+09          #Done 1625
read_in_single_file_data_and_plot("sonnet_sims\\V8_(5-8)IDC1625_CC_760.csv", "") #QR = 48660    f0 = 2.14421e+09

# Arms 5 to 8 at 1450
read_in_single_file_data_and_plot("sonnet_sims\\V8_(5-8)IDC1450_CC_720.csv", "") #QR = 49894    f0 = 2.17516e+09          #Done 1450
read_in_single_file_data_and_plot("sonnet_sims\\V8_(5-8)IDC1450_CC_725.csv", "") #QR = 49343    f0 = 2.17513e+09

# Arms 5 to 8 at 1275
read_in_single_file_data_and_plot("sonnet_sims\\V8_(5-8)IDC1275_CC_695.csv", "") #QR = 50082    f0 = 2.20741e+09          #Done 1275
read_in_single_file_data_and_plot("sonnet_sims\\V8_(5-8)IDC1275_CC_700.csv", "") #QR = 49460    f0 = 2.20738e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(5-8)IDC1275_CC_730.csv", "") #QR = 46345    f0 = 2.20716e+09

# Arms 5 to 8 at 1100
read_in_single_file_data_and_plot("sonnet_sims\\V8_(5-8)IDC1100_CC_680.csv", "") #QR = 49797    f0 = 2.23928e+09          #Done 1100
read_in_single_file_data_and_plot("sonnet_sims\\V8_(5-8)IDC1100_CC_700.csv", "") #QR = 47458    f0 = 2.23914e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(5-8)IDC1100_CC_720.csv", "") #QR = 45412    f0 = 2.23899e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(5-8)IDC1100_CC_740.csv", "") #QR = 43407    f0 = 2.23885e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(5-8)IDC1100_CC_750.csv", "") #QR = 42544    f0 = 2.23878e+09



Arms_5_8 = {}
Arms_5_8["IDCL"] = [               1975,         1800,         1625,         1450,         1275,         1100]
Arms_5_8["CCL"] =  [Arms_1_4["CCL"][-1],          775,          745,          720,          695,          680]
Arms_5_8["f0"] =   [ Arms_1_4["f0"][-1],  2.11348e+09,  2.14431e+09,  2.17516e+09,  2.20741e+09,  2.23928e+09]
Arms_5_8["QR"] =   [ Arms_1_4["QR"][-1],        50181,        50063,        49894,        50082,        49797]

'''
############
Arms 9 to 12
############
'''
# Arms 9 to 12 at 1975
# Arms 9 to 12 at 1800
read_in_single_file_data_and_plot("sonnet_sims\\V8_(9-12)IDC1800_CC_645.csv", "") #QR = 49918    f0 = 2.27905e+09          #Done 1800
read_in_single_file_data_and_plot("sonnet_sims\\V8_(9-12)IDC1800_CC_650.csv", "") #QR = 49331    f0 = 2.27901e+09

# Arms 9 to 12 at 1625
read_in_single_file_data_and_plot("sonnet_sims\\V8_(9-12)IDC1625_CC_620.csv", "") #QR = 49919    f0 = 2.31839e+09          #Done 1625

# Arms 9 to 12 at 1450
read_in_single_file_data_and_plot("sonnet_sims\\V8_(9-12)IDC1450_CC_600.csv", "") #QR = 49692    f0 = 2.35801e+09          #Done 1450

# Arms 9 to 12 at 1275
read_in_single_file_data_and_plot("sonnet_sims\\V8_(9-12)IDC1275_CC_575.csv", "") #QR = 49623    f0 = 2.39971e+09          #Done 1275

# Arms 9 to 12 at 1100
read_in_single_file_data_and_plot("sonnet_sims\\V8_(9-12)IDC1100_CC_555.csv", "") #QR = 50950    f0 = 2.44108e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(9-12)IDC1100_CC_560.csv", "") #QR = 50195    f0 = 2.44104e+09          #Done 1100
read_in_single_file_data_and_plot("sonnet_sims\\V8_(9-12)IDC1100_CC_570.csv", "") #QR = 48876    f0 = 2.44095e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(9-12)IDC1100_CC_590.csv", "") #QR = 46359    f0 = 2.44077e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(9-12)IDC1100_CC_610.csv", "") #QR = 43861    f0 = 2.44059e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(9-12)IDC1100_CC_625.csv", "") #QR = 42391    f0 = 2.44045e+09



Arms_9_12 = {}
Arms_9_12["IDCL"] = [               1975,         1800,         1625,         1450,         1275,         1100]
Arms_9_12["CCL"] =  [Arms_5_8["CCL"][-1],          645,          620,          600,          575,          560]
Arms_9_12["f0"] =   [ Arms_5_8["f0"][-1],  2.27905e+09,  2.31839e+09,  2.35801e+09,  2.39971e+09,  2.44104e+09]
Arms_9_12["QR"] =   [ Arms_5_8["QR"][-1],        49918,        49919,        49692,        49623,        50195]

'''
#############
Arms 13 to 16
#############
'''
# Arms 13 to 16 at 1975
# Arms 13 to 16 at 1800
read_in_single_file_data_and_plot("sonnet_sims\\V8_(13-16)IDC1800_CC_530.csv", "") #QR = 50128    f0 = 2.49230e+09          #Done 1800

# Arms 13 to 16 at 1625
read_in_single_file_data_and_plot("sonnet_sims\\V8_(13-16)IDC1625_CC_505.csv", "") #QR = 50419    f0 = 2.54362e+09          #Done 1625
read_in_single_file_data_and_plot("sonnet_sims\\V8_(13-16)IDC1625_CC_510.csv", "") #QR = 49536    f0 = 2.54356e+09

# Arms 13 to 16 at 1450
read_in_single_file_data_and_plot("sonnet_sims\\V8_(13-16)IDC1450_CC_480.csv", "") #QR = 51050    f0 = 2.59576e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(13-16)IDC1450_CC_485.csv", "") #QR = 50093    f0 = 2.59571e+09          #Done 1450
read_in_single_file_data_and_plot("sonnet_sims\\V8_(13-16)IDC1450_CC_490.csv", "") #QR = 48835    f0 = 2.59565e+09

# Arms 13 to 16 at 1275
read_in_single_file_data_and_plot("sonnet_sims\\V8_(13-16)IDC1275_CC_470.csv", "") #QR = 49901    f0 = 2.65083e+09          #Done 1275

# Arms 13 to 16 at 1100
read_in_single_file_data_and_plot("sonnet_sims\\V8_(13-16)IDC1100_CC_450.csv", "") #QR = 50246    f0 = 2.70589e+09          #Done 1100
read_in_single_file_data_and_plot("sonnet_sims\\V8_(13-16)IDC1100_CC_470.csv", "") #QR = 46953    f0 = 2.70565e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(13-16)IDC1100_CC_490.csv", "") #QR = 43978    f0 = 2.7054e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(13-16)IDC1100_CC_510.csv", "") #QR = 41206    f0 = 2.70516e+09



Arms_13_16 = {}
Arms_13_16["IDCL"] = [                1975,         1800,         1625,         1450,         1275,         1100]
Arms_13_16["CCL"] =  [Arms_9_12["CCL"][-1],          530,          505,          485,          470,          450]
Arms_13_16["f0"] =   [ Arms_9_12["f0"][-1],  2.49230e+09,  2.54362e+09,  2.59571e+09,  2.65083e+09,  2.70589e+09]
Arms_13_16["QR"] =   [ Arms_9_12["QR"][-1],        50128,        50419,        50093,        49901,        50246]

'''
#############
Arms 17 to 20
#############
'''
# Arms 17 to 20 at 1975
# Arms 17 to 20 at 1800
read_in_single_file_data_and_plot("sonnet_sims\\V8_(17-20)IDC1800_CC_420.csv", "") #QR = 50286    f0 = 2.77425e+09          #Done 1800
read_in_single_file_data_and_plot("sonnet_sims\\V8_(17-20)IDC1800_CC_425.csv", "") #QR = 49241    f0 = 2.77419e+09

# Arms 17 to 20 at 1625
read_in_single_file_data_and_plot("sonnet_sims\\V8_(17-20)IDC1625_CC_390.csv", "") #QR = 54905    f0 = 2.84365e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(17-20)IDC1625_CC_395.csv", "") #QR = 52320    f0 = 2.84358e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(17-20)IDC1625_CC_396.csv", "") #QR = 51237    f0 = 2.84357e+09          #Done 1625 CLOSEST
read_in_single_file_data_and_plot("sonnet_sims\\V8_(17-20)IDC1625_CC_397.csv", "") #QR = 53881    f0 = 2.84355e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(17-20)IDC1625_CC_398.csv", "") #QR = 45344    f0 = 2.84354e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(17-20)IDC1625_CC_400.csv", "") #QR = 47107    f0 = 2.84351e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(17-20)IDC1625_CC_405.csv", "") #QR = 48877    f0 = 2.84344e+09

# Arms 17 to 20 at 1450
read_in_single_file_data_and_plot("sonnet_sims\\V8_(17-20)IDC1450_CC_375.csv", "") #QR = 50495    f0 = 2.9147e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(17-20)IDC1450_CC_380.csv", "") #QR = 49522    f0 = 2.91462e+09          #Done 1450
read_in_single_file_data_and_plot("sonnet_sims\\V8_(17-20)IDC1450_CC_385.csv", "") #QR = 48584    f0 = 2.91454e+09

# Arms 17 to 20 at 1275
read_in_single_file_data_and_plot("sonnet_sims\\V8_(17-20)IDC1275_CC_355.csv", "") #QR = 50773    f0 = 2.99073e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(17-20)IDC1275_CC_360.csv", "") #QR = 49734    f0 = 2.99065e+09          #Done 1275
read_in_single_file_data_and_plot("sonnet_sims\\V8_(17-20)IDC1275_CC_365.csv", "") #QR = 48727    f0 = 2.99057e+09

# Arms 17 to 20 at 1100
read_in_single_file_data_and_plot("sonnet_sims\\V8_(17-20)IDC1100_CC_345.csv", "") #QR = 49209    f0 = 3.06698e+09          #Done 1100
read_in_single_file_data_and_plot("sonnet_sims\\V8_(17-20)IDC1100_CC_355.csv", "") #QR = 47230    f0 = 3.0668e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(17-20)IDC1100_CC_380.csv", "") #QR = 42741    f0 = 3.06634e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(17-20)IDC1100_CC_390.csv", "") #QR = 41094    f0 = 3.06616e+09



Arms_17_20 = {}
Arms_17_20["IDCL"] = [                 1975,         1800,         1625,         1450,         1275,         1100]
Arms_17_20["CCL"] =  [Arms_13_16["CCL"][-1],          420,          396,          380,          360,          345]
Arms_17_20["f0"] =   [ Arms_13_16["f0"][-1],  2.77425e+09,  2.84357e+09,  2.91462e+09,  2.99065e+09,  3.06698e+09]
Arms_17_20["QR"] =   [ Arms_13_16["QR"][-1],        50286,        51237,        49522,        49734,        49209]

'''
#############
Arms 21 to 24
#############
'''
# Arms 21 to 24 at 1975
# Arms 21 to 24 at 1800
read_in_single_file_data_and_plot("sonnet_sims\\V8_(21-24)IDC1800_CC_290.csv", "") #QR = 64689    f0 = 3.16109e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(21-24)IDC1800_CC_300.csv", "") #QR = 64625    f0 = 3.16089e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(21-24)IDC1800_CC_305.csv", "") #QR = 51825    f0 = 3.16078e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(21-24)IDC1800_CC_306.csv", "") #QR = 51089    f0 = 3.16076e+09          #Done 1800 CLOSEST
read_in_single_file_data_and_plot("sonnet_sims\\V8_(21-24)IDC1800_CC_307.csv", "") #QR = 54913    f0 = 3.16074e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(21-24)IDC1800_CC_310.csv", "") #QR = 42398    f0 = 3.16067e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(21-24)IDC1800_CC_325.csv", "") #QR = 40994    f0 = 3.16037e+09

# Arms 21 to 24 at 1625
read_in_single_file_data_and_plot("sonnet_sims\\V8_(21-24)IDC1625_CC_290.csv", "") #QR = 58794    f0 = 3.25684e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(21-24)IDC1625_CC_292.csv", "") #QR = 51553    f0 = 3.25680e+09          #Done 1625 CLOSEST
read_in_single_file_data_and_plot("sonnet_sims\\V8_(21-24)IDC1625_CC_293.csv", "") #QR = 51971    f0 = 3.25677e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(21-24)IDC1625_CC_295.csv", "") #QR = 48641    f0 = 3.25673e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(21-24)IDC1625_CC_300.csv", "") #QR = 48316    f0 = 3.25662e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(21-24)IDC1625_CC_310.csv", "") #QR = 42686    f0 = 3.25639e+09

# Arms 21 to 24 at 1450
read_in_single_file_data_and_plot("sonnet_sims\\V8_(21-24)IDC1450_CC_270.csv", "") #QR = 51955    f0 = 3.35647e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(21-24)IDC1450_CC_272.csv", "") #QR = 53367    f0 = 3.35642e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(21-24)IDC1450_CC_273.csv", "") #QR = 51163    f0 = 3.35639e+09          #Done 1450 CLOSEST
read_in_single_file_data_and_plot("sonnet_sims\\V8_(21-24)IDC1450_CC_274.csv", "") #QR = 47759    f0 = 3.35637e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(21-24)IDC1450_CC_275.csv", "") #QR = 48431    f0 = 3.35634e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(21-24)IDC1450_CC_280.csv", "") #QR = 41995    f0 = 3.35622e+09

# Arms 21 to 24 at 1275
read_in_single_file_data_and_plot("sonnet_sims\\V8_(21-24)IDC1275_CC_245.csv", "") #QR = 49908    f0 = 3.46382e+09          #Done 1275
read_in_single_file_data_and_plot("sonnet_sims\\V8_(21-24)IDC1275_CC_250.csv", "") #QR = 49126    f0 = 3.46368e+09

# Arms 21 to 24 at 1100
read_in_single_file_data_and_plot("sonnet_sims\\V8_(21-24)IDC1100_CC_230.csv", "") #QR = 50356    f0 = 3.5716e+09          #Done 1100
read_in_single_file_data_and_plot("sonnet_sims\\V8_(21-24)IDC1100_CC_245.csv", "") #QR = 46552    f0 = 3.57113e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(21-24)IDC1100_CC_260.csv", "") #QR = 43058    f0 = 3.57066e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(21-24)IDC1100_CC_276.csv", "") #QR = 39756    f0 = 3.57016e+09



Arms_21_24 = {}
Arms_21_24["IDCL"] = [                 1975,         1800,        1625,         1450,         1275,        1100]
Arms_21_24["CCL"] =  [Arms_17_20["CCL"][-1],          306,         292,          273,          245,         230]
Arms_21_24["f0"] =   [ Arms_17_20["f0"][-1],  3.16076e+09,  3.25680e+09,  3.35639e+09,  3.46382e+09,  3.5716e+09]
Arms_21_24["QR"] =   [ Arms_17_20["QR"][-1],        51089,       51553,        51163,        49908,       50356]

'''
#############
Arms 25 to 28
#############
'''
# Arms 25 to 28 at 1975
# Arms 25 to 28 at 1800
read_in_single_file_data_and_plot("sonnet_sims\\V8_(25-28)IDC1800_CC_205.csv", "") #QR = 50261    f0 = 3.68914e+09          #Done 1800
read_in_single_file_data_and_plot("sonnet_sims\\V8_(25-28)IDC1800_CC_210.csv", "") #QR = 48905    f0 = 3.68896e+09

# Arms 25 to 28 at 1625
read_in_single_file_data_and_plot("sonnet_sims\\V8_(25-28)IDC1625_CC_185.csv", "") #QR = 50618    f0 = 3.8094e+09          #Done 1625
read_in_single_file_data_and_plot("sonnet_sims\\V8_(25-28)IDC1625_CC_190.csv", "") #QR = 49135    f0 = 3.8092e+09

# Arms 25 to 28 at 1450
read_in_single_file_data_and_plot("sonnet_sims\\V8_(25-28)IDC1450_CC_165.csv", "") #QR = 52077    f0 = 3.93075e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(25-28)IDC1450_CC_170.csv", "") #QR = 50545    f0 = 3.93053e+09          #Done 1450
read_in_single_file_data_and_plot("sonnet_sims\\V8_(25-28)IDC1450_CC_175.csv", "") #QR = 48647    f0 = 3.9303e+09

# Arms 25 to 28 at 1275
read_in_single_file_data_and_plot("sonnet_sims\\V8_(25-28)IDC1275_CC_155.csv", "") #QR = 50519    f0 = 4.05583e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(25-28)IDC1275_CC_157.csv", "") #QR = 50013    f0 = 4.05573e+09          #Done 1275

# Arms 25 to 28 at 1100
read_in_single_file_data_and_plot("sonnet_sims\\V8_(25-28)IDC1100_CC_140.csv", "") #QR = 51485    f0 = 4.1748e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(25-28)IDC1100_CC_145.csv", "") #QR = 49843    f0 = 4.17453e+09          #Done 1100
read_in_single_file_data_and_plot("sonnet_sims\\V8_(25-28)IDC1100_CC_150.csv", "") #QR = 48347    f0 = 4.17426e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(25-28)IDC1100_CC_170.csv", "") #QR = 42666    f0 = 4.17318e+09
read_in_single_file_data_and_plot("sonnet_sims\\V8_(25-28)IDC1100_CC_190.csv", "") #QR = 37892    f0 = 4.1721e+09



Arms_25_28 = {}
Arms_25_28["IDCL"] = [                 1975,         1800,        1625,         1450,         1275,         1100]
Arms_25_28["CCL"] =  [Arms_21_24["CCL"][-1],          205,         185,          170,          157,          145]
Arms_25_28["f0"] =   [ Arms_21_24["f0"][-1],  3.68914e+09,  3.8094e+09,  3.93053e+09,  4.05573e+09,  4.17453e+09]
Arms_25_28["QR"] =   [ Arms_21_24["QR"][-1],        50261,       50618,        50545,        50013,        49843]





'''
###############################################################################
                        PLOTTING AND FITTING ALL THE DATA                       
###############################################################################
'''
plt.close("all")

names = ["Arms_1_4",
          "Arms_5_8",
          "Arms_9_12",
          "Arms_13_16",
          "Arms_17_20",
          "Arms_21_24",
          "Arms_25_28"]

all_data = {}
all_data[names[0]] = Arms_1_4
all_data[names[1]] = Arms_5_8
all_data[names[2]] = Arms_9_12
all_data[names[3]] = Arms_13_16
all_data[names[4]] = Arms_17_20
all_data[names[5]] = Arms_21_24
all_data[names[6]] = Arms_25_28

all_data_fits = {}

all_in_one_plot = True

if all_in_one_plot:
    fig=plt.figure(str("ALL Fitting"), figsize=(18,10))
    rows = 2
    cols = 2
    grid = plt.GridSpec(rows, cols, top=0.95, bottom=0.092, left=0.05, right=0.95, hspace=0.0, wspace=0.2)
    
    ax0 = plt.subplot(grid[0:2, 0])
    ax1 = plt.subplot(grid[0, 1])
    ax2 = plt.subplot(grid[1, 1], sharex=ax1)

for i, name in enumerate(names):
    
    data_IDCL = all_data[name]["IDCL"]
    data_CCL  = all_data[name]["CCL"]
    data_f0   = all_data[name]["f0"]
    all_data_fits[name] = {}
    
    
    if not all_in_one_plot:
        fig=plt.figure(str(name + " Fitting"), figsize=(18,10))
        rows = 2
        cols = 2
        grid = plt.GridSpec(rows, cols, top=0.95, bottom=0.092, left=0.05, right=0.95, hspace=0.0, wspace=0.2)
        
        ax0 = plt.subplot(grid[0:2, 0])
        ax1 = plt.subplot(grid[0, 1])
        ax2 = plt.subplot(grid[1, 1], sharex=ax1)
        
        scatter_marker = "o"
        scatter_marker_size = 50
    else:
        
        scatter_marker = "o" if i%2==0 else "x"
        scatter_marker_size = 50
        
        
    # Axes 0 CCL against IDCL
    ax0.scatter(data_IDCL, data_CCL, marker=scatter_marker)
    
    CCL_IDCL_polyfit = np.polyfit(data_IDCL, data_CCL, deg=3)
    CCL_IDCL_polyfunc = np.poly1d(CCL_IDCL_polyfit)
    x_ax = np.linspace(1100, 1975, 100)
    label_text = str(name + " fit = {0}".format(CCL_IDCL_polyfit))
    ax0.plot(x_ax, CCL_IDCL_polyfunc(x_ax), linewidth=0.3, label=label_text)
    
    all_data_fits[name]["CCL_IDCL"] = CCL_IDCL_polyfit
    
    ax0.set_title("CCL vs IDCL", loc="left")
    ax0.set_xlabel("IDCL     (um)")
    ax0.set_ylabel("CCL     (um)")
    ax0.grid(alpha=0.3)
    ax0.legend(loc="best")

    # Axes 1 IDCL against f0
    ax1.scatter(data_f0, data_IDCL, marker=scatter_marker)
    
    IDCL_f0_polyfit = np.polyfit(data_f0, data_IDCL, deg=3)
    IDCL_f0_polyfunc = np.poly1d(IDCL_f0_polyfit)
    x_ax = np.linspace(min(data_f0), max(data_f0), 100)
    label_text = str(name + " fit = {0}".format(IDCL_f0_polyfit))
    ax1.plot(x_ax, IDCL_f0_polyfunc(x_ax), linewidth=0.3, label=label_text)
    
    all_data_fits[name]["IDCL_f0"] = IDCL_f0_polyfit
    
    ax1.set_title("IDCL vs f0", loc="left")
    ax1.set_xlabel("f0     (Hz)")
    ax1.set_ylabel("IDCL     (um)")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="best")

    # Axes 2 CCL against f0
    ax2.scatter(data_f0, data_CCL, marker=scatter_marker)
    
    CCL_f0_polyfit = np.polyfit(data_f0, data_CCL, deg=3)
    CCL_f0_polyfunc = np.poly1d(CCL_f0_polyfit)
    x_ax = np.linspace(min(data_f0), max(data_f0), 100)
    label_text = str(name + " fit = {0}".format(CCL_f0_polyfit))
    ax2.plot(x_ax, CCL_f0_polyfunc(x_ax), linewidth=0.3, label=label_text)
    
    all_data_fits[name]["CCL_f0"] = CCL_f0_polyfit
    
    ax2.set_title("CCL vs f0", loc="left")
    ax2.set_xlabel("f0     (Hz)")
    ax2.set_ylabel("CCL     (um)")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="best")


    fig.show()



'''
###############################################################################
                      PLOTTING AND FITTING SPECIFIC DATA                      
###############################################################################
'''
# arms_name = "Arms_1_4"
# arms_name = "Arms_5_8"
# arms_name = "Arms_9_12"
# arms_name = "Arms_13_16"
arms_name = "Arms_17_20"
# arms_name = "Arms_21_24"
# arms_name = "Arms_25_28"

specific_data = all_data[arms_name]

specific_data_IDCL = specific_data["IDCL"]
specific_data_CCL  = specific_data["CCL"]
specific_data_f0   = specific_data["f0"]


fig=plt.figure(str(arms_name + " Fitting"), figsize=(18,10))
rows = 2
cols = 2
grid = plt.GridSpec(rows, cols, top=0.95, bottom=0.092, left=0.05, right=0.95, hspace=0.0, wspace=0.2)

ax0 = plt.subplot(grid[0:2, 0])
ax1 = plt.subplot(grid[0, 1])
ax2 = plt.subplot(grid[1, 1], sharex=ax1)


# Axes 0 CCL against IDCL
ax0.scatter(specific_data_IDCL, specific_data_CCL)

CCL_IDCL_polyfit = np.polyfit(specific_data_IDCL, specific_data_CCL, deg=3)
CCL_IDCL_polyfunc = np.poly1d(CCL_IDCL_polyfit)
x_ax = np.linspace(1100, 1975, 100)
ax0.plot(x_ax, CCL_IDCL_polyfunc(x_ax), linewidth=0.3, label=str("fit = {0}".format(CCL_IDCL_polyfit)))

ax0.set_title("CCL vs IDCL", loc="left")
ax0.set_xlabel("IDCL     (um)")
ax0.set_ylabel("CCL     (um)")
ax0.grid(alpha=0.3)
ax0.legend(loc="best")


# Axes 1 IDCL against f0
ax1.scatter(specific_data_f0, specific_data_IDCL)

IDCL_f0_polyfit = np.polyfit(specific_data_f0, specific_data_IDCL, deg=3)
IDCL_f0_polyfunc = np.poly1d(IDCL_f0_polyfit)
x_ax = np.linspace(min(specific_data_f0), max(specific_data_f0), 100)
ax1.plot(x_ax, IDCL_f0_polyfunc(x_ax), linewidth=0.3, label=str("fit = {0}".format(IDCL_f0_polyfit)))

ax1.set_title("IDCL vs f0", loc="left")
ax1.set_xlabel("f0     (Hz)")
ax1.set_ylabel("IDCL     (um)")
ax1.grid(alpha=0.3)
ax1.legend(loc="best")


# Axes 2 CCL against f0
ax2.scatter(specific_data_f0, specific_data_CCL)

CCL_f0_polyfit = np.polyfit(specific_data_f0, specific_data_CCL, deg=3)
CCL_f0_polyfunc = np.poly1d(CCL_f0_polyfit)
x_ax = np.linspace(min(specific_data_f0), max(specific_data_f0), 100)
ax2.plot(x_ax, CCL_f0_polyfunc(x_ax), linewidth=0.3, label=str("fit = {0}".format(CCL_f0_polyfit)))

ax2.set_title("CCL vs f0", loc="left")
ax2.set_xlabel("f0     (Hz)")
ax2.set_ylabel("CCL     (um)")
ax2.grid(alpha=0.3)
ax2.legend(loc="best")


fig.show()





def testing_get_idc_cc(f0):
    """
    Parameters
    ----------
    f0 : float, int
        The desired frequency (**in Hz**) for the KID.
    
    Returns
    -------
    IDC_array : list
        This is a list of length 28. Each element is a float representing the
        length of an IDC arm in um. The first element in this list is the
        length of arm 1, the second element is the length of the second arm
        and so on...
    
    CCL : float
        This is the length of the coupling capacitor in um.
    """
    min_freq = 1970120000.0
    max_freq = 4174530000.0
    
    if f0 < min_freq or f0 > max_freq:
        print("error occured")
        raise(Exception("Desired frequency out of range. Should be within {0} -> {1}".format(min_freq, max_freq)))
    
    Arms_1_4_min_max_freqs   = [1970120000.0, 2080770000.0]
    Arms_5_8_min_max_freqs   = [2080770000.0, 2239280000.0]
    Arms_9_12_min_max_freqs  = [2239280000.0, 2441040000.0]
    Arms_13_16_min_max_freqs = [2441040000.0, 2705890000.0]
    Arms_17_20_min_max_freqs = [2705890000.0, 3066980000.0]
    Arms_21_24_min_max_freqs = [3066980000.0, 3571600000.0]
    Arms_25_28_min_max_freqs = [3571600000.0, 4174530000.0]
    
    
    # Fits from data.
    Arms_1_4_IDC_f0_polyfit   = [ 4.40593480e-23, -2.67909667e-13,  5.34980654e-04, -3.49055119e+05]
    Arms_5_8_IDC_f0_polyfit   = [ 1.87200482e-23, -1.21162839e-13,  2.55765811e-04, -1.74274315e+05]
    Arms_9_12_IDC_f0_polyfit  = [ 3.12896341e-24, -2.11432074e-14,  4.31831099e-05, -2.38375201e+04]
    Arms_13_16_IDC_f0_polyfit = [ 9.30798289e-25, -6.46468470e-15,  1.14603758e-05, -1.01767002e+03]
    Arms_17_20_IDC_f0_polyfit = [ 1.14601699e-25, -4.38019706e-16, -2.76247107e-06,  1.03869882e+04]
    Arms_21_24_IDC_f0_polyfit = [-9.89518123e-26,  1.33651102e-15, -7.32943707e-06,  1.47377098e+04]
    Arms_25_28_IDC_f0_polyfit = [-2.78899336e-25,  3.28681434e-15, -1.43346369e-05,  2.39520423e+04]
    
    Arms_1_4_CC_f0_polyfit   = [ 1.02325925e-22, -6.15094874e-13,  1.23020222e-03, -8.17665744e+05]
    Arms_5_8_CC_f0_polyfit   = [-2.24898154e-23,  1.49655879e-13, -3.32533402e-04,  2.47412710e+05]
    Arms_9_12_CC_f0_polyfit  = [-3.91010408e-24,  2.86521516e-14, -7.04177628e-05,  5.85968409e+04]
    Arms_13_16_CC_f0_polyfit = [-3.27444516e-24,  2.59330629e-14, -6.87761882e-05,  6.15470736e+04]
    Arms_17_20_CC_f0_polyfit = [-1.09047056e-24,  9.81438047e-15, -2.96571843e-05,  3.04441295e+04]
    Arms_21_24_CC_f0_polyfit = [-6.20264683e-25,  6.32823387e-15, -2.16963227e-05,  2.52534933e+04]
    Arms_25_28_CC_f0_polyfit = [-1.84259106e-25,  2.25955100e-15, -9.33504064e-06,  1.31424615e+04]
    
    
    # Making fits into polynomial functions.
    Arms_1_4_IDC_f0_polyfunc   = np.poly1d(Arms_1_4_IDC_f0_polyfit)
    Arms_5_8_IDC_f0_polyfunc   = np.poly1d(Arms_5_8_IDC_f0_polyfit)
    Arms_9_12_IDC_f0_polyfunc  = np.poly1d(Arms_9_12_IDC_f0_polyfit)
    Arms_13_16_IDC_f0_polyfunc = np.poly1d(Arms_13_16_IDC_f0_polyfit)
    Arms_17_20_IDC_f0_polyfunc = np.poly1d(Arms_17_20_IDC_f0_polyfit)
    Arms_21_24_IDC_f0_polyfunc = np.poly1d(Arms_21_24_IDC_f0_polyfit)
    Arms_25_28_IDC_f0_polyfunc = np.poly1d(Arms_25_28_IDC_f0_polyfit)
    
    Arms_1_4_CC_f0_polyfunc   = np.poly1d(Arms_1_4_CC_f0_polyfit)
    Arms_5_8_CC_f0_polyfunc   = np.poly1d(Arms_5_8_CC_f0_polyfit)
    Arms_9_12_CC_f0_polyfunc  = np.poly1d(Arms_9_12_CC_f0_polyfit)
    Arms_13_16_CC_f0_polyfunc = np.poly1d(Arms_13_16_CC_f0_polyfit)
    Arms_17_20_CC_f0_polyfunc = np.poly1d(Arms_17_20_CC_f0_polyfit)
    Arms_21_24_CC_f0_polyfunc = np.poly1d(Arms_21_24_CC_f0_polyfit)
    Arms_25_28_CC_f0_polyfunc = np.poly1d(Arms_25_28_CC_f0_polyfit)
    
    
    # Getting the IDC and CC lengths from the relevant polynomial functions.
    
    if f0 >= Arms_1_4_min_max_freqs[0] and f0 <= Arms_1_4_min_max_freqs[1]:
        
        IDCL = Arms_1_4_IDC_f0_polyfunc(f0)
        CCL = Arms_1_4_CC_f0_polyfunc(f0)
        
        IDC_array = np.zeros(28)
        
        IDC_array[0:4] = IDCL
        IDC_array[4:] = 1975
        
        return IDC_array, CCL
        
    elif f0 >= Arms_5_8_min_max_freqs[0] and f0 <= Arms_5_8_min_max_freqs[1]:
        
        IDCL = Arms_5_8_IDC_f0_polyfunc(f0)
        CCL = Arms_5_8_CC_f0_polyfunc(f0)
        
        IDC_array = np.zeros(28)
        
        IDC_array[0:4] = 1100
        IDC_array[4:8] = IDCL
        IDC_array[8:] = 1975
        
        return IDC_array, CCL
        
    elif f0 >= Arms_9_12_min_max_freqs[0] and f0 <= Arms_9_12_min_max_freqs[1]:
        
        IDCL = Arms_9_12_IDC_f0_polyfunc(f0)
        CCL = Arms_9_12_CC_f0_polyfunc(f0)
        
        IDC_array = np.zeros(28)
        
        IDC_array[0:8] = 1100
        IDC_array[8:12] = IDCL
        IDC_array[12:] = 1975
        
        return IDC_array, CCL
        
    elif f0 >= Arms_13_16_min_max_freqs[0] and f0 <= Arms_13_16_min_max_freqs[1]:
        
        IDCL = Arms_13_16_IDC_f0_polyfunc(f0)
        CCL = Arms_13_16_CC_f0_polyfunc(f0)
        
        IDC_array = np.zeros(28)
        
        IDC_array[0:12] = 1100
        IDC_array[12:16] = IDCL
        IDC_array[16:] = 1975
        
        return IDC_array, CCL
        
    elif f0 >= Arms_17_20_min_max_freqs[0] and f0 <= Arms_17_20_min_max_freqs[1]:
        
        IDCL = Arms_17_20_IDC_f0_polyfunc(f0)
        CCL = Arms_17_20_CC_f0_polyfunc(f0)
        
        IDC_array = np.zeros(28)
        
        IDC_array[0:16] = 1100
        IDC_array[16:20] = IDCL
        IDC_array[20:] = 1975
        
        return IDC_array, CCL
        
    elif f0 >= Arms_21_24_min_max_freqs[0] and f0 <= Arms_21_24_min_max_freqs[1]:
        
        IDCL = Arms_21_24_IDC_f0_polyfunc(f0)
        CCL = Arms_21_24_CC_f0_polyfunc(f0)
        
        IDC_array = np.zeros(28)
        
        IDC_array[0:20] = 1100
        IDC_array[20:24] = IDCL
        IDC_array[24:] = 1975
        
        return IDC_array, CCL
        
    elif f0 >= Arms_25_28_min_max_freqs[0] and f0 <= Arms_25_28_min_max_freqs[1]:
        
        IDCL = Arms_25_28_IDC_f0_polyfunc(f0)
        CCL = Arms_25_28_CC_f0_polyfunc(f0)
        
        IDC_array = np.zeros(28)
        
        IDC_array[0:24] = 1100
        IDC_array[24:] = IDCL
        
        return IDC_array, CCL
    
    else:
        print("error occured")
        return Exception("error in obtaining IDCL and CCL")


def time_test():
    import time
    test_f0s = np.linspace(1970120000, 4174530000, 435*4)
    time_taken_list = []
    for f in test_f0s:
        start_time = time.time()
        IDCLs, CCL = testing_get_idc_cc(f)
        time_taken = time.time() - start_time
        time_taken_list.append(time_taken)
    
    new_times = [i for i in time_taken_list if i != 0]
    avg_time = np.average(new_times)
    
    print("The average time was: " + str(avg_time))