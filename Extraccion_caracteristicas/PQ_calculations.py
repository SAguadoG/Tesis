"""
This file includes a class for done all PQ calculation over data.
Config is loaded at object creation an then only calculation is invoked.
mode:
1 process all data
2 time window input data
config:
config file, which must include Fs for window analysis,
and in some situations other parameters
window_s:
seconds of each window
use_acum:
keep in a acum segments not included in windows
parameters:
list of keywords for config analysis
k - kurtosis
s - skewness
v - variance, needed nominal voltage value and probe scale if applied
rms - rms value of input data, neded probe scale if applied
pqi - PQindex, normal k s v values
all harmonic calculations use Hanning window
thd - total harmonic distortion
dft - returns all dft rms amplitudes vector
dft_f - returns frequency axis of dft
thdg - total harmonic distortion using harmonic groups (cuadractic sum all +-4
bins and half +-5 bin)
Yg - Values of harmonic groups
thdsg - Total harmonic distortion using harmonic sub groups (harmonic component
and adjacent)
Ysg - Values of harmonic sub groups
thdsgi - total harmonic distortion of interharmonic sub-groups ( cuadratic sum
of inter harmonic components but adjacents to harmonic components), thd
calculated respect to main harmonic sub-group
Ysgi - Value of inter harmonic sub-groups
thdsub - thd of sub-harmononic group (all bins under fundamental but adjacent),
calculatic in relation of main harmonic sub-grups
Ygsub- Value of sup-harmonic group
thdyb - thd of 2.1-9khz harmonic groups (grouped in 200Hz groups),
calculated in relation with main harmonic suf-group
Yb - Values of 2.1-9kHz harmonic groups
thdsup - thd of harmonics groups over 9kHz, grouped in 1kHz, in relation with
 fundamental sub-group
Ysup - Value of hamonic groups over 9kHz

config parameters:
Vrms - voltage rms of power system
"""

import numpy as np

class PqCalculation():

    def __init__(self, config,mode=1, window_s=0,use_acum=False, parameters=[]):
        # config file
        self.config = config
        # keep variables in self
        self.mode = mode
        self.window_s = window_s
        self.parameters = parameters
        if mode == 2 and (self.config['Fs']<=0 or window_s <= 0):
            raise Exception('Must be indicated Fs and Window in mode 2, windowed data')
        #itialize acum
        self.data_acum_x = np.array([])
        self.data_acum_t = np.array([], dtype='datetime64[ns]')
        self.use_acum = use_acum

    # fuction to clear acum
    def clear_acum(self):
        self.data_acum = np.array([])

    #fuction to config acum
    def acum_config(self,use_acum=False):
        self.use_acum = use_acum

    # master fuction to launch analysis
    def analyze(self,x_signal,t_signal,return_t0_N=False):
        # check vector length
        equal = x_signal.shape[0] == t_signal.shape[0]
        if not equal:
            raise Exception(
                'vectors does not have same length')
        # set fuction window or not
        if self.mode == 2:
            # window function return a list of dict
            output = self.window_data( x_signal, t_signal)
        else:
            # signal function return a dict, for normalize output, it
            # is turned into a list
            output = [self.calculate( x_signal, t_signal[0])]

        # if required get and return t0 and N
        if return_t0_N:
            # get t0 and N
            t0 = t_signal[0]
            N = t_signal.shape[0]

            return output, t0, N

        else:

            return output

    # function to launch alysis when windowed
    def window_data(self,x_signal,t_signal,parameters=[],window_s=0):
        # check in window_s are given to function, if no, self.window_s
        # are used
        if window_s == 0:
            window_s = self.window_s
        # check in parameters are given to function, if no, self.parameters
        # are used
        if not parameters:
            parameters = self.parameters
        #calculate window lenght in number of points
        window_length = int(window_s*self.config['Fs'])
        #if use data acum
        if self.use_acum:
            x_signal = np.append(self.data_acum_x,x_signal)
            t_signal = np.append(self.data_acum_t,t_signal)
        # take signal length
        length = x_signal.shape[0]
        # calculate number of segments, with integer divission
        N = int(length // window_length)
        # take segments and acum results
        output = []
        for i_segment in range(0,N):
            # really range is frm i*M to (i+1)*M-1, but python does not take
            # last index, so one more should be indicated
            x_segment = x_signal[i_segment*window_length:(i_segment+1)*window_length]
            t_segment0 = t_signal[i_segment*window_length]
            output.append(self.calculate(x_segment,t_segment0,parameters))
        # if use data acum
        if self.use_acum:
            # if there are data not taken in windows
            if (N)*window_length<length:
                #add to acum
                self.data_acum_x = np.append(self.data_acum_x,x_signal[(N)*window_length:])
                self.data_acum_t = np.append(self.data_acum_t,t_signal[(N)*window_length:])


        return output



    #function to calculate, which is invoked from windowed
    def calculate(self,x_signal,t0,parameters=[]):
        # check in parameters are given to fuction, if no, slef.parameters
        # are used
        if not parameters:
            parameters = self.parameters
        # initialize output with t0
        output = {'t0':t0}
        # if any HOS variable required
        if any(item in parameters for item in ['k','s','v','pqi']):
            # HOS are calculated
            hos_dict = self.hos(x_signal)
            # and to output are given only the reqeuired
            if 'v' in parameters:
                key = 'Variance'
                output.update({key:hos_dict[key]})
            if 's' in parameters:
                key = 'Skewness'
                output.update({key: hos_dict[key]})
            if 'k' in parameters:
                key = 'Kurtosis'
                output.update({key:hos_dict[key]})
            # in addition, if PQi is reqeuired, calcualtion is done
            #with HOS data
            if 'pqi' in parameters:
                key = 'PQ_index'
                output.update({key:self.pqindex(hos_dict)})
        # if RMS reqeuired, is calculated
        if 'rms' in parameters:
            RMS = np.sqrt(np.mean(np.array(x_signal) ** 2))
            key = 'RMS'
            output.update({key: RMS})
        # if any harmonic calculation is required
        if any(item in parameters for item in ['thd','dft', 'thdg', 'Yg',
                                                    'thdsg','Ysg','thdsgi',
                                                    'Ysgi','thdsub','Ygsub',
                                                    'Yb','thdyb','Ysup',
                                                    'thdsup']):
            # dft is calculated
            dft, f = self.my_dft( x_signal, self.config['Fs'], window_type='hann')
            # harmonics are calculated with dft. this returns only requeired parameters
            harm_out = self.harm_une(dft,f,parameters)
            output.update(harm_out)


        return output

    #funtion to calculate HOS using cumulatns
    def hos(self,x_signal):
        # normalize signal
        norm_signal = x_signal / (self.config['Vrms'] * np.sqrt(2))
        #create empty output dict
        HOS_dict = dict()
        try:
            # centre data, substracting mean
            mean_value = np.mean(norm_signal)
            unbias_data = norm_signal - mean_value
            N = unbias_data.size
            #calculate power 2 3 4 of data, needed for cumulants
            unbias_data_2 = unbias_data * unbias_data
            unbias_data_3 = unbias_data_2 * unbias_data
            unbias_data_4 = unbias_data_3 * unbias_data
            Variance = np.sum(unbias_data_2) / N
            Skewness = (np.sum(unbias_data_3) / N) / (np.power(Variance, 3 / 2))
            Kurtosis = (np.sum(unbias_data_4) / N - 3 * np.power(Variance, 2)) / (np.power(Variance, 2))
            HOS_dict['Variance'] = Variance
            HOS_dict['Skewness'] = Skewness
            HOS_dict['Kurtosis'] = Kurtosis
        except Exception as e:
            print(e)

        return HOS_dict

    #function to calculate PQ indez from HOS and normal values in config
    def pqindex(self, HOS_dict):
        # PQi is intialized as -1, in order to return error value if error
        pq_index = -1
        try:
            # absolute difference to normal values are calculated, and sum
            pq_index = np.absolute(HOS_dict['Variance'] - self.config['variance_normal']) + np.absolute(
                HOS_dict['Skewness'] - self.config['skewness_normal']) + np.absolute(
                HOS_dict['Kurtosis'] - self.config['kurtosis_normal'])
        except Exception as e:
            print(e.__str__())
        return pq_index


    # personal calculation of fft, with window option
    def my_dft(self, x, Fs, window_type=''):
        # needed from special package
        from scipy.fftpack import fft
        # take signal size pt
        N = x.size
        # spectral window
        window_type = window_type.lower()
        # hanning is the indicated by regulation when Fs has no sinc with
        # power signal
        if window_type == 'hann' or window_type == 'hanning':
            #window is generated
            window = np.hanning(N)
            # and signal is modulated and a factor is applied, related to
            # the window, to get propper amplitude
            x = x * window * 2
        # hamming is implemented too, but not needed for PQ, now
        elif window_type == 'hamm' or window_type == 'hamming':
            window = np.hamming(N)
            x = x * window * 1.8
        else:
            pass
        #calculate dft using fft
        yf = fft(x)

        # Calculate frequency increase on each step
        df = Fs/N
        # calculate length of frequency axis
        f_len = N // 2 + 1
        # caclulate frequency axis, for each step, increase df
        f = np.arange(0,f_len)*df
        # yf are complex, this change values to real, and correct amplitudes
        dft = 2.0 / N * np.abs(yf[:N // 2 + 1])
        return dft, f

    # harmonic calculations using dft according to regulation
    def harm_une(self,dft,f,parameters='thd'):

        #create output dict
        output = {}
        #chek if df is 5Hz
        df = f[1]-f[0]
        if np.abs(df-5)>0.1:
            raise Exception('df must be 5 Hz')

        # dft is scaled to be rms values instead amplitude values
        dft = dft / np.sqrt(2)
        # output directly dft
        if any(item in parameters for item in ['dft']):
            output['dft'] = dft
        if any(item in parameters for item in ['dft_f']):
            output['dft_f'] = f
        # execute if required thd
        if any(item in parameters for item in ['thd']):
            # first THD or THDy, calculated with spectral components, up to 50th
            # harm, firsm index are calculated, from freq 50 to 50*50+1,
            # due tu last value is not take with increment 50, and as df
            # is 5, all is divided by 5
            harm_index_1_50 = np.arange(50/5,50*50/5+1,50/5,int)
            thd = np.sqrt(np.sum(np.power(dft[harm_index_1_50[1:]]/dft[harm_index_1_50[0]],2)))
            output['thd'] = thd

        # if requeired thdg Yg, harmonic groups or thd based in groups
        if any(item in parameters for item in ['thdg', 'Yg']):
            # harmonic groups calculation, accroding to une 61000-4-7 it uses
            # groups are squared sum of harmonic components and
            # surrondings n +-4 components, and half value for spectral component
            # n +-5 being n the spectrla index
            harm_index_1_50 = np.arange(50 / 5, 50 * 50 / 5 + 1, 50 / 5, int)
            Yg=[]
            for h_index in harm_index_1_50:
                #+1 extra in range is due open interval, last value not taken
                Ygh = np.sqrt(0.5*np.power(dft[h_index-5],2)\
                      +0.5*np.power(dft[h_index+5],2)\
                      +np.sum(np.power(dft[range(h_index-4,h_index+4+1)],2)))
                # faster save as list
                Yg.append(Ygh)
            #and then change into array
            Yg = np.array(Yg)
            # alculation of thd
            thdg = np.sqrt(np.sum(np.power(Yg[1:] / Yg[0], 2)))
            output['thdg'] = thdg
            output['Yg'] = Yg

        # harmonic suggroups are nedded for other calculations but only are saver in output if required
        # harmonic subgroups calculation, accroding to une 61000-4-7 it uses
        # subgroups are squared sum of harmonic components and
        # surrondings n +-1 components, specral componentes just surroindings
        harm_index_1_50 = np.arange(50 / 5, 50 * 50 / 5 + 1, 50 / 5, int)
        Ysg = []
        for h_index in harm_index_1_50:
            # +1 extra in range is due open interval, last value not taken
            Ysgh = np.sqrt(np.sum(np.power(dft[range(h_index - 1, h_index + 1+1)], 2)))
            #same saved as list
            Ysg.append(Ysgh)
        #and then change into array
        Ysg = np.array(Ysg)
        thdsg = np.sqrt(np.sum(np.power(Ysg[1:] / Ysg[0], 2)))
        if any(item in parameters for item in ['thdsg', 'Ysg']):
            output['thdsg'] = thdsg
            output['Ysg'] = Ysg


        # if required interharmonic subgropuos thdsgi Ysgi
        if any(item in parameters for item in ['thdsgi', 'Ysgi']):
            # calculation of interharmonic sub-gropups, square sum of groups,
            # without the just adjacet to harmonics
            harm_index_1_50 = np.arange(50 / 5, 50 * 50 / 5 + 1, 50 / 5, int)
            Ysgi = []
            for i_h_index in range(len(harm_index_1_50)-1):
                # +1 extra in range is due open interval, last value not taken
                Ysgih = np.sqrt(np.sum(
                    np.power(dft[range(harm_index_1_50[i_h_index]+2,
                                       harm_index_1_50[i_h_index+1]-1)], 2)))
                # same saved as list
                Ysgi.append(Ysgih)
            # and then change into array
            Ysgi = np.array(Ysgi)
            # consider the fundamental harmonic sub-group for the THD
            thdsgi = np.sqrt(np.sum(np.power(Ysgi / Ysg[0], 2)))
            output['thdsgi'] = thdsgi
            output['Ysgi'] = Ysgi

        # if required subharmonics
        if any(item in parameters for item in ['thdsub', 'Ygsub']):
            # same philosopy for sub-harmonic group calculation, all
            # bins under fundamental, just adjacent
            Ygsub = np.sqrt(np.sum(np.power(dft[range(0,harm_index_1_50[0]-1)], 2)))
            # for total distortion, compared with fundamental subgroup, otherwise,
            # same bis are considered
            thdsub = np.sqrt((np.power(Ygsub/ Ysg[0], 2)))
            output['thdsub'] = thdsub
            output['Ygsub'] = Ygsub

        # if required high freq harmonics up to 9khz thdyb Yb
        if any(item in parameters for item in ['thdyb', 'Yb']):
            # calculate centre of harmonics over harmonic of order 50
            # and up to 9kHz.
            if f.max()>9e3:
                harm_index_50_9khz = np.arange(2100 / 5, 9e3/ 5 + 1, 200 / 5, int)
            else:
                harm_index_50_9khz = np.arange(2100 / 5, f.max(), 200 / 5, int)
            # list for keep grouped high frequency harmonics
            Yb = []
            for i_h_index in range(len(harm_index_50_9khz)):
                # +1 extra in range is due open interval, last value not taken
                # should be taken from -95 hx to +100 hz around centre freq
                Ybh = np.sqrt(np.sum(
                    np.power(dft[range(harm_index_50_9khz[i_h_index] -95//5,
                                       harm_index_50_9khz[i_h_index] +100//5+1)], 2)))
                # same saved as list
                Yb.append(Ybh)
            # and then change into array
            Yb = np.array(Yb)
            # thd using fundamental sub-group
            thdyb = np.sqrt(np.sum(np.power(Yb[1:] / Ysg[0], 2)))
            output['Yb'] = Yb
            output['thdyb'] = thdyb

        # if required supraharmonics, from 9kHz
        if any(item in parameters for item in ['thdsup', 'Ysup']):
            # calculate centre of harmonics over 9kHz
            harm_index_9khz_end = np.arange(9e3 / 5, f.max()/5, 1000 / 5, int)
            # list for keep grouped high frequency harmonics
            Ysup = []
            for i_h_index in range(len(harm_index_9khz_end)):
                # +1 extra in range is due open interval, last value not taken
                # should be taken from -95 hx to +100 hz around centre freq
                Ysuph = np.sqrt(np.sum(
                    np.power(dft[range(harm_index_9khz_end[i_h_index] - 495 // 5,
                                       harm_index_9khz_end[
                                           i_h_index] + 500 // 5 + 1)], 2)))
                # same saved as list
                Ysup.append(Ysuph)
            # and then change into array
            Ysup = np.array(Ysup)
            # thd using fundamental sub-group
            thdsup = np.sqrt(np.sum(np.power(Ysup[1:] / Ysg[0], 2)))
            output['Ysup'] = Ysup
            output['thdsup'] = thdsup

        return output

    # SK calculation
    def sk_calc(self, dft_mat):
        # dft_mat 2d numpy array, 0 realizations 1 frequencies
        # get number of realizations for calcuation
        M = dft_mat.shape[0]
        #initialize vector for results
        sk_vector = np.zeros(dft_mat.shape[1])
        # for each frequency, amplitude of all realizations are taken
        # and SK equation is applied
        for i_freq in range(dft_mat.shape[1]):
            sk_vector[i_freq] = M / (M-1) *( ((M+1)*np.sum(np.power(dft_mat[:,i_freq],4))) / np.power(np.sum(np.power(dft_mat[:, i_freq], 2)), 2) - 2)
        return sk_vector,M

    # SK calculation over PQ signal
    def pq_sk(self,x_signal,t_signal=[],window_s=0.2):
        # calculate dft realizations, with 0.2s length
        dft_list = self.window_data(x_signal,t_signal,parameters=['dft','dft_f'],window_s=window_s)
        # check if t_signal for asign value
        if t_signal:
            t0 = t_signal[0]
        else:
            t0 = -1

        dft_mat = np.array([])
        for i_dft in range(dft_list.__len__()):
            if i_dft == 0:
                dft_mat = dft_list[i_dft]['dft']
            else:
                dft_mat = np.vstack((dft_mat,dft_list[i_dft]['dft']))

        sk_data,M = self.sk_calc(dft_mat)
        output = {'t0':t0,
                  'SK':sk_data,
                  'f':dft_list[0]['dft_f'],
                  'M':M}
        return output


if __name__ == '__main__':
    # crear vectores de datos, inicializar, inyectar en for para probar
    from config import PROCESS_CONFIG
    import datetime
    from scipy import signal
    import time as time_module

    # initialize analyzer
    analyzer = PqCalculation(PROCESS_CONFIG,mode=2,use_acum=True, window_s=0.2, parameters=['v','s','k','rms','pqi','thd','Ysup','Yb','Yg','Ysg','Ysgi','Ygsub','dft'])

    base = datetime.datetime.now()  # get time at measurement start
    # get dt of each point
    dT = 1 / PROCESS_CONFIG['Fs']
    # get T of measure
    dT_meas = round(dT * PROCESS_CONFIG['N'])  # time among measures in us
    # initial point for a time vector of a T before start
    base1 = base - datetime.timedelta(seconds=dT_meas)
    # create a refecerence time vector, from previous time interval.
    t = np.array([base1 + datetime.timedelta(seconds=dT * i) for i in
                  range(PROCESS_CONFIG['N'])], dtype='datetime64')
    # for generate signal, a time vector is used starting in zero, not
    # t
    t_signal = np.array([dT * i for i in range(PROCESS_CONFIG['N'])]) \
               - (dT * PROCESS_CONFIG['N'])
    # random initial phases are calculated for signal and pulse
    initial_phase_sig = np.random.uniform(0, 2 * np.pi)
    initial_phase_pul = np.random.uniform(0, 2 * np.pi)

    # set random frequency for main signal
    f = np.random.uniform(49.9, 50.1)
    print(f)

    # continuous execution flag is set
    flag_countinue_acqusition = True

    while flag_countinue_acqusition:
        # get initial time for dinamic delay
        initial_time = time_module.time()

        # update t_signal with T increment
        t_sigal_ant = t_signal
        t_signal = t_sigal_ant + dT * PROCESS_CONFIG['N']
        #     calculate signal
        sinusoid = 230*np.sqrt(2) * np.sin(
            2 * np.pi * f * t_signal + initial_phase_sig) + np.random.normal(
            scale=0.03, size=PROCESS_CONFIG['N'])+2.3*np.sqrt(2) * np.sin(
            2 * np.pi * 3*f * t_signal )
        # calculate pulse GPS-PPS
        # pulse = 5 * (
        #             1 + signal.square(2 * np.pi * t_signal + initial_phase_pul,
        #                               duty=0.0001)) + np.random.normal(
        #     scale=0.03, size=PROCESS_CONFIG['N'])
        # cretate output array
        # output = np.array([sinusoid, pulse])
        # update time vector
        t = t + np.timedelta64(dT_meas, 's')
        # sk = analyzer.pq_sk(sinusoid, t)
        out = analyzer.analyze(sinusoid, t)
        # dft,f = analyzer.my_dft(sinusoid,PROCESS_CONFIG['Fs'],window_type='hann')
        # import matplotlib.pyplot as plt
        # plt.plot(sk['f'],sk['SK'])
        # plt.show()
        # from tests import plot
        # plot(sk['f'],sk['SK'])
        # out = analyzer.harm_une( dft, f,['thd','Ysup','Yb','Yg','Ysg','Ysgi','Ygsub'])
        time_elapsed = time_module.time() - initial_time
        while dT_meas-time_elapsed > 0:
            time_module.sleep(dT_meas-time_elapsed)
            time_elapsed = time_module.time() - initial_time

    print('end')
