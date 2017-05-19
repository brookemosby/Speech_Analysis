from Signal_Analysis import Signal
from scipy.io import wavfile
import numpy as np
import pytest

class Test_Signal():
    
    def test_get_F_0( self ):
        def testing_sig(rate,wave,true_freq):
            sig=Signal(wave,rate)
            est_freq=sig.get_F_0(.1,.1)
            assert abs(est_freq-true_freq)<10, 'Estimated frequency not within allotted range.'
        
        #for here look at how conrad uses pytest to test for value/type errors
        #also need to add in two more test cases for extra type/value error added
        wave_function = lambda x, frequency: np.sin(2*np.pi*x*frequency)
        domain=np.linspace(0,1,1000)
        rate=1000
        wave=wave_function(domain,50)
        sig=Signal(wave,rate)
        with pytest.raises(Exception) as excinfo:
            sig.get_F_0(.1,.1)
        assert excinfo.typename == 'ValueError'
        assert excinfo.value.args[0] == "The maximum pitch cannot be greater than the Nyquist Frequency."
        domain=np.linspace(0,2,10000)
        rate=10000
        wave=wave_function(domain,50)
        sig=Signal(wave,rate)
        with pytest.raises(Exception) as excinfo:
            sig.get_F_0(.1,.1,min_pitch=0)
        assert excinfo.typename == 'ValueError'
        assert excinfo.value.args[0] == "The minimum pitch cannot be equal or less than zero."
        domain=np.linspace(0,2,10000)
        rate=10000
        wave=wave_function(domain,50)
        sig=Signal(wave,rate)
        with pytest.raises(Exception) as excinfo:
            sig.get_F_0(.1,.1, max_num_candidates=1 )
        assert excinfo.typename == 'ValueError'
        assert excinfo.value.args[0] == "The minimum number of candidates is 2."  
                                 
        domain=np.linspace(0,2,10000)
        rate=10000
        wave=wave_function(domain,50)
        sig=Signal(wave,rate)
        assert sig.get_F_0(.1,.1,min_pitch=300)==0
                          

        
                                 
        
        rate, wave = wavfile.read( '03-01-01-01-01-01-10.wav' )
        testing_sig(rate,wave,231.2)
        rate, wave = wavfile.read( '03-01-06-01-01-02-04.wav' )
        testing_sig(rate,wave,338.2)
        rate, wave = wavfile.read( 'OAF_youth_sad.wav' )
        testing_sig(rate,wave,172.5)
        rate, wave = wavfile.read( 'YAF_kite_sad.wav' )
        testing_sig(rate,wave,211.9)

        
        #we want to test for multiple mins, maxs, and both types of thresholds, multiple for loops, make sure that when we test mins/maxs we don't eliminate true vals....
        #also we will probs want to make new signals