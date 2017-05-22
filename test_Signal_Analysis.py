from Signal_Analysis import Signal
from scipy.io import wavfile
import numpy as np
import pytest

class Test_Signal():
    
    def test_get_F_0( self ):
        def testing_sig(rate,wave,true_freq):
            sig=Signal(wave,rate)
            est_freq=sig.get_F_0()
            assert abs(est_freq-true_freq)<10, 'Estimated frequency not within allotted range.'

        wave_function = lambda x, frequency: np.sin(2*np.pi*x*frequency)
        domain=np.linspace(0,2,10000)
        rate=10000
        wave=wave_function(domain,50)
        sig=Signal(wave,rate)
        
        #Here we test all the exceptions using the generic sin wave created above 
        with pytest.raises(Exception) as excinfo:
            sig.get_F_0(max_pitch=20000)
        assert excinfo.typename == 'ValueError'
        assert excinfo.value.args[0] == "The maximum pitch cannot be greater than the Nyquist Frequency."

        with pytest.raises(Exception) as excinfo:
            sig.get_F_0(min_pitch=0)
        assert excinfo.typename == 'ValueError'
        assert excinfo.value.args[0] == "The minimum pitch cannot be equal or less than zero."
        
        with pytest.raises(Exception) as excinfo:
            sig.get_F_0(max_num_candidates=1 )
        assert excinfo.typename == 'ValueError'
        assert excinfo.value.args[0] == "The minimum number of candidates is 2."  
                                 
        with pytest.raises(Exception) as excinfo:
            sig.get_F_0(octave_cost=4.5 )
        assert excinfo.typename == 'ValueError'
        assert excinfo.value.args[0] == "octave_cost must be between 0 and 1."
        
        with pytest.raises(Exception) as excinfo:
            sig.get_F_0('Testing')
        assert excinfo.typename == 'TypeError'
        assert excinfo.value.args[0] == "min_pitch, max_pitch and octave_cost must be scalars, max_num_candidates must be an int and HNR must be a bool."
            
        #This test the case that we have no valid candidates
        assert sig.get_F_0(min_pitch=300)==0
        
        #These cases test that the calculated answer is within an acceptable range (+/- 10 hz)
        rate, wave = wavfile.read( '03-01-01-01-01-01-10.wav' )
        testing_sig(rate,wave,231.2)
        
        rate, wave = wavfile.read( '03-01-06-01-01-02-04.wav' )
        testing_sig(rate,wave,338.2)
        
        rate, wave = wavfile.read( 'OAF_youth_sad.wav' )
        testing_sig(rate,wave,172.5)
        
        rate, wave = wavfile.read( 'YAF_kite_sad.wav' )
        testing_sig(rate,wave,211.9)
