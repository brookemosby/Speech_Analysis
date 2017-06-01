from Signal_Analysis import Signal
from scipy.io import wavfile
import numpy as np
import pytest

class Test_Signal():
    
    def test_get_F_0( self ):
        def testing_sig( rate, wave, true_freq ):
            sig = Signal( wave, rate )
            est_freq = sig.get_F_0()
            assert abs( est_freq - true_freq ) < 2, 'Estimated frequency not within allotted range.'

        wave_function = lambda x, frequency:(1 + .3 * np.sin( 2 * np.pi * x * frequency ) ) * np.sin( 4 * np.pi * x * frequency )
        domain = np.linspace( 0, 2, 10000 )
        rate = 10000
        wave = wave_function( domain, 50 )
        sig = Signal( wave, rate )
        
        #Here we test all the exceptions using the generic sin wave created above 
        with pytest.raises( Exception ) as excinfo:
            sig.get_F_0( max_pitch = 20000 )
        assert excinfo.typename == 'ValueError'
        assert excinfo.value.args[ 0 ] == "The maximum pitch cannot be greater than the Nyquist Frequency."
        with pytest.raises( Exception ) as excinfo:
            sig.get_F_0( HNR = 20000 )
        assert excinfo.typename == 'TypeError'
        assert excinfo.value.args[ 0 ] == "HNR must be a bool."
        with pytest.raises( Exception ) as excinfo:
            sig.get_F_0( min_pitch = 0 )
        assert excinfo.typename == 'ValueError'
        assert excinfo.value.args[ 0 ] == "The minimum pitch cannot be equal or less than zero."
        
        with pytest.raises( Exception ) as excinfo:
            sig.get_F_0( max_num_candidates = 1 )
        assert excinfo.typename == 'ValueError'
        assert excinfo.value.args[ 0 ] == "The minimum number of candidates is 2."  
                                 
        with pytest.raises( Exception ) as excinfo:
            sig.get_F_0( octave_cost = 4.5 )
        assert excinfo.typename == 'ValueError'
        assert excinfo.value.args[ 0 ] == "octave_cost must be between 0 and 1."
        
        with pytest.raises( Exception ) as excinfo:
            sig.get_F_0( silence_threshold = 4.5 )
        assert excinfo.typename == 'ValueError'
        assert excinfo.value.args[ 0 ] == "silence_threshold must be between 0 and 1."
        
        with pytest.raises( Exception ) as excinfo:
            sig.get_F_0( voicing_threshold = 4.5 )
        assert excinfo.typename == 'ValueError'
        assert excinfo.value.args[ 0 ] == "voicing_threshold must be between 0 and 1."
        
        with pytest.raises( Exception ) as excinfo:
            sig.get_F_0( 'Testing' )
        assert excinfo.typename == 'TypeError'
        assert excinfo.value.args[ 0 ] == "min_pitch, max_pitch, and max_num_candidates must be an int, octave_cost, silence_threshold, and voicing_threshold must be a float"
        #This test the case that we have no valid candidates
        assert sig.get_F_0( min_pitch = 400, max_num_candidates = 3 ) == 0
        
        #These cases test that the calculated answer is within an acceptable range (+/- 2 hz)
        rate, wave = wavfile.read( '03-01-01-01-01-01-10.wav' )
        testing_sig( rate, wave, 231.2 )
        
        rate, wave = wavfile.read( '03-01-06-01-01-02-04.wav' )
        testing_sig( rate, wave, 338.2 )
        
        rate, wave = wavfile.read( 'OAF_youth_sad.wav' )
        testing_sig( rate, wave, 172.5 )
        
        rate, wave = wavfile.read( 'YAF_kite_sad.wav' )
        testing_sig( rate, wave, 211.9 )

    
    def test_get_HNR( self ):
        def testing_sig( rate, wave, true_HNR ):
            sig = Signal( wave, rate )
            est_HNR = sig.get_HNR()
            assert abs( est_HNR - true_HNR ) < 1, 'Estimated frequency not within allotted range.'
        wave_function = lambda x, frequency: np.sin( 2 * np.pi * x * frequency )
        domain = np.linspace( 0, 2, 10000 )
        rate = 10000
        wave = wave_function( domain, 50 )
        sig = Signal( wave, rate )
        
        with pytest.raises( Exception ) as excinfo:
            sig.get_HNR( min_pitch = 0 )
        assert excinfo.typename == 'ValueError'
        assert excinfo.value.args[ 0 ] == "The minimum pitch cannot be equal or less than zero."
        
        with pytest.raises( Exception ) as excinfo:
            sig.get_HNR( silence_threshold = 4.5 )
        assert excinfo.typename == 'ValueError'
        assert excinfo.value.args[ 0 ] == "silence_threshold must be between 0 and 1."

        #These cases test that the calculated answer is within an acceptable range (+/- 1 dB)
        rate, wave = wavfile.read( '03-01-01-01-01-01-10.wav' )
        testing_sig( rate, wave, 12.195 )
        
        rate, wave = wavfile.read( '03-01-06-01-01-02-04.wav' )
        testing_sig( rate, wave, 8.589 )
        
        rate, wave = wavfile.read( 'OAF_youth_sad.wav' )
        testing_sig( rate, wave, 16.838 )
        
        rate, wave = wavfile.read( 'YAF_kite_sad.wav' )
        testing_sig( rate, wave, 14.929 )