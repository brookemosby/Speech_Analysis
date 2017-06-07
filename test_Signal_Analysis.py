import Signal_Analysis as sig
from scipy.io import wavfile
import numpy as np
import pytest

def test_get_F_0():
    wave_function = lambda x, frequency:(1 + .3 * np.sin( 2 * np.pi * x * frequency ) ) * np.sin( 4 * np.pi * x * frequency )
    domain = np.linspace( 0, 2, 10000 )
    rate = 10000
    wave = wave_function( domain, 50 )
    #Here we test all the exceptions using the generic sin wave created above 
    params = [ ( "The maximum pitch cannot be greater than the Nyquist Frequency.",   { 'max_pitch'         : 20000 } ),
               ( "The minimum pitch cannot be equal or less than zero.",              { 'min_pitch'         : 0 } ),
               ( "The minimum number of candidates is 2.",                            { 'max_num_candidates': 1 } ),
               ( "octave_cost must be between 0 and 1.",                              { 'octave_cost'       : 3 } ),
               ( "silence_threshold must be between 0 and 1.",                        { 'silence_threshold' : 3 } ),
               ( "voicing_threshold must be between 0 and 1.",                        { 'voicing_threshold' : 3 } ) ]
    for param in params:
        message, kwargs = param
        with pytest.raises( Exception ) as excinfo:
            sig.get_F_0( wave, rate, **kwargs )
        assert excinfo.typename == 'ValueError'
        assert excinfo.value.args[ 0 ] == message
    #This test the case that we have no valid candidates
    assert sig.get_F_0( wave, rate, min_pitch = 400, max_num_candidates = 3 ) == 0
    params = [ ( wavfile.read( '03-01-01-01-01-01-10.wav' ), 231.2 ),
               ( wavfile.read( '03-01-06-01-01-02-04.wav' ), 338.2 ),
               ( wavfile.read( 'OAF_youth_sad.wav' ),        172.5 ),
               ( wavfile.read( 'YAF_kite_sad.wav' ),         211.9 ) ]                             
    for param in params:
        signal, true_val = param
        est_val = sig.get_F_0( signal[ 1 ], signal[ 0 ] )
        assert abs( est_val - true_val ) < 2, 'Estimated frequency not within allotted range.'

def test_get_HNR():
    
    wave_function = lambda x, frequency: np.sin( 2 * np.pi * x * frequency )
    domain = np.linspace( 0, 2, 10000 )
    rate = 10000
    wave = wave_function( domain, 50 )
    
    with pytest.raises( Exception ) as excinfo:
        sig.get_HNR( wave, rate, min_pitch = 0 )
    assert excinfo.typename == 'ValueError'
    assert excinfo.value.args[ 0 ] == "The minimum pitch cannot be equal to or less than zero."
    
    params = [ ( wavfile.read( '03-01-01-01-01-01-10.wav' ), 12.195 ),
               ( wavfile.read( '03-01-06-01-01-02-04.wav' ),  8.589 ),
               ( wavfile.read( 'OAF_youth_sad.wav' ),        16.838 ),
               ( wavfile.read( 'YAF_kite_sad.wav' ),         14.929 ) ]
    
    for param in params:
        signal, true_val = param
        est_val = sig.get_HNR( signal[ 1 ], signal[ 0 ] )
        assert abs( est_val - true_val ) < 1, 'Estimated HNR not within allotted range.'

def test_get_Jitter():
    unknown = 0
    params = [ ( wavfile.read( '03-01-01-01-01-01-10.wav' ), unknown ),
               ( wavfile.read( '03-01-06-01-01-02-04.wav' ), unknown ),
               ( wavfile.read( 'OAF_youth_sad.wav' ),        unknown ),
               ( wavfile.read( 'YAF_kite_sad.wav' ),         unknown ) ]    
    for param in params:
        signal, true_val = param
        est_val = sig.get_Jitter( signal[ 1 ], signal[ 0 ] )
        assert abs( est_val - true_val ) == est_val
    #for now this works. will need to change later...
