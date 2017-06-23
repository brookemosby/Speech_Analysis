import Signal_Analysis as sig
from scipy.io import wavfile
import numpy as np
import pytest
import time

rate1, wave1 = wavfile.read( '03-01-01-01-01-01-10.wav' )
rate2, wave2 = wavfile.read( '03-01-06-01-01-02-04.wav' )
rate3, wave3 = wavfile.read( 'OAF_youth_sad.wav' )
rate4, wave4 = wavfile.read( 'YAF_kite_sad.wav' )

def test_get_F_0():
    wave_function = lambda x, frequency:(1 + .3 * np.sin( 2 * np.pi * x * frequency ) ) * np.sin( 4 * np.pi * x * frequency )
    domain = np.linspace( 0, 2, 10000 )
    rate = 10000
    wave = wave_function( domain, 50 )
    #Here we test all the exceptions using the generic sin wave created above 
    params = [ ( "The maximum pitch cannot be greater than the Nyquist Frequency.",   { 'max_pitch'         : 20000 } ),
               ( "The minimum pitch cannot be equal or less than zero.",              { 'min_pitch'         : 0 } ),
               ( "The minimum number of candidates is 2.",                            { 'max_num_cands': 1 } ),
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
    assert sig.get_F_0( wave, rate, min_pitch = 400, max_num_cands = 3 ) == 0
    params = [ ( wave1, rate1, 228.892 ),
               ( wave2, rate2, 349.443 ),
               ( wave3, rate3, 182.571 ),
               ( wave4, rate4, 229.354 ) ]
    
    for param in params:
        wave, rate, true_val = param
        start = time.clock()
        est_val = sig.get_F_0( wave, rate )
        #assert time.clock() - start < 1
        assert abs( est_val - true_val ) < 4.5, 'Estimated frequency not within allotted range.'

def test_get_HNR():
    
    wave_function = lambda x, frequency: np.sin( 2 * np.pi * x * frequency )
    domain = np.linspace( 0, 2, 10000 )
    rate = 10000
    wave = wave_function( domain, 50 )
    """
    with pytest.raises( Exception ) as excinfo:
        sig.get_HNR( wave, rate, min_pitch = 0 )
    assert excinfo.typename == 'ValueError'
    assert excinfo.value.args[ 0 ] == "The minimum pitch cannot be equal to or less than zero."
    """
    
    params = [ ( wave1, rate1, 13.935 ),
               ( wave2, rate2, 9.246 ),
               ( wave3, rate3, 16.781 ),
               ( wave4, rate4, 16.440 ),
               ( np.random.random(10000)*.001, 5000, 0 )]
                           
    for param in params:
        wave, rate, true_val = param
        start = time.clock()
        est_val = sig.get_HNR( wave, rate )
        assert time.clock() - start < 1
        assert abs( est_val - true_val ) < 3, 'Estimated HNR not within allotted range.'

def test_get_Jitter():
    wave_function = lambda x, frequency: np.sin( 2 * np.pi * x * frequency )
    domain = np.linspace( 0, 2, 10000 )
    rate = 10000
    wave = wave_function( domain, 50 )
    
    params = [ ( wave1, rate1, np.array( [ 1.849, 82.600E-6, 0.853, 0.962, 2.560 ] ) ),
               ( wave2, rate2, np.array( [ 2.035, 60.799E-6, 0.824, 0.910, 2.471 ] ) ),
               ( wave3, rate3, np.array( [ 1.472, 75.910E-6, 0.600, 0.640, 1.800 ] ) ),
               ( wave4, rate4, np.array( [ 2.308, 97.224E-6, 1.184, 1.061, 3.551 ] ) ) ] 
    for param in params:
        wave, rate, true_val = param
        start = time.clock()
        est_val = sig.get_Jitter( wave, rate )
        est_val=np.array( list( est_val.values() ) )
        print(est_val)
        #assert time.clock() - start < 6
        assert np.allclose( est_val , true_val, atol=0, rtol=.01 ) 
    #for now this works. will need to change later...
