import Signal_Analysis as sig
from scipy.io import wavfile
import numpy as np
import pytest

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
    params = [ ( wave1, rate1, { 'accurate'  : False }, 228.892 ),
               ( wave2, rate2, { 'accurate'  : False }, 349.444 ),
               ( wave3, rate3, { 'accurate'  : False }, 182.571 ),
               ( wave4, rate4, { 'accurate'  : False }, 229.355 ),
               ( wave1, rate1, { 'accurate'  : True  }, 229.936 ),
               ( wave2, rate2, { 'accurate'  : True  }, 345.517 ),
               ( wave3, rate3, { 'accurate'  : True  }, 183.453 ),
               ( wave4, rate4, { 'accurate'  : True  }, 229.729 ),
               ( wave1, rate1, { 'time_step' : 0     }, 229.862 ),
               ( wave2, rate2, { 'time_step' : 0     }, 348.380 ),
               ( wave3, rate3, { 'time_step' : 0     }, 182.465 ),
               ( wave4, rate4, { 'time_step' : 0     }, 229.198 ) ]
    
    for param in params:
        wave, rate, kwargs, true_val = param
        est_val = sig.get_F_0( wave, rate, **kwargs )
        assert abs( est_val - true_val ) < 5, 'Estimated frequency not within allotted range.'

def test_get_HNR():
    
    wave_function = lambda x, frequency: np.sin( 2 * np.pi * x * frequency )
    domain = np.linspace( 0, 2, 10000 )
    rate = 10000
    wave = wave_function( domain, 50 )
    with pytest.raises( Exception ) as excinfo:
        sig.get_HNR( wave, rate, min_pitch = 0 )
    assert excinfo.typename == 'ValueError'
    assert excinfo.value.args[ 0 ] == "The minimum pitch cannot be equal to or less than zero."
    
    with pytest.raises( Exception ) as excinfo:
        sig.get_HNR( wave, rate, silence_threshold = 3 )
    assert excinfo.typename == 'ValueError'
    assert excinfo.value.args[ 0 ] == "silence_threshold must be between 0 and 1."
    
    params = [ ( wave1, rate1, 13.083 ),
               ( wave2, rate2,  9.628 ),
               ( wave3, rate3, 17.927 ),
               ( wave4, rate4, 16.206 ),
               ( np.zeros(500), 500, 0 )]
                           
    for param in params:
        wave, rate, true_val = param
        est_val = sig.get_HNR( wave, rate )
        assert abs( est_val - true_val ) < .75, 'Estimated HNR not within allotted range.'

def test_get_Jitter():
    wave_function = lambda x, frequency: np.sin( 2 * np.pi * x * frequency )
    domain = np.linspace( 0, 2, 10000 )
    rate = 10000
    wave = wave_function( domain, 50 )
    
    params = [ ( wave1, rate1, np.array( [ 0.046211, 0.000207, 0.023501, 0.028171, 0.070503 ] ) ),
               ( wave2, rate2, np.array( [ 0.049284, 0.000148, 0.026462, 0.025010, 0.079386 ] ) ),
               ( wave3, rate3, np.array( [ 0.027097, 0.000141, 0.014425, 0.013832, 0.043274 ] ) ),
               ( wave4, rate4, np.array( [ 0.034202, 0.000143, 0.019735, 0.018335, 0.059206 ] ) ) ] 
    for param in params:
        wave, rate, true_val = param
        est_val = sig.get_Jitter( wave, rate )
        print(est_val)
        est_val=np.array( list( est_val.values() ) )
        
        #assert time.clock() - start < 6
        assert np.allclose( est_val , true_val, atol = 0, rtol = .2 ) 
    #for now this works. will need to change later...
