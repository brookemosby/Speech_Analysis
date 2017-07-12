from scipy.io import matlab
import numpy as np
import pytest
import Signal_Analysis as sig

dict1 = matlab.loadmat( 'tests/03-01-01-01-01-01-10.mat' )
rate1 = dict1[ 'Fs' ][ 0 ][ 0 ]
wave1 = dict1[ 'y' ]
dict2 = matlab.loadmat( 'tests/03-01-06-01-01-02-04.mat' )
rate2 = dict2[ 'Fs' ][ 0 ][ 0 ]
wave2 = dict2[ 'y' ]
dict3 = matlab.loadmat( 'tests/OAF_youth_sad.mat' )
rate3 = dict3[ 'Fs' ][ 0 ][ 0 ]
wave3 = dict3[ 'y' ]
dict4 = matlab.loadmat( 'tests/YAF_kite_sad.mat' )
rate4 = dict4[ 'Fs' ][ 0 ][ 0 ]
wave4 = dict4[ 'y' ]

def test_get_F_0():
    #Here we test all the exceptions using wave1 & rate1 created above 
    params = [ ( "The maximum pitch cannot be greater than the Nyquist Frequency.",   { 'max_pitch'         : 200000 } ),
               ( "The minimum pitch cannot be equal or less than zero.",              { 'min_pitch'         : 0 } ),
               ( "The minimum number of candidates is 2.",                            { 'max_num_cands': 1 } ),
               ( "octave_cost must be between 0 and 1.",                              { 'octave_cost'       : 3 } ),
               ( "silence_threshold must be between 0 and 1.",                        { 'silence_threshold' : 3 } ),
               ( "voicing_threshold must be between 0 and 1.",                        { 'voicing_threshold' : 3 } ) ]

    for param in params:
        message, kwargs = param
        with pytest.raises( Exception ) as excinfo:
            sig.get_F_0( wave1, rate1, **kwargs )
        assert excinfo.typename == 'ValueError'
        assert excinfo.value.args[ 0 ] == message
    #This test the case that we have no valid candidates
    assert sig.get_F_0( wave1, rate1, min_pitch = 550, max_num_cands = 3 ) == 0
                      
    #Testing values that came from Praat for each signal, using the standard values (with the exception
    #that time_step is set equal to .04 in Praat )                      
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
    #Here we test all the exceptions using the generic sine wave created above 
    with pytest.raises( Exception ) as excinfo:
        sig.get_HNR( wave1, rate1, min_pitch = 0 )
    assert excinfo.typename == 'ValueError'
    assert excinfo.value.args[ 0 ] == "The minimum pitch cannot be equal to or less than zero."
    with pytest.raises( Exception ) as excinfo:
        sig.get_HNR( wave1, rate1, silence_threshold = 3 )
    assert excinfo.typename == 'ValueError'
    assert excinfo.value.args[ 0 ] == "silence_threshold must be between 0 and 1."
    
    #Testing values that came from Praat for each signal, using the standard values 
    params = [ ( wave1, rate1, 13.083 ),
               ( wave2, rate2,  9.628 ),
               ( wave3, rate3, 17.927 ),
               ( wave4, rate4, 16.206 ),
               ( np.zeros(500), 500, 0 )]
                           
    for param in params:
        wave, rate, true_val = param
        est_val = sig.get_HNR( wave, rate )
        assert abs( est_val - true_val ) < .75, 'Estimated HNR not within allotted range.'
        
def test_get_Pulses():
    with pytest.raises( Exception ) as excinfo:
        sig.get_Pulses( wave1, rate1, include_maxima = False, include_minima = False)
    assert excinfo.typename == 'ValueError'
    assert excinfo.value.args[ 0 ] == "At least one of include_minima or include_maxima must set to True." 
    params = [ ( wave1, rate1, .00446, { 'include_maxima' :  True, 'include_minima' : False } ),
               ( wave2, rate2, .00304, { 'include_maxima' :  True, 'include_minima' : False } ),
               ( wave3, rate3, .00521, { 'include_maxima' :  True, 'include_minima' : False } ),
               ( wave4, rate4, .00420, { 'include_maxima' :  True, 'include_minima' : False } ),
               ( wave1, rate1, .00447, { 'include_maxima' :  True, 'include_minima' : True  } ),
               ( wave2, rate2, .00301, { 'include_maxima' :  True, 'include_minima' : True  } ),
               ( wave3, rate3, .00521, { 'include_maxima' :  True, 'include_minima' : True  } ),
               ( wave4, rate4, .00421, { 'include_maxima' :  True, 'include_minima' : True  } ) ] 
    
    for param in params:
        wave, rate, avg_period, kwargs = param
        points = np.diff( sig.get_Pulses( wave, rate, **kwargs ) )
        sum_total = 0
        num_periods = 0
        points = np.hstack(( points[ 0 ], points, points[ -1 ] ))
        for i in range( len( points ) - 2):
            p1 = points[ i ]
            p2 = points[ i + 1 ]
            p3 = points[ i + 2 ]
            ratio_1, ratio_2 = p1 / p2, p2 / p3
            if (ratio_1 < 1.300 and 
                ratio_1 > 0.769 and 
                ratio_2 < 1.300 and 
                ratio_2 > 0.769 and 
                p2 < .02 and
                p2 > .0001 ):
                sum_total += p2
                num_periods += 1
        est_avg_period = sum_total / num_periods 
        assert abs( est_avg_period - avg_period ) < .00015
        
def test_get_Jitter():
    #Testing values that came from Praat for each signal, by going from sound-> PointProcess (peaks)
    #and using the jitter default values
    params = [ ( wave1, rate1, np.array( [ 0.046211, 0.000207, 0.023501, 0.028171, 0.070503 ] ) ),
               ( wave2, rate2, np.array( [ 0.049284, 0.000148, 0.026462, 0.025010, 0.079386 ] ) ),
               ( wave3, rate3, np.array( [ 0.027097, 0.000141, 0.014425, 0.013832, 0.043274 ] ) ),
               ( wave4, rate4, np.array( [ 0.034202, 0.000143, 0.019735, 0.018335, 0.059206 ] ) ) ] 
    
    for param in params:
        wave, rate, true_val = param
        est_val = sig.get_Jitter( wave, rate )
        est_val=np.array( list( est_val.values() ) )
        #we allow a 10.5% error tolerance
        assert np.allclose( est_val , true_val, atol = 0, rtol = .105 ) 
