from scipy.io import matlab
import numpy as np
import pytest
import afx.features.signal as sig


dict1 = matlab.loadmat( 'tests/features/data/03-01-01-01-01-01-10.mat' )
r1 = dict1[ 'Fs' ][ 0 ][ 0 ]
sig1 = dict1[ 'y' ].ravel()
dict2 = matlab.loadmat( 'tests/features/data/03-01-06-01-01-02-04.mat' )
r2 = dict2[ 'Fs' ][ 0 ][ 0 ]
sig2 = dict2[ 'y' ].ravel()
dict3 = matlab.loadmat( 'tests/features/data/OAF_youth_sad.mat' )
r3 = dict3[ 'Fs' ][ 0 ][ 0 ]
sig3 = dict3[ 'y' ].ravel()
dict4 = matlab.loadmat( 'tests/features/data/YAF_kite_sad.mat' )
r4 = dict4[ 'Fs' ][ 0 ][ 0 ]
sig4 = dict4[ 'y' ].ravel()

def test_get_F_0():
    #Here we test all the exceptions using wave1 & rate1 created above 
    params = [ ( "max_pitch can't be larger than Nyquist Frequency.",         
               { 'max_pitch'     : 40000 } ),
               ( "min_pitch has to be greater than zero.",         
               { 'min_pitch'     : 0     } ),
               ( "octave_cost isn't in [ 0, 1 ]",   
               { 'octave_cost'   : 3     } ),
               ( "silence_thres isn't in [ 0, 1 ]", 
               { 'silence_thres' : 3     } ),
               ( "voicing_thres isn't in [ 0, 1 ]", 
               { 'voicing_thres' : 3     } ) ]

    for param in params:
        message, kwargs = param
        with pytest.raises( Exception ) as excinfo:
            sig.get_F_0( sig1, r1, **kwargs )
        assert excinfo.typename == 'ValueError'
        assert excinfo.value.args[ 0 ] == message

    zero = np.zeros( 10000 ) 
    r  = 5000                  
    #Testing values that came from Praat for each signal, using the standard 
    #values                   
    params = [ ( sig1, r1, { 'accurate'  : 0, 'min_pitch' : 75 }, 229.862 ),
               ( sig2, r2, { 'accurate'  : 0, 'min_pitch' : 75 }, 348.380 ),
               ( sig3, r3, { 'accurate'  : 0, 'min_pitch' : 75 }, 182.465 ),
               ( sig4, r4, { 'accurate'  : 0, 'min_pitch' : 75 }, 229.198 ),
               ( sig1, r1, { 'accurate'  : 1, 'min_pitch' : 75 }, 229.631 ),
               ( sig2, r2, { 'accurate'  : 1, 'min_pitch' : 75 }, 347.990 ),
               ( sig3, r3, { 'accurate'  : 1, 'min_pitch' : 75 }, 182.539 ),
               ( sig4, r4, { 'accurate'  : 1, 'min_pitch' : 75 }, 229.192 ),
               ( zero, r,  { 'accurate'  : 1, 'min_pitch' : 75 },   0.0   )]
    
    for param in params:
        wave, rate, kwargs, true_val = param
        est_val = sig.get_F_0( wave, rate, **kwargs )[ 0 ]
        assert abs( est_val - true_val ) < 3, 'Frequency not accurate.'
def test_get_HNR():
    #Here we test all the exceptions
    with pytest.raises( Exception ) as excinfo:
        sig.get_HNR( sig1, r1, min_pitch = 0 )
    assert excinfo.typename == 'ValueError'
    assert excinfo.value.args[ 0 ] == "min_pitch has to be greater than zero."
    with pytest.raises( Exception ) as excinfo:
        sig.get_HNR( sig1, r1, silence_threshold = 3 )
    assert excinfo.typename == 'ValueError'
    assert excinfo.value.args[ 0 ] == "silence_threshold isn't in [ 0, 1 ]."
    
    #Testing values that came from Praat for each signal, using the standard 
    #values 
    params = [ ( sig1, r1, 13.102 ),
               ( sig2, r2,  9.660 ),
               ( sig3, r3, 17.940 ),
               ( sig4, r4, 16.254 ),
               ( np.zeros(500), 500, 0 )]
                           
    for param in params:
        wave, rate, true_val = param
        est_val = sig.get_HNR( wave, rate )
        assert abs( est_val - true_val ) < .3, 'HNR not accurate'
        
def test_get_Pulses():
    with pytest.raises( Exception ) as e:
        sig.get_Pulses( sig1, r1, include_max = 0, include_min = 0 )
    assert e.typename == 'ValueError'
    assert e.value.args[ 0 ] == "include_min and include_max can't both be False"
    params = [ ( sig1, r1, .00446, { 'include_max' : 1, 'include_min' : 0 } ),
               ( sig2, r2, .00304, { 'include_max' : 1, 'include_min' : 0 } ),
               ( sig3, r3, .00521, { 'include_max' : 1, 'include_min' : 0 } ),
               ( sig4, r4, .00420, { 'include_max' : 1, 'include_min' : 0 } ),
               ( sig1, r1, .00447, { 'include_max' : 1, 'include_min' : 1 } ),
               ( sig2, r2, .00301, { 'include_max' : 1, 'include_min' : 1 } ),
               ( sig3, r3, .00521, { 'include_max' : 1, 'include_min' : 1 } ),
               ( sig4, r4, .00421, { 'include_max' : 1, 'include_min' : 1 } ) ] 
    
    
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
    #Testing values that came from Praat for each signal, by going from 
    #sound-> PointProcess (peaks) and using the jitter default values
    
    params = [ ( sig1, r1,
                np.array( [ 0.04621, 0.000207, 0.02350, 0.02817, 0.07050 ] ) ),
               ( sig2, r2,
                np.array( [ 0.04928, 0.000148, 0.02646, 0.02501, 0.07938 ] ) ),
               ( sig3, r3, 
                np.array( [ 0.02709, 0.000141, 0.01442, 0.01383, 0.04327 ] ) ),
               ( sig4, r4, 
                np.array( [ 0.03420, 0.000143, 0.01973, 0.01833, 0.05920 ] ) )] 
    #4.2% relative error tolerance
    error = 0.042
    for param in params:
        
        wave, rate, true = param
        est = sig.get_Jitter( wave, rate )

        assert abs( est[ 'local'           ] - true[ 0 ] )  < error * true[ 0 ]
        assert abs( est[ 'local, absolute' ] - true[ 1 ] )  < error * true[ 1 ]
        assert abs( est[ 'rap'             ] - true[ 2 ] )  < error * true[ 2 ]
        assert abs( est[ 'ppq5'            ] - true[ 3 ] )  < error * true[ 3 ]
        assert abs( est[ 'ddp'             ] - true[ 4 ] )  < error * true[ 4 ]
