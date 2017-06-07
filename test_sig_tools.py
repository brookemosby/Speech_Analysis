import numpy as np
import sig_tools as st
def test_estimated_autocorrelation():
    """
    domain = np.linspace( 0, 30, 500 )
    sine = np.sin( domain )
    sine -= sine.mean()
    sine *= np.hanning(500)
    r_a = st.estimated_autocorrelation( sine )
    r_w = st.estimated_autocorrelation( np.hanning( 500 ) )
    r_x = r_a / r_w
    r_x = r_x[ : 250 ]
    assert np.allclose( r_x, .5 * np.cos( np.linspace( 0, 15, 250, endpoint = False ) ), atol = .05 )
    """
    pass
    
def test_sinc_interp():
    """
    domain = np.linspace( 0, 5, 500 )
    sine = np.sin( domain )
    sine_1 = st.sinc_interp( sine, domain, domain )    
    assert np.allclose( sine_1, sine)
    """
    pass
    
def test_find_max():
    """
    domain = np.linspace( 0, 5, 500 )
    sine = np.sin( domain )
    maximums, maximizers = st.find_max( sine, domain, np.inf )
    assert np.max( maximums ) == np.max( sine )
    assert maximizers[ np.argmax( maximums ) ] == domain[ np.argmax( sine ) ]
    """
    pass