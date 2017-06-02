import numpy as np
#TODO: ***GO ON GITHUB LOOK AT EXAMPLES OF CLASS CODE, MAY NOT WANT EVERYTHING IN ONE GARGANTUAN FILE***
#TODO: ***START OPTIMIZNG CODE. BE CAREFUL AND TEST OFTEN CODE IS DELICATE***

def gauss_window( length, T, k ):
    times=np.linspace(-.5*T,1.5*T,length)
    gauss_func= lambda t: (np.e**(-k*(t/T-.5)**2)-np.e**(-k))/(1-np.e**(-k))
    return gauss_func(times)

def estimated_autocorrelation( x ):
    """
    This function accepts a signal and calculates an estimation of the autocorrelation, 
    based off the given algorithm (steps 3.5-3.9), described below 
    1. append half the window length of zeros
    2. append zeros until the segment length is a power of 2, calculated with log.
    3. take the FFT
    4. square samples in the signal
    5. then again take the FFT
    
    Args:
        x (numpy.ndarray): an array of the signal that autocorrelation is calculated from.
    
    Returns: 
        a (numpy.ndarray): an array of the normalized autocorrelation of the signal.
    
    """
    N = len( x )
    x = np.hstack( ( x, np.zeros( int( N / 2 ) ) ) )
    x = np.hstack( ( x, np.zeros( 2 ** ( int( np.log2( N ) + 1 ) ) - N ) ) )            
    s = np.fft.fft( x )
    a = np.real( np.fft.fft( s * np.conjugate( s ) ) )
    a = a[ :N ]
    return a


    
def sinc_interp( x, s, u ):
    """
    This function uses sinc interpolation to upsample x.
    
    Args:
        x (numpy.ndarray): an array of the signal to be interpolated
        s (numpy.ndarray): an array of the sampled domain
        u (numpy.ndarray): an array of the new sampled domain
        
    Returns:
        y (numpy.ndarray): an array of the interpolated signal.
        
    """
    #Find the period    
    T = s[ 1 ] - s[ 0 ]
    #This creates an array of values to use in our interpolation
    sincM = np.tile( u, ( len( s ), 1 ) ) - np.tile( s[ :, np.newaxis ], ( 1, len( u ) ) )
    #This calculates interpolated array
    y = np.dot( x, np.sinc( sincM / T ) )
    return y

def find_max( arr, time_array, max_num_candidates ):
    """
    This function finds the first max_num_candidates maxima of an array.
    
    Args:
        arr (numpy.ndarray): an array of the signal we are calculating the peaks of.
        time_array (numpy.ndarray): an array of the corresponding points in time that the
            signal was sampled at
        
    Returns:
        maxima_values (numpy.ndarray): an array of the calculated maximum values
        maxima_places (numpy.ndarray): an array of the corresponding calculated places 
            where the maxima occur.
    """
    maxima_values = []
    maxima_places = []
    index = 0
    while index < len( arr ) and len( maxima_values ) < max_num_candidates :
        #Typically the first peak of the autocorrelation is the peak representing the 
        #frequency, so max_num_candidates is most accurate when equal to 2.
        best_max = 0
        best_place = 0
        
        while index < len( arr ) and arr[ index ] > 0:
            
            if arr[ index ] > best_max:
                # keep track of the maximums and maximizers of the signal
                best_max = arr[ index ]
                best_place = time_array[ index ]
            index += 1
            
        maxima_values.append( best_max )
        maxima_places.append( best_place )
        while index < len( arr ) and arr[ index ] <= 0 and len( maxima_values ) < max_num_candidates:
            #if the signal dips below zero, we ignore and keep iterating until it 
            #reaches a value above zero.
            index += 1
                
    return np.array( maxima_values ), np.array( maxima_places )