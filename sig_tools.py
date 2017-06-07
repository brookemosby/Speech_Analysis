import numpy as np

def segment_signal( window_len, time_step, sig ):
    """
    This functions accepts a signal, and partitions it based off parameters passed in.
    
    Args:
        window_len (float): The length of one partition of the signal.
        time_step (float): The length of time between each sample.
        sig (numpy.ndarray): The signal to be partitioned.
        
    Returns:
        segmented_signal (list): a list composed of numpy.ndarray, each index 
        corresponding to a partition.
    """
    
    frame_len = window_len / time_step
    
    #there has to be at least one frame
    num_frames = max( 1, int( len( sig ) / frame_len + .5 ) ) 
    
    segmented_signal = [ sig[ int( i * frame_len ) : int( ( i + 1 ) * frame_len ) ] 
                                                 for i in range( num_frames + 1 ) ]
    
    #This eliminates an empty list that could be created at the end
    if len( segmented_signal[ - 1 ] ) == 0:
        segmented_signal = segmented_signal[ : -1 ]
    return segmented_signal
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
        x (numpy.ndarray): an array of the signal that autocorrelation is calculated 
        from.
    
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
    This function uses sinc interpolation to up-sample or down-sample x.
    
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
        time_array (numpy.ndarray): an array of the corresponding points in time that
        the signal was sampled at
        
    Returns:
        maxima_values (numpy.ndarray): an array of the calculated maximum values
        maxima_places (numpy.ndarray): an array of the corresponding places where the 
        maxima occur.
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