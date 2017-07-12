import numpy as np
import scipy.fftpack as sf

def gaussian( length, T ):
    """
    This fucntion creates an array of points corresponding to a gaussian window
    
    Args:
        length (int): The length that the returned array should be.
        T (float): The time period that the gaussian window is being created for.
        
    Returns:
        numpy.ndarray: an array of points corresponding to a gaussian window of time T.
        
    """
    t = np.linspace( 0, T, length )
    return ( np.e ** ( -12 * ( t / T - .5 ) ** 2 ) - np.e ** -12 ) / ( 1 - np.e ** -12 )

def segment_signal( window_len, time_step, sig, rate ):
    """
    This functions accepts a signal, and partitions it based off parameters passed in.
    
    Args:
        window_len (float): The length of one partition of the signal.
        time_step (float): The length of time between each window.
        sig (numpy.ndarray): The signal to be partitioned.
        rate (int): The rate of samples taken per second.
        
    Returns:
        list: a list composed of numpy.ndarray, each index corresponding to a partition.
    """
    
    frame_len = int( window_len * rate )
    time_len = int( time_step * rate )
    
    #there has to be at least one frame
    num_frames = max( 1, int( len( sig ) / time_len + .5 ) ) 
    
    segmented_signal = [ sig[ int( i * time_len ) : int( i  * time_len ) + frame_len ]  
                                                 for i in range( num_frames + 1 ) ]
    
    #This eliminates an empty list that could be created at the end
    if len( segmented_signal[ - 1 ] ) == 0:
        segmented_signal = segmented_signal[ : -1 ]
    return segmented_signal

def estimated_autocorrelation( x ):
    """
    This function accepts a signal and calculates an estimation of the autocorrelation, 
    based off the given algorithm (steps 3.5-3.9):
        http://www.fon.hum.uva.nl/david/ba_shs/2010/Boersma_Proceedings_1993.pdf
    described below 
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
    s = sf.fft( x )
    a = np.real( sf.fft( s * np.conjugate( s ) ) )
    a = a[ :N ]
    return a

def viterbi( cands, strengths, v_unv_cost, oct_jump_cost ):
    """
    Calculates smallest costing path through list of candidates, and returns path.
    Detailed description can be found at step 4 of algorithm described in:
        http://www.fon.hum.uva.nl/david/ba_shs/2010/Boersma_Proceedings_1993.pdf
    
    Args:
        cands (list): a list of tuples, each tuple containing possible candidates, with the index 
        of the tuple of candidates corresponding to that segmented frame.
        
        strengths (list): a list of tuples, each tuple containing strengths that correspond to the 
        candidates in the previously passed in list.
            
        v_unv_cost (float): the voiced/unvoiced cost provided by the user, used to determine if an 
        interval is voiced or unvoiced.
            
        oct_jump_cost (float): the octave jump cost provided by the user, used to determined if an 
        interval can jump octaves.
    Returns:
        numpy.ndarray : an array of the best candidates per frame based off of the path with the least
        cost.
    """
    best_total_cost = np.inf
    best_total_path = []
    
    #for each initial candidate find the path of least cost, then of those paths, choose the one 
    #with the least cost.
    for a in range( len( cands[ 0 ] ) ):
        start_val = cands[ 0 ][ a ]
        total_path = [ start_val ]
        #the starting cost is minus the strength of that candidate
        total_cost = -1 * strengths[ 0 ][ a ]
        level = 1
        
        while level < len( cands ) :
            
            prev_val = total_path[ -1 ]
            best_cost = np.inf
            best_val  = np.inf
            for j in range( len( cands[ level ] ) ):
                cur_val = cands[ level ][ j ] 
                
                if prev_val == np.inf and cur_val == np.inf:
                    cost = 0
                elif prev_val == np.inf or cur_val == np.inf:
                    cost = v_unv_cost 
                else:
                    cost = oct_jump_cost * abs( np.log2( prev_val / cur_val ) ) 
                
                #The cost for any given candidate is given by the transition cost, minus the strength
                #of the given candidate
                cost -= ( strengths[ level ][ j ] )
                
                if cost <= best_cost:
                    best_cost = cost
                    best_val = cur_val
                    
            total_path.append( best_val )
            total_cost += best_cost
            level += 1
        if total_cost < best_total_cost:
            best_total_cost = total_cost
            best_total_path = total_path

    return np.array( best_total_path )
            