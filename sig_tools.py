import numpy as np
import scipy.fftpack as sf
"""
def gaussian(length,T,k=12):
    t=np.linspace(0,T,length)
    return (np.e**(-k*(t/T-.5)**2)-np.e**(-k))/(1-np.e**(-k))
"""
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
    s = sf.fft( x )
    a = np.real( sf.fft( s * np.conjugate( s ) ) )
    a = a[ :N ]
    return a
def viterbi(cands,strengths,num_cands,v_unv_cost,oct_jump_cost):
    best_total_cost=np.inf
    best_total_path=[]
    for a in range(len(cands[-1])):
        end_val=cands[-1][a]
        level=len(cands)-1
        best_path=[]
        best_path.append(end_val)
        total_cost=0
        while level>0:
            prev_level=level-1
            cur_val=best_path[-1]
            j=cands[level].index(cur_val)
            best_val=0
            best_cost=0
            
            if len(cands[prev_level])==1:
                best_val=np.inf
                if cur_val<np.inf:
                    best_cost=v_unv_cost
            else:
                for i in range(len(cands[prev_level])):
                    prev_val=cands[prev_level][i]
                    if prev_val==np.inf and cur_val==np.inf:
                        cost=-1*(strengths[level][j]+strengths[prev_level][i])
                    elif prev_val==np.inf or cur_val==np.inf:
                        cost=v_unv_cost-(strengths[level][j]+strengths[prev_level][i])
                    else:
                        cost=oct_jump_cost*abs(np.log2(cur_val/prev_val))-(strengths[level][j]+strengths[prev_level][i])
                    if cost<=best_cost:
                        best_cost=cost
                        best_val=prev_val
                        
            best_path.append(best_val)
            total_cost+=best_cost
            level-=1
        if total_cost<best_total_cost:
            best_total_cost=total_cost
            best_total_path=best_path
    return best_total_path
            
            
            
def sinc_interp( x, s, u, time_step ):
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
    T = time_step
    #This creates an array of values to use in our interpolation
    sincM = np.tile( u, ( len( s ), 1 ) ) - np.tile( s[ :, np.newaxis ], ( 1, len( u ) ) )
    #This calculates interpolated array
    y = np.dot( x, np.sinc( sincM / T ) )
    return y