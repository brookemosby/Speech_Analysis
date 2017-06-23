import sig_tools as st
import numpy as np
from scipy import fftpack as sf
import peakutils as pu


def get_F_0( signal, rate, min_pitch = 75, max_pitch = 600, max_num_cands = 15,
            voicing_threshold = .45, silence_threshold = .01, octave_cost = .01,
            voiced_unvoiced_cost = .005, octave_jump_cost = .015, jitter = False):
    """
    Compute Fundamental Frequency (F0).
    Algorithm filters out values higher than the Nyquist Frequency, then segments the signal 
    into frames containing at least 3 periods of the minimum pitch. For each frame it then 
    calculates autocorrelation of the signal. After autocorrelation is calculated the maxima 
    values are found. Once these values have been chosen the best candidate for the F0 is 
    picked and then returned.
    This algorithm is adapted from 
    http://www.fon.hum.uva.nl/david/ba_shs/2010/Boersma_Proceedings_1993.pdf
    
    Args:
        signal (numpy.ndarray): The signal the fundamental frequency will be calculated from.
        
        rate (int): the rate per seconds that the signal was sampled at.
        
        min_pitch (float): (default value: 75) minimum value to be returned as pitch, cannot 
        be less than or equal to zero
        
        max_pitch (float): (default value: 600) maximum value to be returned as pitch, cannot
        be greater than Nyquist Frequency
        
        max_num_cands (int): (default value: 15) maximum number of candidates to be 
        considered for each frame, unvoiced candidate (i.e. F_0 equal to zero) is always 
        considered.
        
        silence_threshold (float): (default value: 0.01) frames that do not contain amplitudes
        above this threshold (relative to the global maximum amplitude), are probably silent.
        
        voicing_threshold (float): (default value: 0.45) the strength of the unvoiced candidate,
        relative to the maximum possible autocorrelation. To increase the number of unvoiced 
        decisions, increase this value.
        
        octave_cost (float): (default value: 0.01 per octave) degree of favouring of 
        high-frequency candidates, relative to the maximum possible autocorrelation. This is 
        necessary because in the case of a perfectly periodic signal, all undertones of F0 are 
        equally strong candidates as F0 itself. To more strongly favour recruitment of 
        high-frequency candidates, increase this value.
        
        octave_jump_cost (float): (default value: 0.15) degree of disfavouring of pitch changes, 
        relative to the maximum possible autocorrelation. To decrease the number of large 
        frequency jumps, increase this value. 
        
        voiced_unvoiced_cost (float): (default value: 0.005) degree of disfavouring of 
        voiced/unvoiced transitions, relative to the maximum possible autocorrelation. To 
        decrease the number of voiced/unvoiced transitions, increase this value.
        
        #TODO:add in a jitter parameter
        
    Returns:
        float: The median F0 of the signal.
        
    Raises:
        ValueError: The maximum pitch cannot be greater than the Nyquist Frequency.
        ValueError: The minimum pitch cannot be equal or less than zero.
        ValueError: The minimum number of candidates is 2.
        ValueError: octave_cost must be between 0 and 1.
        ValueError: silence_threshold must be between 0 and 1.
        ValueError: voicing_threshold must be between 0 and 1.

    Example:
    ::
        from scipy.io import wavfile as wav
        import Signal_Analysis as sig
        rate, wave = wav.read( 'example_audio_file.wav' )
        sig.get_F_0( wave, rate )
    """
    
    time_step = 1. / rate
    total_time = time_step * len( signal )
    Nyquist_Frequency = 1. / ( time_step * 2 )
    
    #checking to make sure values are valid
    if Nyquist_Frequency < max_pitch:
        raise ValueError( "The maximum pitch cannot be greater than the Nyquist Frequency." )
    if max_num_cands <2 :
        raise ValueError( "The minimum number of candidates is 2.")
    if octave_cost < 0 or octave_cost > 1:
        raise ValueError( "octave_cost must be between 0 and 1." )            
    if voicing_threshold < 0 or voicing_threshold> 1:
        raise ValueError( "voicing_threshold must be between 0 and 1." ) 
    if min_pitch <= 0:
        raise ValueError( "The minimum pitch cannot be equal or less than zero." )
    if silence_threshold < 0 or silence_threshold > 1:
        raise ValueError( "silence_threshold must be between 0 and 1." )
        
    #filtering by Nyquist Frequency and segmenting signal 
    upper_bound = .9 * Nyquist_Frequency
    intial_len = len( signal )
    zeros_pad = 2 ** ( int( np.log2( len( signal ) ) ) + 1 ) - len( signal )
    signal = np.hstack( ( signal, np.zeros( zeros_pad ) ) )
    fft_signal = sf.fft( signal )
    fft_signal = fft_signal * ( fft_signal < upper_bound )
    sig = sf.ifft( fft_signal )
    sig = sig[ :intial_len ]
    
    #segmenting signal into windows that contain 3 periods of minimum pitch
    window_len = 3.0 / min_pitch
    segmented_signal = st.segment_signal( window_len, time_step, sig )
    global_peak = max( abs( sig ) )    
    
    #initializing list of candidates for F_0, and their strengths
    best_cands = []
    strengths = []
    for index in range( len( segmented_signal ) ):
        
        time_begin = index * window_len
        time_end = min( ( index + 1 ) * window_len, total_time )
        window_len = time_end - time_begin
        segment = segmented_signal[ index ]
        
        local_peak = max( abs( segment ) )
        
        #calculating autocorrelation, based off steps 3.2-3.10
        segment = segment - segment.mean()
        window = np.hanning( len( segment ) )
        segment *= window
        r_a = st.estimated_autocorrelation( segment )
        r_w = st.estimated_autocorrelation( window )
        r_x = r_a / r_w
        r_x /= r_x[ 0 ]
        #eliminating infinite values created by dividing by numbers close to zero
        r_x = r_x[ np.isfinite( r_x ) ]
        r_x = r_x[: int( len( r_x ) / 2 ) ]
        #creating an array of the points in time corresponding to our sampled autocorrelation
        #of the signal (r_x)
        time_array = np.linspace( 0, window_len / 2, len( r_x ) )
        upsamp = np.linspace( 0 , window_len / 2, len( r_x ) * 2 )
        r_x = st.sinc_interp( r_x, time_array, upsamp, time_step )
        time_array = upsamp
        
        #finding maximizers, and maximums and eliminating values that don't produce a 
        #pitch in the allotted range.
        i=pu.indexes(r_x)
        maxima_values, maxima_places = r_x[  i ], time_array[ i ]
        max_place_possible = 1. / min_pitch
        min_place_possible = 1. / max_pitch

        maxima_values = maxima_values[ maxima_places >= min_place_possible ]
        maxima_places = maxima_places[ maxima_places >= min_place_possible ]
        
        maxima_values = maxima_values[ maxima_places <= max_place_possible ]
        maxima_places = maxima_places[ maxima_places <= max_place_possible ]
        
        if len( maxima_values ) > 0 :
            #finding the max_num_cands-1 maximizers, and maximums, then calculating their
            #strengths (eq. 23 & 24) and accounting for silent candidate
            maxima_places = np.array( [ maxima_places[ i ] for i in np.argsort( maxima_values )[
                    -1 * ( max_num_cands - 1 ) : ] ] )
            maxima_values = np.array( [ maxima_values[ i ] for i in np.argsort( maxima_values )[
                    -1 * ( max_num_cands - 1 ) : ] ] )
            strengths_1 = [ max_val - octave_cost ** 2 * np.log2( min_pitch * max_place ) for 
                    max_val, max_place in zip( maxima_values, maxima_places ) ]
            strengths_1.append( voicing_threshold + max( 0, 2 - ( ( local_peak / global_peak ) / 
                    ( silence_threshold / ( 1 + voicing_threshold ) ) ) ) )
            
            #np.inf is our silent candidate
            maxima_places = np.hstack( ( maxima_places, np.inf ) )
            best_cands.append( list( maxima_places ) )
            strengths.append( strengths_1 )
        else:
            #if there are no available maximums, only account for silent candidate
            best_cands.append( [ np.inf ] )
            strengths.append( [ voicing_threshold + max( 0, 2 - ( ( local_peak / global_peak ) /
                    ( silence_threshold / ( 1 + voicing_threshold ) ) ) ) ] )
                
    #using viterbi algorithm to find a path through best set of candidates        
    f_0 = st.viterbi( best_cands, strengths, max_num_cands, voiced_unvoiced_cost, octave_jump_cost )
    
    f_0 = np.array( f_0 )

    
    f_0 = f_0[ f_0 < np.inf ]
    if jitter:
        return f_0
    if len( f_0 ) == 0:
        return 0
    else:
        f_0 = 1.0 / f_0
        return np.median( f_0 )





def get_HNR( signal, rate, min_pitch = 75, silence_threshold = .01, periods_per_window = 6):
    """
    Compute Fundamental Frequency (F_0).
    Algorithm filters out values higher than the Nyquist Frequency, then segments the signal 
    into frames containing at least 3 periods of the minimum pitch. For each frame it then 
    calculates autocorrelation of the signal. After autocorrelation is calculated the maxima 
    values are found. Once these values have been chosen the best candidate for the F_0 is 
    picked and then returned.
    This algorithm is adapted from 
    http://www.fon.hum.uva.nl/david/ba_shs/2010/Boersma_Proceedings_1993.pdf
    
    Args:
        signal (numpy.ndarray): The signal the fundamental frequency will be calculated from.
        
        rate (int): the rate per seconds that the signal was sampled at.
        
        min_pitch (float): (default value: 75) minimum value to be returned as pitch, cannot be 
        less than or equal to zero
                  
        silence_threshold (float): (default value: 0.01) frames that do not contain amplitudes 
        above this threshold (relative to the global maximum amplitude), are considered silent.

        periods_per_window (float): (default value: 6) 4.5 is best for speech: HNR values up to
        37 dB are guaranteed to be detected reliably; 6 periods per window raises this figure to 
        more than 60 dB, but the algorithm becomes more sensitive to dynamic changes in the 
        signal.
        
    Returns:
        float: The mean HNR of the signal.
        
    Raises:
        ValueError: The minimum pitch cannot be equal or less than zero.
        ValueError: silence_threshold must be between 0 and 1.

    Example:
    ::
        from scipy.io import wavfile as wav
        import Signal_Analysis as sig
        rate, wave = wav.read( 'example_audio_file.wav' )
        sig.get_F_0( wave, rate )
    """
    
    time_step = 1. / rate
    total_time = time_step * len( signal )
    Nyquist_Frequency = 1. / ( time_step * 2 )
    max_pitch=Nyquist_Frequency
    
    #checking to make sure values are valid
    if min_pitch <= 0:
        raise ValueError( "The minimum pitch cannot be equal to or less than zero." )
    if silence_threshold < 0 or silence_threshold > 1:
        raise ValueError( "silence_threshold must be between 0 and 1." )
        
    #filtering by Nyquist Frequency and segmenting signal 
    upper_bound = .9 * Nyquist_Frequency
    intial_len = len( signal )
    zeros_pad = 2 ** ( int( np.log2( len( signal ) ) ) + 1 ) - len( signal )
    signal = np.hstack( ( signal, np.zeros( zeros_pad ) ) )
    fft_signal = sf.fft( signal )
    fft_signal = fft_signal * ( fft_signal < upper_bound )
    sig = sf.ifft( fft_signal )
    sig = sig[ :intial_len ]
    window_len = periods_per_window / min_pitch
    segmented_signal = st.segment_signal( window_len, time_step, sig )
    
    #initializing list of candidates for HNR
    best_cands = []
    global_peak = max( abs( sig ) )
    for index in range( len( segmented_signal ) ):
        
        time_begin = index * window_len
        time_end = min( ( index + 1 ) * window_len, total_time )
        window_len = time_end - time_begin
        segment = segmented_signal[ index ]
        local_peak = max( abs( segment ) )
        
        #calculating autocorrelation, based off steps 3.2-3.10
        segment = segment - segment.mean()
        window = np.hanning( len( segment ) )
        segment *= window
        r_a = st.estimated_autocorrelation( segment )
        r_w = st.estimated_autocorrelation( window )
        r_x = r_a / r_w
        r_x /= r_x[ 0 ]
        
        #creating an array of the points in time corresponding to our sampled autocorrelation
        #of the signal (r_x)
        time_array = np.linspace( 0, window_len, len( r_x ) )
        
        #finding maximizers, and maximums and eliminating values that don't produce a 
        #pitch in the allotted range.
        max_place_possible = 1. / min_pitch
        min_place_possible = 1. / max_pitch

        r_x = r_x[ time_array >= min_place_possible ]
        time_array = time_array[ time_array >= min_place_possible ]
        
        r_x = r_x[ time_array <= max_place_possible ]
        time_array = time_array[ time_array  <= max_place_possible ]
        
        #values greater than one produce nan values
        r_x = r_x[ r_x <1 ]
        
        #eq. 23 & 24 with octave_cost, and voicing_threshold set to zero
        strengths = [ max( r_x ), max( 0, 2 - ( ( local_peak / global_peak ) / ( silence_threshold ) ) ) ]
        #if the maximum strength is the unvoiced candidate, then .5 corresponds to HNR of 0
        if np.argmax( strengths ):
            best_cands.append( .5 )  
        else:
            best_cands.append( max( r_x ) )
    
    best_cands = np.array( best_cands )
    best_cands = best_cands[ best_cands >= .5 ]
    if len(best_cands) == 0:
        return 0
    #eq. 4
    best_cands = 10 * np.log10( best_cands / ( 1 - best_cands ) )
    best_candidate = np.mean( best_cands )
    return best_candidate
        




def get_Jitter( signal, rate, period_floor = .0001, period_ceiling = .02, max_period_factor = 1.3 ):
    """
    Compute Jitter, random pertubations in period length.
    Algorithm filters out values higher than the Nyquist Frequency, then segments the signal
    into frames containing at least 3 periods of period ceiling. For each frame it 
    calculates absolute jitter, relative jitter, relative average perturbation (rap), the 5-
    point period pertubation quotient (ppq5), and the difference of differences of periods (ddp).
    After each type of jitter has been calculated for each frame the best candidate for each type
    is chosen and returned in a dictionary.
    This algorithm is adapted from 
    http://www.lsi.upc.edu/~nlp/papers/far_jit_07.pdf
    
    Args:
        signal (numpy.ndarray): The signal the fundamental frequency will be calculated from.
        
        rate (int): the rate per seconds that the signal was sampled at.
        
        period_floor (float): (default value: .0001) the shortest possible interval that will be 
        used in the computation of jitter, in seconds. If an interval is shorter than this, it 
        will be ignored in the computation of jitter (and the previous and next intervals will 
        not be regarded as consecutive). This setting will normally be very small, say 0.1 ms.
        
        period_ceiling (float): (default value: .02) the longest possible interval that will be 
        used in the computation of jitter, in seconds. If an interval is longer than this, it 
        will be ignored in the computation of jitter (and the previous and next intervals will 
        not be regarded as consecutive). For example, if the minimum frequency of periodicity 
        is 50 Hz, set this setting to 0.02 seconds; intervals longer than that could be regarded
        as voiceless stretches and will be ignored in the computation.
        
        max_period_factor (float): (default value: 1.3) the largest possible difference between 
        consecutive intervals that will be used in the computation of jitter. If the ratio of the
        durations of two consecutive intervals is greater than this, this pair of intervals will 
        be ignored in the computation of jitter (each of the intervals could still take part in 
        the computation of jitter in a comparison with its neighbour on the other side).
        
    Returns:
        dict: a dictionary with keys: 'local', 'local, absolute', 'rap', 'ppq5', and 'ddp' and 
        values corresponding to each type of jitter.
        
    Raises:
        
    Example:
    ::
        from scipy.io import wavfile as wav
        import Signal_Analysis as sig
        rate, wave = wav.read( 'example_audio_file.wav' )
        sig.get_Jitter( wave, rate )
    
    """    
    
    #***Q***: how are we supposed to be calculating the periods that we calculate jitters 
    #from? should we be using default values for calculating periods?
    periods = get_F_0( signal, rate, jitter = True )

    min_period_factor = 1.0 / max_period_factor
    period_variation = []
    
    #***Q***: in praat it says to ignore values below period floor and above period
    #ceiling and any that have a transition ratio greater than max_period_factor, do 
    #we want to do the same?
    
    for i in range( len( periods ) - 1 ):
        if periods[ i ] > period_ceiling or periods[ i ] < period_floor:
            pass
        elif periods[ i + 1 ] > period_ceiling or periods[ i + 1 ] < period_floor:
            pass
        else:
            ratio = periods[ i + 1 ] / periods[ i ]
            if ratio < max_period_factor and ratio > min_period_factor:
                period_variation.append( abs( periods[ i + 1 ] - periods[ i ] ) )
        
    sum_total = 0
    periods = np.hstack( ( periods, periods[ -1 ] ) )
    ignore_next = False
    
    for i in range( len( periods ) - 1 ):
        if periods[ i ] > period_ceiling or periods[ i ] < period_floor:
            pass
        elif periods[ i + 1 ] > period_ceiling or periods[ i + 1 ] < period_floor:
            pass
        elif ignore_next:
            
            ratio = periods[ i + 1 ] / periods[ i ]
            if ratio < max_period_factor and ratio > min_period_factor:
                ignore_next = False
            else:
                ignore_next = True
        else:
            ratio = periods[ i + 1 ] / periods[ i ]
            if ratio < max_period_factor and ratio > min_period_factor:
                sum_total += periods[ i ]
            else:
                ignore_next = True
                
    avg_period = sum_total / len( periods )
    
    absolute = np.mean( period_variation )
    relative = np.mean( period_variation ) / avg_period * 100 

    #have yet to finish these types of jitter(need to account for period ceiling,
    #period floor and max_period_factor, assuming that we are using these values)
    
    top    = np.hstack( ( periods,     [ 0, 0 ] ) )
    middle = np.hstack( ( [ 0 ], periods, [ 0 ] ) )
    bottom = np.hstack( ( [ 0, 0 ],     periods ) )
    
    neighbors_matrix = np.vstack( ( top, middle, bottom ) )
    avg_neighbors = np.zeros( len( neighbors_matrix ) - 2 )
    for i in range( len( neighbors_matrix ) - 2 ):
        col = neighbors_matrix[ 0:, i + 1 ]
        zeros = list( col ).count( 0 )
        if zeros < 3:
            avg_neighbors[ i ] = sum( col ) / ( 3 - zeros )
            
    rap = np.mean( abs( periods  - avg_neighbors) ) / avg_period * 100 
    
    top     = np.hstack( ( periods,     [ 0, 0, 0, 0 ] ) )
    upper_q = np.hstack( ( [ 0 ], periods, [ 0, 0, 0 ] ) )
    middle  = np.hstack( ( [ 0, 0 ], periods, [ 0, 0 ] ) )
    lower_q = np.hstack( ( [ 0, 0, 0 ], periods, [ 0 ] ) )
    bottom  = np.hstack( ( [ 0, 0, 0, 0 ],     periods ) )
    
    neighbors_matrix = np.vstack( ( top, upper_q, middle, lower_q, bottom ) )
    avg_neighbors = np.zeros( len( neighbors_matrix ) - 4 )
    for i in range( len( neighbors_matrix ) - 4 ):
        col = neighbors_matrix[ 0:, i + 2 ]
        zeros = list( col ).count( 0 )
        if zeros < 5:
            avg_neighbors[ i ] = sum( col ) / ( 5 - zeros )                      
                        
    ppq5 = np.mean( abs( periods  - avg_neighbors) ) / avg_period * 100 
            
    return {  'local' : relative, 'local, absolute' : absolute, 'rap' : rap, 'ppq5' : ppq5, 'ddp' : 3 * rap }



























#TODO: for jitter we want to return all 5 types, in a dictionary with key being the type of jitter
#TODO:look up fixtures in python testing, maybe ask david if not sure about it...
#TODO: need to document test file better 
#TODO: get code put up github/travis/coveralls...   
#TODO: clone repo, put on github-> goes in super ai->afx->features (in my fork, ask when ready to merge fork)
#TODO: travis and coveralls, he will get a key for me to put it on privately(only for coveralls), which I don't have yet so maybe don't work on this.
#which I can put up once I have finished all the features
#TODO: update github pages so that I can post other things on there besides just titanic