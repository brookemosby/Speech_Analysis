import numpy as np
import peakutils as pu


def get_F_0( signal, rate, time_step = .04, min_pitch = 75, max_pitch = 600, max_num_cands = 15,
            silence_threshold = .03, voicing_threshold = .45, octave_cost = .01, octave_jump_cost = .35,
            voiced_unvoiced_cost = .14, accurate = False, pulse = False ):
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
        
        time_step (float): (default value: 0.04) the measurement interval (frame duration), in seconds. 
        If you supply 0, Praat will use a time step of 0.75 / (min_pitch), e.g. 0.01 seconds if the 
        minimum pitch is 75 Hz; in this example, algorithm computes 100 pitch values per second.
        
        min_pitch (float): (default value: 75) minimum value to be returned as pitch, cannot 
        be less than or equal to zero
        
        max_pitch (float): (default value: 600) maximum value to be returned as pitch, cannot
        be greater than Nyquist Frequency
        
        max_num_cands (int): (default value: 15) maximum number of candidates to be 
        considered for each frame, unvoiced candidate (i.e. F_0 equal to zero) is always 
        considered.
        
        silence_threshold (float): (default value: 0.03) frames that do not contain amplitudes
        above this threshold (relative to the global maximum amplitude), are probably silent.
        
        voicing_threshold (float): (default value: 0.45) the strength of the unvoiced candidate,
        relative to the maximum possible autocorrelation. To increase the number of unvoiced 
        decisions, increase this value.
        
        octave_cost (float): (default value: 0.01 per octave) degree of favouring of 
        high-frequency candidates, relative to the maximum possible autocorrelation. This is 
        necessary because in the case of a perfectly periodic signal, all undertones of F0 are 
        equally strong candidates as F0 itself. To more strongly favour recruitment of 
        high-frequency candidates, increase this value.
        
        octave_jump_cost (float): (default value: 0.35) degree of disfavouring of pitch changes, 
        relative to the maximum possible autocorrelation. To decrease the number of large 
        frequency jumps, increase this value. 
        
        voiced_unvoiced_cost (float): (default value: 0.14) degree of disfavouring of 
        voiced/unvoiced transitions, relative to the maximum possible autocorrelation. To 
        decrease the number of voiced/unvoiced transitions, increase this value.
        
        accurate (bool): (default value: False) if false, the window is a Hanning window with a physical 
        length of 3 / (min_pitch). If on, the window is a Gaussian window with a physical length of 
        6 / (min_pitch), i.e. twice the effective length.
        
        pulse (bool): (default value: False) if false, returns the median F_0, if True, returns the
        frequencies for each frame in a list and also a list of tuples containing the beginning time of 
        the frame, and the ending time of the frame. The indicies in each list correspond to each other.
        
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
    Nyquist_Frequency = rate /  2.0
    global_peak = max( abs( signal ) ) 
    upper_bound = .95 * Nyquist_Frequency
    initial_len = len( signal )
    zeros_pad = 2 ** ( int( np.log2( len( signal ) ) ) + 1 ) - len( signal )
    signal = np.hstack( ( signal, np.zeros( zeros_pad ) ) )
    fft_signal = np.fft.fft( signal )
    for i in range( int( upper_bound ), int( initial_len / 2 ) ):
        fft_signal[ i ] = 0
    sig = np.fft.ifft( fft_signal )
    sig = sig[ :initial_len ]
    
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
        
    #segmenting signal into windows that contain 3 periods of minimum pitch
    if time_step == 0:
        time_step = .75 / min_pitch
    
    if accurate:
        window_len = 6.0 / min_pitch
    else:
        window_len = 3.0 / min_pitch
            
    octave_jump_cost     *= .01 / time_step
    voiced_unvoiced_cost *= .01 / time_step 

    #Segmenting signal
    frame_len = int( window_len * rate )
    time_len = int( time_step * rate )
    
    #there has to be at least one frame
    num_frames = max( 1, int( len( sig ) / time_len + .5 ) ) 
    
    segmented_signal = [ sig[ int( i * time_len ) : int( i  * time_len ) + frame_len ]  
                                                 for i in range( num_frames + 1 ) ]
    
    #This eliminates an empty list that could be created at the end
    if len( segmented_signal[ - 1 ] ) == 0:
        segmented_signal = segmented_signal[ : -1 ]
     
    
    #initializing list of candidates for F_0, and their strengths
    best_cands = []
    strengths  = []
    if pulse:
        time_vals = []
    for index in range( len( segmented_signal ) ):
        
        segment = segmented_signal[ index ]
        window_len = len( segment ) / float( rate )
        local_peak = max( abs( segment ) )
        if pulse:
            time_vals.append( ( index * time_step, index * time_step + window_len ) )

        if accurate:
            t = np.linspace( 0, window_len, len( segment ) )
            window = ( np.e ** ( -12 * ( t / window_len - .5 ) ** 2 ) - np.e ** -12 ) / ( 1 - np.e ** -12 )
        else:
            window = np.hanning( len( segment ) )
            
        #calculating autocorrelation, based off steps 3.2-3.10
        segment = segment - segment.mean()
        segment *= window
        """
        Calculates an estimation of the autocorrelation,based off the given algorithm (steps 3.5-3.9):
            http://www.fon.hum.uva.nl/david/ba_shs/2010/Boersma_Proceedings_1993.pdf
        described below 
        1. append half the window length of zeros
        2. append zeros until the segment length is a power of 2, calculated with log.
        3. take the FFT
        4. square samples in the signal
        5. then again take the FFT
        """
        N = len( segment )
        x = np.hstack( ( segment, np.zeros( int( N / 2 ) ) ) )
        x = np.hstack( ( x, np.zeros( 2 ** ( int( np.log2( N ) + 1 ) ) - N ) ) )            
        x_fft = np.fft.fft( x )
        r_a = np.real( np.fft.fft( x_fft * np.conjugate( x_fft ) ) )
        r_a = r_a[ :N ]
        
        N = len( window )
        x = np.hstack( ( window, np.zeros( int( N / 2 ) ) ) )
        x = np.hstack( ( x, np.zeros( 2 ** ( int( np.log2( N ) + 1 ) ) - N ) ) )            
        x_fft = np.fft.fft( x )
        r_w = np.real( np.fft.fft( x_fft * np.conjugate( x_fft ) ) )
        r_w = r_w[ :N ]
        
        r_x = r_a / r_w
        r_x /= r_x[ 0 ]
        #creating an array of the points in time corresponding to our sampled autocorrelation
        #of the signal (r_x)
        time_array = np.linspace( 0, window_len, len( r_x ) )
        i = pu.indexes( r_x )
        maxima_values, maxima_places = r_x[ i ], time_array[ i ]
        max_place_possible = 1.0 / min_pitch
        min_place_possible = 1.0 / max_pitch
        
        maxima_values = maxima_values[ maxima_places >= min_place_possible ]
        maxima_places = maxima_places[ maxima_places >= min_place_possible ]
        
        maxima_values = maxima_values[ maxima_places <= max_place_possible ]
        maxima_places = maxima_places[ maxima_places <= max_place_possible ]
        
        if len( maxima_values ) > 0:
            #finding the max_num_cands-1 maximizers, and maximums, then calculating their
            #strengths (eq. 23 & 24) and accounting for silent candidate
            maxima_places = np.array( [ maxima_places[ i ] for i in np.argsort( maxima_values )[
                    -1 * ( max_num_cands - 1 ) : ] ] )
            maxima_values = np.array( [ maxima_values[ i ] for i in np.argsort( maxima_values )[
                    -1 * ( max_num_cands - 1 ) : ] ] )
            strengths_1 = [ max_val - octave_cost * np.log2( min_pitch * max_place ) for 
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
    """
    Calculates smallest costing path through list of candidates, and returns path.
    Detailed description can be found at step 4 of algorithm described in:
        http://www.fon.hum.uva.nl/david/ba_shs/2010/Boersma_Proceedings_1993.pdf
    """
    best_total_cost = np.inf
    best_total_path = []
    #for each initial candidate find the path of least cost, then of those paths, choose the one 
    #with the least cost.
    for a in range( len( best_cands[ 0 ] ) ):
        start_val = best_cands[ 0 ][ a ]
        total_path = [ start_val ]
        #the starting cost is minus the strength of that candidate
        total_cost = -1 * strengths[ 0 ][ a ]
        level = 1
        
        while level < len( best_cands ) :
            
            prev_val = total_path[ -1 ]
            best_cost = np.inf
            best_val  = np.inf
            for j in range( len( best_cands[ level ] ) ):
                cur_val = best_cands[ level ][ j ] 
                
                if prev_val == np.inf and cur_val == np.inf:
                    cost = 0
                elif prev_val == np.inf or cur_val == np.inf:
                    cost = voiced_unvoiced_cost 
                else:
                    cost = octave_jump_cost * abs( np.log2( prev_val / cur_val ) ) 
                
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

    f_0 = np.array( best_total_path )    
    print(f_0)
    if pulse:
        removed = 0
        for i in range( len( f_0 ) ):
            if f_0[ i ] == np.inf:
                time_vals.remove( time_vals[ i - removed ] )
                removed += 1
                
    f_0 = f_0[ f_0 < np.inf ]
    if pulse:
        return f_0, time_vals
    
    if len( f_0 ) == 0:
        return 0
    else:
        f_0 = 1.0 / f_0
        return np.median( f_0 )





def get_HNR( signal, rate, time_step =.01, min_pitch = 75, silence_threshold = .1, periods_per_window = 4.5):
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
                  
        silence_threshold (float): (default value: 0.1) frames that do not contain amplitudes 
        above this threshold (relative to the global maximum amplitude), are considered silent.

        periods_per_window (float): (default value: 4.5) 4.5 is best for speech: HNR values up to
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
    
    
    
    #checking to make sure values are valid
    if min_pitch <= 0:
        raise ValueError( "The minimum pitch cannot be equal to or less than zero." )
    if silence_threshold < 0 or silence_threshold > 1:
        raise ValueError( "silence_threshold must be between 0 and 1." )
        
    #filtering by Nyquist Frequency and segmenting signal 
    Nyquist_Frequency = rate / 2
    max_pitch = Nyquist_Frequency
    
    upper_bound = .95 * Nyquist_Frequency
    initial_len = len( signal )
    zeros_pad = 2 ** ( int( np.log2( len( signal ) ) ) + 1 ) - len( signal )
    signal = np.hstack( ( signal, np.zeros( zeros_pad ) ) )
    fft_signal = np.fft.fft( signal )
    for i in range( int( upper_bound ), int( initial_len / 2 ) ):
        fft_signal[ i ] = 0
    sig = np.fft.ifft( fft_signal )
    sig = sig[ :initial_len ]
    
    global_peak = max( abs( signal ) ) 
    
    window_len = periods_per_window / float( min_pitch )
    

    #Segmenting signal
    frame_len = int( window_len * rate )
    time_len = int( time_step * rate )
    
    #there has to be at least one frame
    num_frames = max( 1, int( len( sig ) / time_len + .5 ) ) 
    
    segmented_signal = [ sig[ int( i * time_len ) : int( i  * time_len ) + frame_len ]  
                                                 for i in range( num_frames + 1 ) ]
    
    #This eliminates an empty list that could be created at the end
    if len( segmented_signal[ - 1 ] ) == 0:
        segmented_signal = segmented_signal[ : -1 ]
    
    #initializing list of candidates for HNR
    best_cands = []
    
    for index in range( len( segmented_signal ) ):
        
        segment = segmented_signal[ index ]
        window_len = len( segment ) / float( rate )
        local_peak = max( abs( segment ) )
        #calculating autocorrelation, based off steps 3.2-3.10
        segment = segment - segment.mean()
        window = np.hanning( len( segment ) )
        segment *= window
        """
        Calculates an estimation of the autocorrelation,based off the given algorithm (steps 3.5-3.9):
            http://www.fon.hum.uva.nl/david/ba_shs/2010/Boersma_Proceedings_1993.pdf
        described below 
        1. append half the window length of zeros
        2. append zeros until the segment length is a power of 2, calculated with log.
        3. take the FFT
        4. square samples in the signal
        5. then again take the FFT
        """
        N = len( segment )
        x = np.hstack( ( segment, np.zeros( int( N / 2 ) ) ) )
        x = np.hstack( ( x, np.zeros( 2 ** ( int( np.log2( N ) + 1 ) ) - N ) ) )            
        x_fft = np.fft.fft( x )
        r_a = np.real( np.fft.fft( x_fft * np.conjugate( x_fft ) ) )
        r_a = r_a[ :N ]
        
        N = len( window )
        x = np.hstack( ( window, np.zeros( int( N / 2 ) ) ) )
        x = np.hstack( ( x, np.zeros( 2 ** ( int( np.log2( N ) + 1 ) ) - N ) ) )            
        x_fft = np.fft.fft( x )
        r_w = np.real( np.fft.fft( x_fft * np.conjugate( x_fft ) ) )
        r_w = r_w[ :N ]
        r_x = r_a / r_w
        r_x /= r_x[ 0 ]
        
        #creating an array of the points in time corresponding to our sampled autocorrelation
        #of the signal (r_x)
        time_array = np.linspace( 0, window_len, len( r_x ) )
        i = pu.indexes( r_x )
        maxima_values, maxima_places = r_x[ i ], time_array[ i ]
        max_place_possible = 1. / min_pitch
        min_place_possible = 1. / max_pitch

        maxima_values = maxima_values[ maxima_places >= min_place_possible ]
        maxima_places = maxima_places[ maxima_places >= min_place_possible ]
        
        maxima_values = maxima_values[ maxima_places <= max_place_possible ]
        maxima_places = maxima_places[ maxima_places <= max_place_possible ]
        
        maxima_values = np.hstack(( maxima_values, 1.0 / maxima_values ))
        maxima_values = maxima_values[ maxima_values < 1 ]
        

        #eq. 23 & 24 with octave_cost, and voicing_threshold set to zero
        if len( maxima_values ) > 0:
            strengths = [ max( maxima_values ), max( 0, 2 - ( ( local_peak / global_peak ) / ( silence_threshold ) ) ) ]
        #if the maximum strength is the unvoiced candidate, then .5 corresponds to HNR of 0
            if np.argmax( strengths ):
                best_cands.append( .5 )  
            else:
                best_cands.append( strengths[ 0 ] )
        else:
            best_cands.append( .5 )
    
    best_cands = np.array( best_cands )
    best_cands = best_cands[ best_cands > .5 ]
    if len(best_cands) == 0:
        return 0
    #eq. 4
    best_cands = 10 * np.log10( best_cands / ( 1 - best_cands ) )
    best_candidate = np.mean( best_cands )
    return best_candidate
    



    
def get_Pulses(signal, rate, min_pitch = 75, max_pitch = 600, include_maxima = False, include_minima = True ):
    """
    This algorithm examines voiced intervals of a signal, and creates a list of points that correspond
    to the sequence of glottal closures in vocal-fold vibration.
    adapted from: https://pdfs.semanticscholar.org/16d5/980ba1cf168d5782379692517250e80f0082.pdf
    
    Args:
        signal (numpy.ndarray): The signal the fundamental frequency will be calculated from.
        
        rate (int): the rate per seconds that the signal was sampled at.
        
        min_pitch (float): (default value: 75) minimum value to be returned as pitch, cannot 
        be less than or equal to zero
        
        max_pitch (float): (default value: 600) maximum value to be returned as pitch, cannot
        be greater than Nyquist Frequency     
        
    Returns:
        list: a list of points in a time series that correspond to a signal periodicity
    
    Raises:
        ValueError: At least one of include_minima or include_maxima must set to True.
        
    Example:
    ::
        from scipy.io import wavfile as wav
        import Signal_Analysis as sig
        rate, wave = wav.read( 'example_audio_file.wav' )
        sig.get_Pulses( wave, rate )
    
    """
    #first calculate F_0 estimates for each voiced interval
    if not include_maxima and not include_minima:
        raise ValueError( "At least one of include_minima or include_maxima must set to True." )
    period, interval = get_F_0( signal, rate, time_step = 3.0 / min_pitch,
                               min_pitch = min_pitch, max_pitch = max_pitch, pulse = True )
    points=[]
    total_time = len( signal ) / rate
    time_arr = np.linspace( 0, total_time, len( signal ) )
    #Then for each voiced interval calculate all pulses
    for i in range( len( period ) ):
        time_start, time_stop = interval[ i ]
        #finding the starting index for this voiced interval
        t_start_index = int( time_start * rate )
        T_0 = period[ i ]
        #assigning the start of our frame to the start of the voiced interval
        f_start_index = t_start_index
        frame_start = time_arr[ f_start_index ]
        #assigning the end of our frame to the elapsed time of the period, with some cushion room
        frame_stop  = frame_start + 1.25 * T_0
        f_stop_index = int( frame_stop * rate + .5 )
        #finding our minima, maxima, or absolute maxima in this frame dependent on what parameters
        #were passed in
        frame = signal[ f_start_index : f_stop_index + 1 ]
        print( frame, f_start_index, f_stop_index )
        if include_minima and not include_maxima:
            t_index = np.argmin( frame ) + f_start_index
        elif include_maxima and not include_minima:
            t_index = np.argmax( frame ) + f_start_index
        else:
            t_index = np.argmax( abs( frame ) ) + f_start_index
        t = time_arr[ t_index ]
        points.append( t )
        
        #until we have reached the end of our voiced interval, continued looking at frames and picking
        #the best candidate for glottal pulse
        while frame_stop < time_stop:
            frame_start = t + 0.75 * T_0 
            frame_stop  = t + 1.25 * T_0 
            f_start_index = int( frame_start * rate )
            f_stop_index  = int( frame_stop  * rate + .5 )
            if include_minima and not include_maxima:
                t_index = np.argmin( signal[ f_start_index : f_stop_index + 1 ]  ) + f_start_index
            elif include_maxima and not include_minima:
                t_index = np.argmax( signal[ f_start_index : f_stop_index + 1 ]  ) + f_start_index
            else:
                t_index = np.argmax( abs( signal[ f_start_index : f_stop_index + 1 ] )  ) + f_start_index
            t = time_arr[ t_index ]
            points.append( t )
            
    #return the array of pulses (sorted and unique)
    points = np.array( sorted( list( set( points ) ) ) )
    return points



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
        
    Example:
    ::
        from scipy.io import wavfile as wav
        import Signal_Analysis as sig
        rate, wave = wav.read( 'example_audio_file.wav' )
        sig.get_Jitter( wave, rate )
    
    """    
    pulses = get_Pulses( signal, rate )
    periods = np.diff( pulses )
    
    min_period_factor = 1.0 / max_period_factor
    period_variation = []
    
    #finding local, absolute
    #described at: http://www.fon.hum.uva.nl/praat/manual/PointProcess__Get_jitter__local__absolute____.html
    for i in range( len( periods ) - 1 ):
        p1 = periods[ i ]
        p2 = periods[ i + 1 ]
        ratio = p2 / p1
        if (ratio < max_period_factor and 
            ratio > min_period_factor and 
            p1 < period_ceiling and
            p2 < period_ceiling and
            p1 > period_floor and
            p2 > period_floor ):
                period_variation.append( abs( periods[ i + 1 ] - periods[ i ] ) )
                
    absolute = np.mean( period_variation )
    
    #finding local, 
    #described at: http://www.fon.hum.uva.nl/praat/manual/PointProcess__Get_jitter__local____.html
    sum_total = 0
    num_periods = 0
    
    periods = np.hstack(( periods[ 0 ], periods, periods[ -1 ] ))
    for i in range( len( periods ) - 2):
        p1 = periods[ i ]
        p2 = periods[ i + 1 ]
        p3 = periods[ i + 2 ]
        ratio_1, ratio_2 = p1 / p2, p2 / p3
        if (ratio_1 < max_period_factor and 
            ratio_1 > min_period_factor and 
            ratio_2 < max_period_factor and 
            ratio_2 > min_period_factor and 
            p2 < period_ceiling and
            p2 > period_floor ):
            
            sum_total += p2
            num_periods += 1
    periods = periods[ 1 : -1 ]
    avg_period = sum_total / num_periods 
    relative = absolute / avg_period
    
    #finding rap
    #described at: http://www.fon.hum.uva.nl/praat/manual/PointProcess__Get_jitter__rap____.html
    sum_total = 0
    num_periods = 0
    
    for i in range( len( periods ) - 2):
        p1 = periods[ i ]
        p2 = periods[ i + 1 ]
        p3 = periods[ i + 2 ]
        ratio_1, ratio_2 = p1 / p2, p2 / p3
        if (ratio_1 < max_period_factor and 
            ratio_1 > min_period_factor and 
            ratio_2 < max_period_factor and 
            ratio_2 > min_period_factor and 
            p1 < period_ceiling and
            p2 < period_ceiling and
            p3 < period_ceiling and
            p1 > period_floor and
            p2 > period_floor and
            p3 > period_floor ):
            
            sum_total += abs( p2 - ( p1 + p2 + p3 ) / 3.0 )
            num_periods += 1
    rap = ( sum_total / num_periods ) / avg_period 
          
    #finding ppq5
    #described at: http://www.fon.hum.uva.nl/praat/manual/PointProcess__Get_jitter__ppq5____.html
    sum_total = 0
    num_periods = 0
    
    for i in range( len( periods ) - 4):
        p1 = periods[ i ]
        p2 = periods[ i + 1 ]
        p3 = periods[ i + 2 ]
        p4 = periods[ i + 3 ]
        p5 = periods[ i + 4 ]
        ratio_1, ratio_2, ratio_3, ratio_4 = p1 / p2, p2 / p3, p3 / p4, p4 / p5
        if (ratio_1 < max_period_factor and 
            ratio_1 > min_period_factor and 
            ratio_2 < max_period_factor and 
            ratio_2 > min_period_factor and 
            ratio_3 < max_period_factor and 
            ratio_3 > min_period_factor and 
            ratio_4 < max_period_factor and 
            ratio_4 > min_period_factor and 
            p1 < period_ceiling and
            p2 < period_ceiling and
            p3 < period_ceiling and
            p4 < period_ceiling and
            p5 < period_ceiling and
            p1 > period_floor and
            p2 > period_floor and
            p3 > period_floor and
            p4 > period_floor and
            p5 > period_floor):
            
            sum_total += abs( p3 - ( p1 + p2 + p3 +p4 + p5 ) / 5.0 )
            num_periods += 1
            
    ppq5 = ( sum_total / num_periods ) / avg_period
            
    return {  'local' : relative, 'local, absolute' : absolute, 'rap' : rap, 'ppq5' : ppq5, 'ddp' : 3 * rap }























#TODO: get code put up github/travis?/coveralls...   
#TODO: clone repo, put on github-> goes in super ai->afx->features (in my fork, ask when ready to merge fork)
#TODO: need to get a key for me to put it on privately(only for coveralls)
#which I can put up once I have finished all the features
#TODO: update github pages so that I can post other things on there besides just titanic