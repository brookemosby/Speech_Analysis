import numpy as np
import peakutils as pu
def get_F_0( signal, rate, time_step = 0.0, min_pitch = 75, max_pitch = 600, max_num_cands = 15,
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
        
        time_step (float): (default value: 0.0) the measurement interval (frame duration), in seconds. 
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
    if min_pitch <= 0:
        raise ValueError( "The minimum pitch cannot be equal or less than zero." )
    if max_num_cands < max_pitch/min_pitch: max_num_cands = int(max_pitch/min_pitch)
    
    total_time = len( signal ) / float( rate )
    tot_time_arr = np.linspace( 0, total_time, len( signal ) )
    
    #segmenting signal into windows that contain 3 periods of minimum pitch
    if accurate: periods_per_window = 6.0
    else:        periods_per_window = 3.0
    
    #degree of oversampling is 4    
    if time_step <= 0: time_step = ( periods_per_window / 4.0 ) / min_pitch
    window_len = periods_per_window / min_pitch
    #correcting for time_step       
    octave_jump_cost     *= .01 / time_step
    voiced_unvoiced_cost *= .01 / time_step 
    Nyquist_Frequency = rate /  2.0
    upper_bound = .95 * Nyquist_Frequency
    initial_len = len( signal )
    zeros_pad = 2 ** ( int( np.log2( len( signal ) ) ) + 1 ) - len( signal )
    signal = np.hstack( ( signal, np.zeros( zeros_pad ) ) )
    fft_signal = np.fft.fft( signal )
    fft_signal[ int( upper_bound ) : -int(upper_bound)] = 0
    sig = np.fft.ifft( fft_signal )
    sig = sig[ :initial_len ].real
    #checking to make sure values are valid
    if min_pitch < periods_per_window / total_time:
        raise ValueError( "To analyse this sound min_pitch must not be less than ", str(periods_per_window / total_time ) )
    if Nyquist_Frequency < max_pitch:
        raise ValueError( "The maximum pitch cannot be greater than the Nyquist Frequency." )
    if octave_cost < 0 or octave_cost > 1:
        raise ValueError( "octave_cost must be between 0 and 1." )            
    if voicing_threshold < 0 or voicing_threshold> 1:
        raise ValueError( "voicing_threshold must be between 0 and 1." ) 

    if silence_threshold < 0 or silence_threshold > 1:
        raise ValueError( "silence_threshold must be between 0 and 1." )
        
    #Segmenting signal
    frame_len = int( window_len * rate + .5 )
    time_len  = int( time_step  * rate + .5 )
        
    #initializing list of candidates for F_0, and their strengths
    best_cands, strengths = [], []
    if pulse: time_vals = []
    global_peak = max( abs( sig - sig.mean() ) )
    
    start_index = 0
    while start_index < len( sig ) - frame_len :
        segment = sig[ start_index : start_index + frame_len ]
        if accurate:
            t = np.linspace( 0, window_len, len( segment ) )
            window = ( np.e ** ( -12 * ( t / window_len - .5 ) ** 2 ) - np.e ** -12 ) / ( 1 - np.e ** -12 )
            interpolation_depth = 0.25
        else: 
            window = np.hanning( len( segment ) )    
            interpolation_depth = 0.50
        
        if pulse:
            start_time = tot_time_arr[ start_index +             int( 0.25 * time_len ) ]
            stop_time  = tot_time_arr[ start_index + frame_len - int( 0.25 * time_len ) ]
            time_vals.append( ( start_time, stop_time ) )
            
        longest_period_index = int( rate / min_pitch )
        half_period_index = int( longest_period_index / 2.0 + 1 )
        
        period_cushion      = segment[    half_period_index  : - half_period_index   ]  

        local_mean = period_cushion.mean() 
        segment = segment - local_mean
        segment *= window
        half_period_cushion = segment[ longest_period_index  : -longest_period_index ]
        local_peak = max( abs( half_period_cushion ) )
        intensity = local_peak / global_peak
        
        if local_peak == 0:
            #shortcut -> complete silence only candidate is silent candidate
            best_cands.append( [ np.inf ] )
            strengths.append( [ voicing_threshold + 2 ] )
            start_index += time_len
        else:
            #calculating autocorrelation, based off steps 3.2-3.10
         
            N = len( segment )
            nsampFFT = 2 ** int( np.log2( ( 1.0 + interpolation_depth ) * N + 1 ) )
            window  = np.hstack( (   window, np.zeros( nsampFFT - N ) ) ) 
            segment = np.hstack( (  segment, np.zeros( nsampFFT - N ) ) )
            x_fft = np.fft.fft( segment )
            r_a = np.real( np.fft.fft( x_fft * np.conjugate( x_fft ) ) )
            r_a = r_a[ : N ]
                       
            x_fft = np.fft.fft( window )
            r_w = np.real( np.fft.fft( x_fft * np.conjugate( x_fft ) ) )
            r_w = r_w[ : N ]
            r_x = r_a / r_w
            r_x /= r_x[ 0 ]

            #eliminating frequencies below min_pitch
            r_x = r_x[ : int( len( r_x ) / periods_per_window ) ]
            #creating an array of the points in time corresponding to our sampled autocorrelation
            #of the signal (r_x)
            time_array = np.linspace( 0 , window_len /periods_per_window, len( r_x ) )
            peaks = pu.indexes( r_x , thres = 0)
            
            maxima_values, maxima_places = r_x[ peaks ], time_array[ peaks ]
            
            max_place_possible  = 1.0 / min_pitch
            min_place_possible  = 1.0 / max_pitch
            #to silence formants
            min_place_possible2 = 0.5 / max_pitch
            
            maxima_places = maxima_places[ maxima_values > 0.5 * voicing_threshold ]
            maxima_values = maxima_values[ maxima_values > 0.5 * voicing_threshold ]  
            
            for i in range( len( maxima_values ) ):
                #reflecting values > 1 through 1.
                if maxima_values[ i ] > 1.0 : maxima_values[ i ] = 1.0 / maxima_values[ i ]
              
            local_strength = [ max_val - octave_cost * np.log2(  max_place * min_pitch ) for 
                        max_val, max_place in zip( maxima_values, maxima_places ) ]
            
            if len( maxima_values ) > 0.0 :
                #finding the max_num_cands-1 maximizers, and maximums, then calculating their
                #strengths (eq. 23 & 24) and accounting for silent candidate
                maxima_places = np.array( [ maxima_places[ i ] for i in np.argsort( local_strength )[
                        -max_num_cands + 1 : ] ] )
                maxima_values = np.array( [ maxima_values[ i ] for i in np.argsort( local_strength )[
                        -max_num_cands + 1 : ] ] )
                
                local_strength = list(np.sort( local_strength )[ -max_num_cands + 1 : ] )
                local_strength.append( voicing_threshold + max( 0, 2 - ( intensity / 
                        ( silence_threshold / ( 1 + voicing_threshold ) ) ) ) )
                
                #np.inf is our silent candidate
                maxima_places = np.hstack( ( maxima_places, np.inf ) )
                best_cands.append( list( maxima_places ) )
                strengths.append( local_strength )
            else:
                #if there are no available maximums, only account for silent candidate
                best_cands.append( [ np.inf ] )
                strengths.append( [ voicing_threshold + max( 0, 2 - ( intensity /
                        ( silence_threshold / ( 1 + voicing_threshold ) ) ) ) ] )
            start_index += time_len
            
    #Calculates smallest costing path through list of candidates (forwards), and returns path.
    best_total_cost, best_total_path = -np.inf, []
    #for each initial candidate find the path of least cost, then of those paths, choose the one 
    #with the least cost.
    for a in range( len( best_cands[ 0 ] ) ):
        start_val = best_cands[ 0 ][ a ]
        total_path = [ start_val ]
        level = 1
        prev_delta = strengths[ 0 ][ a ]
        maximum = -np.inf
        while level < len( best_cands ) :
            prev_val = total_path[ -1 ]
            best_val  = np.inf
            for j in range( len( best_cands[ level ] ) ):
                cur_val   = best_cands[ level ][ j ] 
                cur_delta =  strengths[ level ][ j ]
                cost = 0
                cur_unvoiced  = ( cur_val  == np.inf or cur_val  < min_place_possible2 )
                prev_unvoiced = ( prev_val == np.inf or prev_val < min_place_possible2 )
                
                if cur_unvoiced:
                    #both voiceless
                    if prev_unvoiced: cost = 0 
                    #voiced-to-unvoiced transition
                    else:             cost = voiced_unvoiced_cost 
                else:
                    #unvoiced-to-voiced transition
                    if prev_unvoiced: cost = voiced_unvoiced_cost
                    #both are voiced
                    else:             cost = octave_jump_cost * abs( np.log2( cur_val / prev_val ) )
                            
                #The cost for any given candidate is given by the transition cost, minus the strength
                #of the given candidate
                
                value = prev_delta - cost + cur_delta
                if value > maximum: maximum, best_val = value, cur_val
                    
            prev_delta = maximum        
            total_path.append( best_val )
            level += 1
            
        if maximum > best_total_cost: best_total_cost, best_total_path = maximum, total_path
            
    f_0_forward = np.array( best_total_path )        
    
    #Calculates smallest costing path through list of candidates (backwards), and returns path.
    
    best_total_cost, best_total_path2 = -np.inf, []
    #for each initial candidate find the path of least cost, then of those paths, choose the one 
    #with the least cost.
    for a in range( len( best_cands[ -1 ] ) ):
        start_val = best_cands[ -1 ][ a ]
        total_path = [ start_val ]
        level = len(best_cands)-2
        prev_delta = strengths[ -1][ a ]
        maximum = -np.inf
        while level >-1 :
            prev_val = total_path[ -1 ]
            best_val  = np.inf
            for j in range( len( best_cands[ level ] ) ):
                cur_val   = best_cands[ level ][ j ] 
                cur_delta =  strengths[ level ][ j ]
                cost = 0
                cur_unvoiced  = ( cur_val  == np.inf or cur_val  < min_place_possible2 )
                prev_unvoiced = ( prev_val == np.inf or prev_val < min_place_possible2 )
                
                if cur_unvoiced:
                    #both voiceless
                    if prev_unvoiced: cost = 0 
                    #voiced-to-unvoiced transition
                    else:             cost = voiced_unvoiced_cost 
                else:
                    #unvoiced-to-voiced transition
                    if prev_unvoiced: cost = voiced_unvoiced_cost
                    #both are voiced
                    else:             cost = octave_jump_cost * abs( np.log2( cur_val / prev_val ) )
                    
                #The cost for any given candidate is given by the transition cost, minus the strength
                #of the given candidate
                
                value = prev_delta - cost + cur_delta
                if value > maximum: maximum, best_val = value, cur_val
                    
            prev_delta = maximum        
            total_path.append( best_val )
            level -= 1
            
        if maximum > best_total_cost: best_total_cost, best_total_path2 = maximum, total_path
            
    f_0_backward = np.array( best_total_path2 ) 
    #reversing f_0_backward so the initial value corresponds to first frequency
    f_0_backward = f_0_backward[ -1 : : -1 ] 

    f_0 = np.array( [ min( i, j ) for i, j in zip( f_0_forward, f_0_backward ) ] )
    
    if pulse:
        removed = 0
        for i in range( len( f_0 ) ):
            if f_0[ i ] > max_place_possible or f_0[ i] < min_place_possible:
                time_vals.remove( time_vals[ i - removed ] )
                removed += 1
      
    for i in range( len( f_0 ) ):
        #if f_0 is voiceless assign occurance of peak to inf
        if f_0[ i ] > max_place_possible or f_0[ i ] < min_place_possible : f_0[ i ] = np.inf
              
    f_0 = f_0[ f_0 < np.inf ]
    if pulse:               return f_0, time_vals, signal, global_peak
    if len( f_0 ) == 0:     return 0
    else:                   return np.median( 1.0 / f_0 )   


def get_HNR( signal, rate, time_step = 0, min_pitch = 75, silence_threshold = .1, 
             periods_per_window = 4.5):
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

        time_step (float): (default value: 0.0) the measurement interval (frame duration), in seconds. 
        If you supply 0, Praat will use a time step of 0.75 / (min_pitch), e.g. 0.01 seconds if the 
        minimum pitch is 75 Hz; in this example, algorithm computes 100 pitch values per second.

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
    if time_step <= 0: time_step = ( periods_per_window / 4.0 ) / min_pitch        
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
        
        maxima_values = np.hstack( ( maxima_values, 1.0 / maxima_values ) )
        maxima_values = maxima_values[ maxima_values < 1.0 ]
        
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
    

def get_Pulses( signal, rate, min_pitch = 75, max_pitch = 600, include_maxima = False, 
                include_minima = True ):
    """
    This algorithm examines voiced intervals of a signal, and creates a list of points that correspond
    to the sequence of glottal closures in vocal-fold vibration.
    adapted from: https://pdfs.semanticscholar.org/16d5/980ba1cf168d5782379692517250e80f0082.pdf
    This algorithm produces a relative error of 4% when compared to Praat
    
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
        
    period, intervals, signal, global_peak = get_F_0( signal, rate, min_pitch = min_pitch, 
                                        max_pitch = max_pitch, pulse = True)
    
    points, voiced_intervals =   [], []
    f_times, v_times = np.array( [] ), np.array( [] )
    total_time = np.linspace( 0, len( signal ) / float( rate ), len( signal ) )
    
    for interval in intervals:
        start, stop = interval
        #finding all midpoints for each interval
        f_times = np.hstack( ( f_times, ( start + stop ) / 2.0 ) )
    i = 0  
    while i < len( intervals ) - 1 :
        start, stop = intervals[ i ]
        int_start, prev_stop = intervals[ i ]
        while start <= prev_stop and i < len( intervals ) - 1 :
            prev_start, prev_stop = intervals[ i ]
            i += 1
            start, stop = intervals[ i ]
        if i == len( intervals ) - 1:
            v_times = np.hstack( ( v_times, 
                                  np.linspace( int_start, stop,      int( ( stop      - int_start ) * rate ) ) ) )
            voiced_intervals.append( ( int_start, stop ) )
        else:
            v_times = np.hstack( ( v_times, 
                                  np.linspace( int_start, prev_stop, int( ( prev_stop - int_start ) * rate ) ) ) )	
            voiced_intervals.append( ( int_start, prev_stop ) )
    
    periods_interp = np.interp( v_times, f_times, period )
    
    for interval in voiced_intervals:
        start, stop = interval
        midpoint = ( start + stop ) / 2.0
        midpoint_index = np.argmin( abs( v_times - midpoint ) )
        midpoint = v_times[ midpoint_index ]
        T_0 = periods_interp[ midpoint_index ]
        frame_start = midpoint - T_0
        frame_stop  = midpoint + T_0
        while frame_start > start :
            frame_start_index = np.argmin( abs( total_time - frame_start ) )
            frame_stop_index  = np.argmin( abs( total_time - frame_stop  ) )
            frame = signal[ frame_start_index : frame_stop_index ]
            
            if include_maxima and include_minima: p_index = np.argmax( abs( frame ) ) + frame_start_index
            elif include_maxima:                  p_index = np.argmax( frame )        + frame_start_index
            else:                                 p_index = np.argmin( frame )        + frame_start_index 
                                                                     
            if abs( signal[ p_index ] ) > .02333 * global_peak: points.append( total_time[ p_index ] )
                
            t = total_time[ p_index ]
            t_index = np.argmin( abs( v_times - t ) )
            T_0 = periods_interp[ t_index ]
            frame_start = t - 1.25 * T_0
            frame_stop  = t - 0.80 * T_0
            
        T_0 = periods_interp[ midpoint_index ]    
        frame_start = midpoint - T_0
        frame_stop  = midpoint + T_0  
        
        while frame_stop < stop :
            frame_start_index = np.argmin( abs( total_time - frame_start ) )
            frame_stop_index  = np.argmin( abs( total_time - frame_stop  ) )
            frame = signal[ frame_start_index : frame_stop_index ]
            
            if include_maxima and include_minima: p_index = np.argmax( abs( frame ) ) + frame_start_index
            elif include_maxima:                  p_index = np.argmax( frame )        + frame_start_index
            else:                                 p_index = np.argmin( frame )        + frame_start_index 
                                                                     
            if abs( signal[ p_index ] ) > .02333 * global_peak: points.append( total_time[ p_index ] )  
            
            t = total_time[ p_index ]
            t_index = np.argmin( abs( v_times - t ) )
            T_0 = periods_interp[ t_index ]
            frame_start = t + 0.80 * T_0
            frame_stop  = t + 1.25 * T_0 
            
    return np.array( sorted( list( set( points ) ) ) )


def get_Jitter( signal, rate, period_floor = .0001, period_ceiling = .02, max_period_factor = 1.3, 
                pulses = None ):
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
    if pulses is None: pulses = get_Pulses( signal, rate )
    periods = np.diff( pulses )
    
    min_period_factor = 1.0 / max_period_factor
    
    #finding local, absolute
    #described at: http://www.fon.hum.uva.nl/praat/manual/PointProcess__Get_jitter__local__absolute____.html
    sum_total = 0
    num_periods = len( pulses ) - 1
    for i in range( len( periods ) - 1 ):
        p1, p2 = periods[ i ], periods[ i + 1 ]
        
        ratio = p2 / p1
        if (ratio < max_period_factor and ratio > min_period_factor and 
            p1    < period_ceiling    and p1    > period_floor      and
            p2    < period_ceiling    and p2    > period_floor      ):
            
                sum_total += abs( periods[ i + 1 ] - periods[ i ] ) 
        else: num_periods -= 1
                
    absolute = sum_total / ( num_periods - 1 )
    
    #finding local, 
    #described at: http://www.fon.hum.uva.nl/praat/manual/PointProcess__Get_jitter__local____.html
    sum_total = 0
    num_periods = 0
    
    #duplicating edges so there is no need to test edge cases
    periods = np.hstack(( periods[ 0 ], periods, periods[ -1 ] ))
    
    for i in range( len( periods ) - 2):
        p1, p2, p3 = periods[ i ], periods[ i + 1 ], periods[ i + 2 ]
        
        ratio_1, ratio_2 = p1 / p2, p2 / p3
        if (ratio_1 < max_period_factor and ratio_1 > min_period_factor and 
            ratio_2 < max_period_factor and ratio_2 > min_period_factor and 
            p2      < period_ceiling    and p2      > period_floor      ):
            
            sum_total += p2
            num_periods += 1
            
    periods = periods[ 1 : -1 ]
    avg_period = sum_total / ( num_periods ) 
    relative = absolute / avg_period
    
    #finding rap
    #described at: http://www.fon.hum.uva.nl/praat/manual/PointProcess__Get_jitter__rap____.html
    sum_total = 0
    num_periods = 0
    
    for i in range( len( periods ) - 2 ):
        p1, p2, p3 = periods[ i ], periods[ i + 1 ], periods[ i + 2 ]
        
        ratio_1, ratio_2 = p1 / p2, p2 / p3
        if (ratio_1 < max_period_factor and ratio_1 > min_period_factor and 
            ratio_2 < max_period_factor and ratio_2 > min_period_factor and 
            p1      < period_ceiling    and p1      > period_floor      and
            p2      < period_ceiling    and p2      > period_floor      and
            p3      < period_ceiling    and p3      > period_floor      ):
            
            sum_total += abs( p2 - ( p1 + p2 + p3 ) / 3.0 )
            num_periods += 1
    rap = ( sum_total / num_periods ) / avg_period 
          
    #finding ppq5
    #described at: http://www.fon.hum.uva.nl/praat/manual/PointProcess__Get_jitter__ppq5____.html
    sum_total = 0
    num_periods = 0
    
    for i in range( len( periods ) - 4 ):
        p1, p2, p3, p4, p5 = periods[ i ], periods[ i + 1 ], periods[ i + 2 ], periods[ i + 3 ], periods[ i + 4 ]
        
        ratio_1, ratio_2, ratio_3, ratio_4 = p1 / p2, p2 / p3, p3 / p4, p4 / p5
        if (ratio_1 < max_period_factor and ratio_1 > min_period_factor and 
            ratio_2 < max_period_factor and ratio_2 > min_period_factor and 
            ratio_3 < max_period_factor and ratio_3 > min_period_factor and 
            ratio_4 < max_period_factor and ratio_4 > min_period_factor and 
            p1      < period_ceiling    and p1      > period_floor      and
            p2      < period_ceiling    and p2      > period_floor      and
            p3      < period_ceiling    and p3      > period_floor      and
            p4      < period_ceiling    and p4      > period_floor      and
            p5      < period_ceiling    and p5      > period_floor      ):
            
            sum_total += abs( p3 - ( p1 + p2 + p3 +p4 + p5 ) / 5.0 )
            num_periods += 1
            
    ppq5 = ( sum_total / num_periods ) / avg_period
            
    return {  'local' : relative, 'local, absolute' : absolute, 'rap' : rap, 'ppq5' : ppq5, 'ddp' : 3 * rap }

