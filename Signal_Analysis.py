import sig_tools as st
import numpy as np
#TODO: update github pages so that I can post other things on there besides just titanic
def get_F_0( signal, rate, min_pitch = 75, max_pitch = 550, max_num_candidates = 2, octave_cost = .01, 
            voicing_threshold = .4, silence_threshold = .01):
    """
    Compute Fundamental Frequency (F_0).
    Algorithm filters out values higher than the Nyquist Frequency, then segments the signal 
    into frames containing at least 3 periods of the minimum pitch. For each frame it then 
    calculates autocorrelation of the signal. After autocorrelation is calculated the maxima 
    values of the interpolation are found. Once these values have been chosen the best 
    candidate for the F_0 is picked and then returned.
    This algorithm is adapted from 
    http://www.fon.hum.uva.nl/david/ba_shs/2010/Boersma_Proceedings_1993.pdf
    
    Args:
        min_pitch (float): minimum value to be returned as pitch, cannot be less than or 
                  equal to zero
        max_pitch (float): maximum value to be returned as pitch, cannot be greater than 
                  Nyquist Frequency
        max_num_candidates (int): maximum number of candidates to be considered for each 
                           frame, unvoiced candidate (i.e. F_0 equal to zero) is always 
                           considered.
        octave_cost (float): value between 0 and 1 that aids in determining which candidate
                    for the frame is best. Higher octave_cost favors higher F_0.
        voicing_threshold (float): Threshold that peak of autocorrelation of signal must be 
                          greater than to be considered for maxima, and used to calculate 
                          strength of voiceless candidate. The higher the value the more 
                          likely the F_0 will be returned as voiceless.
        silence_threshold (float): Used to calculate strength of voiceless candidate, the 
                          higher the value the more likely the F_0 will be returned as 
                          voiceless.
    Returns:
        float: The F_0 of the signal +/- 2 hz.
        
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
    if max_num_candidates <2 :
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
    upper_bound = .95 * Nyquist_Frequency
    fft_signal = np.fft.fft( signal )
    fft_signal = fft_signal * ( fft_signal < upper_bound )
    sig = np.fft.ifft( fft_signal )
    global_peak = max( abs( sig ) )
    window_len = 3.0 / min_pitch
    segmented_signal = st.segment_signal( window_len, time_step, sig )
        
    #initializing list of candidates for F_0
    best_cands = []
    
    for index in range( len( segmented_signal ) ):
        time_begin = index * window_len
        time_end = min( ( index + 1 ) * window_len, total_time )
        window_len = time_end - time_begin
        segment = segmented_signal[ index ]
        local_peak = max( abs( segment ) )
        
        """
        For each segment we follow the given algorithm (steps 3.2-3.10), by
        1. Subtracting the mean of the segment
        2. Multiply the segment by the hanning window
        3. Calculate the autocorrelation of the windowed signal (r_a)
        4. Calculate the autocorrelation of the window (r_w)
        5. Divide the r_a by r_w to estimate the autocorrelation of the segment (r_x)
        """
        segment = segment - segment.mean()
        window = np.hanning( len( segment ) )
        segment *= window
        r_a = st.estimated_autocorrelation( segment )
        r_w = st.estimated_autocorrelation( window )
        r_x = r_a / r_w
        """
        Only consider the first half of the autocorrelation because for lags longer than a 
        half of the window length, it becomes less reliable there for signals with few 
        periods per window, stated in algorithm pg. 104.
        Eliminate points in the autocorrelation that are not finite (caused by dividing by 
        a number close to zero)
        """
        r_x = r_x[ np.isfinite( r_x ) ]
        r_len = int( len( r_x ) / 2)
        
        #creating an array of the points in time corresponding to our sampled autocorrelation
        #of the signal (r_x)
        time_array = np.linspace( 0, window_len / 2, r_len )
        r_x = r_x [ : r_len ]
        """
            Sidenote: In the algorithm it states to upsample the signal by a factor of 2 
        (step 3.11, referenceing eq. 22 where this is stated) to get a more accurate answer,
        however in practice most of the signals contain too much noise and once upsampled 
        the noise is exaggerated. By downsampling the peaks are cleaner and it becomes easier
        to pick the best peak that represents the frequency.
        """
        #we down sample the signal using sinc_interpolation, and eliminate any nan
        down_sampled = np.linspace( 0, window_len / 2, r_len / 2 )
        vals = np.nan_to_num( st.sinc_interp( r_x , time_array, down_sampled ) )
    
        #finding maximizers, and maximums and eliminating values that don't produce a 
        #pitch in the allotted range.
        maxima_values, maxima_places = st.find_max( vals, down_sampled, max_num_candidates )
        
        max_place_possible = min( 1. / min_pitch, window_len / 2 )
        min_place_possible = 1. / max_pitch
        
        maxima_values = maxima_values[ maxima_places <= max_place_possible ]
        maxima_places = maxima_places[ maxima_places <= max_place_possible ]

        maxima_values = maxima_values[ maxima_places >= min_place_possible ]
        maxima_places = maxima_places[ maxima_places >= min_place_possible ]
        
        #we only want to consider maximum greater than voicing_threshold, otherwise 
        #autocorrelation is not strong enough here to provide accurate data
        maxima_places = maxima_places[ maxima_values > voicing_threshold ]
        maxima_values = maxima_values[ maxima_values > voicing_threshold ]
        
        """
        Here we check our list to make sure its not empty, then calculate strengths 
        based off the formula given in the algorithm, (i.e. 
        R = r( tau_max ) - OctaveCost * log_2( MinimumPitch * tau_max ) (eq. 24)
        for voiced candidate and
        R = VoicingThreshold + max( 0, 2 - ( ( local absolute peak ) / ( global absolute peak ) ) 
        / ( SilenceThreshold / ( 1 + VoicingThreshold ) ) )
        (eq. 23) for unvoiced candidate)
        and append the strongest maxizer to our list of candidates.
        
            Sidenote:In the algorithm given it defines a way to calculate the best 
        candidate by iterating through a list of all possible candidates for each segment and 
        calculating the cost associated with it, (step 4) however this is too expensive to 
        calculate. Instead we find the best candidate per frame and choose the one of the
        candidates of highest value.
        """
        if len( maxima_values ) > 0:
            # equation number 24
            strengths = [ val / local_peak - octave_cost * np.log2( min_pitch * place ) 
                        for place, val in zip( maxima_places, maxima_values ) ]
        else:
            strengths = []
        #include unvoiced candidate
        maxima_places = np.hstack( ( maxima_places, 0 ) )
        
        # equation number 23
        strengths.append( voicing_threshold + max( 0, 2 - ( local_peak / global_peak ) / 
                                    ( silence_threshold / ( 1 + voicing_threshold ) ) ) )
        best_cands.append( maxima_places[ np.argmax( strengths ) ] )
        
    best_cands = np.array( best_cands )
    best_cands = best_cands[ best_cands > 0 ]
    if len( best_cands ) == 0:
        #if there are no candidates that fit criteria then assume the signal is unvoiced, 
        #i.e. return 0.
        return 0
    """
    Return the candidate that is in the 85th percentile, instead of the highest valued 
    candidates, which are usually anomalies caused by changes in amplitude in the signal 
    (autocorrelation PDAS are very sensitive to changes in amplitudes.)
    """
    best_candidate = sorted( best_cands )[ int( .85 * len( best_cands ) ) ]      
    return 1. / best_candidate
        
def get_HNR( signal, rate, min_pitch = 90, silence_threshold = .01, ):   
    """
    Compute Harmonic to Noise Ratio (HNR).
    Algorithm filters out values higher than the Nyquist Frequency, then segments the signal
    into frames containing at least 6 periods of the minimum pitch. For each frame it 
    calculates autocorrelation of the signal. After autocorrelation is calculated and
    upsampled, the maxima values of the interpolation are found. Once these values have been 
    chosen the largest HNR is picked and then returned.
    This algorithm is adapted from 
    http://www.fon.hum.uva.nl/david/ba_shs/2010/Boersma_Proceedings_1993.pdf
    
    Args:
        min_pitch (float): minimum value to be returned as pitch, cannot be less than or 
                  equal to zero
        
    Returns:
        float: The HNR of the signal, 0 if HNR is the signal is majority noise, or perfectly 
        periodic.
        
    Raises:
        ValueError: The minimum pitch cannot be equal or less than zero.
        
    Example:
    ::
        from scipy.io import wavfile as wav
        import Signal_Analysis as sig
        rate, wave = wav.read( 'example_audio_file.wav' )
        sig.get_HNR( wave, rate )
    """
    
    time_step = 1. / rate
    total_time = time_step * len( signal )
    Nyquist_Frequency = 1. / ( time_step * 2 )
    if min_pitch <= 0:
        raise ValueError( "The minimum pitch cannot be equal to or less than zero." )
        
    #filtering by Nyquist Frequency and segmenting signal
    upper_bound = .95 * Nyquist_Frequency
    fft_signal = np.fft.fft( signal )
    fft_signal = fft_signal * ( fft_signal < upper_bound )
    sig = np.fft.ifft( fft_signal )
    global_peak = max( abs( sig ) ) 
    window_len = 6.0 / min_pitch
    segmented_signal = st.segment_signal( window_len, time_step, sig )
        
    #initializing list of candidates for HNR
    best_cands = []
    
    for index in range( len( segmented_signal ) ):
        
        segment = segmented_signal[ index ]
        local_peak = max( abs( segment ) )
        std_ratio = np.std( segment ) / np.std( sig ) 
        """
        if the standard deviation of the segment is 130% greater than the standard deviation
        of the signal or the local peak is less than 5% of the global peak, there is a large
        enough change in amplitude to give counterfeit peak heights (because autocorrelation
        is very sensitive to change in amplitude.), in these cases, skip the segment in 
        question.
        """
        if std_ratio < 1.3 and local_peak / global_peak > .05 :
            time_begin = index * window_len
            time_end = min( ( index + 1 ) * window_len, total_time )
            window_len = time_end - time_begin
            
            """
            For each segment we follow the given algorithm (steps 3.2-3.10), by
            1. Subtracting the mean of the segment
            2. Multiply the segment by the hamming window (we use hamming to prevent errors
               caused by division by numbers close to zero)
            3. Calculate the autocorrelation of the windowed signal (r_a)
            4. Calculate the autocorrelation of the window (r_w)
            5. Divide the r_a signal by r_w to estimate the autocorrelation of the segment 
                (r_x).
            """
            segment = segment - segment.mean()
            window = np.hamming( len( segment ) )
            segment *= window
            #normalize autocorrelation
            r_a = st.estimated_autocorrelation( segment )
            r_a = r_a / r_a[ 0 ]
            r_w = st.estimated_autocorrelation( window )
            r_w = r_w / r_w[ 0 ]
            r_x = r_a / r_w
            """    
            Only consider the first half of the autocorrelation because for lags longer than a 
            half of the window length, it becomes less reliable there for signals with few 
            periods per window, stated in algorithm pg. 104.
            Eliminate points in the autocorrelation that are not finite (caused by dividing 
            by a number close to zero)
            """
            r_x = r_x[ np.isfinite( r_x ) ]
            r_len = int( len( r_x ) / 2 ) 
            
            #creating an array of the points in time corresponding to our sampled autocorrelation
            #of the signal (r_x)
            time_array = np.linspace( 0, window_len / 2, r_len )
            r_x = r_x [ : r_len ]

            #we upsample the signal using sinc_interpolation, and eliminate any nan
            up_sampled = np.linspace( 0, window_len / 2, r_len * 2 )
            vals = np.nan_to_num( st.sinc_interp( r_x , time_array, up_sampled) )

            if len( vals.nonzero()[ 0 ] ) != 0:
                all_maxima, all_places = st.find_max( vals, up_sampled, 15 )
                #eliminate the initial/highest peak
                all_maxima = all_maxima[ 1: ]
                all_places = all_places[ 1: ]
                #eliminate any values less than .5 (will give negative HNR)
                all_places = all_places[ all_maxima > .5 ]    
                all_maxima = all_maxima[ all_maxima > .5 ]
                #eliminate any values equal to or greater than one (will produce nan)
                all_places = all_places[ all_maxima < 1 ]    
                all_maxima = all_maxima[ all_maxima < 1 ]
                if len( all_maxima ) > 0 :
                    best_cands.append( max( all_maxima ) ) 
    #TODO:test case for if best_cands==0 so that we have 100% test coverage.
    if len( best_cands ) == 0:
        return 0 
    best_cands = max( best_cands )
    #(eq. 4)
    return 10 * np.log10( best_cands / ( 1 - best_cands ) )

def get_Jitter( signal, rate, min_pitch = 90, method = 'absolute' ):
    time_step = 1. / rate
    total_time = time_step * len( signal )
    Nyquist_Frequency = 1. / ( time_step * 2 )
        
    #filtering by Nyquist Frequency and segmenting signal 
    upper_bound = .95 * Nyquist_Frequency
    fft_signal = np.fft.fft( signal )
    fft_signal = fft_signal * ( fft_signal < upper_bound )
    sig = np.fft.ifft( fft_signal )
    window_len = 3.0 / min_pitch
    segmented_signal = st.segment_signal( window_len, time_step, sig )
        
    #initializing list of candidates for F_0
    best_cands = []
    
    for index in range( len( segmented_signal ) ):
        time_begin = index * window_len
        time_end = min( ( index + 1 ) * window_len, total_time )
        window_len = time_end - time_begin
        segment = segmented_signal[ index ]
        time_arr = np.linspace( 0, window_len, len( segment ) )
        down_sampled = np.linspace( 0, window_len, len( segment ) / 10 )
        segment = np.nan_to_num( st.sinc_interp( segment , time_arr, down_sampled ) )
        time_arr = down_sampled
        maxima_v, maxima_p = st.find_max( segment, time_arr, np.inf )
        
        if segment[ 0 ] < 0:
            maxima_v = maxima_v[ 1 : ]
            maxima_p = maxima_p[ 1 : ]
            
        if method == 'absolute':
            best_cands.append( np.mean( abs( np.diff( maxima_p, 2 ) ) ) )
        elif method == 'relative':
            best_cands.append( np.mean( abs( np.diff( maxima_p, 2 ) ) ) / np.mean( np.diff( maxima_p ) ) * 100 )
        elif method == 'rap':
            best_cands.append( 0 )
        elif method == 'ppq5':
            best_cands.append( 0 )
    return min( best_cands )
        #maybe calculate all forms of jitter and have it return certain forms based off of parameters given... that would be interesting.
        #TODO:look up fixtures in python testing, maybe ask david if not sure about it...
        #2 or less peaks cause problems
        #need to look more into...
        #pick least jitter, do we want values???
        #TODO: look at a way to implement a peak picking system that focuses on largest peaks, look at pu...
        #TODO: need to document test file better 
        #TODO: get code put up github/travis/coveralls...   
        #TODO: clone repo, put on github-> goes in super ai->afx->features (in my fork, ask when ready to merge fork)
        #TODO: travis and coveralls, he will get a key for me to put it on privately, which I don't have yet so maybe don't work on this.
