import sig_tools as st
import numpy as np
from matplotlib import pyplot as plt
import peakutils as pu
from scipy import signal as ss
#TODO: update github pages so that I can post other things on there besides just titanic
#TODO: look at HNR, attempt cepstrum one more time, also look at auto without windowing some
#more to see if there is more I can do to edit/refine that method... doesn't seem to get
#above 13 ish dB
#TODO:look at using gaussian window for HNR
class Signal():
    def __init__( self, signal, rate ):
        self.signal = signal
        self.rate = rate
        #min_pitch = 75...
    def get_F_0( self, min_pitch = 75, max_pitch = 550, max_num_candidates = 2, octave_cost = .01, 
                voicing_threshold = .4, silence_threshold = .01, HNR = False):
        """
        Compute Fundamental Frequency (F_0).
        Algorithm uses Fast Fourier Transform (FFT) to filter out values higher than the Nyquist
        Frequency. Then it segments the signal into frames containing at least 3 periods of the 
        minimum pitch. For each frame it then again uses FFT to calculate autocorrelation of the 
        signal. After autocorrelation is calculated it is down sampled with sinc interpolation, 
        and then the maxima values of the interpolation are found. After these values have been 
        chosen the best candidate for the F_0 is picked and then returned.
        This algorithm is adapted from 
        http://www.fon.hum.uva.nl/david/ba_shs/2010/Boersma_Proceedings_1993.pdf
        
        Args:
            min_pitch (float): minimum value to be returned as pitch, cannot be less than or 
                      equal to zero
            max_pitch (float): maximum value to be returned as pitch, cannot be greater than 
                      Nyquist Frequency
            max_num_candidates (int): maximum number of candidates to be considered for each 
                               frame, unvoiced candidate (i.e. fundamental frequency equal to 
                               zero) is always considered.
            octave_cost (float): value between 0 and 1 that aids in determining which candidate
                        for the frame is best. Higher octave_cost favors higher frequencies.
            voicing_threshold (float): Threshold that peak of autocorrelation of signal must be 
                              greater than to be considered for maxima, and used to calculate 
                              strength of voiceless candidate. The higher the value the more 
                              likely the F_0 will be returned as voiceless.
            silence_threshold (float): Used to calculate strength of voiceless candidate, the 
                              higher the value the more likely the F_0 will be returned as 
                              voiceless.
            HNR (bool): Determines if Harmonics-to-Noise ratio is returned
            
        Returns:
            float: The F_0 of the signal +/- 2 hz.
            
        Raises:
            TypeError: HNR must be a bool
            TypeError: min_pitch, max_pitch, and max_num_candidates must be an int, octave_cost,
                        silence_threshold, and voicing_threshold must be a float.
            ValueError: The maximum pitch cannot be greater than the Nyquist Frequency.
            ValueError: The minimum pitch cannot be equal or less than zero.
            ValueError: The minimum number of candidates is 2.
            ValueError: octave_cost must be between 0 and 1.
            ValueError: silence_threshold must be between 0 and 1.
            ValueError: voicing_threshold must be between 0 and 1.
        
        Example:
        ::
            
            from scipy.io import wavfile as wav
            rate, wave= wav.read( 'example_audio_file.wav' )
            sig = Signal( wave, rate )
            sig.get_F_0()
            
        """
        
        time_step=1./self.rate
        total_time = time_step * len( self.signal )
        Nyquist_Frequency = 1. / ( time_step * 2 )
        
        #checking to make sure values are valid
        #check for type errors of values passed in 
        if type(HNR) != bool:
            raise TypeError( "HNR must be a bool." )
        if not HNR:    
            if ( type( min_pitch ) != int or 
                 type( max_pitch ) != int or 
                 type( max_num_candidates ) != int or 
                 type( octave_cost ) != float or 
                 type( silence_threshold ) != float or
                 type( voicing_threshold ) != float ):
                
                raise TypeError( "min_pitch, max_pitch, and max_num_candidates must be an int, octave_cost, silence_threshold, and voicing_threshold must be a float" )   
                   
            if Nyquist_Frequency < max_pitch:
                raise ValueError( "The maximum pitch cannot be greater than the Nyquist Frequency." )
            if max_num_candidates <2 :
                raise ValueError( "The minimum number of candidates is 2.")
            if octave_cost < 0 or octave_cost > 1:
                raise ValueError( "octave_cost must be between 0 and 1." )            
            if voicing_threshold < 0 or voicing_threshold> 1:
                raise ValueError( "voicing_threshold must be between 0 and 1." ) 
        else:
            if type(min_pitch) != int or type(silence_threshold)!=float:
                raise ValueError( "min_pitch must be an int, and silence_threshold must be a float." )

        #these values need to be checked for F_0 and HNR calculations   
        if min_pitch <= 0:
            raise ValueError( "The minimum pitch cannot be equal or less than zero." )
        if silence_threshold < 0 or silence_threshold > 1:
            raise ValueError( "silence_threshold must be between 0 and 1." )
            
        #filtering by Nyquist Frequency (preproccesing step)
        upper_bound = .95 * Nyquist_Frequency
        fft_signal = np.fft.fft( self.signal )
        fft_signal = fft_signal * ( fft_signal < upper_bound )
        sig = np.fft.ifft( fft_signal )
        
        global_peak = max( abs( sig ) )
        
        #finding the window_len in seconds, finding frame len (as an integer of how many points 
        #will be in a window), finding number of frames/ windows that we will need to segment 
        #the signal into then segmenting signal
        if HNR:
            window_len = 6.0 / min_pitch
            voicing_threshold = 0
            octave_cost = 0
        else:
            window_len = 3.0 / min_pitch
            
        frame_len = window_len / time_step
        num_frames = max( 1, int( len( sig ) / frame_len + .5 ) ) 
        #there has to be at least one frame
        segmented_signal = [ sig[ int( i * frame_len ) : int( ( i + 1 ) * frame_len ) ] 
                                                     for i in range( num_frames + 1 ) ]
        
        #This eliminates an empty list that could be created at the end
        if len( segmented_signal[ len( segmented_signal ) - 1 ] ) == 0:
            segmented_signal = segmented_signal[ : -1 ]
            
        #initializing list of candidates for F_0
        best_cands = []
        corrs_cand_vals = []
        bestest=0
        for index in range( len( segmented_signal ) ):
            time_begin, time_end = index * window_len, min( ( index + 1 ) * window_len, total_time )
            window_len = time_end - time_begin
            segment = segmented_signal[ index ]
            local_peak = max( abs( segment ) )
            """
            For each segment we follow the given algorithm (steps 3.2-3.10), by
            1. Subtracting the mean of the segment
            2. Multiply the segment by the hanning window
            3. Calculate the autocorrelation of the windowed signal (r_a)
            4. Calculate the autocorrelation of the window (r_w)
            5. Divide the autocorrelation of the windowed signal by autocorrelation of the window
                    to estimate the autocorrelation of the segment (r_x)
            """
            segment = segment - segment.mean()
            segment *= np.hanning( len( segment ) )
            r_a = st.estimated_autocorrelation( segment )
            r_w = st.estimated_autocorrelation( np.hanning( len( segment ) ) )
            r_x = r_a / r_w
            
            #eliminating points in the autocorrelation that are not finite (cause by dividing by 
            #a number close to zero)
            r_x = r_x[ np.isfinite( r_x ) ]
            r_len = len( r_x )
            
            #creating an array of the points in time corresponding to our sampled autocorrelation
            #of the signal (r_x)
            time_array = np.linspace( 0, window_len, r_len )

            #Only consider the first half of the autocorrelation because for lags longer than a 
            #half of the window length, it becomes less reliable there for signals with few 
            #periods per window, stated in algorithm pg. 104.
            first_half = np.ones( int( r_len / 2 ) )
            second_half = np.zeros( r_len - int( r_len / 2 ) )
            limited_window = np.hstack( ( first_half, second_half  ) )
            r_x = r_x * limited_window

            """
                Sidenote: In the algorithm it states to upsample the signal by a factor of 2 
            (step 3.11, referenceing eq. 22 where this is stated) to get a more accurate answer,
            however in practice most of the signals contain too much noise and once upsampled, 
            the noise is exaggerated. By downsampling the peaks are cleaner and it becomes easier
            to pick the best peak that represents the frequency.
            """
            #we down sample the signal using sinc_interpolation, and eliminate any nan
            down_sampled_time_array = np.linspace( 0, window_len, r_len /2 )
            vals = np.nan_to_num( st.sinc_interp( r_x , time_array, down_sampled_time_array ) )
            time_array = down_sampled_time_array
            
            if len( vals.nonzero()[ 0 ] ) != 0:
                #finding maximizers, and maximums and eliminating values that don't produce a 
                #pitch in the allotted range.
                maxima_values, maxima_places = st.find_max( vals, time_array, max_num_candidates )
                
                max_place_possible = min( 1. / min_pitch, window_len / 2 )
                min_place_possible = 1. / max_pitch
                
                
                corrs_maxima_vals = maxima_values[ maxima_places <= max_place_possible ]
                top_vals_elim = maxima_places[ maxima_places <= max_place_possible ]

                maxima_values = corrs_maxima_vals[ top_vals_elim >= min_place_possible ]
                maxima_places = top_vals_elim[ top_vals_elim >= min_place_possible ]
                
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
                candidate by iterating through a list of all possible candidates for each segment
                and calculating the cost associated with it, (step 4) however assuming it takes 
                a milisecond per iteration, there are 4 candidates per segment and 20 segments, 
                this would take approximately 34 years to calculate. Instead we find the best 
                candidate per frame and choose the one of the candidates of highest value.
                """
                strengths=[]
                max_val=vals[0]
                vals=vals/max_val                
                all_maxima,all_places=st.find_max(vals,time_array,2)

                corrs_maxima_vals = all_maxima[ all_places <= max_place_possible ]
                top_vals_elim = all_places[ all_places <= max_place_possible ]

                all_maxima = corrs_maxima_vals[ top_vals_elim >= min_place_possible ]
                all_places = top_vals_elim[ top_vals_elim >= min_place_possible ]
                
                all_places=all_places[all_maxima<1]
                all_maxima=all_maxima[all_maxima<1]

                #plt.plot(time_array,vals)
                
                #plt.plot(all_places,all_maxima,'o')
                #plt.plot(time_array,np.ones(len(time_array))*.879)
               
                #plt.show()
                if len(all_maxima)>0:
                    if max(all_maxima)>bestest:
                        bestest=max(all_maxima)
                    
                if len( maxima_values ) > 0:
                    """
                    By definition, the best candidate for the acoustic pitch period of a sound 
                    can be found from the position of the maximum of the autocorrelation function
                    of the sound, while the degree of periodicity (the harmonics-to-noise ratio)
                    of the sound can be found from the relative height of this maximum (pg. 97)
                    """
                    # equation number 24
                    strengths = [ val / local_peak - octave_cost * np.log2( min_pitch * place ) 
                                for place, val in zip( maxima_places, maxima_values ) ]
                    #next two lines include unvoiced candidate
                maxima_places = np.hstack( ( maxima_places, 0 ) )
                maxima_values = np.hstack( ( maxima_values, 0) )
                # equation number 23
                strengths.append( voicing_threshold + max( 0, 2 - ( local_peak / global_peak ) / 
                                            ( silence_threshold / ( 1 + voicing_threshold ) ) ) )
                corrs_cand_vals.append( maxima_values[ np.argmax( strengths ) ] / max( vals ) )
                best_cands.append( maxima_places[ np.argmax( strengths ) ] )
        best_cands = np.array( best_cands )
        corrs_cand_vals = np.array( corrs_cand_vals )
        corrs_cand_vals = corrs_cand_vals[ best_cands > 0 ]
        best_cands = best_cands[ best_cands > 0 ]
        if len( best_cands ) == 0:
            #if there are no candidates that fit criteria then assume the signal is unvoiced, 
            #i.e. return 0.
            return 0
        if HNR:
            #TODO: ***LEARN MORE ABOUT HNR TO DETERMINE HOW TO CALCULATE IT***
            pass
        """
        Return the candidate that is in the 85th percentile, instead of the highest valued 
        candidates, which are more often than not anomalies caused by changes in amplitude in the
        signal (autocorrelation PDAS are very sensitive to changes in amplitudes.)
        """
        best_candidate=sorted( best_cands )[ int( .85 * len( best_cands ) ) ]      
        return 1. / best_candidate
        
        
    def get_HNR( self, min_pitch = 90, silence_threshold = .01, ):   
        """
        Compute Harmonic to Noise Ratio (HNR).
        Algorithm uses Fast Fourier Transform (FFT) to filter out values higher than the Nyquist
        Frequency. Then it segments the signal into frames containing at least 6 periods of the 
        minimum pitch. For each frame it then again uses FFT to calculate autocorrelation of the
        signal.  After autocorrelation is calculated it is down sampled with sinc interpolation,
        and then the maxima values of the interpolation are found. After these values have been 
        chosen the best candidate for the HNR is picked and then returned.
        This algorithm is adapted from 
        http://www.fon.hum.uva.nl/david/ba_shs/2010/Boersma_Proceedings_1993.pdf
        
        Args:
            min_pitch (float): minimum value to be returned as pitch, cannot be less than or 
                      equal to zero
            silence_threshold (float): Used to calculate strength of voiceless candidate, the 
                              higher the value the more likely the F_0 will be returned as 
                              voiceless.
            
        Returns:
            float: The HNR of the signal
            
        Raises:
            TypeError: min_pitch must be an int, and silence_threshold must be a float.
            ValueError: The minimum pitch cannot be equal or less than zero.
            ValueError: silence_threshold must be between 0 and 1.
            
        Example:
        ::
            
            from scipy.io import wavfile as wav
            rate, wave= wav.read( 'example_audio_file.wav' )
            sig = Signal( wave, rate )
            sig.get_HNR()
            
        """
        #TODO:***need to update test_Signal_Analysis to get 100% coverage***
        #TODO: need ground values for HNR
        
        return self.get_F_0( min_pitch = min_pitch, silence_threshold = silence_threshold, HNR=True )
        #get code put up github/travis/coveralls...   
        #TODO: clone repo, put on github-> goes in super ai->afx->features (in my fork, ask when ready to merge fork)
        #if everything is similar enough can be put in one class, else seperate it into different classes/files
        #TODO: travis and coveralls, he will get a key for me to put it on privately, which I don't have yet so maybe don't work on this.
        #TODO:write tests for sig_tools
