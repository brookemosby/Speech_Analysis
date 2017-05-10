import numpy as np
import peakutils as pu
import warnings
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore")

class Signal():
    def __init__(self, signal):
        self.signal = signal
        self.F_0 = None
        #maybe include sample rate in here ... no need to pass into functions
        
    def get_F_0(self, min_pitch, max_pitch, voicing_threshold, silence_threshold, 
                time_step = .01, max_num_candidates = 4, octave_cost = .01, voiced_unvoiced_cost = 0, octave_jump_cost = 0, HNR = False ):
        """
        Insert awesome description here
        """
        #end goal is to include a variable for HNR =True/False then we can do it all in one function and then get_HNR would call this function passing in HNR=True
        
        #first plot
        #plotting signal
        sig_len = len( self.signal )
        domain = np.linspace( 0, sig_len * time_step, sig_len )
        plt.subplot(221)
        plt.plot( domain, self.signal, label = 'signal' )
        
        total_time = time_step * len( self.signal )
        Nyquist_Frequency = 1. / ( time_step * 2 )#if we decide to use sample rate then we would change this to nf=sample_rate/2
        
        #checking to make sure values are valid                         
        if Nyquist_Frequency < max_pitch:
            raise ValueError( "The maximum pitch cannot be greater than the Nyquist Frequency." )#do we want to raise a ValueError, or just change max_pitch to nyguist frequency and print message?
        if min_pitch == 0:
            raise ValueError( "The minimum pitch cannot be zero." ) #double check, we may not want to raise a value error, maybe just change window_len to length of signal in this case
            
        #filtering by Nyquist Frequency (preproccesing step)
        upper_bound = .95 * Nyquist_Frequency
        fft_signal = np.fft.fft( self.signal )
        mask = fft_signal < upper_bound
        fft_signal *= mask
        signal = np.fft.ifft( fft_signal )
        
        global_peak = max( abs( signal ) )
        #plotting the global peak with the signal
        plt.plot( domain, np.ones( sig_len ) * global_peak )
        
        
        #note->if HNR: then put HNR window len in list with regular window len and do same for both(iterate through list) so code duplication is reduced


        #finding the window_len in seconds, finding frame len (as an integer of how many points will be in a window), finding number of frames/ windows that we will need to segment the signal into
        #then segmenting signal
        window_len = 3.0 / min_pitch
        frame_len = window_len / time_step
        num_frames = max( 1, int( len( signal ) / frame_len + .5 ) ) #there has to be at least one frame
        segmented_signal = [ signal[ int( i * frame_len ) : int( ( i+1 ) * frame_len ) ] for i in range( num_frames + 1 ) ]
        
        #plotting the segments on the signal
        for i in range( num_frames ):
            plt.plot( domain[ int( i*frame_len ) ] * np.ones( 50 ), np.linspace( -2, 2, 50 ) )
        plt.legend()
    
        #intializing variables before for loop so they are not localized 
        cands_for_all_frames = [ ]
        
        #these functions are defined out side of the loop, so that they won't be defined multiple times
        def estimated_autocorrelation( x ):
            """
            Accepts segment finds autocorrelation of a segment, returns result
            """
            n = len( x )
            variance = x.var()
            x = x - x.mean()
            r = np.correlate( x, x, mode = 'full' )[ -n: ]
            result = r / ( variance * ( np.arange( n, 0, -1 ) ) )
            return result
        def sinc_interp( x, s, u ):
            """
            Interpolates x, sampled at "s" instants
            Output y is sampled at "u" instants ("u" for "upsampled")
            """
            
            if len( x ) != len( s ):
                raise Exception( 'x and s must be the same length' )
            
            # Find the period    
            T = s[ 1 ] - s[ 0 ]
            sincM = np.tile( u, ( len( s ), 1 ) ) - np.tile( s[ :, np.newaxis ], ( 1, len( u ) ) )
            y = np.dot( x, np.sinc( sincM / T ) )
            return y
                
        #this calculates maxima of autocorrelation for each segment.
        for index in range( len( segmented_signal ) ):
            #this is to ignore the occasional empty segments (when the signal can be segmented into equal parts it leaves 1 empty array at the end)
            if len( segmented_signal[ index ] ) == 0:
                break
            
            else:
                #finding where the time begins and ends for each segmentso that we can create time domains easier
                time_begin, time_end = index * window_len, min( ( index + 1 ) * window_len, total_time )
                window_len = time_end - time_begin
                
                candidate_strength = [ ]
                local_peak = max( abs( segmented_signal[ index ] ) )
                
                segment = segmented_signal[ index ]
                r_x = estimated_autocorrelation( segment )
                
                plt.subplot( 222 )
                plt.plot( np.linspace( time_begin, time_end, len( r_x ) ), r_x, label = 'r_x' )
                plt.legend()
                
                time_array = np.linspace( time_begin, time_end, len( r_x ) ) 
                
                #do we want to multiply by hanning window to window the interpolation (as done in paper)?
                vals = np.nan_to_num( sinc_interp( r_x, time_array, np.linspace( time_begin, time_end, len( r_x ) * 8 ) ) ) #* np.hanning( len( r_x ) * 2 ) 
                time_array = np.linspace( time_begin, time_end, len( vals ) )
                
                plt.subplot( 223 )#??
                plt.plot( time_array, vals, label = 'r_x interpolated' )
                plt.legend()
                
                if len( vals.nonzero()[ 0 ] ) != 0:          
                    #pu.indexes returns the indexes of peaks above the normalized threshold of .5
                    indexes = pu.indexes( vals, .5 )
                    maxima_places, maxima_values = time_array[ indexes ], vals[ indexes ]
                    max_place_possible = 1. / min_pitch
                    min_place_possible = 1. / max_pitch
                    
                    #plot all maxima places on plot above
                    plt.plot( maxima_places, maxima_values, 'o' )
                    
                    #deleting maxima that don't yield a frequency between min and max pitch
                    top_vals_elim = maxima_places[ maxima_places < max_place_possible ]
                    corrs_maxima_vals = maxima_values[ maxima_places < max_place_possible ]
                    maxima_places = top_vals_elim[ top_vals_elim > min_place_possible ]
                    maxima_values = corrs_maxima_vals[ top_vals_elim > min_place_possible ]
                    
                    #plotting maxima after invalid values have been eliminated
                    plt.subplot( 224 )#??
                    plt.plot( time_array, vals, label = 'r_x windowed & interpolated, new bounds for max' )
                    plt.legend()
                    plt.plot( maxima_places, maxima_values, 'o' )
                    
                    #unvoiced candidate, which is always present.          
                    strengths = [ val - octave_cost * np.log2( min_pitch * place ) for place, val in zip( maxima_places, maxima_values ) ]
                    indexes_with_vals = [ ( idx, i ) for idx, i in enumerate( strengths ) ]
                    highest_vals = sorted( indexes_with_vals, reverse = True , key = lambda i:i[ 1 ] )
                    
                    #the number being added below indicates the number of candidates following it, for that specific frame. this helps when computing the cost.
                    if len( highest_vals ) < max_num_candidates - 1 :
                        cands_for_all_frames.append( len( highest_vals ) + 1 )
                    else:
                        cands_for_all_frames.append( max_num_candidates )
                    candidate_strength = [ ( maxima_places[ i[ 0 ] ], i[ 1 ] ) for i in highest_vals[ :max_num_candidates-1 ] ]
                    unvoiced_strength = voicing_threshold + max( 0, 2 - ( ( local_peak ) / float( global_peak ) ) / ( ( silence_threshold ) / ( 1. + voicing_threshold ) ) )
                    candidate_strength.append( [ 0, unvoiced_strength ] ) #unvoiced candidate and strength are added to dictionary
                else:
                    #this addresses the case of an all zero array
                    unvoiced_strength = voicing_threshold + max( 0, 2 - ( ( local_peak ) / float( global_peak ) ) / ( ( silence_threshold ) / ( 1. + voicing_threshold ) ) )
                    cands_for_all_frames.append( 1 )
                    candidate_strength.append( [ 0, unvoiced_strength ] )
                    
                cands_for_all_frames.append( candidate_strength )
                
        plt.show()
        #finding total number of paths possible based off of number of candidates
        total_paths = 1
        for x in cands_for_all_frames[ 0 : : 2 ]:
            total_paths *= x

        def find_all_paths( paths, layer ):
            """This is a recursive function that returns the indices of all possible paths corresponding to the list cans_for_all_frames."""
            if layer == num_frames:
                return paths
            else:
                for i in range( len( paths ) ):
                    num = len( paths ) / cands_for_all_frames[ 0 : : 2 ][ layer ]
                    paths[ i ].append( int( i / num ) )
                total_list = []
                for x in range( cands_for_all_frames[ 0 : : 2 ][ layer ] ):
                    total_list += find_all_paths( paths[ int( x * num ) : int( ( x + 1 ) * num ) ], layer + 1 )
            return total_list
        
        #after we have created a list of paths we will need to iterate through each list, and create a corresponding list of the cost for each path, then choose the path with the lowest cost
        
        all_paths = find_all_paths( [ [] for x in range( total_paths ) ], 0 )
        
        def transition_cost( path ):
            """Finds transition cost for a specific path, returns a value"""
            sum_total = 0
            for x in range( len( path ) - 1 ):
                if path[ x ] == 0 and path[ x+1 ] == 0:
                    pass
                elif path[ x ] == 0 or path[ x + 1 ]==0:
                    sum_total += voiced_unvoiced_cost
                elif path[ x ] != 0 and  path[ x + 1 ] != 0:
                    sum_total += octave_jump_cost * abs( np.log2( path[ x ] / path[ x + 1 ] ) )
            return sum_total
        
        def sum_strengths( path ):
            """ sums all the strengths of a path of different frequencies, returns resulting value."""
            sum_total = 0
            for x in range( len( path ) ):
                sum_total += cands_for_all_frames[ 1 : : 2 ][ x ][ path[ x ] ][ 1 ]
            return sum_total
        
        total_cost = [ transition_cost( path ) - sum_strengths( path ) for path in all_paths ]
        best_path = all_paths[ total_cost.index( min( total_cost ) ) ]
        
        #we iterate through the best path found, and give the maximum of a list of the places that correspond to the best values
        pos = np.arange( len( best_path ) )  
        best_time = max([ cands_for_all_frames [ 1 : : 2 ][ int( p ) ][ index ][ 0 ] for index, p in zip( best_path, pos ) ])
        
        #return corresponding frequency to the maximum time, aka the minimum frequency
        if best_time == 0:
            return 0
        else:
            return  1. / best_time
    
        #ok so what is wrong right now?
        #not returning correct values, multiples of correct value
        #thoughts, increase values of interpolation, multiply by gaussian window....
        #the problem is it is picking later peaks that are repetitions of the original peak.
        
        #write test code
        #then add in HNR (shouldn't be too hard)
        # get code put up github/travis/coveralls...
        #make a python installable package      
            
        #if HNR: window_len_HNR=6.0/min_pitch
        