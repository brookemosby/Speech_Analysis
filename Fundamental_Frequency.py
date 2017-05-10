import numpy as np
import peakutils as pu
import warnings
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore")
#this ignores all warnings, and makes code look sooo much prettier, also kinda naughty...

class Signal():
    def __init__(self, signal):
        self.signal=signal
        self.F_0 =None
        #maybe include sample rate in here ... no need to pass into functions
        
    def get_F_0(self, min_pitch, max_pitch, voicing_threshold, silence_threshold, 
                time_step= .01, max_num_candidates= 4, octave_cost= .01, voiced_unvoiced_cost= 0, octave_jump_cost= 0, HNR= False ):
        """
        If you sample a signal s(t) at fs samples per second, then fN=fs/2 is the Nyquist frequency 
        """
        #end goal is to include a variable for HNR =True/False then we can do it all in one function and then get_HNR would call this function passing in HNR=True
        """
        Step 1. Preprocessing: to remove the sidelobe of the Fourier transform of the
        Hanning window for signal components near the Nyquist frequency, we perform a
        soft upsampling as follows: do an FFT on the whole signal; filter by multiplication in
        the frequency domain linearly to zero from 95% of the Nyquist frequency to 100% of
        the Nyquist frequency; do an inverse FFT of order one higher than the first FFT
        """
        
        """
        The voiced speech of a typical adult male will have a fundamental frequency from 85 to 180 Hz, and that of a typical adult female from 165 to 255 Hz, high pitched voices can be up to 600.
        maybe set this as default min/max pitch, with a wide error range ie (10-700)?
        """
        
        #first plot
        sig_len=len(self.signal)
        domain=np.linspace(0,sig_len*time_step,sig_len)
        plt.subplot(221)
        plt.plot(domain,self.signal,label='signal')
        
        total_time=time_step*len(self.signal)
        Nyquist_Frequency=1./(time_step*2)#if we decide to use sample rate then we would change this to nf=sample_rate/2
        if Nyquist_Frequency<max_pitch:
            raise ValueError("The maximum pitch cannot be greater than the Nyquist Frequency.")
        if min_pitch==0:
            raise ValueError("The minimum pitch cannot be zero.") #double check, we may not want to raise a value error.
        upper_bound=.95*Nyquist_Frequency
        fft_signal=np.fft.fft(self.signal)
        mask=fft_signal<upper_bound
        fft_signal*=mask
        signal=np.fft.ifft(fft_signal)
        """
        Step 2. Compute the global absolute peak value of the signal 
        """
        global_peak=max(abs(signal))
        
        plt.plot(domain,np.ones(sig_len)*global_peak)
        
        
        #if HNR: then put HNR window len in list with regular window len and do same for both(iterate through list) so code duplication is reduced
        """
        Step 3. Because our method is a short-term analysis method, the analysis is
        performed for a number of small segments (frames) that are taken from the signal in
        steps given by the TimeStep parameter (default is 0.01 seconds). For every frame, we
        look for at most MaximumNumberOfCandidatesPerFrame (default is 4) lag-height
        pairs that are good candidates for the periodicity of this frame. This number includes
        the unvoiced candidate, which is always present. The following steps are taken for
        each frame:
                            
                Step 3.1. Take a segment from the signal. The length of this segment (the window
                length) is determined by the MinimumPitch parameter, which stands for the lowest
                fundamental frequency that you want to detect. The window should be just long
                enough to contain three periods (for pitch detection) or six periods (for HNR
                measurements) of MinimumPitch. 
                
        """
        window_len=3.0/min_pitch
        frame_len=window_len/time_step
        num_frames=int(len(signal)/frame_len+.5)
        segmented_signal=[signal[int(i*frame_len):int((i+1)*frame_len)] for i in range(num_frames+1)]
        
        
        for i in range(num_frames):
            plt.plot(domain[int(i*frame_len)]*np.ones(50),np.linspace(-2,2,50))
            
        plt.legend()
            
    
        #intializing variables before for loop so they are not localized 
        cands_for_all_frames=[]
        for index in range(len(segmented_signal)):
            if len(segmented_signal[index])==0:
                #this is to ignore the occasional empty segments (when the signal can be segmented into equal parts it leaves 1 empty array at the end)
                break
            else:

                time_begin,time_end=index*window_len,min((index+1)*window_len,total_time)
                window_len=time_end-time_begin
                candidate_strength=[]#calculate candidate strength after modifying frame to find abs maximas
                local_peak=max(abs(segmented_signal[index]))
                
                segment=segmented_signal[index]
               
                def estimated_autocorrelation(x):
                    n = len(x)
                    variance = x.var()
                    x = x-x.mean()
                    r = np.correlate(x, x, mode = 'full')[-n:]
                    result = r/(variance*(np.arange(n, 0, -1)))
                    return result
                
                r_x=estimated_autocorrelation(segment)
                
                plt.subplot(222)#?
                plt.plot(np.linspace(time_begin,time_end,len(r_x)),r_x,label='r_x')
                plt.legend()
                
                """
                Step 3.11. Find the places and heights of the maxima of the continuous version of
                rx(τ), which is given by equation 22. The only places considered for the maxima are
                those that yield a pitch between MinimumPitch and MaximumPitch. The MaximumPitch 
                parameter should be between MinimumPitch and the Nyquist frequency. The only candidates 
                that are remembered, are the unvoiced candidate, which has a local strength equal to
                R ≡ VoicingThreshold + max (0, 2 − (local absolute peak/global absolute peak)/
                (SilenceThreshold/ ( 1+ VoicingThreshold)) and the voiced candidates with the highest 
                (MaximumNumberOfCandidatesPerFrame minus 1) values of the local strength
                R ≡ r(τ max) − OctaveCost *2log( MinimumPitch * τ max)
                the harmonically amplitude-modulated signal with modulation depth dmod
                x( t ) = (1+ dmod sin2πFt) sin 4πFt
                has an acoustic fundamental frequency of F, whereas its perceived pitch is 2F for
                modulation depths smaller than 20 or 30 percent.The default OctaveCost value is 0.01, 
                corresponding to a criterion of 10%.
                After performing step 2 for every frame, we are left with a number of frequency strength
                pairs (Fni, Rni), where the index n runs from 1 to the number of frames, and i
                is between 1 and the number of candidates in each frame. The locally best candidate
                in each frame is the one with the highest R. But as we can have several approximately
                equally strong candidates in any frame, we can launch on these pairs the global path
                finder, the aim of which is to minimize the number of incidental voiced-unvoiced
                decisions and large frequency jumps
                """
                def sinc_interp(x, s, u):
                    """
                    Interpolates x, sampled at "s" instants
                    Output y is sampled at "u" instants ("u" for "upsampled")
                    
                    """
                    
                    if len(x) != len(s):
                        raise Exception( 'x and s must be the same length')
                    
                    # Find the period    
                    T = s[1] - s[0]
                    
                    sincM = np.tile(u, (len(s), 1)) - np.tile(s[:, np.newaxis], (1, len(u)))
                    y = np.dot(x, np.sinc(sincM/T))
                    return y
                
              
                time_array=np.linspace(time_begin,time_end,len(r_x)) 
                vals=np.nan_to_num(sinc_interp(r_x,time_array,np.linspace(time_begin,time_end,len(r_x)*2)))*np.hanning(2*len(r_x))
                time_array=np.linspace(time_begin,time_end,len(vals))
                
                plt.subplot(223)#??
                plt.plot(time_array,vals,label='r_x interpolated, multiplied by hanning window')
                plt.legend()
                
                
                if len(vals.nonzero()[0])!=0:               
                    indexes=pu.indexes(vals,.5)
                    #I think this all works, now just need to check up on the bottom stuff... see why it is returning bad info.
                    maxima_places,maxima_values=time_array[indexes],vals[indexes]
                    max_place_possible=1./min_pitch
                    min_place_possible=1/max_pitch
                    plt.plot(maxima_places,maxima_values,'o')
                    top_vals_elim=maxima_places[maxima_places<max_place_possible]
                    corrs_maxima_vals=maxima_values[maxima_places<max_place_possible]
                    maxima_places=top_vals_elim[top_vals_elim>min_place_possible]
                    maxima_values=corrs_maxima_vals[top_vals_elim>min_place_possible]
                    
                    #delete any values in maxima_places that do not fit in this range.
                    #1 divided by the maxima places should give us pitch 
                    #in this case we need to check for the 1/maxima_places that fall in the correct range.
                    
                    
                    plt.subplot(224)#??
                    plt.plot(time_array,vals,label='r_x windowed & interpolated, new bounds for max')
                    plt.legend()
                    plt.plot(maxima_places,maxima_values,'o')
                    
                    #unvoiced candidate, which is always present.
                    #F=0 for unvoiced candidate            
                    strengths=[val-octave_cost*np.log2((min_pitch*place)) for place, val in zip(maxima_places,maxima_values)]
                    indexes_with_vals=[(idx,i) for idx,i in enumerate(strengths)]
                    highest_vals=sorted(indexes_with_vals,reverse=True,key=lambda i:i[1])
                    #the number being added below indicates the number of candidates following it, for that specific frame. this helps when computing the cost.
                    if len(highest_vals)<max_num_candidates-1:
                        cands_for_all_frames.append(len(highest_vals)+1)
                    else:
                        cands_for_all_frames.append(max_num_candidates)
                    candidate_strength=[(maxima_places[i[0]],i[1]) for i in highest_vals[:max_num_candidates-1]]
                    unvoiced_strength=voicing_threshold+ max(0, 2-((local_peak)/float(global_peak))/((silence_threshold)/(1.+voicing_threshold)))
                    candidate_strength.append([0,unvoiced_strength]) #unvoiced candidate and strength are added to dictionary
                else:
                    #this addresses the case of an all zero array
                    unvoiced_strength=voicing_threshold+ max(0, 2-((local_peak)/float(global_peak))/((silence_threshold)/(1.+voicing_threshold)))
                    cands_for_all_frames.append(1)
                    candidate_strength.append([0,unvoiced_strength])
                    
                    
                cands_for_all_frames.append(candidate_strength)
                #should I raise value error if max_pitch>nyquist frequency? or set new max_pitch= nyquist frequency <- still a good question, for now raise value error.
                #we only remember candidates that have highest local strength, calculated with the places and heights of the maxima of r_x and of course unvoiced candidate
                #strength calculated with places and heights of local maxima
        # ok so we want to first create a list of list, each index containing all the possible paths.
        """
        Step 4. For every frame n, pn is a number between 1 and the number of candidates
        for that frame. The values {pn | 1 ≤ n ≤ number of frames} define a path through the
        candidates: {(Fnpn, Rnpn) | 1 ≤ n ≤ number of frames}. With every possible path we
        associate a cost.
        The globally best path is the path with the lowest cost. This path might contain some
        candidates that are locally second-choice. We can find the cheapest path with the aid
        of dynamic programming.
        We turn the path finder off by setting the
        VoicedUnvoicedCost and OctaveJumpCost parameters to zero; in this way, the
        algorithm selects the locally best candidate for each frame.
        For HNR measurements, the path finder is turned off, and the OctaveCost and
        VoicingThreshold parameters are zero, too; MaximumPitch equals the Nyquist
        frequency; only the TimeStep, MinimumPitch, and SilenceThreshold parameters are
        relevant for HNR measurements
        """
        total_paths=1
        for x in cands_for_all_frames[0::2]:
            total_paths*=x
        #this creates a path of potential candidates for each frame
        #a list must be passed in, this list must contain total_paths number of empty lists
        #returns a list of indexes that correspond to the list of potential candidates, creates all possible paths.
        def find_all_paths(paths,layer):
            """This is a recursive function that returns the indices of all possible paths corresponding to the list cans_for_all_frames."""
            if layer==num_frames:
                return paths
            else:
                for i in range(len(paths)):
                    num=len(paths)/cands_for_all_frames[0::2][layer]
                    paths[i].append(int(i/num))
                total_list=[]
                for x in range(cands_for_all_frames[0::2][layer]):
                    total_list+=find_all_paths(paths[int(x*num):int((x+1)*num)],layer+1)
            return total_list
        # after we have created a list of paths we will need to iterate through each list, and create a corresponding list of the cost for each path, then choose the path with the lowest cost
        all_paths=find_all_paths([[] for x in range(total_paths)],0)
        def transition_cost(path):
            sum_total=0
            for x in range(len(path)-1):
                if path[x]==0 and path[x+1]==0:
                    pass
                elif path[x]==0 or path[x+1]==0:
                    sum_total+=voiced_unvoiced_cost
                elif path[x]!=0 and  path[x+1]!=0:
                    sum_total+= octave_jump_cost*abs(np.log2(path[x]/path[x+1]))
            return sum_total
        
        def sum_strengths(path):
            sum_total=0
            for x in range(len(path)):
                sum_total+=cands_for_all_frames[1::2][x][path[x]][1]
            return sum_total
        
        
        
        plt.legend()
        plt.show()
        
        
        
        total_cost=[transition_cost(path)-sum_strengths(path) for path in all_paths]
        total_cost=list(np.array(total_cost).real)#changing this to a list of real values
        best_path=all_paths[total_cost.index(min(total_cost))]
        pos=np.arange(len(best_path))
        freqs=[cands_for_all_frames[1::2][int(p)][index][0] for index,p in zip(best_path,pos)]
        if min(freqs)==0:
            return 0
        else:
            return 1/min(freqs)
    
        #ok so what is wrong right now?
        #a-returning the wrong thing, path should line up with number of frames, we are sometimes getting an empty list ie min freq=10, that shouldnt happen.
        
        #write test code
        #then add in HNR (shouldn't be too hard)
        # get code put up github/travis/coveralls...
        #make a python installable package      
            
        #if HNR: window_len_HNR=6.0/min_pitch
        