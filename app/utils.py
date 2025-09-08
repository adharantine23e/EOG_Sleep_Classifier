import numpy as np
from scipy.signal import firwin, firwin2, freqz
from functools import partial
from scipy.fft import rfft, irfft
from typing import Optional

#                                         FILTERING  BASED ON MNE                                           #
#############################################################################################################
def transform(data: np.ndarray, epoch_duration = 30, padding_duration = None):
  sfreq = 100
  samples_per_epoch = int(epoch_duration * sfreq)
  n_epochs = data.shape[1] // samples_per_epoch


  if padding_duration is not None:
    overlap_samples = int(padding_duration * sfreq)
    n_epochs = int(np.ceil(data.shape[1] - overlap_samples) / (samples_per_epoch - overlap_samples))
    epochs = np.zeros((n_epochs, data.shape[0], samples_per_epoch))
    for i in range(n_epochs):
      start_idx = i * (samples_per_epoch - overlap_samples)
      end_idx = start_idx + samples_per_epoch
      if end_idx > data.shape[1]:
        padding = np.zeros((data.shape[0], end_idx - data.shape[1]))
        epoch_data = np.concatenate((data[:, start_idx:data.shape[1]], padding), axis=1)
      else:
        epoch_data = data[:, start_idx:end_idx]
      epochs[i] = epoch_data
  else:
    remaining_samples = data.shape[1] % samples_per_epoch
    if remaining_samples > 2:
        padding = np.zeros((data.shape[0], samples_per_epoch - remaining_samples))
        data = np.concatenate((data, padding), axis = 1)
        n_epochs = data.shape[1] // samples_per_epoch


    epochs = data[: ,:n_epochs *samples_per_epoch].reshape((n_epochs, samples_per_epoch))
  return epochs
def filter_data(data: np.ndarray, sfreq: float = 100.0,
                l_freq: float = 0.3, h_freq: float = 15.0, 
                filter_length= 'auto',
                l_trans_bandwidth = 'auto',
                h_trans_bandwidth = 'auto',
                n_jobs = None,
                method = 'fir',
                copy = True,
                phase = 'zero',
                fir_window= 'hamming',
                fir_design = 'firwin',
                pad = 'reflect_limited',
                *,
                verbose=None,):
  
  filt = create_filter(data, sfreq, l_freq, h_freq, filter_length,
                       l_trans_bandwidth, h_trans_bandwidth, method, 
                       phase, fir_window, fir_design)
  data = overlap_add_filter(data, filt, None, phase, n_jobs, copy, pad)
  return data

def overlap_add_filter(x, h, n_fft= None, phase = 'zero', n_jobs = None, copy = True, pad = 'reflect_limited'):
  x, original_shape = prep_for_filtering(x, copy)
  if len(h) == 1:
    return x*h**2 if phase== 'zero-double' else x*h
  n_edge = max(min(len(h), x.shape[1])-1, 0)
  n_x= x.shape[1] + 2*n_edge

  min_fft = 2*len(h) -1
  if n_fft is None:
    max_fft = n_x
    if max_fft >= min_fft:
      N = 2 ** np.arange(np.ceil(np.log2(min_fft)), np.ceil(np.log2(max_fft)) + 1, dtype=int)
      cost = (np.ceil(n_x / (N - len(h) + 1).astype(np.float64))* N* (np.log2(N) + 1)
            )
      cost += 4e-5 * N * n_x
      n_fft = N[np.argmin(cost)]
    else:
      n_fft = next_fast_len(min_fft)
  if n_fft < min_fft:
    raise ValueError(f"n_fft must be at least {min_fft}, got {n_fft}")
  for p in range(len(x)):

    x[p]= _1d_overlap_filter(x[p], h, len(h), n_edge, phase, pad, n_fft)

  x.shape = original_shape
  return x

def _1d_overlap_filter(x: np.ndarray, h: np.ndarray, n_h: int, n_edge: int, phase: str, pad: str, n_fft: int):
    """Do one-dimensional overlap-add FFT FIR filtering."""
    # pad to reduce ringing
    x_ext = _smart_pad(x, (n_edge, n_edge), pad)
    n_x = len(x_ext)
    x_filtered = np.zeros_like(x_ext)

    n_seg = n_fft - n_h + 1
    n_segments = int(np.ceil(n_x / float(n_seg)))
    shift = ((n_h - 1) // 2 if phase.startswith("zero") else 0) + n_edge

    # Now the actual filtering step is identical for zero-phase (filtfilt-like)
    # or single-pass
    for seg_idx in range(n_segments):
        start = seg_idx * n_seg
        stop = (seg_idx + 1) * n_seg
        seg = x_ext[start:stop]
        seg = np.concatenate([seg, np.zeros(n_fft - len(seg))])

        prod = _fft_multiply_repeated(seg, rfft = rfft, n_fft= n_fft, irfft = irfft , h = h)

        start_filt = max(0, start - shift)
        stop_filt = min(start - shift + n_fft, n_x)
        start_prod = max(0, shift - start)
        stop_prod = start_prod + stop_filt - start_filt
        x_filtered[start_filt:stop_filt] += prod[start_prod:stop_prod]

    # Remove mirrored edges that we added and cast (n_edge can be zero)
    x_filtered = x_filtered[: n_x - 2 * n_edge].astype(x.dtype)
    return x_filtered

def _fft_multiply_repeated(x: np.ndarray, rfft: callable, n_fft: int, irfft: callable, h: np.ndarray):
  h_fft = rfft(h, n_fft)
  x_fft= rfft(x, n_fft)
  x_fft *= h_fft
  x = irfft(x_fft, n_fft)
  return x

def _smart_pad(x: np.ndarray, n_pad: tuple, pad = 'reflect_limited'):
  n_pad = np.asarray(n_pad)
  assert n_pad.shape == (2,)
  if (n_pad ==0).all():
    return x
  elif (n_pad <0).any():
    raise RuntimeError("n_pad must be non-negative")
  l_z_pad = np.zeros(max(n_pad[0] - len(x) +1, 0), dtype= x.dtype)
  r_z_pad = np.zeros(max(n_pad[1] - len(x) +1, 0), dtype= x.dtype)
  return np.concatenate(
      [l_z_pad,
      2* x[0] - x[n_pad[0]: 0: -1],
      x,
      2* x[-1] - x[-2: -n_pad[1] -2: -1],
      r_z_pad
       ]
  )

def create_filter(data: np.ndarray,
                  sfreq: float,
                  l_cut: Optional[float],
                  h_cut: Optional[float],
                  filter_length = 'auto',
                  l_trans_bandwidth = 'auto',
                  h_trans_bandwidth = 'auto',
                  method = 'fir',
                  phase = 'zero',
                  fir_window = 'hamming',
                  fir_design = 'firwin',
                  verbose = False):
  sfreq = float(sfreq)
  if sfreq < 0:
    raise ValueError("The sampling frequency must be positive")
  h_cut = np.array(h_cut, dtype = float).ravel()
  l_cut = np.array(l_cut, dtype = float).ravel()

  freq = [0, sfreq/2.0]
  gain = [1.0, 1.0]

  if l_cut is not None and h_cut is not None:
    if (l_cut < h_cut).any():
      l_cut, h_cut = l_cut.item(), h_cut.item()
      (data, sfreq, f_p1, f_p2, f_s1, f_s2, filter_length, phase, fir_window, fir_design) = triage_filter_param(
          data, sfreq,
          l_freq = l_cut, h_freq = h_cut,
          l_trans_bandwidth= l_trans_bandwidth, h_trans_bandwidth= h_trans_bandwidth,
          filter_length= filter_length, phase= phase,
          fir_window= fir_window, fir_design= fir_design,         
      )

      freq = [f_s1, f_p1, f_p2, f_s2]
      gain = [ 0, 1, 1 ,0 ]
      if f_s2 != sfreq / 2.0:
        freq += [sfreq / 2.0]
        gain += [0]
      if f_s1 != 0:
        freq += [0] 
        gain += [0]
    else:
      if len(l_cut) != len(h_cut):
        raise ValueError("l_cut and h_cut must have the same length")
      print("Setting up band-stop filter [Not what we want]")
      if len(l_cut) == 1:
        l_cut, h_cut = l_cut.item(), h_cut.item()
      (data, sfreq, f_p1, f_p2, f_s1, f_s2, filter_length, phase, fir_window, fir_design) = triage_filter_param(
          data, sfreq,
          l_freq = l_cut, h_freq = h_cut,
          l_trans_bandwidth= l_trans_bandwidth, h_trans_bandwidth= h_trans_bandwidth,
          filter_length= filter_length, phase= phase,
          fir_window= fir_window, fir_design= fir_design, bands = 'arr', reverse = True      
      )
      
      freq = np.r_[f_p1, f_s1, f_s2, f_p2]
      gain = np.r_[
                    np.ones_like(f_p1),
                    np.zeros_like(f_s1),
                    np.zeros_like(f_s2),
                    np.ones_like(f_p2),
                ]
      order = np.argsort(freq)
      print(order)
      freq = freq[order]
      gain = gain[order]
      if freq[0] !=0:
        freq= np.r_[[0.0], freq]
        gain = np.r_[[1.0], gain]
      if freq[-1] != sfreq/2.0:
        freq = np.r_[freq, [sfreq/2.0]]
        gain = np.r_[gain, [1.0]]
      if np.any(np.abs(np.diff(gain, 2))> 1):
        raise ValueError("Stop bands are not sufficiently seperate")

  else:
    raise ValueError(f"You must supply but low and high cut, got l_cut: {l_cut} and h_cut: {h_cut}")
  out = fir_filter(sfreq, freq, gain, filter_length, phase, fir_window, fir_design)
  return out

def firwin_design(N: int, freq: np.ndarray, gain: np.ndarray, window: str, sfreq: float):
  assert freq[0] == 0
  assert len(freq) > 1
  assert len(freq) == len(gain)
  assert N%2 ==1
  h = np.zeros(N)
  prev_freq = freq[-1]
  prev_gain = gain[-1]
  if gain[-1] == 1:
    h[N //2] = 1
  assert prev_gain in (0, 1)
  length_factor = dict(hann= 3.1, hamming=3.3, blackman = 5.0)
  for this_freq, this_gain in zip(freq[::-1][1:], gain[::-1][1:]):
    assert this_gain in (0, 1)
    if this_gain != prev_gain:
      transition = (prev_freq - this_freq) / 2.0
      this_N = int(round(length_factor[window] / transition))
      this_N += 1 - this_N % 2
      if this_N > N:
        raise ValueError(
                    f"The requested filter length {N} is too short for the requested "
                    f"{transition * sfreq / 2.0:0.2f} Hz transition band, which "
                    f"requires {this_N} samples"
                )
      this_h = firwin(this_N, (prev_freq + this_freq)/2.0, window = window,
                      pass_zero= True, fs= freq[-1] * 2)
      assert this_h.shape == (this_N,)
      offset = (N - this_N) // 2
      if this_gain == 0:
        h[offset : N - offset] -= this_h
      else:
        h[offset : N - offset] += this_h
    prev_freq = this_freq
    prev_gain = this_gain
  return h

def filter_attenuation(h: np.ndarray, freq: np.ndarray, gain: np.ndarray):
    """Compute minimum attenuation at stop frequency."""
    _, filt_resp = freqz(h.ravel(), worN=np.pi * freq)
    filt_resp = np.abs(filt_resp)  # use amplitude response
    filt_resp[np.where(gain == 1)] = 0
    idx = np.argmax(filt_resp)
    att_db = -20 * np.log10(np.maximum(filt_resp[idx], 1e-20))
    att_freq = freq[idx]
    return att_db, att_freq

def fir_filter(sfreq: float, freq: np.ndarray, gain: np.ndarray, filter_length: int, phase: str, fir_window: str, fir_design: str):
  assert freq[0] ==0 
  if fir_design == 'firwin2':
    fir_design = firwin2
  else:
    assert fir_design == 'firwin'
    fir_design = partial(firwin_design, sfreq = sfreq)
  min_att_db = 12 if phase == 'minimum-half'  else 20
  freq = np.array(freq) / (sfreq / 2.0)
  if freq[0] != 0 or freq[-1] != 1:
    raise ValueError( f"freq must start at 0 and end an Nyquist ({sfreq / 2.0}), got {freq}")
  gain = np.array(gain)

  N = int(filter_length)
  if N %2 == 0:
    if phase == 'zero':
      raise RuntimeError(f'filter length must be odd if phase = zero, got {N}')
    elif phase == 'zero-double' and gain[-1] == 1:
      N+=1
  h = fir_design(N, freq, gain, window= fir_window)
  assert h.size == N
  att_db, att_freq = filter_attenuation(h, freq, gain)
  if phase == 'zero-double':
    att_db +=6
  if att_db < min_att_db:
    att_freq *= sfreq/ 2.0
  
  return h
    
def triage_filter_param(x: np.ndarray, 
                        sfreq: float,
                        l_freq: float,
                        h_freq: float,
                        l_trans_bandwidth: str,
                        h_trans_bandwidth: str,
                        filter_length: str,
                        phase: str,
                        fir_window: str,
                        fir_design: str,
                        method = 'fir',
                        bands = 'scalar',
                        reverse= False):
  sfreq = float(sfreq)
  report_phase = 'non-linear phase' if phase == 'minimum' else 'zero-phase'
  casuality = "casual" if phase == 'minimum' else "non-casual"
  report_pass = 'one-pass'
  if l_freq  is not None:
    if h_freq is not None:
      kind = 'bandstop' if reverse else 'bandpass'
    else:
      kind = 'highpass'
      assert not reverse
  elif h_freq is not None:
    kind = 'lowpass'
    assert not reverse
  else:
    kind = 'allpass'

  db_cutoff = False
  if bands == 'scalar' or (len(h_freq) == 1 and len(l_freq) == 1):
    if phase == 'zero':
      db_cutoff = "-6 dB"

  l_stop = h_stop = None
  if l_freq is not None:
    if isinstance(l_trans_bandwidth, str):
      if l_trans_bandwidth != 'auto':
        raise ValueError('l_trans_bandwidth must be "auto" if l_freq is not None')
      l_trans_bandwidth = np.minimum(np.maximum(0.25* l_freq, 2.0), l_freq)
    l_trans_rep = np.array(l_trans_bandwidth, float)
    if l_trans_rep.size == 1:
      l_trans_rep = f"{l_trans_rep.item():0.2f}"
    with np.printoptions(precision = 2, floatmode= 'fixed'):
      if db_cutoff:
        l_freq_rep = np.array(l_freq, float)
        if l_freq_rep.size ==1:
          l_freq_rep = f"{l_freq_rep.item():0.2f}"
        cutoff_req = np.array(l_freq - l_trans_bandwidth / 2.0, float)
        if cutoff_req.size == 1:
          cutoff_req = f"{cutoff_req.item():0.2f}"
    l_trans_bandwidth = float(l_trans_bandwidth)
    if np.any(l_trans_bandwidth <=0):
      raise ValueError(f"l_trans_bandwidth must be positive, got {l_trans_bandwidth}")
    l_stop = l_freq - l_trans_bandwidth
    if reverse:
      l_stop += l_trans_bandwidth
      l_freq += l_trans_bandwidth

  if h_freq is not None:
    if isinstance(h_trans_bandwidth, str):
      if h_trans_bandwidth != 'auto':
        raise ValueError('h_trans_bandwidth must be "auto" if h_freq is not None')
      h_trans_bandwidth = np.minimum(np.maximum(0.25 * h_freq, 2.0), sfreq /2.0 - h_freq)
    
    h_trans_rep = np.array(h_trans_bandwidth, float)
    if h_trans_rep.size == 1:
      h_trans_rep = f"{h_trans_rep.item():0.2f}"
    with np.printoptions(precision=2, floatmode='fixed'):
      if db_cutoff:
        h_freq_rep = np.array(h_freq, float)
        if h_freq_rep.size ==1:
          h_freq_rep = f"{h_freq_rep.item():0.2f}"
        cutoff_req = np.array(h_freq - l_trans_bandwidth / 2.0, float)
        if cutoff_req.size == 1:
          cutoff_req = f"{cutoff_req.item():0.2f}"
    h_trans_bandwidth = float(h_trans_bandwidth)
    if np.any(h_trans_bandwidth <= 0):
      raise ValueError(f"h_trans_bandwidth must be positive, got {h_trans_bandwidth}")
    
    h_stop = h_freq + h_trans_bandwidth
    if reverse:
      h_stop -= h_trans_bandwidth
      h_freq -= h_trans_bandwidth
    if np.any(h_stop > sfreq/2):
      raise ValueError(f"h_stop must be less than {sfreq/2}, got {h_stop}")
  
  if isinstance(filter_length, str) and filter_length.lower() == 'auto':
    filter_length = filter_length.lower()
    h_check = l_check = np.inf
    if h_freq is not None:
      h_check = min(np.atleast_1d(h_trans_bandwidth))
    if l_freq is not None:
      l_check = min(np.atleast_1d(l_trans_bandwidth))
    mult_fact = 2.0 if fir_design == 'firwin2' else 1.0
    length_factor = dict(hann= 3.1, hamming=3.3, blackman = 5.0)
    filter_length = f"{length_factor[fir_window]*mult_fact/float(min(h_check, l_check))}s"
    next_pow= None
  else:
    next_pow = isinstance(filter_length, str) and phase == 'zero-double'
  filter_length = to_samples(filter_length, sfreq, phase, fir_design)

  if fir_design == 'firwin' or phase == 'zero':
    filter_length += (filter_length -1) %2
  if filter_length <= 0:
    raise ValueError(f"filter_length must be positive, got {filter_length}")
  
  if next_pow:
    filter_length = 2 ** int(np.ceil(np.log2(filter_length)))
    if fir_design == 'firwin':
      filter_length += (filter_length -1) %2


  return (x,
          sfreq,
          l_freq,
          h_freq,
          l_stop,
          h_stop,
          filter_length,
          phase,
          fir_window,
          fir_design 
          )

def to_samples(filter_length: str, sfreq: float):
  if isinstance(filter_length, str):
    filter_length = filter_length.lower()

    if filter_length.lower().endswith('ms'):
      mult_fact = 1e-3
      filter_length = filter_length[:-2]
    elif filter_length.lower().endswith('s'):
      mult_fact = 1.0
      filter_length = filter_length[:-1]
    else:
      raise ValueError("You suck")
    
    try:
      filter_length = float(filter_length)
    except ValueError:
      raise ValueError("You suck")
    filter_length = max(int(np.ceil(filter_length * mult_fact * sfreq)), 1)
    filter_length += (filter_length -1) % 2
    return float(filter_length)

def prep_for_filtering(x, copy, pick = 1):
  if copy == True:
    x = x.copy()
  original_shape = x.shape
  x = np.atleast_2d(x)
  x.shape = (np.prod(x.shape[:-1]), x.shape[-1])
  return x, original_shape


def next_fast_len(target):
    """Find the next fast size of input data to `fft`, for zero-padding, etc.

    SciPy's FFTPACK has efficient functions for radix {2, 3, 4, 5}, so this
    returns the next composite of the prime factors 2, 3, and 5 which is
    greater than or equal to `target`. (These are also known as 5-smooth
    numbers, regular numbers, or Hamming numbers.)

    Parameters
    ----------
    target : int
        Length to start searching from.  Must be a positive integer.

    Returns
    -------
    out : int
        The first 5-smooth number greater than or equal to `target`.

    """
    from bisect import bisect_left

    hams = (8,9,10,12,15,16,18,20,24,25,27,30,32,36,40,45,48,50,54,60,64,72,75,80,81,90,96,100,108,120,125,128,135,144,150,160,162,180,
            192,200,216,225,240,243,250,256,270,288,300,320,324,360,375,384,400,405,432,450,480,486,500,512,540,576,600,625,640,648,675,
            720,729,750,768,800,810,864,900,960,972,1000,1024,1080,1125,1152,1200,1215,1250,1280,1296,1350,1440,1458,1500,1536,1600,1620,
            1728,1800,1875,1920,1944,2000,2025,2048,2160,2187,2250,2304,2400,2430,2500,2560,2592,2700,2880,2916,3000,3072,3125,3200,3240,
            3375,3456,3600,3645,3750,3840,3888,4000,4050,4096,4320,4374,4500,4608,4800,4860,5000,5120,5184,5400,5625,5760,5832,6000,6075,
            6144,6250,6400,6480,6561,6750,6912,7200,7290,7500,7680,7776,8000,8100,8192,8640,8748,9000,9216,9375,9600,9720,10000,
    )

    if target <= 6:
        return target

    # Quickly check if it's already a power of 2
    if not (target & (target - 1)):
        return target

    # Get result quickly for small sizes, since FFT itself is similarly fast.
    if target <= hams[-1]:
        return hams[bisect_left(hams, target)]

    match = float("inf")  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)

            p2 = 2 ** int(quotient - 1).bit_length()

            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match


#############################################################################################################

#                                         AUGMENTATION METHODOLOGY                                          #
#############################################################################################################
def Gaussian_white_noise(data: np.ndarray, segment: np.ndarray):
  snr_min_db = 8
  snr_max_db = 12

# Calculate the power of the signal based on its variance
  signal_power = np.var(data)
  print('The signal power:',signal_power)

# Calculate the minimum and maximum power of the noise based on the SNR range
  snr_min = 10 ** (snr_min_db / 10)
  snr_max = 10 ** (snr_max_db / 10)

# Generate a random SNR within the specified range
  random_snr = np.random.uniform(snr_min, snr_max)

# Calculate the power of the noise based on the random SNR
  noise_power = signal_power / random_snr

# Generate white Gaussian noise with the same length as the signal
  noise = np.random.normal(0, np.sqrt(noise_power), segment.shape)
  noise = noise.reshape(segment.shape)
# Add the noise to the signal
  signal_with_noise = segment + noise
  return signal_with_noise 

def augmentation_method(data: np.ndarray, labels: np.ndarray, transition_start: int = 5, transition_end: int = 20, batch_size: int = 10):
  values, counts = np.unique(data, return_counts=True)
  desired_samples = max(counts)
  # Calculate the corresponding indices in the signal array for the start and end times
  transition_start_index = int(transition_start * 100)
  transition_end_index = int(transition_end * 100)

  # Create a copy of the signal array
  augmented_signal = data.copy()
  augmented_labels = labels.copy()
  # Loop over each class to balance the samples pass
  for class_label in np.unique(labels):
    # Determine the number of samples for the current class
    class_count = np.sum(labels == class_label)
    if class_count < desired_samples:
      print(f"Aguementing class {class_label}")
      sample_needed  = desired_samples - class_count
      # Find the indices of the current class samples
      class_indices = np.where(labels == class_label)[0]
      # Randomly select existing samples for the transition operation
      selected_indices = np.random.choice(class_indices, size= sample_needed)
      num_batches = int(np.ceil(len(sample_needed) / batch_size))

      for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size
        selected_batch_indices = selected_indices[start_idx:end_idx]
        # Extract the selected segments for the transition operation
        segments = augmented_signal[selected_batch_indices, :]
        # Modify the signal values within the transition range
        transition_duration = transition_end_index - transition_start_index
        transition_range = np.linspace(0, 1, transition_duration)
        segments[:, transition_start_index:transition_end_index] *= np.expand_dims(transition_range, axis=0)
        # Add Gaussian white noise to the segments
        segments = Gaussian_white_noise(augmented_signal, segments)
        # Append the modified segments back into the original data and classes
        augmented_signal = np.concatenate((augmented_signal, segments), axis=0)
        augmented_labels = np.concatenate(augmented_labels, np.full(len(selected_batch_indices), class_label))
      print("Augmentation process done for class ", class_label)

  return augmented_signal, augmented_labels

#############################################################################################################
