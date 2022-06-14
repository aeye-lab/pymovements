from typing import Optional, Sequence, Tuple

import numpy as np

from pymovements.transformations.velocity import pos2vel
from pymovements.typing import ArrayLike


def microsaccades(
        x: ArrayLike,
        v: Optional[ArrayLike] = None,
        eta: Optional[Sequence[float]] = None,
        lam: float = 6,
        min_duration: int = 6,
        sampling_rate: float = 1000,
        pos2vel_method: Optional[str] = 'smooth',
        sigma_method: Optional[str] = 'engbert2015',
        min_eta: float = 1e-10,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Compute (micro-)saccades from raw samples
    adopted from Engbert et al Microsaccade Toolbox 0.9
    von Hans Trukenbrod empfohlen fuer 1000Hz: lam=6, min_duration=6

    :param x: array of shape (N,2) (x und y screen or visual angle coordinates
        of N samples in *chronological* order)
    :param v: TODO
    :param lam: lambda-factor for relative velocity threshold computation
    :param mindur: minimal saccade duration
    :param sampling_rate: sampling frequency of the eyetracker in Hz
    :param threshold: if None: data-driven velocity threshold; if tuple of
        floats: used to compute elliptic threshold
    :param pos2vel_method: TODO
    :param sigma_method: TODO
    :returns:
        - sac - list of arrays of shape (7,): (1) saccade onset, (2) saccade
            offset, (3) peak velocity, (4) horizontal component (dist from first
            to last sample of the saccade), (5) vertical component,
            (6) horizontal amplitude (dist from leftmost to rightmost sample),
            (7) vertical amplitude
        - issac - array of shape (N,): codes whether a sample of x belongs to
            saccade (1) or not (0)
        - radius - horizontal semi-axis of elliptic threshold; vertical
            semi-axis
    """
    x = np.array(x)
    
    if v is None:
        v = pos2vel(x, sampling_rate=sampling_rate, method=pos2vel_method)
    else:
        v = np.array(v)
        if x.shape != v.shape:
            raise ValueError('x.shape and v.shape do not match')
            
    if eta is None:
        eta = compute_sigma(v, method=sigma_method)
    else:
        if len(threshold) != 2:
            raise ValueError('threshold needs to be two-dimensional')
        eta = np.array(eta)

    if (eta < min_eta).any():
        raise ValueError(
            f'Threshold eta does not provide enough variance'
            f' ({eta} < {min_eta})')

    # radius of elliptic threshold
    radius = lam * eta

     # test is <1 iff sample within ellipse
    test = np.power((v[:, 0] / radius[0]), 2) + np.power((v[:, 1] / radius[1]), 2)
    print(f'{test[765]:.100f}')
    print(f'{test[766]:.100f}')
    print(f'{test[767]:.100f}')
    # indices of candidate saccades
    # runtime warning because of nans in test
    # => is ok, the nans come from nans in x
    indx = np.where(np.greater(test,1))[0]
    
    # Initialize saccade variables
    N = len(indx)  # number of saccade candidates
    nsac = 0
    sac = []
    dur = 1
    a = 0  # (potential) saccade onset
    k = 0  # (potential) saccade offset, will be looped over
    issac = np.zeros(len(x)) # codes if row in x is a saccade

    print(indx)

    # Loop over saccade candidates
    while k < N - 1:
        # check for ongoing saccade candidate and increase duration by one
        if indx[k + 1] - indx[k] == 1:
            dur += 1

        # else saccade has ended
        else:
            # check minimum duration criterion (exception: last saccade)
            if dur  >= min_duration:
                nsac += 1
                s = np.zeros(7)  # entry for this saccade
                s[0] = indx[a]  # saccade onset
                s[1] = indx[k]  # saccade offset
                sac.append(s)
                # code as saccade from onset to offset
                issac[indx[a]:indx[k]+1] = 1
                
            a = k + 1  # potential onset of next saccade
            dur = 1  # reset duration  
        k += 1

    # Check minimum duration for last microsaccade
    if dur >= min_duration:
        nsac += 1
        s = np.zeros(7) # entry for this saccade
        s[0] = indx[a] # saccade onset
        s[1] = indx[k] # saccade offset
        sac.append(s)
        # code as saccade from onset to offset
        issac[indx[a]:indx[k]+1] = 1

    sac = np.array(sac)

    if nsac > 0:
        # Compute peak velocity, horizontal and vertical components
        for s in range(nsac): # loop over saccades
            # Onset and offset for saccades
            a = int(sac[s,0]) # onset of saccade s
            b = int(sac[s,1]) # offset of saccade s
            idx = range(a,b+1) # indices of samples belonging to saccade s
            print(list(idx))
            # Saccade peak velocity (vpeak)
            sac[s,2] = np.max(np.sqrt(np.power(v[idx,0],2) + np.power(v[idx,1],2)))
            # saccade length measured as distance between first (onset) and last (offset) sample
            sac[s,3] = x[b,0]-x[a,0] 
            sac[s,4] = x[b,1]-x[a,1] 
            # Saccade amplitude: saccade length measured as distance between leftmost and rightmost (bzw. highest and lowest) sample 
            minx = np.min(x[idx,0]) # smallest x-coordinate during saccade
            maxx = np.max(x[idx,0]) 
            miny = np.min(x[idx,1])
            maxy = np.max(x[idx,1])
            signx = np.sign(np.where(x[idx,0]==maxx)[0][0] - np.where(x[idx,0]==minx)[0][0]) # direction of saccade; np.where returns tuple; there could be more than one minimum/maximum => chose the first one
            signy = np.sign(np.where(x[idx,1]==maxy)[0][0] - np.where(x[idx,1]==miny)[0][0]) #
            sac[s,5] = signx * (maxx-minx) # x-amplitude
            sac[s,6] = signy * (maxy-miny) # y-amplitude

    return sac, issac, radius
    

def compute_sigma(v: np.ndarray, method='engbert2015'):
    """
    Compute variation in velocity (sigma) by taking median-based std of x-velocity

    engbert2003:
    Ralf Engbert and Reinhold Kliegl: Microsaccades uncover the orientation of
    covert attention

    TODO: add detailed descriptions of all methods
    """
    # TODO: use axis instead of explicit x/y-operations

    if method == 'std':
        thx = np.nanstd(v[:,0])
        thy = np.nanstd(v[:,1])

    elif method == 'mad':
        thx = np.nanmedian(np.absolute(v[:,0] - np.nanmedian(v[:,0])))
        thy = np.nanmedian(np.absolute(v[:,1] - np.nanmedian(v[:,1])))
    
    elif method == 'engbert2003':
        thx = np.sqrt(np.nanmedian(np.power(v[:,0], 2))
                      - np.power(np.nanmedian(v[:,0]), 2))
        thy = np.sqrt(np.nanmedian(np.power(v[:,1], 2))
                      - np.power(np.nanmedian(v[:,1]), 2))

    elif method == 'engbert2015':
        thx = np.sqrt(np.nanmedian(np.power(v[:,0] - np.nanmedian(v[:,0]), 2)))
        thy = np.sqrt(np.nanmedian(np.power(v[:,1] - np.nanmedian(v[:,1]), 2)))

    else:
        valid_methods = ['std', 'mad', 'engbert2003', 'engbert2015']
        raise ValueError(
            'Method "{method}" not implemented. Valid methods: {valid_methods}')

    return np.array([thx, thy])

