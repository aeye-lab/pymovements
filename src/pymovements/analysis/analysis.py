# Copyright (c) 2022-2023 The pymovements Project Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This file holds all analysis related functionality.
"""
import numpy as np
from scipy.spatial import distance as distance
from scipy.stats import skew, kurtosis
from scipy.signal import savgol_coeffs
from scipy.special import factorial
from scipy.signal import lfilter


# params:
#   x_dva: raw input of degrees of visual angle for x-axis
#   y_dva: raw input of degrees of visual angle for y-axis
#   n: order of polynomial fit
#   smoothing_window_length: smoothing window length in seconds
#   sampling_rate: sampling rate
def smooth_data(x_dva, y_dva,
                n=2, smoothing_window_length = 0.007,
                sampling_rate = 1000):
    # calculate f, to set up Savitzky-Golar filter setting
    f = np.ceil(smoothing_window_length*sampling_rate)
    if np.mod(f,2)!= 1:
        for i in np.arange(f,100,1):
            if np.mod(i,2) == 1:
                f = i
                break
    if f < 5:
        f = 5
    
    g = np.array([savgol_coeffs(f, n, deriv=d, use='dot') for d in range(n+1)]).T / factorial(np.arange(n+1))
    
    x_smo = lfilter(g[:,0],1, x_dva)
    y_smo = lfilter(g[:,0],1, y_dva)
    
    # calculate the velocities and accelerations
    vel_x = lfilter(g[:,1],1, x_dva) * sampling_rate
    vel_x = vel_x * -1.
    vel_y = lfilter(g[:,1],1, y_dva) * sampling_rate
    vel_y = vel_y * -1.
    vel   = np.sqrt(vel_x**2 + vel_y**2)
    
    acc_x = lfilter(g[:,2],1, x_dva) * sampling_rate**2
    acc_y = lfilter(g[:,2],1, y_dva) * sampling_rate**2
    acc   = np.sqrt(acc_x**2 + acc_y**2)
    
    return {'x_smo':x_smo,
            'y_smo':y_smo,
            'vel_x':vel_x,
            'vel_y':vel_y,
            'vel':vel,
            'acc_x':acc_x,
            'acc_y':acc_y,
            'acc':acc}

# FUNCTION ADOPTED FROM CJH LUDWIG, JANUARY 2002
# This function has a matrix as input that contains four columns: blocknumber, trialnumber, x and y starting positions. 
# It goes through this matrix, normalising each movement and fitting a quadratic, and cubic polynomial on the samples.
# Several metrics of curvature are calculated: Initial deviation (Van Gisbergen et al. 1987), average initial deviation (Sheliga
# et al., 1995), maximum raw deviation (Smit et al., 1990; Doyle and Walker, 2001,2002), and an area-based measure (Doyle and Walker,
# pers. communication). In addition to these existing metrics, we calculate two metrics derived from the curve fitting procedure 


def logical_xor(a, b):
    if bool(a) == bool(b):
        return False
    else:
        return a or b

# function that returns curve metrics for given x- and y- coordinates
# usually called with the x- and y- coordinates of a saccade
#
# params:
#   y_dva: degrees of visual angle (y-axis)
#   x_dva: degrees of visual angle (x-axis)
#   sampling_rate: sampling rate 
def curve_metrics(x_dva,y_dva, sampling_rate):

    metrics = dict()
    x=x_dva
    y=y_dva
    xnorm=[]
    ynorm=[]
   
    if x[-1]>x[0]:
        direction=1 #rightward saccade
    else:
        direction=0 #left saccade
      
    NRsamples=len(x) #number of samples on this trial

    hordisplacement=x[-1]-x[0]
    vertdisplacement=y[0]-y[-1]
    Hstraight=np.sqrt((hordisplacement**2)+(vertdisplacement**2))
    SacAngle=np.arctan2(vertdisplacement,hordisplacement)*(180/np.pi) #calculate the angle of the entire movement
    
    #build up the normalised vectors
    xnorm.append(0) #each movement is normalised so that the starting position coincides with this origin (0,0)
    ynorm.append(0)
    xres=[]
    
    for SampleIndex in np.arange(1,(NRsamples-1),1): #first and last samples never deviate from the straight trajectory!
        hordisplacement= x[SampleIndex]-x[0]
        vertdisplacement= y[0]-y[SampleIndex]
        Hsample=np.sqrt((hordisplacement**2)+(vertdisplacement**2))
        SamAngle=np.arctan2(vertdisplacement,hordisplacement)*180/np.pi
        if SacAngle>SamAngle:
            devdir=1 #clockwise deviation
            DevAngle=SacAngle-SamAngle
        elif SacAngle<SamAngle:
            devdir=-1 #anti-clockwise deviation
            DevAngle=SamAngle-SacAngle
        else:
            devdir=0 #no deviation
            DevAngle=0
            
        Deviation=np.sin(DevAngle*(np.pi/180))*Hsample
        Deviation=Deviation*devdir
        xtrue=np.sqrt((Hsample**2)-(Deviation**2)) #true x-coordinate along the straight line path
        xnorm.append(xtrue)
        ynorm.append(Deviation)
    
    xnorm.append(Hstraight)
    ynorm.append(0)
    # rescale the x-coordinates so that xstart=-1 and xend=1
    for SampleIndex in np.arange(0,NRsamples,1):
        res=-1+((xnorm[SampleIndex]/xnorm[-1])*2)
        xres.append(res)
    
    #    %now calculate the various established curvature metrics
    #    IniDev=atan2(ynorm(3),xnorm(3))*180/pi; %initial deviation metric of Van Gisbergen et al. (1987)
    #    IniAD=mean(ynorm(2:3)); %initial average deviation of Sheliga et al. (1995)
    cPoint = int(np.round(0.005*sampling_rate)) # 5ms
    if cPoint < len(xnorm) or cPoint < 1:
        IniDev=0
        IniAD=0
    if logical_xor(cPoint < len(ynorm),cPoint < 1):
        IniDev=np.arctan2(ynorm[cPoint],xnorm[cPoint])*180/np.pi #initial deviation metric of Van Gisbergen et al. (1987)
        IniAD=np.mean(ynorm[1:cPoint]) #initial average deviation of Sheliga et al. (1995)
    else:
       IniDev=np.arctan2(ynorm[-1],xnorm[-1])*180/np.pi #initial deviation metric of Van Gisbergen et al. (1987)
       IniAD=np.mean(ynorm[1:]) #initial average deviation of Sheliga et al. (1995)
    
    MaxDev = np.max(ynorm)
    MaxIndex = np.argmax(ynorm) #maximum raw deviation (Smit et al., 1990; Doyle and Walker, 2001,2002)
    MinDev = np.min(ynorm)
    MinIndex = np.argmin(ynorm)
    
    if np.abs(MaxDev) > np.abs(MinDev):
        RawDev=MaxDev
        DevIndex=MaxIndex
    elif np.abs(MaxDev) < np.abs(MinDev):
        RawDev=MinDev
        DevIndex=MinIndex
    else:
        if MaxIndex<MinIndex:
            RawDev=MaxDev
            DevIndex=MaxIndex
        else:
            RawDev=MinDev
            DevIndex=MinIndex
            
    RawDev=(RawDev/xnorm[-1])*100
    RawPOC=(xnorm[DevIndex]/xnorm[-1])*100 #raw point of curvature
   
    AreaVector=[] #area based measure (Doyle and Walker, personal communication)
    AreaVector.append(0)
    for AreaIndex in np.arange(1,len(xnorm),1):
        area=(xnorm[AreaIndex]-xnorm[AreaIndex-1])*(ynorm[AreaIndex-1]/2)
        AreaVector.append(area)
   
    CurveArea=(np.sum(AreaVector)/xnorm[-1])*100;
    
    #fit the quadratic function and determine the direction of curvature
    """
    print('############# values: #############')
    print(x)
    print(xres)
    print(y)
    print(ynorm)
    print()
    print()
    """
    pol2 = np.polyfit(xres, ynorm, 2)
    polyval = np.poly1d(pol2)
    ypred2=polyval(xres)
    if pol2[0]<0: #if quadratic coefficient is negative (upward curve), curvature is clockwise
        pol2[0] = np.abs(pol2[0])
    else:
        pol2[0] = pol2[0] * -1 #if quadratic coefficient is positive (downward curve), curvature is anti-clockwise
    
    pol3 = np.polyfit(xres, ynorm, 3) #%derivative of cubic polynomial
    polyval3 = np.poly1d(pol3)
    ypred3=polyval3(xres)
    
    vertdisplacement=ypred3[0]-ypred3[-1]
    Hstraight=np.sqrt((xnorm[-1]**2)+(vertdisplacement**2))
    SacAngle=np.arctan2(vertdisplacement,xnorm[-1])*(180/np.pi)
    
    
    der3=np.polyder(pol3) #derivative of cubic polynomial gives a maximum and minimum
    xder3=[((-1*der3[1])-np.sqrt((der3[1]**2)-(4*der3[0]*der3[2])))/(2*der3[0])]
    xder3.append(((-1*der3[1])+np.sqrt((der3[1]**2)-(4*der3[0]*der3[2])))/(2*der3[0]))
    if ((xder3[0]<xres[0]) or (xder3[0]>xres[-1])): #check whether first maximum/minimum falls within the range of xres
        curve3=0
        POC3=0
    elif not np.all(np.isreal(xder3)):
        curve3=0
        POC3=0
    else:   #%if yes, then calculate curvature
        ymax3=polyval3(xder3[0])
        POC3=(xder3[0]*np.std(xnorm))+np.mean(xnorm)
        POC3=(POC3/xnorm[-1]*100)
        hordisplacement=(xder3[0]*np.std(xnorm))+np.mean(xnorm)
        vertdisplacement= ypred3[0]-ymax3
        Hsample=np.sqrt((hordisplacement**2)+(vertdisplacement**2))
        SamAngle=np.arctan2(vertdisplacement,hordisplacement)*180/np.pi
        if  SacAngle>SamAngle:
            devdir=1
            DevAngle=SacAngle-SamAngle
        elif SacAngle<SamAngle:
            devdir=-1
            DevAngle=SamAngle-SacAngle
        curve3=np.sin(DevAngle*(np.pi/180))*Hsample
        curve3=((curve3*devdir)/xnorm[-1])*100  
    
    # create list
    curve3 = [curve3]
    POC3   = [POC3]
    if ((xder3[1]<xres[0]) or (xder3[1]>xres[-1])): #check whether second maximum/minimum falls within the range of xres
        curve3.append(0)
        POC3.append(0)
    elif not np.all(np.isreal(xder3)):           
        curve3.append(0)
        POC3.append(0)
    else:
        ymax3=polyval3(xder3[1]);
        POC=(xder3[1]*np.std(xnorm))+np.mean(xnorm)
        POC3.append(POC/xnorm[-1]*100)
        hordisplacement=(xder3[1]*np.std(xnorm))+np.mean(xnorm)
        vertdisplacement = ypred3[0]-ymax3
        Hsample=np.sqrt((hordisplacement**2)+(vertdisplacement**2))
        SamAngle=np.arctan2(vertdisplacement,hordisplacement)*180/np.pi
        if SacAngle>SamAngle:
            devdir=1#clockwise deviation
            DevAngle=SacAngle-SamAngle
        elif SacAngle<SamAngle:
            devdir=-1#anti-clockwise deviation
            DevAngle=SamAngle-SacAngle
        curve=np.sin(DevAngle*(np.pi/180))*Hsample
        curve3.append(((curve*devdir)/xnorm[-1])*100)      
    if np.max(np.abs(curve3))> 0:
        MaxDev = np.max(np.abs(curve3))
        MaxIndex = np.argmax(np.abs(curve3))
    else:
        MaxIndex=1
    
    metrics['direction'] = direction
    metrics['IniDev'] = IniDev
    metrics['IniAD'] = IniAD
    metrics['RawDev'] = RawDev
    metrics['RawPOC'] = RawPOC
    metrics['CurveArea'] = CurveArea
    metrics['pol2[0]'] = pol2[0]
    metrics['curve3[0]'] = curve3[0]
    metrics['POC3[0]'] = POC3[0]
    metrics['curve3[1]'] = curve3[1]
    metrics['POC3[1]'] = POC3[1]
    metrics['curve3[MaxIndex]'] = curve3[MaxIndex]
    metrics['POC3[MaxIndex]'] = POC3[MaxIndex]
    return metrics

# creates a feature for a list of values (e.g. mean or standard deviation of values in list)
# params:
#       values: list of values
#       aggregation_function: name of function to be applied to list
# returns:
#       aggregated value
def get_feature_from_list(values, aggregation_function):
    if np.sum(np.isnan(values)) == len(values):
        return np.nan
    if aggregation_function == 'mean':
        return np.nanmean(values)
    elif aggregation_function == 'std':
        return np.nanstd(values)
    elif aggregation_function == 'median':
        return np.nanmedian(values)
    elif aggregation_function == 'skew':
        not_nan_values = np.array(values)[~np.isnan(values)]
        return skew(not_nan_values)
    elif aggregation_function == 'kurtosis':
        not_nan_values = np.array(values)[~np.isnan(values)]
        return kurtosis(not_nan_values)
    else:
        return np.nan
        
        
# compute saccade amplitudes
# params:    
#           saccade_lists: list of list of sccade indexes
#           x_angles: list of angles in x direction
#           y_angles: list of angles in x direction
# returns:
#           list of saccade amplitudes
def get_saccadic_amplitudes(saccade_lists, x_angles=None,
                            y_angles=None):
    vals = []
    for sacc_list in saccade_lists:
        if x_angles is not None:
            amplitude = np.float(distance.euclidean([x_angles[sacc_list[0]], y_angles[sacc_list[0]]],
                                                    [x_angles[sacc_list[-1]], y_angles[sacc_list[-1]]]))
            vals.append(amplitude)
        else:
            vals.append(np.nan)
    return vals


# compute saccade velocities
# params:    
#           saccade_lists: list of list of sccade indexes
#           x_angles: list of angles in x direction
#           y_angles: list of angles in x direction
#           x_vels: list of angular velocities in x direction
#           y_vels: list of angular velocities in y direction
#           aggregation_function: 'max' or 'mean'
#           standardize: flag indicating if we want to standardize the values 100*(speed/(445.9*(1–exp(–0.04844*amplitude–0.1121)))) for mean   
#                                                                             100*(max. speed/(580.4*(1–exp(–0.06771*amplitude–0.1498)))) for max velocity
# returns:
#           list of saccade durations
def get_saccadic_velocities(saccade_lists, x_angles=None,
                            y_angles=None,
                            x_vels=None,
                            y_vels=None,
                            aggregation_function='max',
                            standardize=False):
    vals = []
    for sacc_list in saccade_lists:
        if x_vels is not None:
            velocities = [np.abs(x_vels[sacc_list[i]]) + np.abs(x_vels[sacc_list[i]]) for i in range(len(sacc_list))]
            if aggregation_function == 'max':
                velocity = np.max(velocities)
            elif aggregation_function == 'mean':
                velocity = np.mean(velocities)
            if standardize:
                if x_angles is not None:
                    amplitude = np.float(distance.euclidean([x_angles[sacc_list[0]], y_angles[sacc_list[0]]],
                                                            [x_angles[sacc_list[-1]], y_angles[sacc_list[-1]]]))
                    if aggregation_function == 'max':
                        val = 100 * (velocity / (580.4 * (1 - np.exp(-0.06771 * (amplitude - 0.1498)))))
                        vals.append(val)
                    elif aggregation_function == 'mean':
                        velocity = np.mean(velocities)
                        val = 100 * (velocity / (445.9 * (1 - np.exp(-0.04844 * amplitude - 0.1121))))
                        vals.append(val)
                else:
                    vals.append(np.nan)
            else:
                vals.append(velocity)
        else:
            vals.append(np.nan)
    return vals


## compute vector containing all the saccadic features
# params:    
#           saccade_lists: list of list of sccade indexes
#           x_angles: list of angles in x direction
#           y_angles: list of angles in x direction
#           x_vels: list of angular velocities in x direction
#           y_vels: list of angular velocities in y direction
#           feature_aggregations: list of aggregation functions to apply to list of values
#           feature_prefix: prefix for the featurename
# returns:
#           list of saccade durations
def compute_saccadic_features(saccade_lists, x_angles=None,
                              y_angles=None,
                              x_vels=None, y_vels=None,
                              feature_prefix='',
                              feature_aggregations=['mean', 'std', 'median']):
    if len(feature_prefix) > 0:
        feature_prefix = feature_prefix + '_'

    feature_names = []
    features = []

    # saccadic duration features
    aggregations = ['mean']
    standards = [True, False]
    for aggregation in aggregations:
        for standardize in standards:
            durations = get_saccadic_durations(saccade_lists=saccade_lists, x_angles=x_angles,
                                               y_angles=y_angles,
                                               standardize=standardize)
            for feature_aggregation in feature_aggregations:
                cur_features_suffix = feature_prefix + '_' + feature_aggregation
                if standardize:
                    cur_features_suffix += '_standard'
                cur_features_suffix += '_saccadic_duration_' + aggregation
                feature_names.append(cur_features_suffix)
                features.append(get_feature_from_list(durations, feature_aggregation))

                # saccadic duration features
    aggregations = ['mean']
    standards = [False]
    for aggregation in aggregations:
        for standardize in standards:
            amplitudes = get_saccadic_amplitudes(saccade_lists=saccade_lists, x_angles=x_angles,
                                                 y_angles=y_angles)
            for feature_aggregation in feature_aggregations:
                cur_features_suffix = feature_prefix + '_' + feature_aggregation
                if standardize:
                    cur_features_suffix += '_standard'
                cur_features_suffix += '_saccadic_amplitude_' + aggregation
                feature_names.append(cur_features_suffix)
                features.append(get_feature_from_list(amplitudes, feature_aggregation))

    # saccadic duration velocities
    aggregations = ['mean', 'max']
    standards = [True, False]
    for aggregation in aggregations:
        for standardize in standards:
            velocities = get_saccadic_velocities(saccade_lists=saccade_lists, x_angles=x_angles,
                                                 y_angles=y_angles,
                                                 x_vels=x_vels,
                                                 y_vels=y_vels,
                                                 aggregation_function=aggregation,
                                                 standardize=standardize)
            for feature_aggregation in feature_aggregations:
                cur_features_suffix = feature_prefix + '_' + feature_aggregation
                if standardize:
                    cur_features_suffix += '_standard'
                cur_features_suffix += '_saccadic_velocity_' + aggregation
                feature_names.append(cur_features_suffix)
                features.append(get_feature_from_list(velocities, feature_aggregation))
    return np.array(features), feature_names
    
    
########################################################################################################
#
# compute saccadic feature from paper 'Study of an Extensive Set of Eye Movement Features: Extraction Methods and Statistical Analysis' by Rigas et. al
#             https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7722561/
#
########################################################################################################

#   computes features for a saccade
# params:
#   saccade_list: indexes of saccade
#   pos_h_raw : degrees of visual angle in x-axis (raw)
#   pos_v_raw : degrees of visual angle in y-axis (raw)
#   pos_h : degrees of visual angle in x-axis (smooted)
#   pos_v : degrees of visual angle in y-axis (smooted)
#   vel_h : velocity in degrees of visual angle in x-axis (smooted)
#   vel_v : velocity in degrees of visual angle in y-axis (smooted)
#   acc_h : acceleration in degrees of visual angle in x-axis (smooted)
#   acc_v : acceleration in degrees of visual angle in y-axis (smooted)
#   sampling_rate: sampling rate
#   smoothing_window_length: smoothing window length in seconds
#   min_fixation_duration: minimal fixation duration (in s)
def get_features_for_smoothed_saccade(saccade_list,
                                      pos_h_raw,
                                      pos_v_raw,
                                      pos_h,
                                      pos_v,
                                      vel_h,
                                      vel_v,
                                      acc_h,
                                      acc_v,
                                      sampling_rate=1000,
                                      smoothing_window_length=0.007,
                                      min_fixation_duration=0.030):
    # select saccade
    pos_h_raw = pos_h_raw[saccade_list]
    pos_v_raw = pos_v_raw[saccade_list]
    pos_h = pos_h[saccade_list]
    pos_v = pos_v[saccade_list]
    vel_h = vel_h[saccade_list]
    vel_v = vel_v[saccade_list]
    acc_h = acc_h[saccade_list]
    acc_v = acc_v[saccade_list]

    out_dict = dict()
    # duration (ms)
    out_dict['duration'] = np.max([np.finfo(np.float32).eps, 1000 * (saccade_list[-1] - saccade_list[0]) / sampling_rate])

    # amplitude H, V, R (deg)
    out_dict['amp_v'] = np.abs(pos_h[-1] - pos_h[0])
    out_dict['amp_h'] = np.abs(pos_v[-1] - pos_v[0])
    out_dict['amp_r'] = np.sqrt(out_dict['amp_h'] ** 2 + out_dict['amp_v'] ** 2)

    # saccade horizontal direction
    # +1 -> right, -1 -> left
    out_dict['direction_h'] = np.sign(pos_h[-1] - pos_h[0])

    # saccade travelled distance
    out_dict['trav_dist'] = np.sum(np.sqrt(np.diff(pos_h) ** 2 + np.diff(pos_v) ** 2))

    # saccade efficiancy
    out_dict['efficiency'] = out_dict['amp_r'] / (out_dict['trav_dist'] + np.finfo(np.float32).eps)

    # saccade tail efficiency
    # tail = last 'smoothing_window_length' ms
    if len(pos_h_raw) > smoothing_window_length:
        tail_h = pos_h_raw[len(pos_h_raw) - np.int(np.round(smoothing_window_length * sampling_rate)):]
        tail_v = pos_h_raw[len(pos_v_raw) - np.int(np.round(smoothing_window_length * sampling_rate)):]
    else:
        tail_h = pos_h_raw
        tail_v = pos_h_raw
    
    # if we have really short saccade or really low sampling rate
    if len(tail_h) == 0:
        tail_h = pos_h_raw
        tail_v = pos_h_raw
    
    
    tail_amp = np.sqrt((tail_h[-1] - tail_h[0]) ** 2 + (tail_v[-1] - tail_v[0]) ** 2)
    tail_trav_dist = np.sum(np.sqrt(np.diff(tail_h) ** 2 + np.diff(tail_v) ** 2))
    out_dict['tail_efficiency'] = tail_amp / (np.finfo(float).eps + tail_trav_dist)

    # saccade tail percent incosistent
    # tail last 'smoothing_window_length' ms
    consistent_array = np.zeros([len(tail_h) - 1, 1])
    for m in range(len(tail_h) - 1):
        v1 = np.array([tail_h[m + 1], tail_v[m + 1]]) - np.array([tail_h[m], tail_v[m]])
        v2 = np.array([pos_h_raw[-1], pos_v_raw[-1]]) - np.array([pos_h_raw[0], pos_v_raw[0]])
        Angle = np.arctan2(np.abs(np.linalg.det(np.array([np.transpose(v1), np.transpose(v2)]))),
                           np.dot(np.transpose(v1), np.transpose(v2))) * 180 / np.pi
        if np.abs(Angle) < 60:
            consistent_array[m] = 1

    out_dict['tail_pr_inconsist'] = 100 * (1 - np.sum(consistent_array) / (np.finfo(float).eps + (len(tail_h) - 1)))

    # saccade trajectory curvature features
    if len(pos_h) >= 4:
        metrics = curve_metrics.curve_metrics(pos_h, pos_v, sampling_rate)
    else:
        metrics = dict()
    if 'direction' in metrics:
        out_dict['direction'] = metrics['direction']
    else:
        out_dict['direction'] = np.nan

    if 'IniDev' in metrics:
        out_dict['IniDev'] = metrics['IniDev']
    else:
        out_dict['IniDev'] = np.nan

    if 'IniAD' in metrics:
        out_dict['IniAD'] = metrics['IniAD']
    else:
        out_dict['IniAD'] = np.nan

    if 'RawDev' in metrics:
        out_dict['RawDev'] = metrics['RawDev']
    else:
        out_dict['RawDev'] = np.nan

    if 'RawPOC' in metrics:
        out_dict['RawPOC'] = metrics['RawPOC']
    else:
        out_dict['RawPOC'] = np.nan

    if 'CurveArea' in metrics:
        out_dict['CurveArea'] = metrics['CurveArea']
    else:
        out_dict['CurveArea'] = np.nan

    if 'pol2[0]' in metrics:
        out_dict['pol2[0]'] = metrics['pol2[0]']
    else:
        out_dict['pol2[0]'] = np.nan

    if 'curve3[0]' in metrics:
        out_dict['curve3[0]'] = metrics['curve3[0]']
    else:
        out_dict['curve3[0]'] = np.nan

    if 'POC3[0]' in metrics:
        out_dict['POC3[0]'] = metrics['POC3[0]']
    else:
        out_dict['POC3[0]'] = np.nan

    if 'curve3[1]' in metrics:
        out_dict['curve3[1]'] = metrics['curve3[1]']
    else:
        out_dict['curve3[1]'] = np.nan

    if 'POC3[1]' in metrics:
        out_dict['POC3[1]'] = metrics['POC3[1]']
    else:
        out_dict['POC3[1]'] = np.nan

    if 'curve3[MaxIndex]' in metrics:
        out_dict['curve3[MaxIndex]'] = metrics['curve3[MaxIndex]']
    else:
        out_dict['curve3[MaxIndex]'] = np.nan

    if 'curve3[MaxIndex]' in metrics:
        out_dict['curve3[MaxIndex]'] = metrics['curve3[MaxIndex]']
    else:
        out_dict['curve3[MaxIndex]'] = np.nan

    if 'POC3[MaxIndex]' in metrics:
        out_dict['POC3[MaxIndex]'] = metrics['POC3[MaxIndex]']
    else:
        out_dict['POC3[MaxIndex]'] = np.nan

    # number of velocity local minima
    SVel = np.sqrt(vel_h ** 2 + vel_v ** 2)
    N_localmin = 0;
    for k in range(len(SVel) - 2):
        if SVel[k] > SVel[k + 1] and SVel[k + 2] > SVel[k + 1]:
            N_localmin = N_localmin + 1
    out_dict['num_vel_loc_min'] = N_localmin

    #  peak velocity H, V, R (deg/s)
    out_dict['peak_vel_h'] = np.max(np.abs(vel_h))
    out_dict['peak_vel_v'] = np.max(np.abs(vel_v));
    out_dict['peak_vel_r'] = np.max(np.sqrt(vel_h ** 2 + vel_v ** 2))

    #  mean velocity H, V, R (deg/s)
    out_dict['mean_vel_h'] = out_dict['amp_h'] / (out_dict['duration'] / sampling_rate)
    out_dict['mean_vel_v'] = out_dict['amp_v'] / (out_dict['duration'] / sampling_rate)
    out_dict['mean_vel_r'] = np.sqrt(out_dict['mean_vel_h'] ** 2 + out_dict['mean_vel_v'] ** 2)

    #  velocity profile mean H, V, R (deg/s)
    out_dict['vel_profile_mean_h'] = np.mean(np.abs(vel_h))
    out_dict['vel_profile_mean_v'] = np.mean(np.abs(vel_v))
    out_dict['vel_profile_mean_r'] = np.mean(np.sqrt(vel_h ** 2 + vel_v ** 2))

    #  velocity profile median H, V, R (deg/s)
    out_dict['vel_profile_median_h'] = np.median(np.abs(vel_h))
    out_dict['vel_profile_median_v'] = np.median(np.abs(vel_v))
    out_dict['vel_profile_median_r'] = np.median(np.sqrt(vel_h ** 2 + vel_v ** 2))

    #  velocity profile std H, V, R (deg/s)
    out_dict['vel_profile_std_h'] = np.std(np.abs(vel_h))
    out_dict['vel_profile_std_v'] = np.std(np.abs(vel_v))
    out_dict['vel_profile_std_r'] = np.std(np.sqrt(vel_h ** 2 + vel_v ** 2))

    #  velocity profile skewness H, V, R (deg/s)
    out_dict['vel_profile_skew_h'] = skew(np.abs(vel_h))
    out_dict['vel_profile_skew_v'] = skew(np.abs(vel_v))
    out_dict['vel_profile_skew_r'] = skew(np.sqrt(vel_h ** 2 + vel_v ** 2))

    #  velocity profile kurtosis H, V, R (deg/s)
    out_dict['vel_profile_kurtosis_h'] = kurtosis(np.abs(vel_h))
    out_dict['vel_profile_kurtosis_v'] = kurtosis(np.abs(vel_v))
    out_dict['vel_profile_kurtosis_r'] = kurtosis(np.sqrt(vel_h ** 2 + vel_v ** 2))

    # find liit of acceleration-deceleration phases (via peak velocity for less effect noise)
    vel = np.sqrt(vel_h ** 2 + vel_v ** 2)
    pIdx = np.nanargmax(vel)

    # peak acceleration H, V, R (deg/s^2)
    if pIdx == 0:
        out_dict['peak_acc_h'] = np.max(np.abs(acc_h[0:pIdx + 1]))
        out_dict['peak_acc_v'] = np.max(np.abs(acc_v[0:pIdx + 1]))
        out_dict['peak_acc_r'] = np.max(np.sqrt(acc_h[0:pIdx + 1] ** 2 + acc_v[0:pIdx + 1] ** 2))
    else:
        out_dict['peak_acc_h'] = np.max(np.abs(acc_h[0:pIdx + 1]))
        out_dict['peak_acc_v'] = np.max(np.abs(acc_v[0:pIdx + 1]))
        out_dict['peak_acc_r'] = np.max(np.sqrt(acc_h[0:pIdx + 1] ** 2 + acc_v[0:pIdx + 1] ** 2))

    # peak deceleration H, V, R (deg/s^2)
    if pIdx == len(acc_h) - 1:
        out_dict['peak_deacc_h'] = np.max(np.abs(acc_h[pIdx - 1:]))
        out_dict['peak_deacc_v'] = np.max(np.abs(acc_v[pIdx - 1:]))
        out_dict['peak_deacc_r'] = np.max(np.sqrt(acc_h[pIdx - 1:] ** 2 + acc_v[pIdx - 1:] ** 2))
    else:
        out_dict['peak_deacc_h'] = np.max(np.abs(acc_h[pIdx:]))
        out_dict['peak_deacc_v'] = np.max(np.abs(acc_v[pIdx:]))
        out_dict['peak_deacc_r'] = np.max(np.sqrt(acc_h[pIdx:] ** 2 + acc_v[pIdx:] ** 2))

    # acceleration profile mean H, V, R (deg/s^2)
    out_dict['acc_profile_mean_h'] = np.mean(np.abs(acc_h))
    out_dict['acc_profile_mean_v'] = np.mean(np.abs(acc_v))
    out_dict['acc_profile_mean_r'] = np.mean(np.sqrt(acc_h ** 2 + acc_v ** 2))

    # acceleration profile median H, V, R (deg/s^2)
    out_dict['acc_profile_median_h'] = np.median(np.abs(acc_h))
    out_dict['acc_profile_median_v'] = np.median(np.abs(acc_v))
    out_dict['acc_profile_median_r'] = np.median(np.sqrt(acc_h ** 2 + acc_v ** 2))

    # acceleration profile std H, V, R (deg/s^2)
    out_dict['acc_profile_std_h'] = np.std(np.abs(acc_h))
    out_dict['acc_profile_std_v'] = np.std(np.abs(acc_v))
    out_dict['acc_profile_std_r'] = np.std(np.sqrt(acc_h ** 2 + acc_v ** 2))

    # acceleration profile skew H, V, R (deg/s^2)
    out_dict['acc_profile_skew_h'] = skew(np.abs(acc_h))
    out_dict['acc_profile_skew_v'] = skew(np.abs(acc_v))
    out_dict['acc_profile_skew_r'] = skew(np.sqrt(acc_h ** 2 + acc_v ** 2))

    # acceleration profile kurtosis H, V, R (deg/s^2)
    out_dict['acc_profile_kurtosis_h'] = kurtosis(np.abs(acc_h))
    out_dict['acc_profile_kurtosis_v'] = kurtosis(np.abs(acc_v))
    out_dict['acc_profile_kurtosis_r'] = kurtosis(np.sqrt(acc_h ** 2 + acc_v ** 2))

    # amplitude-duration ratio H, V, R (deg/s)
    out_dict['amp_duration_ratio_h'] = out_dict['amp_h'] / (1 / sampling_rate * out_dict['duration'])
    out_dict['amp_duration_ratio_v'] = out_dict['amp_v'] / (1 / sampling_rate * out_dict['duration'])
    out_dict['amp_duration_ratio_r'] = out_dict['amp_r'] / (1 / sampling_rate * out_dict['duration'])

    # peak velocity-amplitude ratio H, V, R (deg/s/deg)
    out_dict['peak_vel_amp_ratio_h'] = out_dict['amp_h'] / (1 / sampling_rate * out_dict['duration'])
    out_dict['peak_vel_amp_ratio_v'] = out_dict['amp_v'] / (1 / sampling_rate * out_dict['duration'])
    out_dict['peak_vel_amp_ratio_r'] = out_dict['amp_r'] / (1 / sampling_rate * out_dict['duration'])

    # peak velocity-duration ratio (a.k.a. 'saccadic ratio') H, V, R (deg/s^2)
    out_dict['peak_vel_duration_ratio_h'] = out_dict['peak_vel_h'] / (1 / sampling_rate * out_dict['duration'])
    out_dict['peak_vel_duration_ratio_v'] = out_dict['peak_vel_v'] / (1 / sampling_rate * out_dict['duration'])
    out_dict['peak_vel_duration_ratio_r'] = out_dict['peak_vel_r'] / (1 / sampling_rate * out_dict['duration'])

    # peak velocity-mean velocity ratio (a.k.a. 'q-ratio') H, V, R
    out_dict['peak_vel_duration_ratio_h'] = out_dict['peak_vel_h'] / (out_dict['mean_vel_h'] + np.finfo(np.float32).eps)
    out_dict['peak_vel_duration_ratio_v'] = out_dict['peak_vel_v'] / (out_dict['mean_vel_v'] + np.finfo(np.float32).eps)
    out_dict['peak_vel_duration_ratio_r'] = out_dict['peak_vel_r'] / (out_dict['mean_vel_r'] + np.finfo(np.float32).eps)

    # peak velocity-local noise ratio Ratio
    v = np.sqrt(vel_h ** 2 + vel_v ** 2)
    '''
    window_start_idx = int(np.round(saccade_list[0]))
    window_end_idx = int(np.max([0, np.ceil(window_start_idx - min_fixation_duration * sampling_rate)]))
    VelLocNoise = v[window_start_idx:window_end_idx]
    VelLocNoise = np.mean(VelLocNoise) + 3 * np.std(VelLocNoise)
    '''
    VelLocNoise = np.nanmean(v) + 3 * np.nanstd(v)
    out_dict['peak_velocity_loc_noise_ratio_r'] = out_dict['peak_vel_r'] / (VelLocNoise + np.finfo(np.float32).eps)

    # acceleration-deceleration duration ratio
    out_dict['acc_dec_duration_ratio'] = pIdx / (len(vel) - pIdx)

    # peak acceleration-peak deceleration ratio H, V, R
    out_dict['peak_acc_peak_dec_ratio_h'] = out_dict['peak_acc_h'] / (out_dict['peak_deacc_h'] + np.finfo(np.float32).eps)
    out_dict['peak_acc_peak_dec_ratio_v'] = out_dict['peak_acc_v'] / (out_dict['peak_deacc_v'] + np.finfo(np.float32).eps)
    out_dict['peak_acc_peak_dec_ratio_r'] = out_dict['peak_acc_r'] / (out_dict['peak_deacc_r'] + np.finfo(np.float32).eps)

    # ADDITIONAL LOHR FEATURES
    out_dict['pos_trace_sd_h'] = np.std(pos_h)
    out_dict['pos_trace_sd_v'] = np.std(pos_v)
    out_dict['dispersion'] = np.max(pos_h) - np.min(pos_h) + np.max(pos_v) - np.min(pos_v)
    out_dict['sac_angle'] = np.arctan(pos_v[-1] - pos_v[0]) / ((pos_h[-1] - pos_h[0]) + np.finfo(np.float32).eps)
    out_dict['centroid_h'] = np.mean(pos_h)
    out_dict['centroid_v'] = np.mean(pos_v)

    return out_dict


#   computes features for a saccades
# params:
#   saccade_list: list of indices of saccades
#   pos_h_raw : degrees of visual angle in x-axis (raw)
#   pos_v_raw : degrees of visual angle in y-axis (raw)
#   pos_h : degrees of visual angle in x-axis (smooted)
#   pos_v : degrees of visual angle in y-axis (smooted)
#   vel_h : velocity in degrees of visual angle in x-axis (smooted)
#   vel_v : velocity in degrees of visual angle in y-axis (smooted)
#   acc_h : acceleration in degrees of visual angle in x-axis (smooted)
#   acc_v : acceleration in degrees of visual angle in y-axis (smooted)
#   sampling_rate: sampling rate
#   smoothing_window_length: smoothing window length in seconds
#   min_fixation_duration: minimal fixation duration (in s)
def get_features_for_smoothed_saccades(saccade_lists,
                                       pos_h_raw,
                                       pos_v_raw,
                                       pos_h,
                                       pos_v,
                                       vel_h,
                                       vel_v,
                                       acc_h,
                                       acc_v,
                                       sampling_rate=1000,
                                       smoothing_window_length=0.007,
                                       min_fixation_duration=0.030,
                                       feature_prefix='saccade',
                                       feature_aggregations=['mean', 'std', 'median']):
    # print('number of saccades:  ' + str(len(saccade_lists)))

    counter = 0
    for saccade_list in saccade_lists:
        if np.sum(np.isnan(pos_h[saccade_list])) > 0:
            continue
        if counter == 0:
            features = get_features_for_smoothed_saccade(saccade_list,
                                                         pos_h_raw,
                                                         pos_v_raw,
                                                         pos_h,
                                                         pos_v,
                                                         vel_h,
                                                         vel_v,
                                                         acc_h,
                                                         acc_v,
                                                         sampling_rate=sampling_rate,
                                                         smoothing_window_length=smoothing_window_length,
                                                         min_fixation_duration=min_fixation_duration)
            for key in features:
                features[key] = [features[key]]
        else:
            tmp_features = get_features_for_smoothed_saccade(saccade_list,
                                                             pos_h_raw,
                                                             pos_v_raw,
                                                             pos_h,
                                                             pos_v,
                                                             vel_h,
                                                             vel_v,
                                                             acc_h,
                                                             acc_v,
                                                             sampling_rate=sampling_rate,
                                                             smoothing_window_length=smoothing_window_length,
                                                             min_fixation_duration=min_fixation_duration)
            for key in features:
                features[key].append(tmp_features[key])
        counter += 1
    out_features = []
    feature_names = []
    try:
        for key in features:
            cur_list = features[key]
            for aggregation_function in feature_aggregations:
                cur_feature = get_feature_from_list(cur_list, aggregation_function)
                out_features.append(cur_feature)
                feature_names.append(feature_prefix + '_texas_' + aggregation_function + '_' + key)
    except:
        warnings.warn('Warning:no saccades found!')
        out_features = np.array([np.nan for a in range(85*len(feature_aggregations))])
        #raise RuntimeError('no saccades found to process')
    return np.array(out_features), feature_names
    
    
    
########################################################################################################
#
# compute saccadic feature from paper 'Study of an Extensive Set of Eye Movement Features: Extraction Methods and Statistical Analysis' by Rigas et. al
#             https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7722561/
#
########################################################################################################

#   computes features for a fixation
# params:
#   fixation_list: indexes of fixations
#   pos_h_raw : degrees of visual angle in x-axis (raw)
#   pos_v_raw : degrees of visual angle in y-axis (raw)
#   pos_h : degrees of visual angle in x-axis (smooted)
#   pos_v : degrees of visual angle in y-axis (smooted)
#   vel_h : velocity in degrees of visual angle in x-axis (smooted)
#   vel_v : velocity in degrees of visual angle in y-axis (smooted)
#   acc_h : acceleration in degrees of visual angle in x-axis (smooted)
#   acc_v : acceleration in degrees of visual angle in y-axis (smooted)
#   sampling_rate: sampling rate
def get_features_for_smoothed_fixation(fixation_list,
                                      pos_h_raw,
                                      pos_v_raw,
                                      pos_h,
                                      pos_v,
                                      vel_h,
                                      vel_v,
                                      acc_h,
                                      acc_v,
                                      sampling_rate=1000):
    from sklearn import linear_model
    from sklearn.metrics import r2_score                                  
    
    # select saccade
    pos_h_raw = pos_h_raw[fixation_list]
    pos_v_raw = pos_v_raw[fixation_list]
    pos_h = pos_h[fixation_list]
    pos_v = pos_v[fixation_list]
    vel_h = vel_h[fixation_list]
    vel_v = vel_v[fixation_list]
    acc_h = acc_h[fixation_list]
    acc_v = acc_v[fixation_list]
    
    v = np.sqrt(vel_h ** 2 + vel_v ** 2)
    Fix_Vel_P90 = np.percentile(v,90)
    
    out_dict = dict()
    # duration (ms)
    out_dict['duration'] = np.max([np.finfo(np.float32).eps, 1000 * (fixation_list[-1] - fixation_list[0]) / sampling_rate])
    
    # Position Centroid H, V (deg)
    out_dict['PosCentroid_H'] = np.mean(pos_h)
    out_dict['PosCentroid_V'] = np.mean(pos_v)
    
    # Drift Displacement H, V, R (deg)
    out_dict['DriftDisp_H'] = np.abs(pos_h[-1] - pos_h[0])
    out_dict['DriftDisp_V'] = np.abs(pos_v[-1] - pos_v[0])
    out_dict['DriftDisp_R'] = np.sqrt((pos_h[-1]-pos_h[0])**2 + (pos_v[-1]-pos_v[0])**2)
    
    
    # Drift Distance H, V, R (deg)
    out_dict['DriftDist_H'] = np.sum(np.abs(np.diff(pos_h)))
    out_dict['DriftDist_V'] = np.sum(np.abs(np.diff(pos_v)))
    out_dict['DriftDist_R'] = np.sum(np.sqrt(np.diff(pos_h)**2 + np.diff(pos_v)**2))
    
    # Drift Distance H, V, R (deg)
    out_dict['DriftDist_H'] = np.sum(np.abs(np.diff(pos_h)))
    out_dict['DriftDist_V'] = np.sum(np.abs(np.diff(pos_v)))
    out_dict['DriftDist_R'] = np.sum(np.sqrt(np.diff(pos_h)**2 + np.diff(pos_v)**2))
    
    # Drift mean Velocity H, V, R (deg/s)
    out_dict['DriftAvgSpeed_H'] = out_dict['DriftDist_H']/(0.001*out_dict['duration']);
    out_dict['DriftAvgSpeed_V'] = out_dict['DriftDist_V']/(0.001*out_dict['duration']);
    out_dict['DriftAvgSpeed_R'] = out_dict['DriftDist_R']/(0.001*out_dict['duration']); 
    
    # Drift 1-order (linear) fit Slope and R^2 H, V 
    timeData = np.arange(len(pos_h)) / sampling_rate
    lr = linear_model.LinearRegression()
    x = timeData.reshape(-1, 1)
    y = pos_h
    model = lr.fit(x, y)
    pred = model.predict(x)
    r2 = r2_score(pred,y)
    slope = model.coef_[0]
    out_dict['DriftFitLn_Slope_H'] = slope
    out_dict['DriftFitLn_R2_H'] = r2
    x = timeData.reshape(-1, 1)
    y = pos_v
    model = lr.fit(x, y)
    pred = model.predict(x)
    r2 = r2_score(pred,y)
    slope = model.coef_[0]
    out_dict['DriftFitLn_Slope_V'] = slope
    out_dict['DriftFitLn_R2_V'] = r2
    
    
    # Drift 2-order (quadratic) fit R^2 H, V 
    timeData = np.arange(len(pos_h)) / sampling_rate
    timeData = np.array([timeData, np.power(timeData,2)]).T
    x = timeData
    y = pos_h
    model = lr.fit(x, y)
    pred = model.predict(x)
    r2 = r2_score(pred,y)
    out_dict['DriftFitQd_R2_H'] = r2
    x = timeData
    y = pos_v
    model = lr.fit(x, y)
    pred = model.predict(x)
    r2 = r2_score(pred,y)
    out_dict['DriftFitQd_R2_V'] = r2
    
    ''' 
    % Drift step-wise fit parameter percentages H, V
    timeData = (Fix_Start(i):Fix_End(i))'/[Params.samplingFreq];
    Tstep = [timeData power(timeData,2)];
    [~, ~, ~, Xinmodel, ~, ~, ~] = stepwisefit(Tstep, pos_h, 'display', 'off');
    [~, ~, ~, Yinmodel, ~, ~, ~] = stepwisefit(Tstep, pos_v, 'display', 'off');
    Fix_StepLQParam_H(i, 1) = Xinmodel[0]; Fix_StepLQParam_H(i, 2) = Xinmodel(2);
    Fix_StepLQParam_V(i, 1) = Yinmodel[0]; Fix_StepLQParam_V(i, 2) = Yinmodel(2);
   '''
    
    # Velocity Profile Mean H, V, R (deg/s)
    out_dict['VelProfMn_H'] = np.mean(np.abs(vel_h))
    out_dict['VelProfMn_V'] = np.mean(np.abs(vel_v))
    out_dict['VelProfMn_R'] = np.mean(np.sqrt(vel_h**2 + vel_v**2))
    
    # Velocity Profile Median H, V, R (deg/s)
    out_dict['VelProfMd_H'] = np.median(np.abs(vel_h))
    out_dict['VelProfMd_V'] = np.median(np.abs(vel_v))
    out_dict['VelProfMd_R'] = np.median(np.sqrt(vel_h**2 + vel_v**2))
    
    # Velocity Profile Std H, V, R (deg/s)
    out_dict['VelProfSd_H'] = np.std(abs(vel_h))
    out_dict['VelProfSd_V'] = np.std(np.abs(vel_v))
    out_dict['VelProfSd_R'] = np.std(np.sqrt(vel_h**2 + vel_v**2))
    
    # Velocity Profile Skewness H, V, R (deg/s)
    out_dict['VelProfSk_H'] = skew(np.abs(vel_h))
    out_dict['VelProfSk_V'] = skew(np.abs(vel_v))
    out_dict['VelProfSk_R'] = skew(np.sqrt(vel_h**2 + vel_v**2))
    
    # Velocity Profile Kurtosis H, V, R (deg/s)
    out_dict['VelProfKu_H'] = kurtosis(np.abs(vel_h))
    out_dict['VelProfKu_V'] = kurtosis(np.abs(vel_v))
    out_dict['VelProfKu_R'] = kurtosis(np.sqrt(vel_h**2 + vel_v**2))
    
    # Percent Above 90-percentile Velocity Threshold R
    out_dict['PrAbP90VelThr_R'] = 100*np.sum(v > Fix_Vel_P90)/len(v)
    
    '''
    # Percent Crossing 90-percentile Velocity Threshold R
    cross_idx = crossing(v(Fix_Start(i):Fix_End(i)), [], Fix_Vel_P90);
    out_dict['PrCrP90VelThr_R'] = 100*length(cross_idx)/length(v(Fix_Start(i):Fix_End(i)));
    '''
    	
    # Acceleration Profile Mean H, V, R (deg/s^2)
    out_dict['AccProfMn_H'] = np.mean(np.abs(acc_h));
    out_dict['AccProfMn_V'] = np.mean(np.abs(acc_v));
    out_dict['AccProfMn_R'] = np.mean(np.sqrt(acc_h**2 + acc_v**2));
        
    # Acceleration Profile Median H, V, R (deg/s^2)
    out_dict['AccProfMd_H'] = np.median(np.abs(acc_h));
    out_dict['AccProfMd_V'] = np.median(np.abs(acc_v));
    out_dict['AccProfMd_R'] = np.median(np.sqrt(acc_h**2 + acc_v**2));
        
    # Acceleration Profile Std H, V, R (deg/s^2)
    out_dict['AccProfSd_H'] = np.std(np.abs(acc_h));
    out_dict['AccProfSd_V'] = np.std(np.abs(acc_v));
    out_dict['AccProfSd_R'] = np.std(np.sqrt(acc_h**2 + acc_v**2));
        
    # Acceleration Profile Skewness H, V, R (deg/s^2)
    out_dict['AccProfSk_H'] = skew(np.abs(acc_h));
    out_dict['AccProfSk_V'] = skew(np.abs(acc_v));
    out_dict['AccProfSk_R'] = skew(np.sqrt(acc_h**2 + acc_v**2));
        
    # Acceleration Profile Kurtosis H, V, R (deg/s^2)
    out_dict['AccProfKu_H'] = kurtosis(np.abs(acc_h));
    out_dict['AccProfKu_V'] = kurtosis(np.abs(acc_v));
    out_dict['AccProfKu_R'] = kurtosis(np.sqrt(acc_h**2 + acc_v**2));

    return out_dict


#   computes features for a saccades
# params:
#   fixation_lists: list of indices of saccades
#   pos_h_raw : degrees of visual angle in x-axis (raw)
#   pos_v_raw : degrees of visual angle in y-axis (raw)
#   pos_h : degrees of visual angle in x-axis (smooted)
#   pos_v : degrees of visual angle in y-axis (smooted)
#   vel_h : velocity in degrees of visual angle in x-axis (smooted)
#   vel_v : velocity in degrees of visual angle in y-axis (smooted)
#   acc_h : acceleration in degrees of visual angle in x-axis (smooted)
#   acc_v : acceleration in degrees of visual angle in y-axis (smooted)
#   sampling_rate: sampling rate
def get_features_for_smoothed_fixations(fixation_lists,
                                       pos_h_raw,
                                       pos_v_raw,
                                       pos_h,
                                       pos_v,
                                       vel_h,
                                       vel_v,
                                       acc_h,
                                       acc_v,
                                       sampling_rate=1000,
                                       feature_prefix='fixation',
                                       feature_aggregations=['mean', 'std', 'median']):
    # print('number of saccades:  ' + str(len(fixation_list)))

    counter = 0
    for fixation_list in fixation_lists:
        if np.sum(np.isnan(pos_h[fixation_list])) > 0:
            continue
        if counter == 0:
            features = get_features_for_smoothed_fixation(fixation_list,
                                                         pos_h_raw,
                                                         pos_v_raw,
                                                         pos_h,
                                                         pos_v,
                                                         vel_h,
                                                         vel_v,
                                                         acc_h,
                                                         acc_v,
                                                         sampling_rate=sampling_rate)
            for key in features:
                features[key] = [features[key]]
        else:
            tmp_features = get_features_for_smoothed_fixation(fixation_list,
                                                             pos_h_raw,
                                                             pos_v_raw,
                                                             pos_h,
                                                             pos_v,
                                                             vel_h,
                                                             vel_v,
                                                             acc_h,
                                                             acc_v,
                                                             sampling_rate=sampling_rate)
            for key in features:
                features[key].append(tmp_features[key])
        counter += 1
    out_features = []
    feature_names = []
    try:
        for key in features:
            cur_list = features[key]
            for aggregation_function in feature_aggregations:
                cur_feature = get_feature_from_list(cur_list, aggregation_function)
                out_features.append(cur_feature)
                feature_names.append(feature_prefix + '_texas_' + aggregation_function + '_' + key)
    except:
        warnings.warn('Warning:no saccades found!')
        out_features = np.array([np.nan for a in range(49*len(feature_aggregations))])
        #raise RuntimeError('no saccades found to process')
    return np.array(out_features), feature_names



#Gaze entropy measures detect alcohol-induced driver impairment - ScienceDirect
# https://www.sciencedirect.com/science/article/abs/pii/S0376871619302789
# computes the gaze entropy features
# params:
#    fixation_list: list of fixation idx (e.g. by calling get_sacc_fix_lists_dispersion)
#    x_pixel: x-coordinates of data
#    y_pixel: y coordinata of data
#    x_dim: screen horizontal pixels
#    y_dim: screen vertical pixels
#    patch_size: size of patches to use
def get_gaze_entropy_features(fixation_list,
                             x_pixel,
                             y_pixel,
                             x_dim = 1280,
                             y_dim = 1024,
                             patch_size = 64):


    def calc_patch(patch_size,mean):
        return int(np.floor(mean / patch_size))



    def entropy(value):
        return value * (np.log(value) / np.log(2))


    # dictionary of visited patches
    patch_dict = dict()
    # dictionary for patch transitions
    trans_dict = dict()
    pre = None
    for fix_list in fixation_list:
        x_mean = np.mean(x_pixel[fix_list])
        y_mean = np.mean(y_pixel[fix_list])
        patch_x = calc_patch(patch_size,x_mean)
        patch_y = calc_patch(patch_size,y_mean)
        cur_point = str(patch_x) + '_' + str(patch_y)
        if cur_point not in patch_dict:
            patch_dict[cur_point] = 0
        patch_dict[cur_point] += 1
        if pre is not None:
            if pre not in trans_dict:
                trans_dict[pre] = []
            trans_dict[pre].append(cur_point)
        pre = cur_point


    # stationary gaze entropy
    # SGE
    sge = 0
    x_max = int(x_dim / patch_size)
    y_max = int(y_dim / patch_size)
    fix_number = len(fixation_list)
    for i in range(x_max):
        for j in range(y_max):
            cur_point = str(i) + '_' + str(j)
            if cur_point in patch_dict:
                cur_prop = patch_dict[cur_point] / fix_number
                sge += entropy(cur_prop)
    sge = sge * -1
    
    # gaze transition entropy
    # GTE
    gte = 0
    for patch in trans_dict:
        cur_patch_prop = patch_dict[patch] / fix_number
        cur_destination_list = trans_dict[patch]
        (values,counts) = np.unique(cur_destination_list,return_counts = True)
        inner_sum = 0
        for i in range(len(values)):
            cur_val = values[i]
            cur_count = counts[i]
            cur_prob = cur_count / np.sum(counts)
            cur_entropy = entropy(cur_prob)
            inner_sum += cur_entropy
        #print('cur_patch_prop: ' + str(cur_patch_prop))
        #print('inner_sum: ' + str(inner_sum))
        gte += (cur_patch_prop * inner_sum)
    gte = gte * -1
    
    return (np.array([sge,gte],),['fixation_feature_SGE',
                                  'fixation_feature_GTE'])