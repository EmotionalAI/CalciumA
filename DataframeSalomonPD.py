# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 01:51:12 2019

@author: Krzywy
"""
import numpy as np
import pandas as pd

#Merges Calcium traces (ImageJ) with Behavoural scoring (SalomonCoder open software) creating one dataframe 
#SalomonCoder software: https://solomon.andraspeter.com/

#takes ImageJ output, filters and convert it to deltaF/F format
rawIJ = pd.read_csv('Path/to/Calcium/trace/file.csv', delimiter=',', header=0)
#rawIJ.rename(columns = {" " : "frames"}, inplace=True)
rawIJ.insert(0, 'frames', range(1, 1 + len(rawIJ)))
deltaF_IJ = (rawIJ.filter(regex='Me') - rawIJ.filter(regex='Me').mean()) / rawIJ.filter(regex='Me').mean()
deltaF_IJ.insert(0, "Frames", rawIJ["frames"])
#takes corresponding output form Salomon expands (to match calcium framerate) and converts to frames format
RawSal = pd.read_csv('Path/to/salomon/output/file.csv', sep=';',decimal=",", header=0, skip_blank_lines=True)
RawSal = RawSal.dropna()
DupSal = pd.DataFrame(np.repeat(RawSal.values, 2, axis = 0), columns = RawSal.columns) #depending on framerate of Calcium recording u need to expand according number of times (salomon sapling rate[ms]/calcium sampling rate[ms])
#merges IJ output with Behaviour/Location scoring and saves it.
deltaF_IJ.insert(1, 'Beh', DupSal['Behaviours'])
deltaF_IJ.to_csv('Path/to/save/it/in/file.csv', index=False)
