#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioSegmentation as aS
import os
import scipy.io.wavfile as wavfile
# speaker diarization
# all audio sessions are taken in 'audio_data' folder.
dia={}
for i in os.listdir('audio_data'):
  dia[i[:3]]=aS.speaker_diarization('audio_data/'+i,-10) #changed aS.speakerDiarization 
  # speaker diarization outputs speaker number for every .5 sec time frame
  c=0
  dic={}
  for j in dia[i[:3]]:
    dic[c]=j
    c+=1
  del(dia[i[:3]])
  dia[i[:3]]=dic #dictionary of audio session number : speaker number list


# In[ ]:


#silence removal, splitting based on speaker
import os

for i in os.listdir('audio_data'):
  [Fs, x] = aIO.read_audioFile('audio_data/'+i) #changed aIO.readAudioFile 
  segments = aS.silence_removal(x, Fs, 0.020, 0.020,
                                   smooth_window=1.0,
                                   weight=0.3,
                                   plot=False) #changed aS.silenceRemoval and smoothWindow 
  for s in segments:
    seg_name = "{:s}_{:.2f}-{:}.wav".format(i[:3], s[0],
                                            str(int((dia[i[:3]][int((Fs*s[0])/3200)]+dia[i[:3]][int((Fs*s[1])/3200)])/2)))
    #splitting each audio sessions on silences
    try:
      wavfile.write('dat_file/'+str(int((dia[i[:3]][int((Fs*s[0])/3200)]+dia[i[:3]][int((Fs*s[1])/3200)])/2))+'/'+seg_name,
                    Fs, x[int(Fs * s[0]):int(Fs * s[1])])
    except:
      wavfile.write('dat_file/others/'+seg_name, Fs, x[int(Fs * s[0]):int(Fs * s[1])])
    


# In[ ]:


#audio to spectrogram
import spectrust
import re

spect = spectrust.Spectrogram(width=512, height=512)
#spectrograms of size 512X512, RGB
for i in os.listdir("dat_file/others/"):
  spect.generate('dat_file/others/'+i,
                 'raw_data_img/'+str(label_data[int(i[:3])])+'/'+re.sub('.wav','',i)+'.jpg')
  #the labels are taken as a dict - label_data={session_name:Class}


# In[ ]:


#random sampling 3000 images of each class
import numpy as np
import cv2

for i in os.listdir('raw_data_img'):
  for c in range(3000):
    files = os.listdir('raw_data_img/'+i)
    random_files = np.random.choice(files, 3000)
  nm=0
  for j in random_files:
    im=cv2.imread('raw_data_img/'+i+'/'+j)
    cv2.imwrite('data_img/'+i+'/'+str(nm)+'.jpg',im)
    nm+=1


# In[ ]:


#split folders into test-val
import split_folders

split_folders.ratio('data_img', output='final_image_data', seed=1337, ratio=(.8,.20))

