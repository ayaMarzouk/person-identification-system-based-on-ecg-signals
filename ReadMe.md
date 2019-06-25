# ECG Signal Classification

# 1) Introduction:
- Biometrics is a scientific procedure for person identification, one of
those treats is Electrocardiogram (ECG), It is a recording of the
electrical activity of the heart over time. The essential Components we
are are interested are: P, QRS and T waves, which occur in this
temporal order. The physiological and geometrical differences of the
heart among individuals reveal certain uniqueness in their ECG
signals, Thus, it can be used as a tool for person identity detection.

# 2) Input ECG Signal:
![Input Signal](https://drive.google.com/open?id=1eGVmNe73SopIqJETl6ljZdGgzFWA9Lwi)

# 3) Preprocessing Phase:

3.1) Mean Removal:
- We start with calculating the mean value along with the whole signal, then
subtracting that value from each Data instance in the signal.
Why?
- To remove any shifting effect occurs along with the signal.

# 3.2 Bandpass Filter:
Why?
In order to record an ECG signal, electrodes (transducers) are placed at
specific positions on the human body. Artifacts are the unwanted signals
(Noises) that are merged with the ECG signal and sometimes distort it.
Hence, it is necessary to remove those noises or at least reduce them as
much as possible.
- For the sake of reducing the signal noises we used Bandpass Filter :
With the following Parameters:
1) High Cutoff Frequency = 40 Hz
2) Low Cutoff Frequency = 1 Hz
3) Sampling Frequency = 1000 Hz
4) Filter Order:
First Trail with Order = 5, make a huge distortion to the signal, as shown in the
Picture:

![Result or filter with order=5](https://drive.google.com/open?id=1AGrgQWlRfyQg-ThumOmXpRtHqM-vCAWx)

- On the next trail, we decided to go with the Order = 1 & Order =2 for it
achieve both Noise reduction with keeping the main components of the
signal unharmed.
![Result of filter = 1 & 2 ](https://drive.google.com/open?id=1UbO8CNlHnHLBrq9Bw8gE_m-g2R_zM2Vc)

# 3.3 Normalization Phase:
- In this phase, we intend to transform all the training data values (samples)
into a specific range -in our case from (0-1)
Why?
In the machine learning track which we’re using in the training and testing
phase in our project, it’s preferable to cast all the features values to be
around the same range. For the model can correctly interpret the change that
happens for the signal to another even that they may have a wide difference
range of values.

# 3.4 Segmentation Phase:
- In our project, each ECG signal has more than 20 k samples every 205
samples represent one heartbeat, So in order to add up more meaningful
training samples for each person, we divide the whole signal into around 28
segments, Each segment contains 4 heartbeats.
How?
- The first approach was ​manual ​by dividing the end of the signal by
(​4​ (heartbeats counts)* ​205 ​(end of heartbeat cycle))
That was a naive way and may not be accurate.
- The second approach was ​automated one ​using ​QRS detection for ECG
signals:

![The Result](https://drive.google.com/open?id=1OVKZ-JRLeoGN07t2Un_YzqXPeI0bOdhD)

- After detecting QRs, we calculate the distance between each one and it’s
following, afterward, we take the median of the final distance to avoid any
outliers, for there are some misses in the detector. For calculating the final
result (Interval of 4 heartbeats).

# 4) Feature Extraction Phase:
- In this phase, we processed the outcome signal of the previous phase
To extract a feature vector for each segment and concatenate all feature
vectors into one matrix and pass it to the machine learning classifiers for
training and testing phases, This phase is broken into two main stages:

# 4.1) Autocorrelation:
- The Autocorrelation technique can be used to detect repeats or periodicity in
the signal or if it is purely random and it can identify signals that are
embedded signals or noise.

![ Result of autocorrelation](https://drive.google.com/open?id=1OVKZ-JRLeoGN07t2Un_YzqXPeI0bOdhD)

# 4.2) Discrete Cosine Transform (DCT):
- In this step of feature extraction, we aim to compress the output segment out
of the previous step of auto-correlation and right before appending the
output vector to Feature matrix for the whole signal.

- And this was the final step before starting the training phase using
different machine learning algorithms (​SVM, KNN, GaussianNB​)

# 5) Testing Phase:
- After finishing the training model for each one of the classifier, we used
each model to predict to whom the input heartbeat signal is belonging and
here are the winning scores:

