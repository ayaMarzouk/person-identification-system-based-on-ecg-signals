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
