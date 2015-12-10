import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from biosppy.signals import ecg
from biosppy.signals.ecg import hamilton_segmenter
from biosppy.signals import tools as st


def load_signals(folder, count):
	signals = []
	for i in range(count):
		signals.append(load_all_lines_as_ints(folder+'/'+str(i)+'.pd'))
	return signals

def load_all_lines_as_ints(name):
	fh = open(name, 'r')
	result = []
	i = 0
	for line in fh:
		#if i > 2000:
		#	break
		result.append(float(line))
		i += 1
	fh.close()
	return result

def filter_signal(signal, sampling_rate=360):
    signal = np.array(signal)
    order = int(0.3 * sampling_rate)
    filtered, _, _ = st.filter_signal(signal=signal,
                                      ftype='FIR',
                                      band='bandpass',
                                      order=order,
                                      frequency=[3, 45],
                                      sampling_rate=sampling_rate)
    return filtered

def get_rpeaks(signal, sampling_rate=360):
    signal = np.array(signal)
    filtered = filter_signal(signal)
    rpeaks, = hamilton_segmenter(signal=filtered, sampling_rate=sampling_rate)
    return rpeaks#[1:]

def separate(signal):
    rpeaks = get_rpeaks(signal)
    result = []
    for i in range(len(rpeaks)-1):
        result.append(signal[rpeaks[i]:rpeaks[i+1]])
    return result

def get_diff(rpeaks):
    result = []
    for i in range(len(rpeaks)-1):
        result.append(rpeaks[i+1] - rpeaks[i])
    return result

def upd(count):
    return np.array([0 for x in range(count)])

def get_max(samples):
    return max([len(x) for x in samples])

def get_min(samples):
    return min([len(x) for x in samples])

def fit_for_svm(X, smax):
    Xs = []
    for x in X:
        if len(x) < smax:
            umax = smax - len(x)
            x = np.append(x, upd(umax), 0)
        if len(x) > smax:
            umax = smax
            x = x[:umax]
        Xs.append(x)
    return Xs

def knn_rpeaks(filename):
    clf = KNeighborsClassifier()

    normal = load_signals('normal', 16)
    s_tachycardia = load_signals('sinus_tachycardia', 16)
    s_bradycardia = load_signals('sinus_bradycardia', 16)

    filtered_normal = []
    for signal in normal:
        rpeaks = get_rpeaks(signal)
        res = rpeaks
        filtered_normal.append(res)

    filtered_s_tachycardia = []
    for signal in s_tachycardia:
        rpeaks = get_rpeaks(signal)
        res = rpeaks
        filtered_s_tachycardia.append(res)

    filtered_s_bradycardia = []
    for signal in s_bradycardia:
        rpeaks = get_rpeaks(signal)
        res = rpeaks
        filtered_s_bradycardia.append(res)

    X = filtered_normal + filtered_s_tachycardia + filtered_s_bradycardia

    signal = load_all_lines_as_ints(filename)

    predict = [get_rpeaks(signal)]

    smax = get_min(X + predict)
    X = fit_for_svm(X, smax)
    predict = fit_for_svm(predict, smax)

    y = 	['normal' 	 for x in range(16)]
    y +=	['sinus_tachycardia' for x in range(16)]
    y +=	['sinus_bradycardia' for x in range(16)]

    clf.fit(X, y)

    print(clf.predict(predict))

def knn_rpeaks_max(filename):
    clf = KNeighborsClassifier()

    normal = load_signals('normal', 16)
    s_tachycardia = load_signals('sinus_tachycardia', 16)
    s_bradycardia = load_signals('sinus_bradycardia', 16)

    filtered_normal = []
    for signal in normal:
        rpeaks = get_rpeaks(signal)
        res = [max(get_diff(rpeaks))]
        filtered_normal.append(res)

    filtered_s_tachycardia = []
    for signal in s_tachycardia:
        rpeaks = get_rpeaks(signal)
        res = [max(get_diff(rpeaks))]
        filtered_s_tachycardia.append(res)

    filtered_s_bradycardia = []
    for signal in s_bradycardia:
        rpeaks = get_rpeaks(signal)
        res = [max(get_diff(rpeaks))]
        filtered_s_bradycardia.append(res)

    X = filtered_normal + filtered_s_tachycardia + filtered_s_bradycardia

    signal = load_all_lines_as_ints(filename)

    rpeaks = get_rpeaks(signal)
    predict = [max(get_diff(rpeaks))]

    y = 	['normal' 	 for x in range(16)]
    y +=	['sinus_tachycardia' for x in range(16)]
    y +=	['sinus_bradycardia' for x in range(16)]

    clf.fit(X, y)

    print(clf.predict(predict))

knn_rpeaks_max('samples/samplev15.txt')
knn_rpeaks_max('samples/samplemlii5.txt')
knn_rpeaks_max('normal/20.pd')
knn_rpeaks_max('sinus_tachycardia/20.pd')
knn_rpeaks_max('sinus_bradycardia/20.pd')


"""

clf = KNeighborsClassifier()

#signal = np.loadtxt('sinus_bradycardia/1.pd')
#print get_rpeaks(signal)
#print(filter_signal(signal))
normal = load_signals('normal', 16)
s_tachycardia = load_signals('sinus_tachycardia', 16)
s_bradycardia = load_signals('sinus_bradycardia', 16)

filtered_normal = []
for signal in normal:
    rpeaks = get_rpeaks(signal)
    res = rpeaks
    filtered_normal.append(res)

filtered_s_tachycardia = []
for signal in s_tachycardia:
    rpeaks = get_rpeaks(signal)
    res = rpeaks
    filtered_s_tachycardia.append(res)

filtered_s_bradycardia = []
for signal in s_bradycardia:
    rpeaks = get_rpeaks(signal)
    res = rpeaks
    filtered_s_bradycardia.append(res)

X = filtered_normal + filtered_s_tachycardia + filtered_s_bradycardia
#X = normal + s_tachycardia + s_bradycardia
"""
"""
for x in filtered_normal:
    print(x)
print
for x in filtered_s_tachycardia:
    print(x)
print
for x in filtered_s_bradycardia:
    print(x)
"""
"""
signal = load_all_lines_as_ints('samples/samplev15.txt')
signal = load_all_lines_as_ints('normal/20.pd')
signal = load_all_lines_as_ints('sinus_tachycardia/20.pd')
#signal = load_all_lines_as_ints('sinus_bradycardia/20.pd')
#print ()
#print
predict = [get_rpeaks(signal)]
#print(predict)
smax = get_min(X + predict)
X = fit_for_svm(X, smax)
predict = fit_for_svm(predict, smax)
#out_signal = ecg.ecg(signal=s_bradycardia[1], sampling_rate=360, show=True)
#out_signal = ecg.ecg(signal=signal, sampling_rate=360, show=True)
#print predict
#for x in predict:
#    print len(x)
y = 	['normal' 	 for x in range(16)]
y +=	['sinus_tachycardia' for x in range(16)]
y +=	['sinus_bradycardia' for x in range(16)]
#print(len(X), len(y))

clf.fit(X, y)

print(clf.predict(predict))


"""
"""

#print(out_signal['rpeaks'])
for i in separate(s_bradycardia[0]):
    print(len(i))
#print(normal[0][rpeaks[0]:rpeaks[1]])

#print out_signal
"""
