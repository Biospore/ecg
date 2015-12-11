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

def _test_fit(X, smin=None):
    if not smin:
        smin = get_min(X)
    Xs = []
    flag = True
    for x in X:
        while len(x) > smin:
            if flag:
                x = x[1:]
            else:
                x = x[:-1]
        Xs.append(x)
    return Xs, smin

def test_separate(filename, freq):
    clf = svm.SVC()
    clf = KNeighborsClassifier()
    normal = load_signals('normal', 16)
    s_tachycardia = load_signals('sinus_tachycardia', 16)
    s_bradycardia = load_signals('sinus_bradycardia', 16)
    v_tachycardia = load_signals('ventricular_tachycardia', 16)

    filtered_normal = _test_prepare(normal)
    filtered_s_tachycardia = _test_prepare(s_tachycardia)
    filtered_s_bradycardia = _test_prepare(s_bradycardia)
    filtered_v_tachycardia = _test_prepare(v_tachycardia)

    X = filtered_normal + filtered_s_tachycardia + filtered_s_bradycardia + filtered_v_tachycardia

    y = 	['normal' 	 for x in filtered_normal]
    y +=	['sinus_tachycardia' for x in filtered_s_tachycardia]
    y +=	['sinus_bradycardia' for x in filtered_s_bradycardia]
    y +=    ['ventricular_tachycardia' for x in filtered_v_tachycardia]
    X, smin = _test_fit(X)
    clf.fit(X, y)
    """
    signal1 = load_all_lines_as_ints('rsamples/ecg1sbrad1.txt')
    signal2 = load_all_lines_as_ints('rsamples/ecg2sbrad1.txt')
    predict1 = _test_prepare([signal1], 360)
    fitted1, _ = _test_fit(predict1, smin)
    predict2 = _test_prepare([signal2], 360)
    fitted2, _ = _test_fit(predict2, smin)
    """
    signal = load_all_lines_as_ints(filename)
    predict = _test_prepare([signal], freq)
    fitted, _ = _test_fit(predict, smin)

    result = []
    for f in fitted:
        if len(f) < smin:
            continue
        result.append(clf.predict(f)[0])

    return result

def _test_prepare(signals, freq=360):
    result = []
    for signal in signals:
        rpeaks = get_rpeaks(signal, freq)[1:]
        for c in range(len(rpeaks) - 1):
            result.append(signal[rpeaks[c]:rpeaks[c+1]])
    return result

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

def prepare(func, signals):
    result = []
    for signal in signals:
        rpeaks = get_rpeaks(signal)
        result.append([func(get_diff(rpeaks))])
    return result

def train_max(clf):
    normal = load_signals('normal', 16)
    s_tachycardia = load_signals('sinus_tachycardia', 16)
    s_bradycardia = load_signals('sinus_bradycardia', 16)
    v_tachycardia = load_signals('ventricular_tachycardia', 16)

    filtered_normal = prepare(max, normal)
    filtered_s_tachycardia = prepare(max, s_tachycardia)
    filtered_s_bradycardia = prepare(max, s_bradycardia)
    filtered_v_tachycardia = prepare(max, v_tachycardia)

    X = filtered_normal + filtered_s_tachycardia + filtered_s_bradycardia + filtered_v_tachycardia

    y = 	['normal' 	 for x in range(16)]
    y +=	['sinus_tachycardia' for x in range(16)]
    y +=	['sinus_bradycardia' for x in range(16)]
    y +=    ['ventricular_tachycardia' for x in range(16)]

    clf.fit(X, y)
    return clf

def train_min(clf):
    normal = load_signals('normal', 16)
    s_tachycardia = load_signals('sinus_tachycardia', 16)
    s_bradycardia = load_signals('sinus_bradycardia', 16)
    v_tachycardia = load_signals('ventricular_tachycardia', 16)

    filtered_normal = prepare(min, normal)
    filtered_s_tachycardia = prepare(min, s_tachycardia)
    filtered_s_bradycardia = prepare(min, s_bradycardia)
    filtered_v_tachycardia = prepare(min, v_tachycardia)


    X = filtered_normal + filtered_s_tachycardia + filtered_s_bradycardia + filtered_v_tachycardia

    y = 	['normal' 	 for x in range(16)]
    y +=	['sinus_tachycardia' for x in range(16)]
    y +=	['sinus_bradycardia' for x in range(16)]
    y +=    ['ventricular_tachycardia' for x in range(16)]

    clf.fit(X, y)
    return clf

def knn_rpeaks_max(filename, classifier, sampling_rate=360):
    clf = classifier
    signal = load_all_lines_as_ints(filename)

    rpeaks = get_rpeaks(signal, sampling_rate)
    predict = [max(get_diff(rpeaks))]

    return clf.predict(predict)[0]

def knn_rpeaks_min(filename, classifier, sampling_rate=360):
    clf = classifier

    signal = load_all_lines_as_ints(filename)

    rpeaks = get_rpeaks(signal, sampling_rate)
    predict = [min(get_diff(rpeaks))]

    return clf.predict(predict)[0]

class Classifier:
    clf = None
    def __init__(self):
        self.clf = KNeighborsClassifier()

    def fit(self, X, y):
        return self.clf.fit(X, y)

    def predict(self, z):
        return self.clf.predict(z)


#knn_rpeaks_max('samples/samplev110.txt', 360)    #V[x] or ECG1     #works well
#knn_rpeaks_max('samples/samplemlii10.txt', 360)  #MLII or ECG2
#knn_rpeaks_min('samples/samplev110.txt', 360)    #V[x] or ECG1     #works well
#knn_rpeaks_min('samples/samplemlii10.txt', 360)  #MLII or ECG2
#knn_rpeaks('samples/samplev110.txt', 360)    #V[x] or ECG1     #works well
#knn_rpeaks('samples/samplemlii10.txt', 360)  #MLII or ECG2
#knn_rpeaks_max('normal/20.pd')
#knn_rpeaks_max('sinus_tachycardia/20.pd')
#knn_rpeaks_max('sinus_bradycardia/20.pd')
