import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from scipy import ndimage as ndi
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops
from scipy.misc import imread
from skimage.transform import resize
from skimage.color import rgb2lab
import os
import warnings
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from collections import defaultdict
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

'''Convolves complex kernel with image'''
def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)

'''
Calculates Gabor filter. Based on pseudocode from:
https://en.wikipedia.org/wiki/Gabor_filter
'''
def gabor(sigma, theta, f, psi, gamma):
##    sigma = sigma / f
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3 # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation 
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.exp(1j * 2 * np.pi * f * x_theta + psi)
    return gb

'''
Args:
    - image: an m x n or m x n x 3 (RGB) array
    - thetas: a k-length vector of orientations in radians
    - freqs: an l-length vector of frequencies
    
Returns l x k x m x n array of Gabor filter responses for each orientation and frequency pair
'''
def gabor_filter(image, thetas, freqs):
    if len(image.shape) > 2:
        image = rgb2gray(image)
    image = rescale_intensity(image)
    gabor_features = np.zeros((freqs.shape[0], thetas.shape[0], image.shape[0], image.shape[1]))
    for i in range(freqs.shape[0]):
        for j in range(thetas.shape[0]):
            gabor_features[i, j, :, :] = power(image, gabor(2*np.pi, thetas[j], freqs[i], 0, 1))
    return gabor_features

'''
Args:
    - image: an m x n or m x n x 3 (RGB) array
    - pairs: a k-length list of (orientation, frequency) tuples
    
Returns k x m x n array of Gabor filter responses for each orientation and frequency pair
'''
def gabor_filter_pair(image, pairs):
    if len(image.shape) > 2:
        image = rgb2gray(image)
    image = rescale_intensity(image)
    gabor_features = np.zeros((len(pairs), image.shape[0], image.shape[1]))
    for i, (f, t) in enumerate(pairs):
        gabor_features[i, :, :] = power(image, gabor(2*np.pi, t, f, 0, 1))
    return gabor_features

'''
Args:
    - images: an i x j x k x l array, representing a set of i x j images, each of which has size k x l
    - m: the number of blocks along the first dimension of each k x l image
    - n: the number of blocks along the second dimension of each k x l image
    
Returns vector of local Gabor binary pattern histograms for all images, in which each image is split into m x n blocks
'''
def lgbphs(images, m, n):
    rows = images.shape[2] / m
    cols = images.shape[3] / n
    hs = []
    num_bins = 100
    for i in range(images.shape[0]):
        for j in range(images.shape[1]):
            lbp = local_binary_pattern(images[i, j, :, :], 8, 1) / 256
            for k in range(m):
                for l in range(n):
                    startRow = int(k*rows)
                    endRow = int(min((k+1)*rows,lbp.shape[0]))
                    startCol = int(l*cols)
                    endCol = int(min((l+1)*cols, lbp.shape[1]))
                    hs.append(np.histogram(lbp[startRow:endRow, startCol:endCol], bins=num_bins, range=(0, 1))[0] / float(num_bins))
    return np.concatenate(hs)

'''
Args:
    - image: an m x n x 3 array
    - size: a tuple of histogram sizes for all 3 channels

Returns color histogram normalized to a sum of 1
'''
def color_histogram(image, size):
    hist = np.zeros(size)
    r = image[:,:,0]
    g = image[:,:,1]
    b = image[:,:,2]
    for i in range(hist.shape[0]):
        for j in range(hist.shape[1]):
            for k in range(hist.shape[2]):
                hist[i,j,k] = np.sum((r >= i*256/hist.shape[0]) & (r < (i+1)*256/hist.shape[0]) &
                                     (g >= j*256/hist.shape[1]) & (g < (j+1)*256/hist.shape[1]) &
                                     (b >= k*256/hist.shape[2]) & (b < (k+1)*256/hist.shape[2]))
    return hist / image.size

'''
Args
    - im: an h x w x 3 array representing an image
    - m: the number of blocks along the first dimension of the image
    - n: the number of blocks along the second dimension of the image

Returns a 1 x 3 x (3*m*n) matrix of color metrics for each block of each component of the iamge
'''
def color_metrics(im, m, n):
	rowHeight = im.shape[0] // m
	colHeight = im.shape[1] // n
	imComps = [im[:,:,comp] for comp in range(im.shape[2])]
	metrics = []
	for imComp in imComps:
		compMetrics = []
		blockMetrics = []
		for row in range(m):
			for col in range(n):
				rowStart, colStart = row * rowHeight, col * colHeight
				rowEnd, colEnd = (row+1) * rowHeight, (col+1) * colHeight
				imBlock = imComp[rowStart:rowEnd, colStart:colEnd]
				blockMean = np.mean(imBlock)
				blockVar = np.var(imBlock)
				blockSkewness = (np.sum((imBlock - blockMean)**3) / (imBlock.shape[0]*imBlock.shape[1]))/ (blockVar**1.5)
				blockMetrics.append(np.array([blockMean, blockVar, blockSkewness]))
		compMetrics = np.vstack(blockMetrics)
		metrics.append(compMetrics)
	return np.array([np.array(metrics)])

'''
Args:
    - image: an m x n or m x n x 3 (RGB) array

Returns a feature vector representation of image
'''
def extract_features(image):
    features = []
    
    freqs = np.array([0.98, 1.44])
    thetas = np.linspace(0, np.pi, 5)[:-1]
    features.append(lgbphs(gabor_filter(image, thetas, freqs), 3, 3))
    
# Sample calls for including other features
#    features.append(gabor_filter(image, thetas, freqs).reshape((-1,)))
#    features.append(gabor_filter_pair(image, [(0.98, 0), (0.98, np.pi/2), (1.44, np.pi/4), (1.44, 3*np.pi/4)]).reshape((-1,)))
#    features.append(apply_gabor(image, freqs, thetas).reshape((-1,)))
#    features.append(glcm_props(image, 3, 3, [1], thetas, ['contrast', 'homogeneity', 'energy', 'correlation', 'entropy']).reshape((-1,)))
#    features.append(np.fft.fft2(rescale_intensity(rgb2gray(image))).reshape((-1,)))
    if len(image.shape) > 2:
        features.append(color_histogram(image, (8, 8, 8)).reshape((-1,)))

    return np.concatenate(features)

datafile = os.path.join(os.path.dirname(__file__), 'data.pkl')
labelfile = os.path.join(os.path.dirname(__file__), 'labels.pkl')
try:
	with open(datafile, 'rb') as f:
		X = pickle.load(f)
	with open(labelfile, 'rb') as f:
		Y = pickle.load(f)
except:
	image_dir = os.path.join(os.path.dirname(__file__), 'images')
	infected_dir = os.path.join(image_dir, 'infected')
	not_infected_dir = os.path.join(image_dir, 'not_infected')
	infectedImagePaths = [os.path.join(infected_dir, path) for path in os.listdir(infected_dir)]
	uninfectedImagePaths = [os.path.join(not_infected_dir, path) for path in os.listdir(not_infected_dir)]
	#dimSums = np.array([0,0])
	#for path in infectedImagePaths + uninfectedImagePaths:
	#	im = imread(path)
	#	dimSums += im.shape[:2]
	#targetShape = np.around(dimSums / 4 * (len(infectedImagePaths) + len(uninfectedImagePaths))).astype(int)
	targetShape = np.array([340, 160])
	X, Y = [], []
	Xnorm = []
	totalImages = len(infectedImagePaths) + len(uninfectedImagePaths)
	i = 0
	for path in infectedImagePaths:
		if i % 10 == 0:
			print(i, 'of ', totalImages, 'images')
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			im = imread(path)[:,:,:3]
			im = np.around(resize(im, targetShape, preserve_range=True)).astype(int)
			im = rgb2lab(im)
			feat = extract_features(im)
		Xnorm.append(color_metrics(im, 6, 3 ))
		X.append(feat)
		Y.append(True)
		i += 1
	for path in uninfectedImagePaths:
		if i % 10 == 0:
			print(i, 'of ', totalImages, 'images')
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			im = imread(path)[:,:,:3]
			im = np.around(resize(im, targetShape, preserve_range=True)).astype(int)
			feat = extract_features(im)
		Xnorm.append(color_metrics(im, 6, 3))
		X.append(feat)
		Y.append(False)
		i += 1
	X = np.vstack(X)
	Y = np.array(Y)
	Xnorm = np.vstack(Xnorm)
	#means = np.mean(Xnorm, axis=(0,2))
	#stds = np.std(Xnorm, axis=(0,2))
	#for i in range(Xnorm.shape[1]):
	#	for j in range(Xnorm.shape[3]):
	#		Xnorm[:, i, :, j] = (Xnorm[:,i,:,j] - means[i,j])/stds[i,j]
	numImages, numColors, numBlocks, numFeatures = Xnorm.shape
	Xnorm = Xnorm.reshape(numImages, numColors * numBlocks * numFeatures)
	X = np.hstack([X, Xnorm])
	X = scale(X, axis=0)
	with open(datafile, 'wb') as f:
		pickle.dump(X, f, pickle.HIGHEST_PROTOCOL)
	with open(labelfile, 'wb') as f:
		pickle.dump(Y, f, pickle.HIGHEST_PROTOCOL)

numSplits = 100
accs = []
relevances = []
rates = np.zeros((0, 4))
rocAucs = []
rocCurves = []
usedFeatures = defaultdict(lambda: [])
intercepts = []
for i in range(numSplits):
    #XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.25)
    numImages = X.shape[0]
    numTrain = int(0.75 * numImages)
    split = np.random.permutation(numImages)
    trainIndices, testIndices = split[:numTrain], split[numTrain:]
    XTrain, YTrain = X[trainIndices], Y[trainIndices]
    XTest, YTest = X[testIndices], Y[testIndices]
    numTestPositives = np.sum(YTest)

    model = LogisticRegression(penalty='l1', solver='liblinear')
    model.fit(XTrain, YTrain)

    predictions = model.predict(XTest)
    #predictions = pd.Series(model.predict(test_data))
    #predictions.index = test_data.index
    #print('Features:', XTrain.shape)
    #print(np.sum(YTest), YTest[predictions != YTest])
    #print('Accuracy:', model.score(XTest, YTest))
    
    acc = model.score(XTest, YTest)
    accs.append(acc)
    if acc != 1:
        wrong = predictions != YTest
        wrongIndices = np.arange(predictions.shape[0])[wrong]
        print('Wrongly classified images:', split[numTrain + wrongIndices])
    relevantFeatures = (model.coef_ != 0).flatten()
    numRelevantFeatures = np.sum(relevantFeatures)
    relevances.append(numRelevantFeatures)
    used = np.arange(1, XTest.shape[1] + 1)[relevantFeatures]
    for feature in used:
        usedFeatures[feature].append(model.coef_[0,feature-1])
    confMat = confusion_matrix(YTest, predictions)
    totNo, totYes = np.sum(confMat, axis=1)
    #tn, fp, fn, tp
    rates = np.vstack((rates, confMat.ravel()/np.array([totNo, totNo, totYes, totYes])))
    rocAucs.append(roc_auc_score(YTest, predictions))
    rocCurves.append(roc_curve(YTest, predictions, drop_intermediate=False))
    intercepts.append(model.intercept_)

accs = np.array(accs)
minAcc, maxAcc, medAcc = np.amin(accs), np.amax(accs), np.median(accs)
print('Median Accuracy:', medAcc, '\nAccuracy Range: (', minAcc, '-', maxAcc, ')')
relevances = np.array(relevances)
minRel, maxRel, medRel = np.amin(relevances), np.amax(relevances), np.median(relevances)
print('Median Number of Relevant Features:', medRel, '\nNumber of Relevant Features Range: (', minRel, '-', maxRel, ')')
minTn, minFp, minFn, minTp = np.amin(rates, axis=0)
medTn, medFp, medFn, medTp = np.median(rates, axis=0)
maxTn, maxFp, maxFn, maxTp = np.amax(rates, axis=0)
print('True Negative Rate:', medTn, '(', minTn, '-', maxTn, ')')
print('False Positive Rate:', medFp, '(', minFp, '-', maxFp, ')')
print('False Negative Rate:', medFn, '(', minFn, '-', maxFn, ')')
print('True Positive Rate:', medTp, '(', minTp, '-', maxTp, ')')
rocAucs = np.array(rocAucs)
sortedAucIndices = np.argsort(rocAucs)
medRocIndex = sortedAucIndices[rocAucs.shape[0]//2]
minAuc, maxAuc, medAuc = np.amin(rocAucs), np.amax(rocAucs), rocAucs[medRocIndex]
print('Median ROC AUC:', medAuc, '(', minAuc, '-', maxAuc, ')')
fpr, tpr, _ = rocCurves[medRocIndex]
rocCurves = np.array(rocCurves)
plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()

def getCoefStats(coefvals):
    coefvals = sorted(coefvals)
    minVal, maxVal, sumVals = coefvals[0], coefvals[-1], sum(coefvals)
    medVal = coefvals[len(coefvals)//2]
    return (minVal, maxVal, medVal, sumVals)
usedFeatures = dict([(k, getCoefStats(v)) for k, v in usedFeatures.items()])
sortedFeatures = sorted(usedFeatures, key=lambda k: abs(usedFeatures[k][-1]), reverse=True)
print('Feature Importances:')
relevantCoefs = []
for feature in sortedFeatures[:int(maxRel)]:
    minVal, maxVal, medVal, sumVals = usedFeatures[feature]
    relevantCoefs.append(medVal)
    print(feature, ':', medVal, '(', minVal, '-', maxVal, ')', sumVals)

    #pca = PCA(n_components=0.95)
    #pca.fit(X)
    #print('Total explained variance:', np.sum(pca.explained_variance_ratio_))
    #print('Original number of components:', X.shape[1])
    #print('New number of components:', pca.n_components_)
    #X_reduced = pca.transform(X)

    #XReducedTrain, XReducedTest, YReducedTrain, YReducedTest = train_test_split(X_reduced, Y, test_size=0.25, random_state=42)

    #reducedModel = LogisticRegression()
    #reducedModel.fit(XReducedTrain, YReducedTrain)

    #predictions = pd.Series(model.predict(test_data))
    #predictions.index = test_data.index

    #print('Reduced Model accuracy:', reducedModel.score(XReducedTest, YReducedTest))
    
    '''image_dir = os.path.join(os.path.dirname(__file__), 'images')
    infected_dir = os.path.join(image_dir, 'infected')
    not_infected_dir = os.path.join(image_dir, 'not_infected')
    imageIds = []
    for path in os.listdir(infected_dir) + os.listdir(not_infected_dir):
        imageIds.append(int(path.split('.')[0]))
    idsPath = os.path.join(os.path.dirname(__file__), 'imageIds.pkl')
    with open(idsPath, 'wb') as f:
        pickle.dump(imageIds, f, pickle.HIGHEST_PROTOCOL)'''

relevantIndices = np.array([sortedFeatures[:int(maxRel)]]).flatten() - 1
relevantData = X[:,relevantIndices]
relevantCoefs = np.array(relevantCoefs)
pca = PCA(n_components=3)
pca.fit(relevantData)
print('Total explained variance:', np.sum(pca.explained_variance_ratio_))
a,b,c = np.dot(pca.components_, relevantCoefs)
intercepts = np.array(intercepts)
d = np.median(intercepts)
print('Median intercept:', d)
reducedData = pca.transform(relevantData)

x, y, z = reducedData[:,0], reducedData[:,1], reducedData[:,2]
delta = 0.1
planeX, planeY = np.meshgrid(np.arange(np.amin(x), np.amax(x), delta), np.arange(np.amin(y), np.amax(y), delta))
planeZ = np.zeros(planeX.shape)
for py in np.arange(planeX.shape[0]):
    for px in np.arange(planeX.shape[1]):
        xval, yval = planeX[py,px], planeY[py,px]
        zval = (0.5 - a*xval - b*yval - d)/c
        planeZ[py,px] = zval

colors = pd.Series(Y).replace({True: 'r', False:'g'})
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_wireframe(planeX, planeY, planeZ, alpha=0.15)
ax.scatter(x, y, z, s=5, c=colors)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
plt.savefig('Image Data Visualization.png', dpi=300)
plt.show()