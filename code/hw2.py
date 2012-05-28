def generateDataset(N, f, sigma):
    """
    The function generateDataset(N, f, sigma) should return a tuple
    with the 2 vectors x and t. for example:
    
        ti = y(xi) + Normal(mu, sigma)
        # where the xi values are equi-distant on the [0,1] segment (that
        is, x1 = 0, x2=1/N-1, x3=2/N-1..., xN = 1.0)
        mu = 0.0
        sigma = 0.03
        y(x) = sin(x)
    """
    import numpy as np
    vf = np.vectorize(lambda x: f(x) + np.random.normal(0, sigma))
    x = np.linspace(0,1,N)
    return (x, vf(x))


def OptimizeLS(x, t, M):
    import numpy as np
    phi = np.zeros((len(x), M))
    # print 'phi.shape={}, N={}, M={}'.format(phi.shape, len(x), M)
    for n in range(len(x)):
        for m in range(M):
            phi[n][m] = x[n] ** m
    prod = np.dot(phi.T, phi)
    i = np.linalg.inv(prod)
    m = np.dot(i, phi.T)
    w = np.dot(m, t)
    return w
    
def optimizePLS(x, t, M, lamb): # 'lambda' is reserved
    """
    returns the optimal parameters W_{PLS} given M and lambda
    """
    import numpy as np
    phi = np.zeros((len(x), M))
    for n in range(len(x)):
        for m in range(M):
            phi[n][m] = x[n] ** m
    prod = np.dot(phi.T, phi)
    I = np.eye(prod.shape[1]) * lamb
    i = np.linalg.inv(prod + I)
    m = np.dot(i, phi.T)
    W_pls = np.dot(m, t)
    return W_pls

def generateDataset3(N, f, sigma):
    """
    returns 3 pairs of vectors of size N each, (x_test, t_test),
    (x_validate, t_validate) and (x_train, t_train). The target values
    are generated as above with Gaussian noise N(0, sigma). 
    """
    import numpy as np
    
    vf = np.vectorize(lambda x: f(x) + np.random.normal(0, sigma))
    x = np.linspace(0, 1, 3 * N)
    np.random.shuffle(x)
    return tuple((xs, vf(xs)) for xs in [x[:N], x[N:2*N], x[2*N:]])

def lambda_errs(xs, ts, M, error_func):
    # scatter(xs, ts, marker='+', facecolor='g')
    import numpy as np
    errs = []
    lamb_space = np.linspace(-20,5,100)
    for log_lambda in lamb_space:
        lamb = np.e ** log_lambda
        W_pls = optimizePLS(xs, ts, M, lamb)
        errs.append(error_func(W_pls, xs, ts))
    return (np.e ** lamb_space, np.array(errs))

def normalized_errs(w, x, t):
    import numpy as np
    N = len(x)
    M = len(w)
    err = 0
    for i in range(N):
        m_sum = 0
        for m in range(M):
            m_sum += w[m] * (x[i] ** m) # m or m+1 ???????????
                                        # sine it's x^1, x^2, ... x^M 
        t_i_m_sum = t[i] - m_sum
        squared_diff = t_i_m_sum * t_i_m_sum
        err += squared_diff
    err = np.sqrt(err) / np.float64(N)
    return err

def LoptimizePLS(xt, tt, xv, tv, M):
    """
    selects the best value lambda given a dataset for training (xt, tt)
    and a validation test (xv, tv).
    """
    import numpy as np
    lamb_space = np.linspace(-20,5,100) #  100, 1000
    min_err = np.inf
    best_lambda = -1
    for log_lambda in lamb_space:
        lamb = np.e ** log_lambda
        W_pls = optimizePLS(xt, tt, M, lamb)
        tmp = normalized_errs(W_pls, xv, tv)
        if tmp < min_err:
            min_err = tmp
            best_lambda = lamb
    return best_lambda


def bayesianEstimator(x, t, M, alpha, sigma2):
    """
    Given the dataset (x, t) of size N, and the parameters M,
    alpha, and sigma^2 (the variance), returns a tuple of 2 functions
    (m(x), var(x)) which are the mean and variance of the predictive
    distribution inferred from the dataset, based on the parameters
    and the normal prior over w. 
    """
    import numpy as np
    N = len(x)
    def phi(xx):
        return np.array([(xx ** i) for i in range(M+1)])

    # compute S from inv(S)
    aI = alpha * np.eye(M+1)
    S = np.zeros((M+1, M+1))
    for i in range(N):
        phi_xi = phi(x[i])
        S += np.outer(phi_xi, phi_xi.T)
    S = np.linalg.inv(aI + (S / sigma2))
        
    def m(xx):
        phi_t = phi(xx).T        # vector, transose is irrlevant
        sm = np.zeros(M+1)
        for i in range(N):
            sm += phi(x[i])*t[i]
        return (1/sigma2) * np.dot(np.dot(phi_t, S), sm)

    def s2(xx):
        phi_x = phi(xx)
        return sigma2 + np.dot(phi_x.T, np.dot(S, phi_x))
        
    return (m, s2)

def stratifiedSamples(reviews, N=10):
    """
    adapted from hw1. return a tuple (training, test)
    """
    from numpy.random import shuffle # and it's O(n), knuth...
    training = []
    test = []
    frac = 1.0 / N
    short = []
    med = []
    lng = []
        
    for review in reviews:
        r_len = len(review)
        if r_len <= 27:
            short.append(review)
        elif r_len <= 40:
            med.append(review)
        else:
            lng.append(review)

    shuffle(short)
    shuffle(med)
    shuffle(lng)

    for arr in (short, med, lng):
        cut = int(len(arr) * frac)
        test.extend(arr[:cut])
        training.extend(arr[cut:])

    return training, test
    
def word_feats(words):
    return dict([(word, True) for word in words])

def bag_of_words(document):
    from itertools import chain
    words = chain.from_iterable(document)
    return word_feats(words)

def evaluate_features(feature_extractor, N):
    from nltk.corpus import movie_reviews
    from nltk.classify import NaiveBayesClassifier as naive
    from nltk.classify.util import accuracy
    from nltk.metrics import precision, recall, f_measure
    
    negative = movie_reviews.fileids('neg')
    positive = movie_reviews.fileids('pos')
    negfeats = [(feature_extractor(movie_reviews.sents(fileids=[f])),
                 'neg') for f in negative]

    posfeats = [(feature_extractor(movie_reviews.sents(fileids=[f])),
                 'pos') for f in positive]

    negtrain, negtest = stratifiedSamples(negfeats, N)
    postrain, postest = stratifiedSamples(posfeats, N)

    trainfeats = negtrain + postrain
    testfeats = negtest + postest
    classifier = naive.train(trainfeats)

    print 'accuracy: {}'.format(accuracy(classifier, testfeats))

    # Precision, Recall, F-measure
    from collections import defaultdict
    refsets = defaultdict(set)
    testsets = defaultdict(set)

    for i, (feats, label) in enumerate(testfeats):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)
        
    print 'pos precision:', precision(refsets['pos'], testsets['pos'])
    print 'pos recall:', recall(refsets['pos'], testsets['pos'])
    print 'pos F-measure:', f_measure(refsets['pos'], testsets['pos'])
    print 'neg precision:', precision(refsets['neg'], testsets['neg'])
    print 'neg recall:', recall(refsets['neg'], testsets['neg'])
    print 'neg F-measure:', f_measure(refsets['neg'], testsets['neg'])
    
    classifier.show_most_informative_features()
    return classifier
    
    
