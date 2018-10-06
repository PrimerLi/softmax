import numpy as np

class Sample:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __str__(self):
        return ",".join(map(str, self.x)) + ",label = " + str(self.y)

def Z(theta_list, x):
    result = 1.0
    for theta in theta_list[1:]:
        assert(len(theta) == len(x))
        result += np.exp(theta.dot(x))
    return result

def L(samples, theta_list):
    result = 0.0
    N = len(theta_list)
    for sample in samples:
        label = sample.y
        assert(label >= 0 and label < N)
        if (label == 0):
            delta = 0
        else:
            delta = theta_list[label].dot(sample.x)
        result += np.log(Z(theta_list, sample.x)) - delta
    return result/float(len(samples))

def gradient(samples, theta_index, theta_list):
    N = len(theta_list)
    assert(theta_index >= 1 and theta_index <= N-1)
    result = np.zeros(len(theta_list[theta_index]))
    for sample in samples:
        label = sample.y
        if (label == 0):
            delta = 0
        else:
            if (label == theta_index):
                delta = 1
            else:
                delta = 0
        result += (np.exp(theta_list[theta_index].dot(sample.x))/Z(theta_list, sample.x) - delta)*sample.x
    return result/float(len(samples))

def gradient_total(samples, theta_list):
    feature_length = len(theta_list[0])
    result = np.zeros(len(theta_list[0])*(len(theta_list) - 1))
    for i in range(len(theta_list)):
        if (i == 0):
            continue
        else:
            sub_result = gradient(samples, i, theta_list)
            result[(i-1)*feature_length : i*feature_length] = sub_result[:]
    return result

def break_theta(theta_total, feature_length):
    assert(len(theta_total)%feature_length == 0)
    theta_list = []
    theta_list.append(np.zeros(feature_length))
    for i in range(len(theta_total)/feature_length):
        theta_list.append(theta_total[i*feature_length : (i+1)*feature_length])
    return theta_list

def combine_theta(theta_list):
    feature_length = len(theta_list[0])
    theta_total = np.zeros(feature_length*(len(theta_list) - 1))
    for i in range(len(theta_list)):
        if (i == 0):
            continue
        else:
            theta_total[(i-1)*feature_length : i*feature_length] = theta_list[i][:]
    return theta_total

def gradient_descent(eta, samples, theta_list):
    theta_total = np.zeros(len(theta_list[0])*(len(theta_list)-1))
    feature_length = len(theta_list[0])
    for i in range(len(theta_list)):
        if (i == 0):
            continue
        else:
            theta_total[(i-1)*feature_length : i*feature_length] = theta_list[i][:]
    counter = 0
    iterationMax = 100
    error = 0.0
    eps = 1.0e-3
    while(counter < iterationMax):
        counter += 1
        g = gradient_total(samples, theta_list)
        theta_total = theta_total - eta*g
        theta_list = break_theta(theta_total, feature_length)
        error = np.linalg.norm(g)
        print "Iteration_counter = " + str(counter) + ", error = " + str(error)
        if (error < eps):
            break
    return break_theta(theta_total, feature_length)

def adam(alpha, beta_1, beta_2, samples, theta_list):
    theta_total = combine_theta(theta_list)
    feature_length = len(theta_list[0])
    assert(beta_1 >= 0 and beta_1 < 1)
    assert(beta_2 >= 0 and beta_2 < 1)
    assert(alpha > 0)
    m = np.zeros(len(theta_total))
    v = np.zeros(len(theta_total))
    counter = 0
    iterationMax = 5
    eps = 1.0e-8
    error_limit = 1.0e-3
    theta_total_updated = theta_total
    ofile = open("theta_record.txt", "w")
    while(counter < iterationMax):
        counter += 1
        g = gradient_total(samples, theta_list)
        m = beta_1*m + (1 - beta_1)*g
        v = beta_2*v + (1 - beta_2)*g**2
        #m = m/(1 - beta_1**counter)
        #v = v/(1 - beta_2**counter)
        theta_total_updated = theta_total - alpha*m/(np.sqrt(v) + eps)
        theta_list = break_theta(theta_total_updated, feature_length)
        #ofile.write(";".join(map(lambda theta: ",".join(map(str, theta)), theta_list[1:])) + "\n")
        ofile.write(",".join(map(str, theta_list[-1])) + "\n")
        gradient_norm = np.linalg.norm(g)
        error = np.linalg.norm(theta_total - theta_total_updated)
        print "Iteration_counter = ", counter, ", norm of gradient = ", gradient_norm, ", error = ", error
        theta_total = theta_total_updated
        if (error < error_limit):
            break
    ofile.close()
    return theta_list

def probability(sample, theta_list):
    result = []
    partition = Z(theta_list, sample.x)
    for i in range(len(theta_list)):
        if (i == 0):
            result.append(1.0/partition)
        else:
            result.append(np.exp(theta_list[i].dot(sample.x))/partition)
    return result

def read_data(dataFileName):
    import os
    assert(os.path.exists(dataFileName))
    samples = []
    ifile = open(dataFileName, "r")
    for (index, string) in enumerate(ifile):
        if (index == 0):
            header = string.strip("\n")
        else:
            a = map(int, string.strip("\n").split(","))
            label = a[0]
            x = np.zeros(len(a))
            x[0:-1] = np.asarray(a[1:])/255.0
            x[-1] = 1
            samples.append(Sample(x, label))
    ifile.close()
    return samples

def get_label_statistics(samples):
    result = dict()
    for sample in samples:
        if (sample.y in result):
            result[sample.y] += 1
        else:
            result[sample.y] = 1
    return result

def print_dictionary(dictionary):
    keys = dictionary.keys()
    keys = sorted(keys)
    for key in keys:
        print key, dictionary[key]

def main():
    import sys
    if (len(sys.argv) != 2):
        print "trainFileName = sys.argv[1]. "
        return -1
    
    trainFileName = sys.argv[1]
    print "Reading train file ... "
    samples = read_data(trainFileName)
    print "File reading finished. "
    theta_list = []
    feature_length = len(samples[0].x)
    stats = get_label_statistics(samples)
    numberOfCategories = len(stats.keys())
    for i in range(numberOfCategories):
        if (i == 0):
            theta_list.append(np.zeros(feature_length))
        else:
            theta_list.append(np.random.random(feature_length))
    #eta = 1.0e-3
    #theta_list_updated = gradient_descent(eta, samples, theta_list)
    alpha = 0.01
    beta_1 = 0.9
    beta_2 = 0.999
    theta_list_updated = adam(alpha, beta_1, beta_2, samples, theta_list)
    p = probability(samples[0], theta_list_updated)
    for i in range(len(p)):
        print i, p[i]
    print "label = " + str(samples[0].y)
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
