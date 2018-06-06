from software_model.network_data_preprocessor_for_training import NetworkDataPreprocessorForTraining
import sys
from sklearn.tree import DecisionTreeClassifier
def label_y_value(y_value):
    if y_value == [0,0]:
        return "A"
    if y_value == [0,1]:
        return "B"
    if y_value == [1,0]:
        return "C"
    if y_value == [1,1]:
        return "D"

    raise ValueError("Must be of the form [x,y] where x,y in {0,1}; Given: "+str(y_value))


def get_accuracy(trained_classifer, test_x, test_y_labels):
    y_predict_labels = train_classifier.predict(test_x)

    sum = 0;
    total = 0;
    for index, y_prediction in enumerate(y_predict_labels):
        total += 1
        if y_prediction == test_y_labels[index]:
            sum += 1

    return sum/total

def train_classifier(wav_file_names, classifier):

    import random as random
    random.shuffle(wav_file_names)

    training_files = wav_file_names[:int(0.8 * len(wav_file_names))]
    testing_files = wav_file_names[int(0.8 * len(wav_file_names)):]

    print("Reading in training data")
    train_data = NetworkDataPreprocessorForTraining(training_files)
    X_training, y_training_native = train_data.get_all_annotated_chunks()
    y_training_labels = [label_y_value(y_value) for y_value in y_training_native]
    
    print("Reading in test data")
    test_data = NetworkDataPreprocessorForTraining(testing_files)
    X_testing, y_testing_native = test_data.get_all_annotated_chunks()
    y_testing_labels = [label_y_value(y_value) for y_value in y_testing_native]
    
    
    X_training_raw0 = []
    X_training_raw1 = []
    X_training_fft0 = []
    X_training_fft1 = []
    
    print("Spliting Data")
    for x in X_training:
        X_training_raw0.append(x.get_raw0())
        X_training_raw1.append(x.get_raw1())
        X_training_fft0.append(x.get_fft0())
        X_training_fft1.append(x.get_fft1())
    
    print("Initializing Classifiers")
    raw0_classifier = DecisionTreeClassifier()
    raw1_classifier = DecisionTreeClassifier()
    fft0_classifier = DecisionTreeClassifier()
    fft1_classifier = DecisionTreeClassifier()
    
    print("Training raw0_classifier")
    raw0_classifier.fit(X_training_raw0, y_training_labels)
    print("Training raw1_classifier")
    raw1_classifier.fit(X_training_raw1, y_training_labels)
    print("Training fft0_classifier")
    fft0_classifier.fit(X_training_fft0, y_training_labels)
    print("Training fft1_classifier")
    fft1_classifier.fit(X_training_fft1, y_training_labels)
    
    print("Getting Accuracy")
    
    araw0 = get_accuracy(raw0_classifier, X_testing, y_testing_labels)
    print("Accuracy of raw0: ", araw0)
    
    araw1 = get_accuracy(raw1_classifier, X_testing, y_testing_labels)
    print("Accuracy of raw1: ", araw1)
    
    afft0 = get_accuracy(fft0_classifier, X_testing, y_testing_labels)
    print("Accuracy of fft0: ", afft0)
    
    afft1 = get_accuracy(fft1_classifier, X_testing, y_testing_labels)
    print("Accuracy of fft1: ", afft1)