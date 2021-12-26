# -*- coding: utf-8 -*-
"""
@author: Donghui Choe

[External packages version control]
The script has been confirmed to function with the followings:
    python==3.6.10
    numpy==1.13.1
    scipy==1.19.1
    scikit-learn=0.23.1
    pandas==1.0.5

"""

import numpy as np
from sklearn import neighbors
import pandas as pd
import pickle, sys, getopt


def Get_argv(argv):
    try:
        options, args = getopt.getopt(argv, 'hi:g:o:p:n:')
    except getopt.GetoptError:
        print ('Error while reading arguments. Get help by Train_machine_classifier.py -h')
        sys.exit(2)
    if len(options) == 0:
        print ('Error while reading arguments. Get help by Train_machine_classifier.py -h')
        sys.exit(2)
    for option, arg in options:
        if len(options) == 1:
            if option == '-h':
                print ('Train_machine_classifier generates pickled KNN machine classifier required for Term_detector')
                print ('[Arguments]')
                print ("  -i     5' end profile of Term-Seq; gff format")
                print ("  -g     size of genome; integer")
                print ("  -o     output file")
                print ("  -p     positive train set; collection of positions where known or true transcript 3' ends are")
                print ("  -n     negative train set; collection of positions that are not transcript 3' ends")
                sys.exit()
            else:
                print ('Error while reading arguments. Get help by Train_machine_classifier.py -h')
                sys.exit(2)
        elif len(options) == 5:
            if option == '-i':
                if arg == '':
                    print ('Error while reading the "i" argument. Get help by Train_machine_classifier.py -h')
                    sys.exit(2)
                input_file = arg
                print ('\n[Input arguments]')
                print('  Term-Seq profile: '+input_file)
            elif option == '-g':
                if arg == '':
                    print ('Error while reading the "g" argument. Get help by Train_machine_classifier.py -h')
                    sys.exit(2)
                try:
                    genome_size = int(arg)
                except ValueError:
                    print ('Error: The genome size argument only takes integer.')
                    sys.exit(2)
                print("  Genome size     : "+'{:,}'.format(genome_size))
            elif option == '-o':
                if arg == '':
                    print ('Error while reading the "o" argument. Get help by Train_machine_classifier.py -h')
                    sys.exit(2)
                output_file = arg
                print('  Output file     : '+output_file)
            elif option == '-p':
                if arg == '':
                    print ('Error while reading the "p" argument. Get help by Train_machine_classifier.py -h')
                    sys.exit(2)
                positives = arg
                print('  Positive set    : '+positives)
            elif option == '-n':
                if arg == '':
                    print ('Error while reading the "n" argument. Get help by Train_machine_classifier.py -h')
                    sys.exit(2)
                negatives = arg
                print('  Negative set    : '+negatives)
        else:
            print ('Error while reading arguments. Get help by Train_machine_classifier.py -h')
            sys.exit(2)
    return input_file, genome_size, output_file, positives, negatives


def make_signature(position, profile):
    strand = position[0]
    pos = int(position[1:])
    dataset = tuple(profile[strand+str(i)] for i in range(pos-10, pos+11, 1))
    return dataset


def check_pickled(input_file):
    pickled = '.'.join(input_file.split('.')[:-1])+'.pickle'
    try:
        temp = open(pickled, 'rb')
        return True
    except:
        return False


def read_profile(input_file):
    with open(input_file) as infile:
        profile, count = {}, 0
        for line in infile:
            items = line.split('\t')
            profile[items[6]+items[3]] = abs(int(items[5]))
            count += abs(int(items[5]))
    return profile, count


def fill_zeros(profile, genome_size):
    zero_filled_profile = {}
    for position in range(1,genome_size+1, 1):
        for strand in ('+', '-'):
            try:
                zero_filled_profile[strand+str(position)] \
                = profile[strand+str(position)]
            except:
                zero_filled_profile[strand+str(position)]= 0
    return zero_filled_profile

    
def make_dataset(collection, category, profile):
    dataset = []
    for key in collection.keys():
        dataset.append(make_signature(key, profile)+(category,))
    return dataset
        

infile, genome_size, outfile, positive, negative = Get_argv(sys.argv[1:])

print ('\n[Read the profile for machine learning]')
if check_pickled(infile):
    print ('  Found a pickled profile\n  Loading the profile...', end='')
    pickle_in = open('.'.join(infile.split('.')[:-1])+'.pickle', 'rb')
    zfp, read_count = pickle.load(pickle_in)
    pickle_in.close()
    print ('Done')
    print ('  Total read count: '+'{:,}'.format(read_count))
else:
    print ('  Reading the profile...', end = '')
    profile, read_count = read_profile(infile)
    zfp = fill_zeros(profile, genome_size)
    print ('Done')
    print ('  Total read count: '+'{:,}'.format(read_count))
    print ('  Pickling the profile for later use...' , end='')
    pickle_out = open('.'.join(infile.split('.')[:-1])+'.pickle', 'wb')
    pickle.dump((zfp, read_count), pickle_out)
    pickle_out.close()
    print ('Done')


print ('\n[Train a machine classifier]')
print ('  Constructing a training dataset...', end = '')
positive_set = make_dataset(read_profile(positive)[0], 'Term', zfp)
negative_set = make_dataset(read_profile(negative)[0], 'Non-Term', zfp)
index = tuple('profile'+str(i) for i in range(-10,11,1))+('call',)
train_set = pd.DataFrame(positive_set+negative_set, columns = index)
print ('Done')

print ('  Training a machine classifier...', end = '')
X = np.array(train_set.drop(['call'],1))
y = np.array(train_set['call'])

positive_set = pd.DataFrame(positive_set, columns = index)
positive_X = np.array(positive_set.drop(['call'],1))
positive_y = np.array(positive_set['call'])
negative_set = pd.DataFrame(negative_set, columns = index)
negative_X = np.array(negative_set.drop(['call'],1))
negative_y = np.array(negative_set['call']) 

knc = neighbors.KNeighborsClassifier()
knc.fit(X,y)
positive_accuracy = knc.score(positive_X, positive_y)
negative_accuracy = knc.score(negative_X, negative_y)
print ('Done')
print ('   -Positive prediction accuracy:  '+str(round(positive_accuracy*100,2))+'%')
print ('   -Negative prediction accuracy:  '+str(round(negative_accuracy*100,2))+'%')


print ('\n[Save the machine classifier]')
print ('  Saving the machine classifier...', end= '')
pickle_out = open(outfile+'.pickle', 'wb')
pickle.dump(knc, pickle_out)
pickle_out.close()
print ('Done')
print ('  The trained machine classifier is saved as '+outfile+'.pickle')