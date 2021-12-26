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
from os import sep


def Get_argv(argv):
    try:
        options, args = getopt.getopt(argv, 'hi:k:g:o:t:')
    except getopt.GetoptError:
        print ('Error while reading arguments. Get help by Detect_TEPs.py -h')
        sys.exit(2)
    if len(options) == 0:
        print ('Error while reading arguments. Get help by Train_machine_classifier.py -h')
        sys.exit(2)
    for option, arg in options:
        if len(options) == 1:
            if option == '-h':
                print ('Detect_TEPs.py detects TEPs using the trained machine classifier')
                print ('[Arguments]')
                print ("  -i     5' end profile of Term-Seq; gff format")
                print ("  -k     machine classifier; pickled from Train_machine_classifier.py")
                print ("  -g     size of genome; integer")
                print ("  -o     output file")
                print ("  -t     [optional] A threshold detecting TEP. Default = 1 RPM")
                print ("                    Positions with read count below given threshold will be ignored.")
                print ("                    The threshold is defined as #n read per million mapped reads.")
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
                print('  Term-Seq profile  : '+input_file)
            elif option == '-k':
                if arg == '':
                    print ('Error while reading the "k" argument. Get help by Train_machine_classifier.py -h')
                    sys.exit(2)
                knc_name = arg
                print('  Machine classifier: '+knc_name)
            elif option == '-g':
                if arg == '':
                    print ('Error while reading the "g" argument. Get help by Train_machine_classifier.py -h')
                    sys.exit(2)
                try:
                    genome_size = int(arg)
                except ValueError:
                    print ('Error: The genome size argument only takes integer.')
                    sys.exit(2)
                print("  Genome size       : "+'{:,}'.format(genome_size))
            elif option == '-o':
                if arg == '':
                    print ('Error while reading the "o" argument. Get help by Train_machine_classifier.py -h')
                    sys.exit(2)
                output_file = arg
                print('  Output file       : '+output_file)
            elif option == '-t':
                if arg == '':
                    print ('Error while reading the "t" argument. Get help by Train_machine_classifier.py -h')
                    sys.exit(2)
                try:
                    threshold = int(arg)
                except ValueError:
                    print ('Error: The threshold argument only takes integer.')
                    sys.exit(2)
                print("  Threshold         : "+str(threshold))
        else:
            print ('Error while reading arguments. Get help by Train_machine_classifier.py -h')
            sys.exit(2)
    return input_file, knc_name, genome_size, output_file, threshold


def make_measure(position, strand, profile):
    measure = np.array(tuple(profile[strand+str(i)] for i in range(pos-10, pos+11, 1)))
    return measure.reshape(1,-1)


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

infile, knc_name, genome_size, outfile, threshold = Get_argv(sys.argv[1:])

print ('\n[Reading the profile for machine learning]')
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


print ('\n[Loading the machine classifier]')
print ('  Loading the machine classifier...', end='')
knc_pickle = open(knc_name, 'rb')
knc = pickle.load(knc_pickle)
knc_pickle.close()
print ('Done')


print ('\n[Detecting TEPs]')
cutoff = read_count/1000000.0*threshold
print ('  Positions with read density lower than '+str(round(cutoff))+' will be ignored.')
print ('  Detecting TEPs...', end='')

TEP = {}
for pos in range(11,genome_size-9,1):
    for strand in ('+', '-'):
        if zfp[strand+str(pos)] > cutoff:
            measure = make_measure(pos, strand, zfp)
            prediction = list(knc.predict(measure))
            prob = knc.predict_proba(measure).tolist()[0]
            if prob[1] == 1.0 and prediction == ['Term']:
                TEP[strand+str(pos)] = pos, strand, int(strand+str(zfp[strand+str(pos)]))

print ('Done')
print ('Detected TEPs: '+'{:,}'.format(len(TEP)))


print ('\n[Exporting TEPs]')
print ('  Exporting TEPs...', end='')

output_dict = pd.DataFrame.from_dict(TEP)
output_dict = output_dict.T


with open(infile) as input_file:
    line = input_file.readline()
accession, source = line.split('\t')[0:2]

accession = pd.DataFrame([accession for i in range(len(TEP))])
source = pd.DataFrame(['.'.join(knc_name.split('.')[:-1]).split('/')[-1] for i in range(len(TEP))])
filename = pd.DataFrame(['.'.join(infile.split('.')[:-1]).split('/')[-1] for i in range(len(TEP))])
dots = pd.DataFrame(['.' for  i in range(len(TEP))])
TEP_num = pd.DataFrame(['TEP'+str(i) for  i in range(len(TEP))])

positions = output_dict[0]
positions.reset_index(drop=True,inplace=True)
strands = output_dict[1]
strands.reset_index(drop=True,inplace=True)
intensities = output_dict[2]
intensities.reset_index(drop=True,inplace=True)

output_df = pd.concat((accession, source, filename, \
                   positions, positions, intensities, strands, TEP_num), axis=1)

output_df.to_csv(outfile+'.gff', sep='\t', header=False, index=False)
print ('Done')
print ('  Detected TEPs are exported as '+outfile+'.gff')