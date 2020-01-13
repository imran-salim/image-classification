#!/bin/bash

export CLASSPATH=/home/ml/weka-3-8/weka.jar:/home/is39/521/MLAssignment1/out/artifacts/MLAssignment1_jar/MLAssignment1.jar

if [ ! -e /tmp/isml ]; then
	mkdir /tmp/isml
fi

cd /tmp/isml

if [ ! -e svhn.tar.gz ]; then
	cp /home/ml/521/assignment1/svhn.tar.gz .
fi

if [ ! -e svhn ]; then
	tar -xvf svhn.tar.gz
fi

cd svhn

#if [ ! -e testing_10percent.arff ]; then
#	java weka.Run .StratifiedRemoveFolds -c last -i testing.arff -o testing_10percent.arff
#fi

#if [ ! -e training_10percent.arff ]; then
#	java weka.Run .StratifiedRemoveFolds -c last -i training.arff -o training_10percent.arff
#fi
#
#cd /tmp/isml/svhn

java -Xmx10g weka.Run .FilteredClassifier -F ".KMeansImageFilter -size 8 -stride 3 -pool 3 -K 1000 -output-debug-info" -W .MultiClassClassifier -t training.arff -T testing.arff -- -M 3 -W .SGD -- -N -M -F 0 -L 0.0001 -E 100 > /home/is39/521/MLAssignment1/output_svhn.txt

#if [ ! -e testing_1percent.arff ]; then
#	java weka.Run .StratifiedRemoveFolds -c last -i testing.arff | java weka.Run .StratifiedRemoveFolds -c last -o testing_1percent.arff
#fi
#
#if [ ! -e training_1percent.arff ]; then
#	java weka.Run .StratifiedRemoveFolds -c last -i training.arff | java weka.Run .StratifiedRemoveFolds -c last -o training_1percent.arff
#fi
#
#java weka.Run .FilteredClassifier -F ".KMeansImageFilter -output-debug-info" -W .MultiClassClassifier -t training_1percent.arff -T testing_1percent.arff -- -M 3 -W .SGD -- -N -M -F 0 -L 0.0001 -E 100
