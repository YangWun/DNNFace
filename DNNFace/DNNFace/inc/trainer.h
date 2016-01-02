#ifndef TRAINER_H 
#define TRAINER_H 
#include <fstream> 
#include <iostream> 
#include <stdio.h> 
#include <math.h> 
#include <string.h> 
#include <stdlib.h> 
#include "utilLearn.h" 

using namespace std; 
using namespace utilLearn; 

class Trainer 
{ 
public: 
	Trainer(int mini_batch_size, int gibbs_samples, int 
		num_epochs, float learn_rate, int num_classes, float momentum, float 
		lr_beta); 
	~Trainer(); 

	//Training Status 
	void incN(); 
	int getN(){return _n;}; 
	bool epochComplete(); 
	int getEpoch(){return _cur_epoch;}; 
	bool trainComplete(){return _cur_epoch >= _num_epochs;}; 
	float getMomentum(){return _momentum;}; 

	//Providing Training Data 
	void randBatchV(); 
	int nextBatchTrain(); 
	int nextBatchValid(); 
	void setV(int n, int batch); 
	int getTrainSize(){return _train_size;}; 
	int getValidSize(){return _valid_size;}; 
	float getLearnRate(){return _learn_rate;}; 
	int getNumFantasy(){return _mini_batch_size * _gibbs_samples;}; 
	void showTraining(int n); 
	void showCurrent(int b); 
	float* getHostData(){return _train_data;}; 


	int answer(int n){return _train_label_vals[n];}; 
	int ansCurrent(int b){return _mini_batch_label_vals[b];}; 

	int batchClassification(float* d_top_prob, int batch_size); 
	float batchError(float* d_top_prob, int batch_size); 

	//Training Data Calculations 
	float pixelProb(int index); 

	//Loading 
	int loadTrainingData(char* data_file); 
	int loadTrainingLabels(char* label_file); 
	int loadTrainingLabelsMAT(char* label_file); 
	int loadConvertTrainingData(char* data_file); 
	int loadTrainingDataMAT(char* data_file); 

public: 
	//GPU device memory 
	float* d_mini_batch_data; 
	float* d_mini_batch_labels; 

private: 
	//Set validation hold aside 
	void update_valid(){_valid_size = floor((float)_train_size * 
		0.1);}; 


protected: 
	//Training Status Variables 
	int    _mini_batch_size; 
	int    _gibbs_samples; 
	int    _num_epochs;  //max epochs to train on 
	int    _cur_epoch;  //current number of epochs 
	int    _cur_batch; //current batch loaded (in order) 
	int    _next_batch; //next batch to be loaded 
	int    _n;    //training examples seen this epoch 
	float _learn_rate; 
	bool  _using_labels; //set when loaded 
	float   _momentum; 
	float   _lr_beta; 

	//set by loadTrainingData 
	int    _input_size; 
	int    _input_dim_x; 
	int    _input_dim_y; 
	int    _train_size; 
	int    _valid_size; //Number held aside for validation --10% by default 
	int    _num_classes; 

	float*    _train_data; //all training data 
	float*    _mini_batch_data; 
	//>--training labels index
	int*    _train_label_vals; //all training labels 
	float*    _train_labels; 
	float*    _mini_batch_labels; 
	int*    _mini_batch_label_vals; 
}; 

#endif 