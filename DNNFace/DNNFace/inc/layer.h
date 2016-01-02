#ifndef LAYER_H 
#define LAYER_H 

#include <fstream> 
#include <iostream>
#include <iomanip>
#include <stdio.h> 
#include <math.h> 
#include <string.h> 
#include <stdlib.h> 
#include <map> 
#include <cuda.h> 
#include <cuda_runtime.h> 

#include "utilLearn.h" 

using namespace std; 
using namespace utilLearn; 

class Layer 
{ 

public: 
	Layer(int dim_x, int dim_y, int mini_batch_size, int 
		gibbs_samples, bool using_initial); 
	~Layer(); 
	//>--��ʼ������ڵ����ֵΪ0
	void initParams(); 

	//>--�������ڵ����ֵ����(0,1)״ֵ̬
	void randState(float prob); 

	//Host Gets 
	int getSize(){return _size;}; 

	//Device Gets 
	//>--��������״ֵ̬
	int saveState(ofstream* out_file, int loc);
	//>--��������ά��ֵ
	int saveDim(ofstream* out_file, int loc);
	//>--����������ڵ����ֵ
	int saveQ(ofstream* out_file, int loc); 
	int loadQ(ifstream* in_file, int loc); 

	//>--�������ڵ������Ϣ(�ܺͣ����ֵ����Сֵ)
	void checkSparsity(); 

	void printState(bool current, int b, int g);
	void printHrand(int b, int g);

public: 
	float* d_initial_state; 
	float* d_state; 
	float* d_rand; //random numbers used for monte carlo
	//device������ڵ����
	float* d_q; //sparsity estimation 

	float* d_error; //error given input 

protected: 
	int _dim_x; 
	int _dim_y; 
	int _size; 
	int _mini_batch_size; 
	int _gibbs_samples; 

	//host������ڵ����
	float*  _q; //probability estimate 

	bool _using_initial; //set to true if the layers holds a state 

	//display array (��Ӧ����ʾd_state)
	float*  _disp; 

}; 

#endif 