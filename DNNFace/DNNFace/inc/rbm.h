#ifndef RBM_H 
#define RBM_H

#include <fstream> 
#include <iostream> 
#include <stdio.h> 
#include <math.h> 
#include <string.h> 
#include <stdlib.h> 
#include <map> 
#include <cuda.h> 
#include <cuda_runtime.h> 
#include "utilLearn.h" 


//#include <histo.h> 

#include "connection.h" 
#include "layer.h" 


using namespace std; 
using namespace utilLearn; 

class Rbm 
{ 

public: 
	//rbm.cpp 
	Rbm(Layer *vis, Layer *hid, Connection *connection); 
	~Rbm(); 

	//Host Gets 
	int getHSize(){return _hid->getSize();}; 
	int getVSize(){return _vis->getSize();}; 

	//Device Gets (Connections) 
	float* getDevW(){return _connection->d_weight;}; 
	float* getDevWT(){return _connection->d_weight_t;}; 
	float* getDevA(){return _connection->d_a;}; 
	float* getDevB(){return _connection->d_b;}; 
	float* getDevDw(){return _connection->d_dw;}; 
	float* getDevVw(){return _connection->d_vel_weight;}; 

	//Device Gets (Layers) 
	float* getVX(){return _vis->d_state;}; 
	float* getH0(){return _hid->d_initial_state;}; 
	float* getHX(){return _hid->d_state;}; 
	float* getHrand(){return _hid->d_rand;}; 
	float* getHQ(){return _hid->d_q;}; 

	//rbm_init.cpp 
	void initParams(); 
	void save(char* out_file); 
	int load(char* in_file); 
	int saveHDim(ofstream* file, int loc){return _hid->saveDim(file, loc);}; 
	int saveH(ofstream* file, int loc){return _hid->saveState(file, loc);}; 

	void printHidden0(int b, int g){_hid->printState(false, b,g);}; 
	void printVisibleX(int b, int g){_vis->printState(true, b,g);}; 
	void printHiddenX(int b, int g){_hid->printState(true, b,g);}; 
	void printHrand(int b, int g){_hid->printHrand(b,g);};
	//void histogramQ(){_hid->histogramQ();}; 
	//void histogramW(){_connection->histogramW();}; 
	//void histogramA(){_connection->histogramA();}; 
	//void histogramB(){_connection->histogramB();}; 
	//void histogramDw(){_connection->histogramDw();}; 

	//rbm_disp.cpp 
	//void projectH(int index); 

	//rbm_calc.cpp 
	//float calcFreeNRGTrain(); 
	//float calcFreeNRGValid(); 
	//void saveFreeNRG(); 
	void checkSparsityH(){_hid->checkSparsity();}; 




protected: 
	//Objects 
	Connection*  _connection; 
	Layer*  _vis; 
	Layer*   _hid; 


	int    _total_gpu_mem; 

}; 

#endif 