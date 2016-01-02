#ifndef CONNECTION_H 
#define  CONNECTION_H 

#include <fstream> 
#include <iostream>
#include <ctime>
#include <stdio.h> 
#include <string.h> 
#include <stdlib.h> 
#include <map> 
#include <cuda.h> 
#include <cuda_runtime.h> 


#include "utilLearn.h" 

using namespace std; 
using namespace utilLearn; 

class Connection 
{ 

public: 
	Connection(int v_size, int h_size); 
	~Connection(); 

	//Initialization and saving 
	void initParams(); 
	int save(ofstream *o_file, int loc); 
	int load(ifstream *o_file, int loc); 

	//Get 
	int getVSize(){return _v_size;}; 

	float* getWTRow(int hidden_unit){return &_weight_t[hidden_unit 
		* _v_size];}; 

	//Set 
	void setA(int index, float value){_a[index] = value;};
	void cpyA(){
		cudaMemcpy(d_a, _a, _v_size * sizeof(float), cudaMemcpyHostToDevice);}; 
	void setB(int index, float value){_b[index] = value;}; 
	void cpyB(){
		cudaMemcpy(d_b, _b, _h_size * sizeof(float), cudaMemcpyHostToDevice);}; 

	//Print 
	void printW(); 
	void printWT(); 



public: 
	float*    d_a;  //bias to visible unit 
	float*    d_b;  //bias to hidden unit 
	float*    d_weight; 
	float*    d_weight_t; //transposed weights 
	float*    d_vel_weight; //velocity of weight updates 
	float*    d_dw; 

private: 
	int   _v_size; 
	int    _h_size; 
	int    _w_size; 

	float*    _a; 
	float*    _b; 
	float*    _weight; 
	float*    _weight_t; 
	float*    _vel_weight;//velocity of weights 
	float*    _dw; 
}; 

#endif