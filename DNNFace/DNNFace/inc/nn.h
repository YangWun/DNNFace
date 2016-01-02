#ifndef NN_H 
#define NN_H 

#include<list> 


#include "connection.h" 
#include "layer.h" 


using namespace std; 
class Nn 
{ 

public: 
	Nn(Layer *vis, Layer *h0, Layer *h1, Layer *top, Connection * 
		v_h0, Connection * h0_h1, Connection * h1_top); 
	~Nn(); 

	//Host Gets 
	int getTopSize(){return _top->getSize();}; 
	int getH1Size(){return _h1->getSize();}; 
	int getH0Size(){return _h0->getSize();}; 
	int getVSize(){return _vis->getSize();}; 

	//Device Gets (Connections) 
	float* getDevW_0(){return _v_h0->d_weight;}; 
	float* getDevVw_0(){return _v_h0->d_vel_weight;}; 
	float* getDevB_0(){return _v_h0->d_b;}; 
	float* getDevW_1(){return _h0_h1->d_weight;}; 
	float* getDevVw_1(){return _h0_h1->d_vel_weight;}; 
	float* getDevB_1(){return _h0_h1->d_b;}; 
	float* getDevW_2(){return _h1_top->d_weight;}; 
	float* getDevVw_2(){return _h1_top->d_vel_weight;}; 
	float* getDevB_2(){return _h1_top->d_b;}; 

	//Device Gets (Layers) 
	float* getVX(){return _vis->d_state;}; 
	float* getH0(){return _h0->d_state;}; 
	float* getH0In(){return _h0->d_initial_state;}; 
	float* getH1(){return _h1->d_state;}; 
	float* getH1In(){return _h1->d_initial_state;}; 
	float* getTop(){return _top->d_state;}; 


	float* getTopError(){return _top->d_error;}; 
	float* getH1Error(){return _h1->d_error;}; 
	float* getH0Error(){return _h0->d_error;}; 

	int getAnswer(); 

	void showH0(int b){_h0->printState(true, b, 0);}; 
	void showH1(int b){_h1->printState(true, b, 0);}; 
	void showTop(int b){_top->printState(true, b, 0);}; 
	void printTop(int b){_top->printState(true,b, 0);}; 

	void setTopProb(); 

	void saveComplete(char* out_file); 
	int loadComplete(char* in_file); 
protected: 
	//list<Connection*>  _connections; 
	//list<Layer*>    _layers; 

	Layer*  _vis; 
	Layer*  _h0; 
	Layer*  _h1; 
	Layer*  _top; 

	Connection* _v_h0; 
	Connection* _h0_h1; 
	Connection* _h1_top; 
}; 

#endif 