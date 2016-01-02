#ifndef DBN_H 
#define DBN_H 

#include <cuda.h> 
#include <cuda_runtime.h> 



#include "connection.h" 
#include "layer.h" 


using namespace std; 

class Dbn 
{ 

public: 
	Dbn(Layer *vis, Layer *h0, Layer *h1, Connection * v_h0, 
		Connection * h0_h1); 
	~Dbn(); 

	//Host Gets 
	int getH1Size(){return _h1->getSize();}; 
	int getH0Size(){return _h0->getSize();}; 
	int getVSize(){return _vis->getSize();};

	//Device Gets (Connections) 
	float* getDevW_0(){return _v_h0->d_weight;}; 
	float* getDevWT_0(){return _v_h0->d_weight_t;}; 
	float* getDevVw_0(){return _v_h0->d_vel_weight;}; 
	float* getDevB_0(){return _v_h0->d_b;}; 
	float* getDevA_0(){return _v_h0->d_a;}; 
	float* getDevW_1(){return _h0_h1->d_weight;}; 
	float* getDevWT_1(){return _h0_h1->d_weight_t;}; 
	float* getDevVw_1(){return _h0_h1->d_vel_weight;}; 
	float* getDevB_1(){return _h0_h1->d_b;}; 
	float* getDevA_1(){return _h0_h1->d_a;}; 


	//Device Gets (Layers) 
	float* getVX(){return _vis->d_state;}; 
	float* getVPred(){return _vis->d_q;}; 
	float* getH0(){return _h0->d_state;}; 
	float* getH0Rand(){return _h0->d_rand;}; 
	float* getH0In(){return _h0->d_initial_state;}; 
	float* getH0Pred(){return _h0->d_q;}; 
	float* getH1(){return _h1->d_state;}; 
	float* getH1Rand(){return _h1->d_rand;}; 
	float* getH1In(){return _h1->d_initial_state;}; 
	float* getH1Pred(){return _h0->d_q;}; 


	void showVX(int b){_vis->printState(true,b,0);}; 
	void showH0(int b){_h0->printState(true, b, 0);}; 
	void showH1(int b){_h1->printState(true, b, 0);}; 

	void saveLayers(char* out_file_lvl1, char* out_file_lvl2); 
	int loadLayers(char* in_file_lvl1, char* in_file_lvl2); 

protected: 
	Layer*  _vis; 
	Layer*  _h0; 
	Layer*  _h1; 

	Connection* _v_h0; 
	Connection* _h0_h1; 
}; 


#endif 