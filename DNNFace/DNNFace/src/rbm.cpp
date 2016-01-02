#include "../inc/rbm.h" 

/* --------------------------------------------------------------- 
*    CONSTRUCTOR 
* v_size_x    | int | visible layer X dimension (for 
display) 
* v_size_y    | int | visible layer Y dimension (for 
display) 
* h_size_x    | int | hidden layer X dimension (for display) 



* h_size_y    | int | hidden layer Y dimension (for display) 
* mini_batch_size  | int | number of examples per batch 
* gibbs_samples | int | number of fantasy particles 
------------------------------------------------------------------*/ 
Rbm::Rbm(Layer *vis, Layer *hid, Connection *connection) 
{ 
	_vis = vis; 
	_hid = hid; 
	_connection = connection; 

	//Device memory allocation 
	_total_gpu_mem = 0; 

	return; 
} 

/* --------------------------------------------------------------- 
*    DESTRUCTOR 
------------------------------------------------------------------*/ 
Rbm::~Rbm() 
{ 
	//Classes 
	_connection->~Connection(); 
	_vis->~Layer(); 
	_hid->~Layer(); 

	return; 
} 


