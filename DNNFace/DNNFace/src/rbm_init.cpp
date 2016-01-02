#include "../inc/rbm.h" 


/* --------------------------------------------------------------- 
*    INIT PARAMS
* Initializes the paremeters for a new RBM. 
------------------------------------------------------------------*/ 
void Rbm::initParams() 
{ 

	_connection->initParams(); //初始化权重w，显层偏置a，隐层偏置b
	_vis->initParams(); //初始化显层节点概率为0
	_hid->initParams(); //初始化隐层节点概率为0

	//saveFreeNRG(); //Get initial FreeNRG 

	return; 
} 

/* --------------------------------------------------------------- 
*    SAVE 
* Saves the RBM Parameters to a file 
* 
* out_file | char* | pointer to name of file 
------------------------------------------------------------------*/ 
void Rbm::save(char* out_file) 
{ 
	printf("Saving RBM..."); 
	ofstream o_file; 
	o_file.open(out_file, ios::binary); 

	int loc = 0; 
	if(o_file.is_open()) 
	{ 
		loc = _connection->save(&o_file, loc); 

		loc = _hid->saveQ(&o_file, loc); 

		o_file.close(); 
		printf("Completed\n"); 
	} 
	else 
		printf("Failed\n"); 
	return; 
} 

/* --------------------------------------------------------------- 
*    LOAD 
* Loads the RBM Parameters from a file 
* 
* in_file | char* | pointer to name of file 
------------------------------------------------------------------*/ 
int Rbm::load(char* in_file) 
{ 
	printf("Loading RBM from %s...",in_file); 
	ifstream i_file; 
	i_file.open(in_file, ios::binary); 
	if(i_file.is_open()) 
	{ 
		int loc = 0; 

		loc = _connection->load(&i_file, loc); 

		loc = _hid->loadQ(&i_file, loc); 

		i_file.close(); 

		printf("Completed!\n"); 
		return 0; 
	} 
	else 
	{ 
		printf("Failed to open file\n"); 
		return -1; 
	} 
}