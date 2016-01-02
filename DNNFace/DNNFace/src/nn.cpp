#include "../inc/nn.h" 

Nn::Nn(Layer *vis, Layer *h0, Layer *h1, Layer *top, Connection * 
	v_h0, Connection * h0_h1, Connection * h1_top) 
{ 
	/*_layers.push_back(vis); 
	_layers.push_back(h0); 
	_layers.push_back(h1); 
	_layers.push_back(top); 

	_connections.push_back(v_h0); 
	_connections.push_back(h0_h1); 
	_connections.push_back(h1_top);*/ 

	//to do: add verification 

	_vis = vis; 
	_h0 = h0; 
	_h1 = h1; 
	_top = top; 

	_v_h0 = v_h0; 
	_h0_h1 = h0_h1; 
	_h1_top = h1_top; 
} 

Nn::~Nn() 
{ 

} 

void Nn::setTopProb() 
{ 
	float prob = 1 / (float)_top->getSize(); 
	//printf("prob=%f\n",prob); 
	prob = log(prob / (1 - prob)); 
	//printf("log prob=%f\n",prob); 
	for(int i=0;i<_top->getSize();i++) 
	{ 
		_h1_top->setB(i,prob); 
	} 
	_h1_top->cpyB(); 
} 

void Nn::saveComplete(char* out_file) 
{ 
	printf("Saving Net..."); 
	ofstream o_file; 
	o_file.open(out_file, ios::binary); 

	int loc = 0; 
	if(o_file.is_open()) 
	{ 
		loc = _v_h0->save(&o_file, loc); 
		loc = _h0_h1->save(&o_file, loc); 
		loc = _h1_top->save(&o_file, loc); 

		o_file.close(); 
		printf("Completed\n"); 
	} 
	else 
		printf("Failed\n"); 
	return; 
} 

int Nn::loadComplete(char* in_file) 
{ 
	printf("Loading Net from %s...",in_file); 
	ifstream i_file; 
	i_file.open(in_file, ios::binary); 
	if(i_file.is_open()) 
	{ 
		int loc = 0; 

		loc = _v_h0->load(&i_file, loc); 
		loc = _h0_h1->load(&i_file, loc); 
		loc = _h1_top->load(&i_file, loc); 

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