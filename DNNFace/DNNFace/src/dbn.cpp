#include "../inc/dbn.h"
Dbn::Dbn(Layer *vis, Layer *h0, Layer *h1, Connection * v_h0, 
	Connection * h0_h1) 
{ 
	_vis = vis; 
	_h0 = h0; 
	_h1 = h1; 

	_v_h0 = v_h0; 
	_h0_h1 = h0_h1; 
} 

Dbn::~Dbn() 
{ 

} 

void Dbn::saveLayers(char* out_file_lvl1, char* out_file_lvl2) 
{ 
	printf("Saving Layers..."); 
	ofstream o_file; 
	o_file.open(out_file_lvl1, ios::binary); 

	int loc = 0; 
	if(o_file.is_open()) 
	{ 
		loc = _v_h0->save(&o_file, 0); 
		o_file.close(); 
		printf("Completed Layer 1\n"); 
	} 
	else 
		printf("Failed Layer 1\n"); 

	o_file.open(out_file_lvl2, ios::binary); 

	loc = 0; 
	if(o_file.is_open()) 
	{ 
		loc = _h0_h1->save(&o_file, loc); 
		o_file.close(); 
		printf("Completed Layer 2\n"); 
	} 
	else 
		printf("Failed Layer 2\n"); 



	return; 
} 

int Dbn::loadLayers(char* in_file_lvl1, char* in_file_lvl2) 
{ 
	printf("Loading Layer from %s...",in_file_lvl1); 
	ifstream i_file; 
	i_file.open(in_file_lvl1, ios::binary); 
	if(i_file.is_open()) 
	{ 
		int loc = 0; 

		loc = _v_h0->load(&i_file, loc); 

		i_file.close(); 

		printf("Completed!\n"); 
	} 
	else 
	{ 
		printf("Failed to open file\n"); 
		return -1; 
	} 

	printf("Loading Layer from %s...",in_file_lvl2); 
	i_file.open(in_file_lvl2, ios::binary); 
	if(i_file.is_open()) 
	{ 
		int loc = 0; 

		loc = _h0_h1->load(&i_file, loc); 

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