#include "../inc/layer.h" 

Layer::Layer(int dim_x, int dim_y, int mini_batch_size, int 
	gibbs_samples, bool using_initial) 
{ 
	_dim_x = dim_x; 
	_dim_y = dim_y; 
	_size = _dim_x * _dim_y; 

	_mini_batch_size = mini_batch_size; 
	_gibbs_samples = gibbs_samples; 
	_using_initial = using_initial; 

	//Host memory allocation 
	_disp = (float*) malloc(_size * sizeof(float)); 
	_q = (float*)malloc(_size*sizeof(float)); 
	if(_using_initial) 
		dev_alloc(&d_initial_state, _size * _mini_batch_size * gibbs_samples * sizeof(float)); 

	dev_alloc(&d_state, _size * _mini_batch_size * gibbs_samples * 
		sizeof(float)); 
	dev_alloc(&d_error, _size * _mini_batch_size * gibbs_samples * 
		sizeof(float)); 
	dev_alloc(&d_rand, _size * mini_batch_size * gibbs_samples * 
		sizeof(float)); 
	dev_alloc(&d_q,_size *sizeof(float)); 

	return; 
} 

Layer::~Layer() 
{ 
	//Host memory 
	free(_disp); 
	free(_q); 

	//Device memory 
	if(_using_initial) 
		cudaFree(d_initial_state); 

	cudaFree(d_state); 
	cudaFree(d_rand); 
	cudaFree(d_q); 

	return; 
} 

void Layer::initParams() 
{ 
	//Initialize activation probablity estimation to 0 
	memset(_q,0.0,_size*sizeof(float)); 
	cudaMemcpy(d_q, _q, _size * sizeof(float), cudaMemcpyHostToDevice); 

	return; 
} 

int Layer::saveQ(ofstream* out_file, int loc) 
{ 
	//place data on host 
	cudaMemcpy(_q, d_q, _size * sizeof(float), cudaMemcpyDeviceToHost); 

	out_file->seekp(loc); 
	out_file->write((char*)_q, _size * sizeof(float)); 
	loc+=_size * sizeof(float); 
	return loc; 
} 

int Layer::loadQ(ifstream* in_file, int loc) 
{ 
	for(int q=0;q<_size;q++) 
	{ 
		in_file->seekg(loc); 
		in_file->read((char*)&_q[q],sizeof(float)); 
		loc += sizeof(float); 
		//printf("q[%d]=%f\t",q,_q_h[q]); 
	} 
	cudaMemcpy(d_q, _q,_size * sizeof(float),cudaMemcpyHostToDevice); 

	return loc; 
} 


/* --------------------------------------------------------------- 
*    RAND HIDDEN 
* Sets d_state(0,0) to random state w/ probability 
* prob  |  float |  probability of unit being on 
------------------------------------------------------------------*/ 
void Layer::randState(float prob) 
{ 
	float tmp; 
	for(int i=0;i<_size;i++) 
	{ 
		tmp = rand() % 100; 
		_disp[i] = (tmp < (prob*100)); //set target sparsity here 
	} 
	cudaMemcpy(d_state,_disp, _size * sizeof(float), cudaMemcpyHostToDevice); 
	return; 
} 

int Layer::saveState(ofstream* out_file, int loc) 
{ 
	cudaMemcpy(_disp, d_state,_size * sizeof(float),cudaMemcpyDeviceToHost); 

	out_file->seekp(loc); 
	out_file->write((char*)_disp,_size * sizeof(float)); 
	return (loc + (_size * sizeof(float))); 
} 

int Layer::saveDim(ofstream* out_file, int loc) 
{ 
	//Save Hidden layer dimensions 
	out_file->seekp(loc); 
	out_file->write((char*)&_dim_x, sizeof(int)); 

	loc += sizeof(int); 

	out_file->seekp(loc); 
	out_file->write((char*)&_dim_y, sizeof(int)); 

	loc += sizeof(int); 

	return loc; 
} 

void Layer::checkSparsity() 
{ 
	cudaMemcpy(_q, d_q, _size*sizeof(float),cudaMemcpyDeviceToHost); 

	float sum=0; 
	float max=0; 
	float min=1; 
	for(int j=0;j<_size;j++) 
	{ 
		if(_q[j] > max) 
			max = _q[j]; 
		if(_q[j] < min) 
			min = _q[j]; 
		sum+= _q[j]; 
	} 

	printf("Avg q_h:%f\t",sum/(float)_size); 
	printf("Min q_h:%f\t",min); 
	printf("Max q_h:%f\n",max); 

} 


/* --------------------------------------------------------------- 
 *    PRINT HIDDENX 
 * Draws the selected state of the layer 
 * 
 * final | bool | true=current state, false=initial state 
 * b | int | layer batch 
 * g | int | layer fantasy particle 
 ------------------------------------------------------------------*/
void Layer::printState(bool current, int b, int g) 
{ 
  if(b>=_mini_batch_size || b<0) 
  { 
    printf("Selected batch is out of range\n"); 
    return; 
  } 
  else 
  { 
    if(current) 
    { 
      cudaMemcpy(_disp, &d_state[b*_gibbs_samples*_size + g*_size],_size * sizeof(float), cudaMemcpyDeviceToHost);
	  for(int i=0;i<_size;i++)
		  cout<<setw(4)<<_disp[i]<<" ";
	  cout<<endl;
	  return;
    } 
    else 
    { 
      if(!_using_initial) 
      { 
        printf("Attempting to view unused initial state\n"); 
        return; 
      } 
      else 
      { 
     
  cudaMemcpy(_disp,&d_initial_state[b*_gibbs_samples*_size + g*_size],_size * sizeof(float), cudaMemcpyDeviceToHost); 
	  for(int i=0;i<_size;i++)
		  cout<<setw(4)<<_disp[i]<<" ";
	  cout<<endl; 
      } 
    } 
 
  } 
} 


void Layer::printHrand(int b, int g){
	cudaMemcpy(_disp, &d_rand[b*_gibbs_samples*_size + g*_size],_size * sizeof(float), cudaMemcpyDeviceToHost);
	for(int i=0;i<_size;i++)
		cout<<setw(4)<<_disp[i]<<" ";
	cout<<endl;
	return;


}