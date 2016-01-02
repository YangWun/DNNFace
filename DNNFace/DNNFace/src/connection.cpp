#include "../inc/connection.h" 

Connection::Connection(int v_size, int h_size) 
{ 
	_v_size = v_size; 
	_h_size = h_size; 
	_w_size = _v_size * _h_size; 

	//Host memory allocation 
	_weight = (float*) malloc(_w_size * sizeof(float)); 
	_weight_t = (float*) malloc(_w_size * sizeof(float)); 
	_a = (float*) malloc(_v_size * sizeof(float)); 
	_b = (float*) malloc(_h_size * sizeof(float)); 
	_vel_weight = (float*)malloc(_w_size*sizeof(float)); 
	_dw = (float*)malloc(_w_size*sizeof(float)); 

	//Device memory allocation 
	dev_alloc(&d_weight,_w_size *sizeof(float)); 
	dev_alloc(&d_weight_t,_w_size *sizeof(float)); 
	dev_alloc(&d_a, _v_size * sizeof(float)); 
	dev_alloc(&d_b, _h_size * sizeof(float)); 
	dev_alloc(&d_vel_weight,_w_size *sizeof(float)); 
	dev_alloc(&d_dw,_w_size *sizeof(float)); 

	return; 
} 

/* --------------------------------------------------------------- 
*    DESTRUCTOR 
------------------------------------------------------------------*/ 
Connection::~Connection() 
{ 
	//Host memory 
	free(_weight); 
	free(_weight_t); 
	free(_a); 
	free(_b); 
	free(_vel_weight); 
	free(_dw); 

	//Device memory 
	cudaFree(d_weight); 
	cudaFree(d_weight_t); 
	cudaFree(d_a); 
	cudaFree(d_b); 
	cudaFree(d_dw); 
	cudaFree(d_vel_weight); 
} 

void Connection::initParams() 
{ 
	//Setup Weights | i->j | Visible 0 -> Hidden 1 
	srand((unsigned)time(0)); 
	//Approximate Gaussian u=0 z=0.01, 产生Gaussian分布随机数 
	for (int i=0;i<_w_size;i++) 
	{ 
		float central = 0; 
		for(int c=0;c<100;c++) 
		{ 
			float u = (float)rand() / (float)RAND_MAX; 
			central+= (2*u -1)*0.1; 
		} 
		central /= 100; 
		_weight[i] = central; 
	} 

	cudaMemcpy(d_weight, _weight, _w_size * sizeof(float),cudaMemcpyHostToDevice); 
	//free(wij); 

	//Hidden Bias = -2 to encourage sparsity 
	for (int j=0;j<_h_size;j++) 
	{ 
		_b[j] = -2.0; 
	} 

	//Visible Bias = set to 0 
	for (int i=0;i<_v_size;i++) 
	{ 
		_a[i] = 0; 
		//printf("Pi=%f | ai[%d])=%f\n",Pi,i,log(Pi / (1-Pi))); 
	} 

	//Places on device 
	cudaMemcpy(d_a, _a,_v_size * sizeof(float),cudaMemcpyHostToDevice); 
	cudaMemcpy(d_b, _b,_h_size * sizeof(float),cudaMemcpyHostToDevice); 


	//Initialize velocities to 0 
	memset(_vel_weight,0.0,_w_size*sizeof(float)); 
	cudaMemcpy(d_vel_weight, _vel_weight, _w_size * sizeof(float), cudaMemcpyHostToDevice); 


} 

int Connection::save(ofstream *o_file, int loc) 
{ 
	//place data on host 
	cudaMemcpy(_weight, d_weight, _w_size * 
		sizeof(float),cudaMemcpyDeviceToHost); 
	cudaMemcpy(_weight_t, d_weight_t, _w_size * 
		sizeof(float),cudaMemcpyDeviceToHost); 
	cudaMemcpy(_a, d_a,_v_size * 
		sizeof(float),cudaMemcpyDeviceToHost); 
	cudaMemcpy(_b, d_b,_h_size * 
		sizeof(float),cudaMemcpyDeviceToHost); 
	cudaMemcpy(_vel_weight, d_vel_weight, _w_size * sizeof(float), 
		cudaMemcpyDeviceToHost); 
	o_file->write((char*)_weight,_w_size * sizeof(float)); 
	loc += _w_size * sizeof(float); 
	o_file->write((char*)_weight_t,_w_size * sizeof(float)); 
	loc += _w_size * sizeof(float); 
	o_file->seekp(loc); 
	o_file->write((char*)_a,_v_size * sizeof(float)); 
	loc += _v_size * sizeof(float); 
	o_file->seekp(loc); 
	o_file->write((char*)_b,_h_size * sizeof(float)); 
	loc += _h_size * sizeof(float); 
	o_file->seekp(loc); 
	o_file->write((char*)_vel_weight,_w_size * sizeof(float)); 
	loc += _w_size * sizeof(float); 

	return loc; 
} 

int Connection::load(ifstream *i_file, int loc) 
{ 
	//load weights 
	for(int n=0;n<_w_size;n++) 
	{ 
		i_file->seekg(loc); 
		i_file->read((char*)&_weight[n],sizeof(float)); 
		loc += sizeof(float); 
		//printf("w[%d]=%f ",n,w[n]); 
	} 
	//load transposed weights 
	for(int n=0;n<_w_size;n++) 
	{ 
		i_file->seekg(loc); 
		i_file->read((char*)&_weight_t[n],sizeof(float)); 
		loc += sizeof(float); 
		//printf("w[%d]=%f ",n,w[n]); 
	} 
	//load a 
	for(int i=0;i<_v_size;i++) 
	{ 
		i_file->seekg(loc); 
		i_file->read((char*)&_a[i],sizeof(float)); 
		loc += sizeof(float); 
		//printf("a[%d]=%f\t",i,a[i]); 
	} 
	printf("\n"); 
	//load b 
	for(int j=0;j<_h_size;j++) 
	{ 
		i_file->seekg(loc); 
		i_file->read((char*)&_b[j],sizeof(float)); 
		loc += sizeof(float); 
		//printf("b[%d]=%f\t",j,b[j]); 
	}
	//load weights updates velocity
	for(int v=0;v<_w_size;v++) 
	{ 
		i_file->seekg(loc); 
		i_file->read((char*)&_vel_weight[v],sizeof(float)); 
		loc += sizeof(float); 
		//printf("vel[%d]=%f ",v,_vel_weight[v]); 
	} 
	cudaMemcpy(d_weight, _weight, _w_size * 
		sizeof(float),cudaMemcpyHostToDevice); 
	cudaMemcpy(d_weight_t, _weight_t, _w_size * 
		sizeof(float),cudaMemcpyHostToDevice); 
	cudaMemcpy(d_a, _a,_v_size * 
		sizeof(float),cudaMemcpyHostToDevice); 
	cudaMemcpy(d_b, _b,_h_size * 
		sizeof(float),cudaMemcpyHostToDevice); 
	cudaMemcpy(d_vel_weight, _vel_weight, _w_size * 
		sizeof(float),cudaMemcpyHostToDevice); 

	return loc; 
} 

/* --------------------------------------------------------------- 
*    PRINT W 
* Prints the weights in matrix form 
* 
------------------------------------------------------------------*/ 
void Connection::printW() 
{ 
	float* dubs = (float*)malloc(_w_size*sizeof(float)); 
	cudaMemcpy(dubs, d_weight, 
		_w_size*sizeof(float),cudaMemcpyDeviceToHost); 

	printf("\n"); 
	//top left 
	/*for (int i=0;i<10;i++) 
	{ 
	for (int j=0;j<10;j++) 
	{ 
	printf("[%.3f]",dubs[i*_h.size + j]*10); 
	} 
	printf("\n"); 
	}*/ 

	//bottom right 
	for (int i=_v_size - 10;i<_v_size;i++) 
	{ 
		for (int j=_h_size - 10;j<_h_size;j++) 
		{ 
			printf("[%.3f]",dubs[i*_h_size + j]*10); 
		} 
		printf("\n"); 
	} 

	printf("\n"); 
} 

/* --------------------------------------------------------------- 
*    PRINT WT 
* Prints the weights transposed in matrix form 
* 
------------------------------------------------------------------*/ 
void Connection::printWT() 
{ 
	float* dubsT = (float*)malloc(_w_size*sizeof(float)); 
	cudaMemcpy(dubsT, d_weight_t, 
		_w_size*sizeof(float),cudaMemcpyDeviceToHost); 

	printf("\n"); 
	//top left 
	/*for (int i=0;i<10;i++) 
	{ 
	for (int j=0;j<10;j++) 
	{ 
	printf("[%.3f]",dubsT[i*_v.size + j]*10); 
	} 
	printf("\n"); 
	}*/ 

	//bottom right 
	for (int i=_h_size - 10;i<_h_size;i++) 
	{ 
		for (int j=_v_size - 10;j<_v_size;j++) 
		{ 
			printf("[%.3f]",dubsT[i*_v_size + j]*10); 
		} 
		printf("\n"); 
	} 

	printf("\n"); 
}