#include "./inc/connection.h" 
#include "./inc/rbm.h" 
#include "./inc/trainer.h" 
#include <curand.h>
#include "cuPrintf.cu"

using namespace utilLearn; 


//RBM LEARNING PARAMETERS
//>--SAMPLES为GIBS_SAMPLES
#define BATCH      100 
#define SAMPLES      1 
#define STEPS      1 
#define EPOCH      200 

//RBM SIZE PARAMETERS 
#define  VSIZE_X      96 
#define VSIZE_Y      96 
#define VSIZE      9216 


//#define  HSIZE_X      64 
//#define HSIZE_Y      64 
//#define HSIZE      4096 
#define  HSIZE_X      32 
#define HSIZE_Y      32 
#define HSIZE      1024 



char * const param_file = "params/norb-persistent-lvl1.rbm"; 
char * const converted_data_file = "data/norb-images-lvl1.floats";

char * const data_file = "data/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat"; 

//For Transpose 
#define TILE_DIM   16 
#define BLOCK_ROWS    16 
#define  BLOCKS_LAYER  16 


void train_novis(); 


//Helper Functions 
void init_fantasy(); 
void train_mini_persistent(); 
void update_params(); 
void convert(char* out_file); 

//CUDA Functions 
__global__ void transpose(float *w, float *wt); 
__global__ void upPassInitProb(float* v0, float* h0, float* b, float* w); 
__global__ void upPassInit(float* v0, float* h0, float* b, float* w, float* rnd); 
__global__ void upPass(float* vX, float* hX, float* b, float* w, float* rnd); 
__global__ void upPassProb(float* vX, float* hX, float* b, float* w); 
__global__ void downPass(float* vX, float* hX, float* a, float* wt); 
__global__ void updateW(float* v0, float* vX, float* h0, float* hX, float* w, float* vel_w, float momentum, float* q, float* dw, float l_rate); 
__global__ void updateA(float* v0, float* vX, float* a, float l_rate); 
__global__ void updateB(float* h0, float* hX, float* b, float* q, float l_rate); 


//Globals 
Rbm *my_rbm; 
Trainer *my_trainer;
curandGenerator_t d_rand; 

//float total_time; 


int main(int argc, char** argv) 
{ 

	//total_time = 0; 
	//Set GPU 1 (currently not used for display) 
	cudaSetDevice(1); 

	//Set up basic units 
	Connection* my_connection = new Connection(VSIZE, HSIZE); 
	Layer* my_visible = new Layer(VSIZE_X, VSIZE_Y, BATCH, SAMPLES, false); 
	Layer* my_hidden = new Layer(HSIZE_X, HSIZE_Y, BATCH, SAMPLES, true); 
 
	my_rbm = new Rbm(my_visible, my_hidden, my_connection); 
	my_trainer = new Trainer(BATCH, SAMPLES, EPOCH, 0.001, 10, 0.5, 0.0001); 
	//Load Data Set 
	if(my_trainer->loadTrainingDataMAT(data_file) < 0) 
	{ 
		printf("An error occurred loading the training data. Exiting...\n"); 
		return -1; 
	}


	//Set up RBM Parameters 

	printf("Initializing Paramters\n"); 
	my_rbm->initParams(); 
	dim3 grid(my_rbm->getHSize()/TILE_DIM, my_rbm->getVSize()/TILE_DIM), threads(TILE_DIM,BLOCK_ROWS); 
	transpose<<<grid,threads>>>(my_rbm->getDevW(), my_rbm->getDevWT()); 

	printf("Setting visual bias to data probability\n"); 
	//Set visual bias to training data probability 
	for(int i=0;i<my_connection->getVSize();i++) 
	{ 
		//printf("visual bias[%d]=%f\n",i, my_trainer->pixelProb(i)); 
		my_connection->setA(i,my_trainer->pixelProb(i)); 
	} 
	my_connection->cpyA(); //Place on device 

	//Set up Random Initializer 
	curandCreateGenerator(&d_rand, CURAND_RNG_PSEUDO_MTGP32); 
	srand((unsigned)time(0)); 
	int seed = (rand() % 1000); 
	curandSetPseudoRandomGeneratorSeed(d_rand, seed); 

	init_fantasy(); 

	while(!my_trainer->trainComplete()) 
	{ 
		train_mini_persistent(); 
 
		//Update training status 
		my_trainer->incN(); 
		if(my_trainer->epochComplete()) 
		{ 
			printf("Epoch %d Complete!\n", my_trainer->getEpoch()); 
			//my_rbm->save(param_file); 
			//saveFreeNRG(); 
			my_rbm->checkSparsityH(); 
		} 

	} 

	printf("Training run complete!\n"); 
	my_rbm->save(param_file);

	//train_novis(); 
	return 0; 
} 

//>--Used for Persistent CD
void init_fantasy() 
{ 
	my_trainer->randBatchV(); 
	curandGenerateUniform(d_rand, (float *) my_rbm->getHrand(), my_rbm->getHSize()*my_trainer->getNumFantasy()); 

	//Calculate HX 
	dim3 blockDim(BLOCKS_LAYER,SAMPLES,BATCH); 
	dim3 threadDimUp(my_rbm->getHSize()/BLOCKS_LAYER);
	//>--用输入数据初始化HX
	upPassInit<<<blockDim,threadDimUp>>>(my_trainer->d_mini_batch_data,my_rbm->getHX(),my_rbm->getDevB(),my_rbm->getDevW(), my_rbm->getHrand()); 
} 


void train_novis() 
{ 


} 


void train_mini_persistent() 
{ 
	//Select Batch Samples V0 
	my_trainer->randBatchV(); 
	curandGenerateUniform(d_rand, (float *) my_rbm->getHrand(), my_rbm->getHSize()*my_trainer->getNumFantasy()); 



	//Calculate H0 (这一小段代码可以注释掉？--->不能注释掉，因为在更新权重时需要H0)
	dim3 blockDim(BLOCKS_LAYER,SAMPLES,BATCH); 
	dim3 threadDimUp(my_rbm->getHSize()/BLOCKS_LAYER); 
	dim3 threadDimDown(my_rbm->getVSize()/BLOCKS_LAYER); 
	upPassInit<<<blockDim,threadDimUp>>>(my_trainer->d_mini_batch_data,my_rbm->getH0(),my_rbm->getDevB(),my_rbm->getDevW(), my_rbm->getHrand()); 

	//Calculate V1
	//>--由上次迭代产生模型参数HX计算V1(Persistent CD 算法)
	downPass<<<blockDim,threadDimDown>>>(my_rbm->getVX(),my_rbm->getHX(),my_rbm->getDevA(),my_rbm->getDevWT()); 


	//Iterate over gibbs steps HX and VX 
	//for (int g=1;g<STEPS;g++) 
	//{ 
	//	curandGenerateUniform(d_rand, (float *) my_rbm->getHrand(),my_rbm->getHSize()*my_trainer->getNumFantasy()); 
	//	upPass<<<blockDim,threadDimUp>>>(my_rbm->getVX(),my_rbm->getHX(),my_rbm->getDevB(),my_rbm->getDevW(), my_rbm->getHrand()); 
	//	downPass<<<blockDim,threadDimDown>>>(my_rbm->getVX(),my_rbm->getHX(),my_rbm->getDevA(),my_rbm->getDevWT()); 
	//} 



	//Calculate HX (probabilities for update)
	//>--由V1计算HX (此时的HX是概率值，用于W, a, b的更新，非0/1二值)
	upPassProb<<<blockDim,threadDimUp>>>(my_rbm->getVX(),my_rbm->getHX(),my_rbm->getDevB(),my_rbm->getDevW()); 

	float *tmp4=(float*) malloc(my_rbm->getHSize() *my_trainer->getNumFantasy()* sizeof(float));
	cudaMemcpy(tmp4, my_rbm->getHX(), my_rbm->getHSize() * my_trainer->getNumFantasy()*sizeof(float), cudaMemcpyDeviceToHost);
	for(int i=0;i< my_rbm->getHSize();i++){
		cout<<tmp4[i]<<endl;
	}

	update_params(); 


	//>--计算当前迭代的模型参数HX(0/1二值)供下次迭代使用(Persistent CD 算法)
	upPass<<<blockDim,threadDimUp>>>(my_rbm->getVX(),my_rbm->getHX(),my_rbm->getDevB(),my_rbm->getDevW(), my_rbm->getHrand()); 

	bool *tmp=(bool*) malloc(my_rbm->getHSize() * sizeof(bool));
	cudaMemcpy(tmp, my_rbm->getH0(), my_rbm->getHSize() * sizeof(bool), cudaMemcpyDeviceToHost);
	for(int i=0;i< my_rbm->getHSize();i++){
		cout<<tmp[i]<<endl;
	}

	return; 
} 

void update_params() 
{ 


	//Update Parameters 
	dim3 threadDimUp(my_rbm->getHSize()/BLOCKS_LAYER); 
	dim3 threadDimDown(my_rbm->getVSize()/BLOCKS_LAYER); 
	dim3 updateBlockDim(BLOCKS_LAYER,my_rbm->getVSize());

	//my_rbm->checkSparsityH();
	updateW<<<updateBlockDim,threadDimUp>>>(my_trainer->d_mini_batch_data,my_rbm->getVX(),my_rbm->getH0(),my_rbm->getHX(),my_rbm->getDevW() 
		,my_rbm->getDevVw(), my_trainer->getMomentum(), my_rbm->getHQ(), my_rbm->getDevDw(), my_trainer->getLearnRate()); 
	//my_rbm->checkSparsityH();

	float *tmp3=(float*) malloc(my_rbm->getHSize() * my_rbm->getVSize() * sizeof(float));
	cudaMemcpy(tmp3, my_rbm->getDevW(), my_rbm->getHSize() * my_rbm->getVSize() * sizeof(float), cudaMemcpyDeviceToHost);
	for(int i=0;i< my_rbm->getHSize() * my_rbm->getVSize();i++){
		cout<<tmp3[i]<<endl;
	}

	dim3 grid(my_rbm->getHSize()/TILE_DIM, my_rbm->getVSize()/TILE_DIM);
	dim3 threads(TILE_DIM,BLOCK_ROWS); 
	transpose<<<grid,threads>>>(my_rbm->getDevW(), my_rbm->getDevWT());

	float *tmp4=(float*) malloc(my_rbm->getHSize() * my_rbm->getVSize() * sizeof(float));
	cudaMemcpy(tmp4, my_rbm->getDevWT(), my_rbm->getHSize() * my_rbm->getVSize() * sizeof(float), cudaMemcpyDeviceToHost);
	//for(int i=0;i< my_rbm->getHSize() * my_rbm->getVSize();i++){
	//	cout<<tmp4[i]<<endl;
	//}
	for(int i=0;i<my_rbm->getVSize();i++)
		for(int j=0;j<my_rbm->getHSize();j++){
			cout<<tmp3[i*my_rbm->getHSize()+j]<<" "<<tmp4[j*my_rbm->getVSize()+i]<<endl;
		}


	//float *tmp4=(float*) malloc(my_rbm->getHSize() * sizeof(float));
	//cudaMemcpy(tmp4, my_rbm->getHrand(), my_rbm->getHSize() * sizeof(float), cudaMemcpyDeviceToHost);
	//for(int i=0;i< my_rbm->getHSize();i++){
	//	cout<<tmp4[i]<<endl;
	//}

	updateA<<<BLOCKS_LAYER,threadDimDown>>>(my_trainer->d_mini_batch_data,my_rbm->getVX(),my_rbm->getDevA(),my_trainer->getLearnRate()); 
	updateB<<<BLOCKS_LAYER,threadDimUp>>>(my_rbm->getH0(),my_rbm->getHX(),my_rbm->getDevB(), my_rbm->getHQ(), my_trainer->getLearnRate()); 
	return;
} 

void convert(char* out_file) 
{ 

	printf("Converting Data...\n"); 
	ofstream o_file; 
	o_file.open(out_file, ios::binary); 

	int loc = 0; 
	if(o_file.is_open()) 
	{ 
		//Save number of training images 
		int num = my_trainer->getTrainSize(); 
		o_file.seekp(loc); 
		o_file.write((char*)&num, sizeof(int)); 

		loc += sizeof(int); 

		loc = my_rbm->saveHDim(&o_file, loc); 

		dim3 blockDim(BLOCKS_LAYER,1,1); 
		dim3 threadDimUp(my_rbm->getHSize()/BLOCKS_LAYER); 

		for(int i=0;i<num;i++) 
		{ 
			//printf("Converting Image: %d\n",i); 
			//Select Batch Samples V0 
			my_trainer->setV(i,0); 

			//Calculate H0 


			upPassInitProb<<<blockDim,threadDimUp>>>(my_trainer->d_mini_batch_data,my_rbm->getHX(),my_rbm->getDevB(),my_rbm->getDevW()); 
			loc = my_rbm->saveH(&o_file,loc); 


		} 
		o_file.close(); 
		printf("Completed\n"); 
	} 
	else 
		printf("Failed\n"); 
	return; 

} 

/*==================================================================
= 
*       CUDA FUNCTIONS 
===================================================================*
/ 

/* --------------------------------------------------------------- 
*    UP PASS INIT 
* Initial V0->H0 pass. This is necessarily different because 
* all fantasy particles use the same initial V0. 
* 
* v0 | float* | Training examples 
* h0 | float* | Hidden Layers to calculate 
* b  | float* | Bias to hidden units 
* w  | float* | Weights 
* rnd| float* | Random vectors to compete H prob to 
------------------------------------------------------------------*/ 
__global__ void upPassInit(float* v0, float* h0, float* b, float* w, 
	float* rnd) 
{ 
	int h_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 
	int g_off = blockIdx.y; 
	int b_off = blockIdx.z; 
	int t_off = ( (b_off * gridDim.y + g_off) * HSIZE ) + h_idx; 

	float sum = b[h_idx]; 
	//printf("sum = %f \n",b[h_idx]); 
	for(int i=0;i<VSIZE;i++) 
	{ 
		sum += v0[b_off*VSIZE + i] * w[ i*HSIZE + h_idx]; 
	} 
	//printf("sum = %f \n",b[h_idx]); 
	float prob = 1 / (1 + __expf(-1 * sum)); 

	//printf("p(H[%d]=1|v) = %f > %f\n",h_idx, prob, rnd[h_idx + b_offset]); 
	h0[t_off] = (prob > rnd[t_off]); 
} 

__global__ void upPassInitProb(float* v0, float* h0, float* b, 
	float* w) 
{ 
	int h_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 
	int g_off = blockIdx.y; 
	int b_off = blockIdx.z; 
	int t_off = ( (b_off * gridDim.y + g_off) * HSIZE ) + h_idx; 

	float sum = b[h_idx]; 
	//printf("sum = %f \n",b[h_idx]); 
	for(int i=0;i<VSIZE;i++) 
	{ 
		sum += v0[b_off*VSIZE + i] * w[ i*HSIZE + h_idx]; 
	} 
	//printf("sum = %f \n",b[h_idx]); 
	h0[t_off] = 1 / (1 + __expf(-1 * sum)); 
} 

/* --------------------------------------------------------------- 
*    UP PASS 
* Any VX->HX pass. Output is Binary. 
* 
* vX | float* | Visible Layers to use 
* hX | float* | Hidden Layers to calculate 
* b  | float* | Bias to hidden units 
* w  | float* | Weights 
* rnd| float* | Random vectors to compete H prob to 
------------------------------------------------------------------*/ 
__global__ void upPass(float* vX, float* hX, float* b, float* w, float* rnd) 
{ 
	int h_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 
	int g_off = blockIdx.y; 
	int b_off = blockIdx.z * gridDim.y; 
	int t_off = ( (b_off + g_off) * HSIZE ) + h_idx; 

	float sum = b[h_idx]; 
	//printf("sum = %f \n",b[h_idx]); 
	for(int i=0;i<VSIZE;i++) 
	{ 
		sum += vX[(b_off + g_off)*VSIZE + i] * w[ i*HSIZE + 
			h_idx]; 
	} 
	//printf("sum = %f \n",b[h_idx]); 
	float prob = 1 / (1 + __expf(-1 * sum)); 

	//printf("p(H[%d]=1|v) = %f > %f\n",h_idx, prob, rnd[h_idx + b_offset]); 
	hX[t_off] = (prob > rnd[t_off]); 
} 

/* --------------------------------------------------------------- 
*    UP PASS PROB 
* Final VX->HX pass. Output is probability. 
* 
* vX | float* | Visible Layers to use 
* hX | float* | Hidden Layers to calculate 
* b  | float* | Bias to hidden units 
* w  | float* | Weights 
------------------------------------------------------------------*/ 
__global__ void upPassProb(float* vX, float* hX, float* b, float* w) 
{ 
	int h_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 
	int g_off = blockIdx.y;
	int b_off = blockIdx.z * gridDim.y; 
	int t_off = ( (b_off + g_off) * HSIZE ) + h_idx; 

	float sum = b[h_idx]; 
	//printf("sum = %f \n",b[h_idx]); 
	for(int i=0;i<VSIZE;i++) 
	{ 
		sum += vX[(b_off + g_off)*VSIZE + i] * w[ i*HSIZE + 
			h_idx]; 
	} 
	//printf("sum = %f \n",b[h_idx]); 
	hX[t_off] = 1 / (1 + __expf(-1 * sum)); 
} 

/* --------------------------------------------------------------- 
*    DOWN PASS 
* Any HX->VX pass. Output is probability. 
* 
* vX | float* | Visible Layers to calculate 
* hX | float* | Hidden Layers to use 
* a  | float* | Bias to visible units 
* wt | float* | Weights Transposed 
------------------------------------------------------------------*/ 
__global__ void downPass(float* vX, float* hX, float* a, float* wt) 
{ 
	int v_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 
	int g_off = blockIdx.y; 
	int b_off = blockIdx.z * gridDim.y; 
	int t_off = ( (b_off + g_off) * VSIZE ) + v_idx; 

	float sum = a[v_idx]; 
	//printf("sum = %f \n",b[h_idx]); 
	for(int i=0;i<HSIZE;i++) 
	{ 
		//sum += hX[b_off + g_off + i] * w[ i*512 + v_idx]; 
		sum += hX[(b_off + g_off)*HSIZE + i] * wt[ i*VSIZE + v_idx]; 
	} 
	//printf("sum = %f \n",b[h_idx]); 
	vX[t_off] = 1 / (1 + __expf(-1 * sum)); 

} 

/* --------------------------------------------------------------- 
*    UPDATE W 
* Calculates the change to the weights 
* 
* v0     | float* | Visible layer from data 
* vX     | float* | Final Visible layer from model 
* h0     | float* | Hidden layer one pass from data 
* hX     | float* | Hidden layer from model 
* w      | float* | Weights
* momentum | float* | 动量学习率
* q       | float* | problity
* dw      | float* | weights updates
* l_rate  | float  | learning rate 
------------------------------------------------------------------*/ 
__global__ void updateW(float* v0, float* vX, float* h0, float* hX, 
	float* w, float* vel_w, float momentum, float* q, float* dw, float 
	l_rate) 
{ 
	int h_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 
	int v_idx = (blockIdx.y); 
	int v_offset = v_idx * blockDim.x * gridDim.x; 

	float delta = 0.0; 
	float sum_h = 0.0; 

	for(int batch=0;batch<BATCH;batch++) 
	{ 
		for(int gibbs=0;gibbs<SAMPLES;gibbs++) 
		{ 
			int h_off = h_idx + batch*SAMPLES*HSIZE + 
				gibbs*HSIZE; 
			int v_off = v_idx + batch*SAMPLES*VSIZE + 
				gibbs*VSIZE; 

			delta += (v0[v_idx + batch*VSIZE] * h0[h_off]) - 
				(vX[v_off] * hX[h_off]); 
			sum_h += (hX[h_off]); 
		} 
	} 

	//Calculate probability estimate 
	//>--(sum_h / (BATCH * SAMPLES)) is the current estimated sparsity problity
	q[h_idx] = ((.95)*q[h_idx]) + (1-.95)*(sum_h / (BATCH * SAMPLES)); 



	//if(v_idx == 200 && h_idx < 5) 
	//printf("Q = %f\n", q[h_idx]); 
	//if(h_idx + v_offset == 555) 
	//printf("w[%d]=%f += %f\n", h_idx + v_offset, w[h_idx + v_offset], delta); 

	//VELOCITY 
	vel_w[h_idx + v_offset] = momentum * vel_w[h_idx + v_offset] + ( (delta * l_rate) / (SAMPLES * BATCH) ); 

	//DECAY 
	float decay = (0.0005 * w[h_idx + v_offset] ) * l_rate; 


	//SPARSITY 
	// = penalty * ( probability estimation - probability target) 
	float sparsity = 0.0001 * (q[h_idx]-0.1); 

	dw[h_idx + v_offset] = (vel_w[h_idx + v_offset] - decay - sparsity); 
	w[h_idx + v_offset] += (vel_w[h_idx + v_offset] - decay - sparsity); 

	//VELOCITY AND SPARSITY ONLY 
	//dw[h_idx + v_offset] = (vel_w[h_idx + v_offset]  - sparsity); 
	//w[h_idx + v_offset] += (vel_w[h_idx + v_offset]  - sparsity); 

	//w[h_idx + v_offset] += (delta * l_rate) / (SAMPLES * BATCH); 
	//w[h_idx + v_offset] = delta; 

	//dw[h_idx + v_offset] = ((delta * l_rate) / (SAMPLES * BATCH) ) - (decay * l_rate); 
	//w[h_idx + v_offset] += ((delta * l_rate) / (SAMPLES * BATCH) ) - (decay * l_rate); 
} 

/* --------------------------------------------------------------- 
*    UPDATE A 
* Calculates the change to the visible bias 
* 
* v0     | float* | Visible layer from data 
* vX     | float* | Final Visible layer from model 
* a      | float* | Visible bias 
* l_rate  | float  | learning rate 
------------------------------------------------------------------*/ 
__global__ void updateA(float* v0, float* vX, float* a, float 
	l_rate) 
{ 
	int v_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 

	float delta = 0.0; 

	for(int batch=0;batch<BATCH;batch++) 
	{ 
		for(int gibbs=0;gibbs<SAMPLES;gibbs++) 
		{ 
			int v_off = v_idx + batch*SAMPLES*VSIZE + 
				gibbs*VSIZE; 
			delta += (v0[v_idx + batch*VSIZE]) - (vX[v_off]); 
		} 
	} 

	a[v_idx] += ( (delta * l_rate) / (SAMPLES * BATCH) ); 
}
/* --------------------------------------------------------------- 
*    UPDATE B 
* Calculates the change to the hidden bias 
* 
* h0     | float* | Hidden layer one pass from data 
* hX     | float* | Hidden layer from model 
* b      | float* | Hidden bias 
* l_rate  | float  | learning rate 
------------------------------------------------------------------*/ 
__global__ void updateB(float* h0, float* hX, float* b, float* q, 
	float l_rate) 
{ 
	int h_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 

	float delta = 0.0; 

	for(int batch=0;batch<BATCH;batch++) 
	{ 
		for(int gibbs=0;gibbs<SAMPLES;gibbs++) 
		{ 
			int h_off = h_idx + batch*SAMPLES*HSIZE + 
				gibbs*HSIZE; 
			delta += (h0[h_off]) - (hX[h_off]); 
		} 
	} 

	//float sparsity = (0.0001 * ((q[h_idx]-0.04)*(q[h_idx]-0.04)) ); 
	// = penalty * ( probability estimation - probability target) 
	float sparsity = 0.0001 * (q[h_idx]-0.1); 

	//if(h_idx < 5) 
	//printf("sparsity penalty = %f\n",sparsity); 

	b[h_idx] += ( (delta * l_rate) / (SAMPLES * BATCH) )  - sparsity; 
	//b[h_idx] += ( (delta * l_rate) / (SAMPLES * BATCH) ); 
} 

/* --------------------------------------------------------------- 
*    TRANSPOSE 
* Coalesced transpose with no bank conflicts. 
* 
* w  | float* | Weights 
* wt | float* | Weights Transposed 
------------------------------------------------------------------*/ 
__global__ void transpose(float *w, float *wt) 
{ 
	__shared__ float tile[TILE_DIM][TILE_DIM+1]; 
	int xIndex = blockIdx.x * TILE_DIM + threadIdx.x; 
	int yIndex = blockIdx.y * TILE_DIM + threadIdx.y; 
	int index_in = xIndex + (yIndex)*HSIZE; 

	xIndex = blockIdx.y * TILE_DIM + threadIdx.x; 
	yIndex = blockIdx.x * TILE_DIM + threadIdx.y; 
	int index_out = xIndex + (yIndex)*VSIZE; 

	for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) 
	{ 
		tile[threadIdx.y+i][threadIdx.x] = w[index_in+i*HSIZE]; 
	} 

	__syncthreads(); 

	for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) 
	{ 
		wt[index_out+i*VSIZE] = tile[threadIdx.x][threadIdx.y+i]; 
	} 

	//CUPRINTF("\tw[0] is:%f\n", w[0]);
	//CUPRINTF("\twt[0] is:%f\n", wt[0]);
} 

