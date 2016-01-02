#include "./inc/connection.h" 
#include "./inc/nn.h" 
#include "./inc/trainer.h" 
 
 
//NN LEARNING PARAMETERS 
#define BATCH      100 
#define SAMPLES      1 
#define EPOCH      200 
 
//NN SIZE PARAMETERS 
#define VSIZE_X      28 
#define  VSIZE_Y      28 
#define VSIZE      784 
/*#define VSIZE_X      96 
#define  VSIZE_Y      96 
#define VSIZE      9216*/ 
/*#define VSIZE_X      48 
#define  VSIZE_Y      48 
#define VSIZE      2304*/ 

 
#define  H0SIZE_X    32 
#define H0SIZE_Y    16 
#define H0SIZE     512 
/*#define  H0SIZE_X    64 
#define H0SIZE_Y   64 
#define H0SIZE      4096*/ 
//#define  H0SIZE_X    32 
//#define H0SIZE_Y    32 
//#define H0SIZE     1024 


#define  H1SIZE_X    32 
#define H1SIZE_Y    16 
#define H1SIZE     512 
/*#define  H1SIZE_X    64 
#define H1SIZE_Y   64 
#define H1SIZE      4096*/
//#define  H1SIZE_X    32 
//#define H1SIZE_Y    32 
//#define H1SIZE     1024 

 
#define CLASS      10 
 
//NN MODE 
#define MODE      1 //0=train 1=classification 
#define LOAD      2 //0=fresh 1=lower_params 2=complete 
#define TRAIN_EXAMPLE 2 
 
#define  BLOCKS_LAYER  16 
 
//Display Functions 
void train(); 
void test(); 
 
 
void classifyTrain(); 
void classifyValid(); 
void validationError(); 
 
//Helper Functions 
int  load_from_rbm(char* rbm_file, Connection* con); 
void train_mini(); 
 
//File list 
//char * const param_file_v_h0 = "params/persistent-lvl1.rbm"; 
//char * const param_file_h0_h1 = "params/persistent-lvl2.rbm"; 
//char * const param_file_complete = "params/persistent-full.nn"; 
 
char * const param_file_v_h0 = "params/facial-persistent-lvl1.rbm"; 
char * const param_file_h0_h1 = "params/facial-persistent-lvl2.rbm"; 
//char * const param_file_complete = "params/facial-full.nn"; 
char * const param_file_complete = "params/mnist-full.nn"; 
 
//char * const param_file_v_h0 = "params/persistent-lvl1.rbm"; 
//char * const param_file_h0_h1 = "params/persistent-lvl2.rbm"; 
 
//char * const param_file_complete = "params/dbn-full.nn"; 
 
//char * const label_file = "mnist/train-labels.idx1-ubyte";//
//char * const label_file = "data/train-28709_2_48_48-label.bin"; 
//char * const data_file = "data/train-28709_2_48_48-data.bin"; 
//char * const label_file = "data/test-7178_2_48_48-label.bin"; 
//char * const data_file = "data/test-7178_2_48_48-data.bin"; 
char * const label_file = "mnist/t10k-labels.idx1-ubyte"; 
char * const data_file = "mnist/t10k-images.idx3-ubyte"; 



/*char * const label_file = "data/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat"; 
char * const data_file = "data/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat";*/ 
 
//char * const test_label_file = "data/test-7178_2_48_48-label.bin"; 
//char * const test_data_file = "data/test-7178_2_48_48-data.bin"; 
 
//CUDA Functions 
__global__ void vToH0(float* v0, float* h0, float* aj, float* b, float* w); 
__global__ void H0ToH1(float* h0, float* h1, float* aj, float* b, float* w); 
__global__ void H1ToTop(float* h1, float* top, float* b, float* w); 
__global__ void calcErrorTop(float* top, float* label, float* error); 
__global__ void calcErrorH1(float* top_error, float* aj, float* w, float* error); 
__global__ void calcErrorH0(float* h1_error, float* aj, float* w, float* error); 
__global__ void updateW0(float* v0, float* error, float* w, float* vel_w, float momentum, float l_rate); 
__global__ void updateW1(float* h0, float* error, float* w, float* vel_w, float momentum, float l_rate); 
__global__ void updateW2(float* h1, float* error, float* w, float* vel_w, float momentum, float l_rate); 
__global__ void updateBiasH0(float* error, float* bias, float l_rate); 
__global__ void updateBiasH1(float* error, float* bias, float l_rate); 
__global__ void updateBiasTop(float* error, float* bias, float l_rate); 
 
//Globals 
Nn *my_nn; 
Trainer *my_trainer; 
 
int test_iter; 
int example; 
 
//float total_time; 
 
int main(int argc, char** argv) 
{ 

  cudaSetDevice(1); 
 
  example = 0; 
  test_iter = 0; 
  //total_time= 0; 
 
  Layer* visible = new Layer(VSIZE_X, VSIZE_Y, BATCH, SAMPLES, false); 
  Connection* v_to_h0 = new Connection(VSIZE, H0SIZE); 
  Layer* h0 = new Layer(H0SIZE_X, H0SIZE_Y, BATCH, SAMPLES, true); 
  Connection* h0_to_h1 = new Connection(H0SIZE, H1SIZE); 
  Layer* h1 = new Layer(H1SIZE_X, H1SIZE_Y, BATCH, SAMPLES, true); 
  Connection* h1_to_top = new Connection(H1SIZE, CLASS); 
  Layer* top = new Layer(CLASS, 1, BATCH, SAMPLES, false); 
 
  my_nn = new Nn(visible, h0, h1, top, v_to_h0, h0_to_h1, h1_to_top); 
  my_trainer = new Trainer(BATCH, SAMPLES, EPOCH, 0.0001, CLASS, 0.5, 0.00005); 
  //my_trainer = new Trainer(BATCH, SAMPLES, EPOCH, 0.001, CLASS, 0.3, 0.00005); 

  //LOAD=1;
  if(LOAD==0) 
  { 
  //Init top connection 
    v_to_h0->initParams(); 
    h0_to_h1->initParams(); 
    h1_to_top->initParams(); 
    //Set bias towards top to 
    my_nn->setTopProb(); 
  } 
  else if(LOAD==1) 
  { 
    //Load connection information 
    if(  load_from_rbm(param_file_h0_h1, h0_to_h1) < 0) 
    { 
      printf("Failed to load params from %s\n",param_file_h0_h1); 
      return -1; 
    } 
    if(  load_from_rbm(param_file_v_h0, v_to_h0) < 0) 
    { 
      printf("Failed to load params from %s\n",param_file_v_h0); 
      return -1; 
    } 
    //Init top connection 
    h1_to_top->initParams(); 
    //Set bias towards top to 
    my_nn->setTopProb(); 
  } 
  else if(LOAD==2) 
  { 
    my_nn->loadComplete(param_file_complete); 
  } 
 
 
  //if(my_trainer->loadTrainingData(test_data_file) < 0)
  if(my_trainer->loadTrainingData(data_file) < 0) 
  //if(my_trainer->loadTrainingDataMAT(data_file) < 0) 
  { 
    printf("An error occurred loading the training data. Exiting...\n"); 
    return -1; 
  } 
  //if(my_trainer->loadTrainingLabels(test_label_file) < 0)
  if(my_trainer->loadTrainingLabels(label_file) < 0) 
  //if(my_trainer->loadTrainingLabelsMAT(label_file) < 0) 
  { 
    printf("An error occurred loading the training labels. Exiting...\n"); 
    return -1; 
  } 
  
 
  //MODE=0;
  if(MODE==0) 
  { 
    //printf("Correct Label for Item %d: %d\n", TRAIN_EXAMPLE, my_trainer->answer(TRAIN_EXAMPLE)); 
    train(); 
    //glutDisplayFunc(test); 
  } 
  else if(MODE==1) 
    //classifyValid(); 
    classifyTrain(); 

  return 0; 
 
 
} 
 
/*==================================================================
= 
 *       DISPLAY FUNCTIONS 
 
===================================================================*/ 
void test() 
{ 

 
  //my_trainer->setV(example,0); 
 
 
  if(test_iter ==0) 
  { 
    my_trainer->randBatchV(); 
    dim3 blockDim(BLOCKS_LAYER,SAMPLES,BATCH); 
    dim3 threadDim0(my_nn->getH0Size()/BLOCKS_LAYER); 
    vToH0<<<blockDim,threadDim0>>>(my_trainer->d_mini_batch_data, my_nn->getH0(), my_nn->getH0In(), my_nn->getDevB_0(), my_nn->getDevW_0()); 
    dim3 threadDim1(my_nn->getH1Size()/BLOCKS_LAYER); 
    H0ToH1<<<blockDim,threadDim1>>>(my_nn->getH0(), my_nn->getH1(), my_nn->getH1In(), my_nn->getDevB_1(), my_nn->getDevW_1()); 
    H1ToTop<<<BATCH,my_nn->getTopSize()>>>(my_nn->getH1(), my_nn->getTop(), my_nn->getDevB_2(), my_nn->getDevW_2()); 
    //my_trainer->showCurrent(0); 
	} 
  else if(test_iter ==1) 
  { 
    printf("Image Category: %d\n",my_trainer->ansCurrent(0)); 
    float tmp[5]; 
    cudaMemcpy(tmp,my_trainer->d_mini_batch_labels,5*sizeof(float),cudaMemcpyDeviceToHost); 
    printf("[%f][%f][%f][%f][%f]\n", tmp[0], tmp[1], tmp[2], tmp[3], tmp[4]); 
    //my_nn->showTop(0); 
  } 
 
  test_iter++; 
  if(test_iter>=2) 
  { 
    test_iter=0; 
    example++; 
  } 

} 
 
void train() 
{ 

	while(!my_trainer->trainComplete()){
		//cudaEvent_t start, stop; 
		//float time; 
		//cudaEventCreate(&start); 
		//cudaEventCreate(&stop); 

		//cudaEventRecord(start, 0); 
		train_mini(); 
		//cudaEventRecord(stop, 0); 
		//cudaEventSynchronize(stop); 
		//cudaEventElapsedTime(&time, start, stop); 

		//printf("Time for mini_batch (20): %f ms\n", time); 
		//total_time+=time; 

		////Show training examples 
		//my_trainer->setV(TRAIN_EXAMPLE,0); 

		//dim3 blockDim(BLOCKS_LAYER,SAMPLES,BATCH); 
		//dim3 threadDim0(my_nn->getH0Size()/BLOCKS_LAYER); 
		//vToH0<<<blockDim,threadDim0>>>(my_trainer->d_mini_batch_data, my_nn->getH0(), my_nn->getH0In(), my_nn->getDevB_0(), my_nn->getDevW_0()); 

		//dim3 threadDim1(my_nn->getH1Size()/BLOCKS_LAYER); 
		//H0ToH1<<<blockDim,threadDim1>>>(my_nn->getH0(), my_nn->getH1(), my_nn->getH1In(), my_nn->getDevB_1(), my_nn->getDevW_1()); 

		////my_nn->showH1(0); 
		////printf("Top Size = %d", my_nn->getTopSize()); 
		//H1ToTop<<<BATCH,my_nn->getTopSize()>>>(my_nn->getH1(), my_nn->getTop(), my_nn->getDevB_2(), my_nn->getDevW_2()); 
		//my_nn->showTop(0); 


		//Update training status 
		my_trainer->incN(); 
		if(my_trainer->epochComplete()) 
		{ 
			printf("Epoch %d Complete!\n", my_trainer->getEpoch()); 
			validationError(); 
			my_nn->saveComplete(param_file_complete); 
		} 
	}

	printf("Training run complete!\n"); 
	//printf("Average Time for mini_batch(20): %f ms\n", total_time / ( (float)(my_trainer->getTrainSize() - my_trainer->getValidSize()) / (float)BATCH) ); 

} 
 
void classifyTrain() 
{ 
  int batch_size=0; 
  int num_seen = 0; 
  int num_wrong =0; 
 
  while(num_seen < my_trainer->getTrainSize()-my_trainer->getValidSize()) 
  //while(num_seen < my_trainer->getValidSize()) 
  { 
    batch_size = my_trainer->nextBatchTrain(); 
    //batch_size = my_trainer->nextBatchValid(); 
    num_seen += batch_size; 
    dim3 blockDim(BLOCKS_LAYER,SAMPLES,batch_size); 
	dim3 threadDim0(my_nn->getH0Size()/BLOCKS_LAYER); 
 
    vToH0<<<blockDim,threadDim0>>>(my_trainer->d_mini_batch_data, my_nn->getH0(), my_nn->getH0In(), my_nn->getDevB_0(), my_nn->getDevW_0()); 
    dim3 threadDim1(my_nn->getH1Size()/BLOCKS_LAYER); 
 
    H0ToH1<<<blockDim,threadDim1>>>(my_nn->getH0(), my_nn->getH1(), my_nn->getH1In(), my_nn->getDevB_1(), my_nn->getDevW_1()); 
 
    H1ToTop<<<batch_size,my_nn->getTopSize()>>>(my_nn->getH1(), my_nn->getTop(), my_nn->getDevB_2(), my_nn->getDevW_2()); 
 
    num_wrong += my_trainer->batchClassification(my_nn->getTop(),batch_size); 
  } 
  printf("Missed %d out of %d\n",num_wrong, num_seen); 
  //printf("Misclassification rate: %f", (float)num_wrong / (float)num_seen); 
  printf("Recognition rate: %f", 1-((float)num_wrong / (float)num_seen)); 
} 
void classifyValid() 
{ 
  int batch_size=0; 
  int num_seen = 0; 
  int num_wrong =0; 
 
  while(num_seen < my_trainer->getValidSize()) 
  { 
    batch_size = my_trainer->nextBatchValid(); 
    num_seen += batch_size; 
    dim3 blockDim(BLOCKS_LAYER,SAMPLES,batch_size); 
    dim3 threadDim0(my_nn->getH0Size()/BLOCKS_LAYER); 
 
    vToH0<<<blockDim,threadDim0>>>(my_trainer->d_mini_batch_data, my_nn->getH0(), my_nn->getH0In(), my_nn->getDevB_0(), my_nn->getDevW_0()); 
    dim3 threadDim1(my_nn->getH1Size()/BLOCKS_LAYER); 
 
    H0ToH1<<<blockDim,threadDim1>>>(my_nn->getH0(), my_nn->getH1(), my_nn->getH1In(), my_nn->getDevB_1(), my_nn->getDevW_1()); 
 
    H1ToTop<<<batch_size,my_nn->getTopSize()>>>(my_nn->getH1(), my_nn->getTop(), my_nn->getDevB_2(), my_nn->getDevW_2()); 
 
    num_wrong += my_trainer->batchClassification(my_nn->getTop(),batch_size); 
  } 
  printf("Missed %d out of %d\n",num_wrong, num_seen); 
  printf("Misclassification rate: %f", (float)num_wrong / (float)num_seen); 
} 
void validationError() 
{ 
 
  printf("Calculating Avg Error on Validation Set...\n"); 
 
  ofstream o_file; 
  o_file.open("experiments/last_run.err", ios::app); 
  o_file.seekp( ios::end ); 
  //o_file << "Epoch: " << my_trainer->getEpoch() << "\t"; 
 
 
  int batch_size=0; 
  float num_seen = 0; 
  float total_error =0; 
 
  while(num_seen < my_trainer->getValidSize()) 
  { 
    batch_size = my_trainer->nextBatchValid(); 
    num_seen += batch_size; 
    dim3 blockDim(BLOCKS_LAYER,SAMPLES,batch_size); 
    dim3 threadDim0(my_nn->getH0Size()/BLOCKS_LAYER); 
 
    vToH0<<<blockDim,threadDim0>>>(my_trainer->d_mini_batch_data, my_nn->getH0(), my_nn->getH0In(), my_nn->getDevB_0(), my_nn->getDevW_0()); 
    dim3 threadDim1(my_nn->getH1Size()/BLOCKS_LAYER); 
 
    H0ToH1<<<blockDim,threadDim1>>>(my_nn->getH0(), my_nn->getH1(), my_nn->getH1In(), my_nn->getDevB_1(), my_nn->getDevW_1()); 
 
    H1ToTop<<<BATCH,my_nn->getTopSize()>>>(my_nn->getH1(), my_nn->getTop(), my_nn->getDevB_2(), my_nn->getDevW_2()); 
 
    total_error += my_trainer->batchError(my_nn->getTop(),batch_size); 
  } 
 
  //o_file << "Avg Error(valid): " << total_error / num_seen << "\n"; 
 
  o_file << total_error / num_seen << ","; 
 
  o_file.close(); 
 
  printf("Average Error on Validation Set: %f\n",total_error / num_seen); 
} 
 
/*==================================================================
=
*       HELPER FUNCTIONS 
 
===================================================================*/ 
 
void train_mini() 
{ 
  my_trainer->randBatchV(); 
  //my_trainer->showCurrent(0); 
  //printf("\nLabel: %d\t",my_trainer->ansCurrent(0)); 
 
  dim3 blockDim(BLOCKS_LAYER,SAMPLES,BATCH); 
 
  dim3 threadDim0(my_nn->getH0Size()/BLOCKS_LAYER); 
  vToH0<<<blockDim,threadDim0>>>(my_trainer->d_mini_batch_data, my_nn->getH0(), my_nn->getH0In(), my_nn->getDevB_0(), my_nn->getDevW_0()); 
 
  dim3 threadDim1(my_nn->getH1Size()/BLOCKS_LAYER); 
  H0ToH1<<<blockDim,threadDim1>>>(my_nn->getH0(), my_nn->getH1(), my_nn->getH1In(), my_nn->getDevB_1(), my_nn->getDevW_1()); 
 
  //my_nn->printTop(0);

  H1ToTop<<<BATCH,my_nn->getTopSize()>>>(my_nn->getH1(), my_nn->getTop(), my_nn->getDevB_2(), my_nn->getDevW_2()); 
  //my_nn->printTop(0); 
 
 
  calcErrorTop<<<BATCH,my_nn->getTopSize()>>>(my_nn->getTop(), my_trainer->d_mini_batch_labels, my_nn->getTopError()); 
  calcErrorH1<<<blockDim,threadDim1>>>(my_nn->getTopError(), my_nn->getH1In(), my_nn->getDevW_2(), my_nn->getH1Error()); 
  calcErrorH0<<<blockDim,threadDim0>>>(my_nn->getH1Error(), my_nn->getH0In(), my_nn->getDevW_1(), my_nn->getH0Error()); 
 
  updateW0<<<BLOCKS_LAYER,threadDim0>>>(my_trainer->d_mini_batch_data, my_nn->getH0Error(), my_nn->getDevW_0(), my_nn->getDevVw_0(), my_trainer->getMomentum(), my_trainer->getLearnRate()); 
  updateW1<<<BLOCKS_LAYER,threadDim1>>>(my_nn->getH0(), my_nn->getH1Error(), my_nn->getDevW_1(), my_nn->getDevVw_1(), my_trainer->getMomentum(), my_trainer->getLearnRate()); 
  updateW2<<<1,my_nn->getTopSize()>>>(my_nn->getH1(), my_nn->getTopError(), my_nn->getDevW_2(), my_nn->getDevVw_2(), my_trainer->getMomentum(), my_trainer->getLearnRate()); 
 
  updateBiasH0<<<BLOCKS_LAYER,threadDim0>>>(my_nn->getH0Error(), my_nn->getDevB_0(), my_trainer->getLearnRate()); 
  updateBiasH1<<<BLOCKS_LAYER,threadDim1>>>(my_nn->getH1Error(), my_nn->getDevB_1(), my_trainer->getLearnRate()); 
  updateBiasTop<<<1,my_nn->getTopSize()>>>(my_nn->getTopError(), my_nn->getDevB_2(), my_trainer->getLearnRate()); 
} 
 
int  load_from_rbm(char* rbm_file, Connection* con) 
{ 
  printf("Loading RBM from %s...",rbm_file); 
  ifstream i_file; 
  i_file.open(rbm_file, ios::binary); 
  if(i_file.is_open()) 
  { 
    int loc = 0; 
 
    loc = con->load(&i_file, loc); 
 
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
 
/*==================================================================
= 
 *       CUDA FUNCTIONS 
 
===================================================================*/ 

//>--aj为节点在通过激励函数前的值
//>--h0为节点在通过激励函数后的值

__global__ void vToH0(float* v0, float* h0, float* aj, float* b, 
float* w) 
{ 
  int h_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 
  int b_off = blockIdx.z; 
  int t_off = ( b_off * H0SIZE ) + h_idx; 
 
  aj[t_off] = b[h_idx]; 
  //printf("sum = %f \n",b[h_idx]); 
  for(int i=0;i<VSIZE;i++) 
  { 
    aj[t_off] += v0[b_off*VSIZE + i] * w[ i*H0SIZE + h_idx]; 
  } 
  //printf("sum = %f \n",b[h_idx]); 
  h0[t_off] = 1 / (1 + __expf(-1 * aj[t_off])); 
} 
 
__global__ void H0ToH1(float* h0, float* h1, float* aj, float* b, 
float* w) 
{ 
int h_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 
  int b_off = blockIdx.z; 
  int t_off = (b_off * H1SIZE ) + h_idx; 
 
  aj[t_off] = b[h_idx]; 
  //printf("sum = %f \n",b[h_idx]); 
  for(int i=0;i<H0SIZE;i++) 
  { 
    aj[t_off] += h0[b_off*H0SIZE + i] * w[ i*H1SIZE + h_idx]; 
  } 
  //printf("sum = %f \n",b[h_idx]); 
  h1[t_off] = 1 / (1 + __expf(-1 * aj[t_off])); 
} 
 
__global__ void H1ToTop(float* h1, float* top, float* b, float* w) 
{ 
  int h_idx = threadIdx.x; 
  int b_off = blockIdx.x; 
  int t_off = b_off*CLASS + h_idx; 
 
  float sum = b[h_idx]; 
  //printf("sum = %f \n",b[h_idx]); 
  for(int i=0;i<H1SIZE;i++) 
  { 
    sum += h1[b_off*H1SIZE + i] * w[ i*CLASS + h_idx]; 
  } 
  //printf("sum = %f \n",b[h_idx]); 
  //top[t_off] = 1 / (1 + __expf(-1 * sum)); 
  top[t_off] = __expf(sum); 
 
  __syncthreads(); 
 
  sum = 0; 
  for(int k=0;k<CLASS;k++) 
  { 
    sum+=top[b_off*CLASS+k]; 
  } 
 
  __syncthreads(); 
 
  top[t_off] = top[t_off] / sum; 
} 
 
__global__ void calcErrorTop(float* top, float* label, float* error) 
{ 
  int h_idx = threadIdx.x; 
  int b_off = blockIdx.x; 
  int t_off = b_off*CLASS + h_idx; 
 
  error[t_off] =  top[t_off] - label[t_off]; 
  //printf("dk[%d]=%f\t",t_off,error[t_off]); 
}

//calcErrorH1<<<blockDim,threadDim1>>>(my_nn->getTopError(), my_nn->getH1In(), my_nn->getDevW_2(), my_nn->getH1Error());
__global__ void calcErrorH1(float* top_error, float* aj, float* w, 
float* error) 
{ 
  int h_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 
  int g_off = 0; 
  int b_off = blockIdx.z; 
  int t_off = ( (b_off * gridDim.y + g_off) * H1SIZE ) + h_idx; 
 
  float sum=0; 
 
  for(int i=0;i<CLASS;i++) 
  { 
    sum += top_error[b_off*CLASS + i] * w[ i*CLASS + h_idx]; 
  } 
 
  error[t_off] = (  __expf(aj[t_off]) / pow((1 + __expf(aj[t_off])),2)  ) * sum; 
  //if(t_off < 50) 
    //printf("dj[%d]=%f\n",t_off,error[t_off]); 
} 
 
__global__ void calcErrorH0(float* h1_error, float* aj, float* w, 
float* error) 
{ 
  int h_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 
  int g_off = 0; 
  int b_off = blockIdx.z; 
  int t_off = ( (b_off * gridDim.y + g_off) * H0SIZE ) + h_idx; 
 
  float sum=0; 
 
  for(int i=0;i<H1SIZE;i++) 
  { 
    sum += ((h1_error[b_off*H1SIZE + i] * w[ i*H1SIZE + 
h_idx])); 
  } 
 
  //error[t_off] = (  __expf(aj[t_off]/2) / pow((1 + __expf(aj[t_off]/2)),2)  ); 
  error[t_off] = (  __expf(aj[t_off]/2) / pow((1 + __expf(aj[t_off]/2)),2)  ) * sum; 
  //error[t_off] *= 100; 
  //if(t_off < 50) 
 
  //printf("aj=%f\tdi[%d]=%f\n",aj[t_off]/2,t_off,error[t_off]); 
} 
 
__global__ void updateW0(float* v0, float* error, float* w, float* 
vel_w, float momentum, float l_rate) 
{ 
  int h_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 
  float deltaW = 0; 
  //printf("sum = %f \n",b[h_idx]); 
 
  for(int i=0;i<VSIZE;i++) 
  { 
    deltaW = 0; 
    for(int b=0;b<BATCH;b++) 
    { 
      deltaW += error[h_idx + b*H0SIZE]*v0[i + VSIZE*b]; 
    } 
    //printf("w[%d]=%f + %f\n",i*H0SIZE + h_idx, w[i*H0SIZE + h_idx], l_rate*deltaW); 
 
    //VELOCITY 
    vel_w[i*H0SIZE + h_idx] = momentum * vel_w[i*H0SIZE + h_idx] + l_rate*(deltaW/BATCH); 
    w[i*H0SIZE + h_idx] -= vel_w[i*H0SIZE + h_idx]; 
 
    //w[i*H0SIZE + h_idx] -= l_rate*(deltaW/BATCH); 
  } 
} 
 
__global__ void updateW1(float* h0, float* error, float* w, float* 
vel_w, float momentum, float l_rate) 
{ 
  int h_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 
 
  float deltaW = 0; 
  //printf("sum = %f \n",b[h_idx]); 
 
  for(int i=0;i<H0SIZE;i++) 
  { 
    deltaW = 0; 
    for(int b=0;b<BATCH;b++) 
    { 
      deltaW += error[h_idx + b*H1SIZE]*h0[i + H0SIZE*b]; 
    } 
    //printf("w[%d]=%f + %f\n",i*H1SIZE + h_idx, w[i*H1SIZE + h_idx], l_rate*deltaW); 
    vel_w[i*H1SIZE + h_idx] = momentum * vel_w[i*H1SIZE + h_idx] + l_rate*(deltaW/BATCH); 
    w[i*H1SIZE + h_idx] -= vel_w[i*H1SIZE + h_idx]; 
 
    //w[i*H1SIZE + h_idx] -=  l_rate*(deltaW/BATCH); 
  } 
} 
 
__global__ void updateW2(float* h1, float* error, float* w, float* 
vel_w, float momentum, float l_rate) 
{ 
  int h_idx = threadIdx.x; 
  float deltaW = 0; 
  //printf("sum = %f \n",b[h_idx]); 
 
  for(int i=0;i<H1SIZE;i++) 
  { 
    deltaW = 0; 
    for(int b=0;b<BATCH;b++) 
    { 
      deltaW += error[h_idx + b*CLASS]*h1[i + H1SIZE*b]; 
    } 
    //printf("w[%d->%d]=%f - %f\n",i, h_idx, w[i*CLASS + h_idx], l_rate*deltaW); 
    vel_w[i*CLASS + h_idx] = momentum * vel_w[i*CLASS + h_idx] + l_rate*(deltaW/BATCH); 
    w[i*CLASS + h_idx] -= vel_w[i*CLASS + h_idx]; 
 
    //w[i*CLASS + h_idx] -= l_rate*(deltaW/BATCH); 
  } 
} 
 
__global__ void updateBiasH0(float* error, float* bias, float 
l_rate) 
{ 
  int h_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 
 
  float deltaB = 0; 
  //printf("sum = %f \n",b[h_idx]); 
 
  for(int b=0;b<BATCH;b++) 
  { 
    deltaB += error[h_idx + b*H0SIZE]; 
  } 
  //printf("b[%d]=%f + %f\n",h_idx, bias[h_idx], l_rate*deltaB); 
  bias[h_idx] -= l_rate*(deltaB/BATCH); 
} 
 
__global__ void updateBiasH1(float* error, float* bias, float 
l_rate) 
{ 
  int h_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 
 
  float deltaB = 0; 
  //printf("sum = %f \n",b[h_idx]); 
 
  for(int b=0;b<BATCH;b++) 
  { 
    deltaB += error[h_idx + b*H1SIZE]; 
  } 
  //printf("w[%d]=%f + %f\n",i*H0SIZE + h_idx, w[i*H0SIZE + h_idx], l_rate*deltaW); 
  bias[h_idx] -= l_rate*(deltaB/BATCH); 
  } 
 
__global__ void updateBiasTop(float* error, float* bias, float 
l_rate) 
{ 
  int h_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 
 
  float deltaB = 0; 
  //printf("sum = %f \n",b[h_idx]); 
 
  for(int b=0;b<BATCH;b++) 
  { 
    deltaB += error[h_idx + b*CLASS]; 
  } 
  //printf("b[->%d]=%f - %f\n", h_idx, bias[h_idx], l_rate*deltaB); 
  bias[h_idx] -= l_rate*(deltaB/BATCH); 
} 