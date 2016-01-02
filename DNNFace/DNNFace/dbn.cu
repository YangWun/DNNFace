#include "./inc/connection.h" 
#include "./inc/dbn.h" 
#include "./inc/trainer.h" 
#include <curand.h> 
 
 
 
using namespace utilLearn; 
 
 
//NN LEARNING PARAMETERS 
#define BATCH      100 
#define SAMPLES      1 
#define STEPS      1 
#define EPOCH      400 
 
//NN SIZE PARAMETERS 
#define VSIZE_X      28 
#define  VSIZE_Y      28 
#define VSIZE      784 
/*#define VSIZE_X      96 
#define  VSIZE_Y      96 
#define VSIZE      9216*/ 
 
#define  H0SIZE_X    32 
#define H0SIZE_Y    16 
#define H0SIZE     512 
//#define  H0SIZE_X    64 
//#define H0SIZE_Y   64 
//#define H0SIZE      4096 
 
#define  H1SIZE_X    32 
#define H1SIZE_Y    16 
#define H1SIZE     512 
//#define  H1SIZE_X    64 
//#define H1SIZE_Y   64 
//#define H1SIZE      4096 
 
//NN MODE 
#define MODE      0 //0=train 1=classification 
#define TRAIN_EXAMPLE 1232 
 
//char * const param_file_1 = "params/persistent-lvl1.rbm"; 
//char * const param_file_2 = "params/persistent-lvl2.rbm"; 
//char * const data_file = "data/train-images.idx3-ubyte"; 
 
char * const param_file_1 = "params/mnist-persistent-lvl1.rbm"; 
char * const param_file_2 = "params/mnist-persistent-lvl2.rbm"; 
//char * const data_file = "data/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat"; 
 char * const data_file = "mnist/train-images.idx3-ubyte";
//For Transpose 
#define TILE_DIM   16 
#define BLOCK_ROWS    16 
 
#define  BLOCKS_LAYER  16 
 
//Display Functions 
void train(); 
 
//Helper Functions 
void train_mini(); 
void update_params(); 
 
 
//CUDA Functions 
__global__ void transpose(float *w, float *wt); 
 
__global__ void upPassInit(float* v0, float* h0, float* b, float* w, float* rnd); 
__global__ void upPass(float* vX, float* hX, float* b, float* w, float* rnd); 
__global__ void upPassProb(float* vX, float* hX, float* b, float* w); 
__global__ void downPass(float* vX, float* hX, float* a, float* wt, float* rnd); 
__global__ void downPassProb(float* vX, float* hX, float* a, float* wt); 
 
__global__ void updateW0T(float* v0, float* v0Pred, float* h0, float* wt, float l_rate); 
__global__ void updateW0(float* v0X, float* h0X, float* h0Pred, float* w, float l_rate); 
__global__ void updateW1(float* h0, float* h0X, float* h1, float* h1X, float* w, float l_rate); 
 
__global__ void updateA0(float* v0, float* v0Pred, float* a, float l_rate); 
__global__ void updateB0(float* h0X, float* h0XPred, float* b, float l_rate); 
__global__ void updateA1(float* h0, float* h0X, float* a, float l_rate); 
__global__ void updateB1(float* h1, float* h1X, float* b, float l_rate); 
 
//Globals 
Dbn *my_dbn; 
Trainer *my_trainer; 
curandGenerator_t d_rand; 
 
//float total_time; 
 
int main(int argc, char** argv) 
{ 
//total_time = 0; 
  //Set GPU 1 (currently not used for display) 
  cudaSetDevice(1); 
 
  //Set up basic units 
  Layer* visible = new Layer(VSIZE_X, VSIZE_Y, BATCH, SAMPLES, false); 
  Connection* v_to_h0 = new Connection(VSIZE, H0SIZE); 
  Layer* h0 = new Layer(H0SIZE_X, H0SIZE_Y, BATCH, SAMPLES, true); 
  Connection* h0_to_h1 = new Connection(H0SIZE, H1SIZE); 
  Layer* h1 = new Layer(H1SIZE_X, H1SIZE_Y, BATCH, SAMPLES, true); 
 
  my_dbn = new Dbn(visible, h0, h1, v_to_h0, h0_to_h1); 
  my_trainer = new Trainer(BATCH, SAMPLES, EPOCH, 0.0005, 5, 0.5, 0.0000); 
 
  if(my_trainer->loadTrainingData(data_file) < 0) 
  //if(my_trainer->loadTrainingDataMAT(data_file) < 0) 
  { 
    printf("An error occurred loading the training data. Exiting...\n"); 
    return -1; 
  } 
 
 
  if(my_dbn->loadLayers(param_file_1, param_file_2) < 0) 
  { 
    printf("An error occurred loading the parameters. Exiting...\n"); 
    return -1; 
  } 
 
 
  //Set up Random Initializer 
  curandCreateGenerator(&d_rand, CURAND_RNG_PSEUDO_MTGP32); 
  srand((unsigned)time(0)); 
  int seed = (rand() % 1000); 
  curandSetPseudoRandomGeneratorSeed(d_rand, seed); 
 

 
  train(); 
 
  return 0; 
 
} 
 
/*==================================================================
= 
 *       DISPLAY FUNCTIONS 
 
===================================================================*/ 
 
//Trains the RBM with a selected visual feeback 
void train() 
{ 

   while(!my_trainer->trainComplete()) {
  //cudaEvent_t start, stop; 
  //float time; 
  //cudaEventCreate(&start); 
  //cudaEventCreate(&stop); 
 
  //cudaEventRecord(start, 0); 
  train_mini(); 
  //cudaEventRecord(stop, 0); 
  //cudaEventSynchronize(stop); 
  //cudaEventElapsedTime(&time, start, stop); 
 
  //printf("Time for mini_batch (100): %f ms\n", time); 
  //total_time+=time; 
 
 
//  my_trainer->setV(TRAIN_EXAMPLE,0); 
//  dim3 blockDim(BLOCKS_LAYER,SAMPLES,BATCH); 
// 
//  //V->H0 
//  dim3 threadDimH0(my_dbn->getH0Size()/BLOCKS_LAYER); 
//  curandGenerateUniform(d_rand, (float *) my_dbn->getH0Rand(), 
//my_dbn->getH0Size()*my_trainer->getNumFantasy()); 
//  upPassInit<<<blockDim,threadDimH0>>>(my_trainer->d_mini_batch_data,my_dbn->getH0In(),my_dbn->getDevB_0(),my_dbn->getDevW_0(), my_dbn->getH0Rand()); 
//  //H0->H1 
//  dim3 threadDimH1(my_dbn->getH1Size()/BLOCKS_LAYER); 
//  curandGenerateUniform(d_rand, (float *) my_dbn->getH1Rand(), my_dbn->getH1Size()*my_trainer->getNumFantasy()); 
//  upPass<<<blockDim,threadDimH1>>>(my_dbn->getH0In(),my_dbn->getH1In(),my_dbn->getDevB_1(),my_dbn->getDevW_1(), my_dbn->getH1Rand()); 
//  //H0<-H1 
//  curandGenerateUniform(d_rand, (float *) my_dbn->getH0Rand(), my_dbn->getH0Size()*my_trainer->getNumFantasy()); 
//  downPass<<<blockDim,threadDimH0>>>(my_dbn->getH0(),my_dbn->getH1In(),my_dbn->getDevA_1(),my_dbn->getDevWT_1(), my_dbn->getH0Rand()); 
//  //H0f->H1f 
//  curandGenerateUniform(d_rand, (float *) my_dbn->getH1Rand(),my_dbn->getH1Size()*my_trainer->getNumFantasy()); 
//  upPass<<<blockDim,threadDimH1>>>(my_dbn->getH0(),my_dbn->getH1(),my_dbn->getDevB_1(),my_dbn->getDevW_1(), my_dbn->getH1Rand()); 
//  //H0f<-H1f 
//  curandGenerateUniform(d_rand, (float *) my_dbn->getH0Rand(), my_dbn->getH0Size()*my_trainer->getNumFantasy()); 
//  downPass<<<blockDim,threadDimH0>>>(my_dbn->getH0(),my_dbn->getH1(),my_dbn->getDevA_1(),my_dbn->getDevWT_1(), my_dbn->getH0Rand()); 
//  //Vf<-H0f 
//  dim3 threadDimV(my_dbn->getVSize()/BLOCKS_LAYER); 
//  downPassProb<<<blockDim,threadDimV>>>(my_dbn->getVX(),my_dbn->getH0(),my_dbn->getDevA_0(),my_dbn->getDevWT_0()); 
// 
//  my_dbn->showVX(0); 
// 
// 
//  glFlush(); 
 
 
  //Update training status 
  my_trainer->incN(); 
  if(my_trainer->epochComplete()) 
  { 
    printf("Epoch %d Complete!\n", my_trainer->getEpoch()); 
    my_dbn->saveLayers(param_file_1, param_file_2); 
  } 
 
}
 

    printf("Training run complete!\n"); 
    //printf("Average Time for mini_batch(100): %f ms\n", total_time / ( (float)(my_trainer->getTrainSize() - my_trainer->getValidSize()) / (float)BATCH) ); 
  
 
} 
 
//Trains a single mini-batch 
void train_mini() 
{ 
  //Select Batch Samples V0
  my_trainer->randBatchV(); 
 
 
  //WAKE PHASE 
  dim3 blockDim(BLOCKS_LAYER,SAMPLES,BATCH); 
 
  //V->H0 
  dim3 threadDimH0(my_dbn->getH0Size()/BLOCKS_LAYER); 
  curandGenerateUniform(d_rand, (float *) my_dbn->getH0Rand(), my_dbn->getH0Size()*my_trainer->getNumFantasy()); 
  upPassInit<<<blockDim,threadDimH0>>>(my_trainer->d_mini_batch_data,my_dbn->getH0In(),my_dbn->getDevB_0(),my_dbn->getDevW_0(), my_dbn->getH0Rand()); 
 
  //H0->H1 
  dim3 threadDimH1(my_dbn->getH1Size()/BLOCKS_LAYER); 
  curandGenerateUniform(d_rand, (float *) my_dbn->getH1Rand(), my_dbn->getH1Size()*my_trainer->getNumFantasy()); 
  upPass<<<blockDim,threadDimH1>>>(my_dbn->getH0In(),my_dbn->getH1In(),my_dbn->getDevB_1(),my_dbn->getDevW_1(), my_dbn->getH1Rand()); 
 
  //H0<-H1 
  curandGenerateUniform(d_rand, (float *) my_dbn->getH0Rand(), my_dbn->getH0Size()*my_trainer->getNumFantasy()); 
  downPass<<<blockDim,threadDimH0>>>(my_dbn->getH0(),my_dbn->getH1In(),my_dbn->getDevA_1(),my_dbn->getDevWT_1(), my_dbn->getH0Rand()); 
 
  //Additional Gibbs steps 
  for (int g=1;g<STEPS;g++) 
  { 
    curandGenerateUniform(d_rand, (float *) my_dbn->getH1Rand(),my_dbn->getH1Size()*my_trainer->getNumFantasy()); 
    upPass<<<blockDim,threadDimH1>>>(my_dbn->getH0(),my_dbn->getH1(),my_dbn->getDevB_1(),my_dbn->getDevW_1(), my_dbn->getH1Rand()); 
    curandGenerateUniform(d_rand, (float *) my_dbn->getH0Rand(),my_dbn->getH0Size()*my_trainer->getNumFantasy()); 
    downPass<<<blockDim,threadDimH0>>>(my_dbn->getH0(),my_dbn->getH1(),my_dbn->getDevA_1(),my_dbn->getDevWT_1(), 
my_dbn->getH0Rand()); 
  } 
  //SLEEP PHASE 
 
  //H0f->H1f 
  curandGenerateUniform(d_rand, (float *) my_dbn->getH1Rand(),my_dbn->getH1Size()*my_trainer->getNumFantasy()); 
  upPass<<<blockDim,threadDimH1>>>(my_dbn->getH0(),my_dbn->getH1(),my_dbn->getDevB_1(),my_dbn->getDevW_1(), my_dbn->getH1Rand()); 
  //H0f<-H1f 
  curandGenerateUniform(d_rand, (float *) my_dbn->getH0Rand(), 
my_dbn->getH0Size()*my_trainer->getNumFantasy()); 
  downPass<<<blockDim,threadDimH0>>>(my_dbn->getH0(),my_dbn->getH1(),my_dbn->getDevA_1(),my_dbn->getDevWT_1(), my_dbn->getH0Rand()); 
 
  //Vf<-H0f 
  dim3 threadDimV(my_dbn->getVSize()/BLOCKS_LAYER); 
  downPassProb<<<blockDim,threadDimV>>>(my_dbn->getVX(),my_dbn->getH0(),my_dbn->getDevA_0(),my_dbn->getDevWT_0()); 
 
  //Predictions 
  downPassProb<<<blockDim,threadDimV>>>(my_dbn->getVPred(),my_dbn->getH0In(),my_dbn->getDevA_0(),my_dbn->getDevWT_0()); 
 
  upPassProb<<<blockDim, threadDimH0>>>(my_dbn->getVX(), my_dbn->getH0Pred(), my_dbn->getDevB_0(), my_dbn->getDevW_0()); 
 
  update_params(); 
 
  return; 
} 
 
void update_params() 
{ 
  //Update Parameters 
  dim3 threadDimV(my_dbn->getVSize()/BLOCKS_LAYER); 
  dim3 threadDimH0(my_dbn->getH0Size()/BLOCKS_LAYER); 
  dim3 threadDimH1(my_dbn->getH1Size()/BLOCKS_LAYER); 
 
  dim3 updateBlockW0(BLOCKS_LAYER,my_dbn->getVSize()); 
  dim3 updateBlockW0T(BLOCKS_LAYER,my_dbn->getH0Size()); 
  dim3 updateBlockW1(BLOCKS_LAYER,my_dbn->getH0Size()); 
 
  updateW0<<<updateBlockW0,threadDimH0>>>(my_dbn->getVX(),my_dbn->getH0(),my_dbn->getH0Pred(),my_dbn->getDevW_0() 
      ,my_trainer->getLearnRate()); 
 
  updateW0T<<<updateBlockW0T,threadDimV>>>(my_trainer->d_mini_batch_data,my_dbn->getVX(),my_dbn->getH0In(),my_dbn->getDevWT_0() 
      ,my_trainer->getLearnRate()); 
 
  updateW1<<<updateBlockW1,threadDimH1>>>(my_dbn->getH0In(),my_dbn->getH0(),my_dbn->getH1In(),my_dbn->getH1(),my_dbn->getDevW_1() 
      ,my_trainer->getLearnRate()); 
  dim3 grid(my_dbn->getH1Size()/TILE_DIM, my_dbn->getH0Size()/TILE_DIM), threads(TILE_DIM,BLOCK_ROWS); 
  transpose<<<grid,threads>>>(my_dbn->getDevW_1(), my_dbn->getDevWT_1()); 
 
  updateA0<<<BLOCKS_LAYER,threadDimV>>>(my_trainer->d_mini_batch_data, my_dbn->getVX(), my_dbn->getDevA_0(), my_trainer->getLearnRate()); 
  updateB0<<<BLOCKS_LAYER,threadDimH0>>>(my_dbn->getH0In(), my_dbn->getH0(), my_dbn->getDevB_0(), my_trainer->getLearnRate()); 
  updateA1<<<BLOCKS_LAYER,threadDimH0>>>(my_dbn->getH0In(), my_dbn->getH0(), my_dbn->getDevA_1(), my_trainer->getLearnRate()); 
  updateB1<<<BLOCKS_LAYER,threadDimH1>>>(my_dbn->getH1In(), my_dbn->getH1(), my_dbn->getDevB_1(), my_trainer->getLearnRate()); 
 
} 
 
/*==================================================================
= 
 *       CUDA FUNCTIONS 
 
===================================================================*/ 
 
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
  int t_off = ( (b_off * gridDim.y + g_off) * H0SIZE ) + h_idx; 
 
  float sum = b[h_idx]; 
  //printf("sum = %f \n",b[h_idx]); 
  for(int i=0;i<VSIZE;i++) 
  { 
    sum += v0[b_off*VSIZE + i] * w[ i*H0SIZE + h_idx]; 
  } 
  //printf("sum = %f \n",b[h_idx]); 
  float prob = 1 / (1 + __expf(-1 * sum)); 
  //printf("p(H[%d]=1|v) = %f > %f\n",h_idx, prob, rnd[h_idx + b_offset]); 
  h0[t_off] = (prob > rnd[t_off]); 
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
__global__ void upPass(float* vX, float* hX, float* b, float* w, 
float* rnd) 
{ 
  int h_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 
  int g_off = blockIdx.y; 
  int b_off = blockIdx.z * gridDim.y; 
  int t_off = ( (b_off + g_off) * H1SIZE ) + h_idx; 
 
  float sum = b[h_idx]; 
  //printf("sum = %f \n",b[h_idx]); 
  for(int i=0;i<H0SIZE;i++) 
  { 
    sum += vX[(b_off + g_off)*H0SIZE + i] * w[ i*H1SIZE + h_idx]; 
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
  int t_off = ( (b_off + g_off) * H0SIZE ) + h_idx; 
 
  float sum = b[h_idx]; 
  //printf("sum = %f \n",b[h_idx]); 
  for(int i=0;i<VSIZE;i++) 
  { 
    sum += vX[(b_off + g_off)*VSIZE + i] * w[ i*H0SIZE + 
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
__global__ void downPass(float* vX, float* hX, float* a, float* wt, 
float* rnd) 
{ 
  int v_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 
  int g_off = blockIdx.y; 
  int b_off = blockIdx.z * gridDim.y; 
  int t_off = ( (b_off + g_off) * H0SIZE ) + v_idx; 
 
  float sum = a[v_idx]; 
  //printf("sum = %f \n",b[h_idx]); 
  for(int i=0;i<H1SIZE;i++) 
  { 
    //sum += hX[b_off + g_off + i] * w[ i*512 + v_idx]; 
    sum += hX[(b_off + g_off)*H1SIZE + i] * wt[ i*H0SIZE + v_idx]; 
  } 
  //printf("sum = %f \n",b[h_idx]); 
  float prob = 1 / (1 + __expf(-1 * sum)); 
 
  vX[t_off] = (prob > rnd[t_off]); 
} 
 
__global__ void downPassProb(float* vX, float* hX, float* a, float* 
wt) 
{ 
  int v_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 
  int g_off = blockIdx.y; 
  int b_off = blockIdx.z * gridDim.y; 
  int t_off = ( (b_off + g_off) * VSIZE ) + v_idx; 
 
  float sum = a[v_idx]; 
  //printf("sum = %f \n",b[h_idx]); 
  for(int i=0;i<H0SIZE;i++) 
  { 
    //sum += hX[b_off + g_off + i] * w[ i*512 + v_idx]; 
    sum += hX[(b_off + g_off)*H0SIZE + i] * wt[ i*VSIZE + v_idx]; 
  } 
  //printf("sum = %f \n",b[h_idx]); 
  vX[t_off] = 1 / (1 + __expf(-1 * sum)); 
} 
 
__global__ void updateW0T(float* v0, float* v0Pred, float* h0, 
float* wt, float l_rate) 
{ 
  int v_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 
  int h_idx = (blockIdx.y); 
  int h_offset = h_idx * blockDim.x * gridDim.x; 
 
  float delta = 0.0; 
 
  for(int batch=0;batch<BATCH;batch++) 
  { 
    for(int gibbs=0;gibbs<SAMPLES;gibbs++) 
    { 
      int h_off = h_idx + batch*SAMPLES*H0SIZE + gibbs*H0SIZE; 
      int v_off = v_idx + batch*SAMPLES*VSIZE + gibbs*VSIZE; 
 
      delta += h0[h_off] * (v0[v_idx + batch*VSIZE]- v0Pred[v_off]); 
    } 
  } 
  //DECAY 
  float decay = (0.0005 * wt[v_idx + h_offset] ) * l_rate; 
 
  wt[v_idx + h_offset] += ((delta * l_rate) / (SAMPLES * BATCH)) - decay; 
} 
 
__global__ void updateW0(float* v0X, float* h0X, float* h0Pred, 
float* w, float l_rate) 
{ 
  int h_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 
  int v_idx = (blockIdx.y); 
  int v_offset = v_idx * blockDim.x * gridDim.x; 
 
  float delta = 0.0; 
  for(int batch=0;batch<BATCH;batch++) 
  { 
    for(int gibbs=0;gibbs<SAMPLES;gibbs++) 
    { 
      int h_off = h_idx + batch*SAMPLES*H0SIZE + gibbs*H0SIZE; 
      int v_off = v_idx + batch*SAMPLES*VSIZE + gibbs*VSIZE; 
 
      delta += v0X[v_off] * (h0X[h_off]- h0Pred[h_off]); 
    } 
  } 
  //DECAY 
  float decay = (0.0005 * w[h_idx + v_offset] ) * l_rate; 
 
  w[h_idx + v_offset] += ((delta * l_rate) / (SAMPLES * BATCH)) - decay; 
} 
 
 
__global__ void updateW1(float* h0, float* h0X, float* h1, float* 
h1X, float* w, float l_rate) 
{ 
  { 
    int h_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 
    int v_idx = (blockIdx.y); 
    int v_offset = v_idx * blockDim.x * gridDim.x; 
 
    float delta = 0.0; 
 
    for(int batch=0;batch<BATCH;batch++) 
    { 
      for(int gibbs=0;gibbs<SAMPLES;gibbs++) 
      { 
        int h_off = h_idx + batch*SAMPLES*H1SIZE + gibbs*H1SIZE; 
        int v_off = v_idx + batch*SAMPLES*H0SIZE + gibbs*H0SIZE; 
 
        delta += (h0[v_off] * h1[h_off]) - (h0X[v_off] * h1X[h_off]); 
      } 
    } 
    //DECAY 
    float decay = (0.0005 * w[h_idx + v_offset] ) * l_rate; 
 
    w[h_idx + v_offset] += ((delta * l_rate) / (SAMPLES * BATCH)) - decay; 
  } 
} 
__global__ void updateA0(float* v0, float* v0Pred, float* a, float 
l_rate) 
{ 
  int v_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 
 
  float delta = 0.0; 
 
  for(int batch=0;batch<BATCH;batch++) 
  { 
    for(int gibbs=0;gibbs<SAMPLES;gibbs++) 
    { 
      int v_off = v_idx + batch*SAMPLES*VSIZE + gibbs*VSIZE; 
      delta += (v0[v_idx + batch*VSIZE]) - (v0Pred[v_off]); 
    } 
  } 
 
  a[v_idx] += ( (delta * l_rate) / (SAMPLES * BATCH) ); 
} 
 
__global__ void updateB0(float* h0X, float* h0XPred, float* b, float 
l_rate) 
{ 
  int h_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 
 
  float delta = 0.0; 
 
  for(int batch=0;batch<BATCH;batch++) 
  { 
    for(int gibbs=0;gibbs<SAMPLES;gibbs++) 
    { 
      int h_off = h_idx + batch*SAMPLES*H0SIZE + gibbs*H0SIZE; 
      delta += (h0X[h_off]) - (h0XPred[h_off]); 
    } 
  } 
 
  b[h_idx] += ( (delta * l_rate) / (SAMPLES * BATCH) ); 
} 
__global__ void updateA1(float* h0, float* h0X, float* a, float 
l_rate) 
{ 
  int v_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 
 
  float delta = 0.0; 
 
  for(int batch=0;batch<BATCH;batch++) 
  { 
    for(int gibbs=0;gibbs<SAMPLES;gibbs++) 
    { 
	int v_off = v_idx + batch*SAMPLES*H0SIZE + gibbs*H0SIZE; 
      delta += (h0[v_off]) - (h0X[v_off]); 
    } 
  } 
 
  a[v_idx] += ( (delta * l_rate) / (SAMPLES * BATCH) ); 
} 
 
__global__ void updateB1(float* h1, float* h1X, float* b, float 
l_rate) 
{ 
  int h_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 
 
  float delta = 0.0; 
 
  for(int batch=0;batch<BATCH;batch++) 
  { 
    for(int gibbs=0;gibbs<SAMPLES;gibbs++) 
    { 
      int h_off = h_idx + batch*SAMPLES*H1SIZE + gibbs*H1SIZE; 
      delta += (h1[h_off]) - (h1X[h_off]); 
    } 
  } 
 
  b[h_idx] += ( (delta * l_rate) / (SAMPLES * BATCH) ); 
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
    int index_in = xIndex + (yIndex)*H1SIZE; 
 
    xIndex = blockIdx.y * TILE_DIM + threadIdx.x; 
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y; 
    int index_out = xIndex + (yIndex)*H0SIZE; 
 
  for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) 
  { 
    tile[threadIdx.y+i][threadIdx.x] = w[index_in+i*H1SIZE]; 
	} 
 
  __syncthreads(); 
 
  for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) 
  { 
    wt[index_out+i*H0SIZE] = 
tile[threadIdx.x][threadIdx.y+i]; 
  } 
} 