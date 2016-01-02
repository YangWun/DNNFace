#ifndef UTIL_LEARN_H 
#define UTIL_LEARN_H 

#include <cuda.h> 
#include <cuda_runtime.h> 


namespace utilLearn{ 

	/* --------------------------------------------------------------- 
	*    SWAP 4 
	* Non-Intel -> Intel Byte formatting for 4 bytes 
	* c | char* | pointer to 4 char array 
	------------------------------------------------------------------*/ 
	static void swap4(char* c) 
	{ 
		char tmp[4]; 
		tmp[0] = c[3]; 
		tmp[1] = c[2]; 
		c[3] = c[0]; 
		c[2] = c[1]; 
		c[1] = tmp[1]; 
		c[0] = tmp[0]; 
		return; 
	} 

	/* --------------------------------------------------------------- 
	*   DEV ALLOC 
	* Helper function to allocate device memory. Helps keep code 
	clean 
	* and keeps running count of allocated memory. 
	* 
	* d   | float**  | device location to allocate 
	* size | int   | total bytes to allocate 
	------------------------------------------------------------------*/ 
	static void dev_alloc(float** d, int size) 
	{ 
		cudaMalloc((void**)d,size); 
		//_total_gpu_mem += size; 
		printf("Allocating %f MBytes on GPU.\n",((float) size / 1024) 
			/ 1024); 
		return; 
	} 
	static void dev_alloc(int** d, int size) 
	{ 
		cudaMalloc((void**)d,size); 
		//_total_gpu_mem += size; 
		printf("Allocating %f MBytes on GPU.\n",((float) size / 1024) 
			/ 1024); 
		return; 
	} 

	/* --------------------------------------------------------------- 
	*   SHOW 
	* Draws the layer passed to it 
	* lay  | float*| pointer to units to display 
	* x  | int | width of layer 
	* y  | int | height of layer 
	------------------------------------------------------------------*/ 
	//static void show(float* lay, int x, int y) 
	//{ 

	//	float px_size = 2.0/(float)x; 
	//	for(int i=0;i<y;i++) 
	//	{ 
	//		for(int j=0;j<x;j++) 
	//		{ 
	//			glColor3f(lay[i*x + j],lay[i*x + j],lay[i*x + 
	//				j]); 
	//			float v_off = 1.0-(float)(i+1)*px_size; 
	//			float h_off = -1.0+(float)j*px_size; 
	//			glBegin(GL_POLYGON); 
	//			glVertex2f(h_off, v_off); 
	//			glVertex2f(h_off, v_off + px_size); 
	//			glVertex2f(h_off + px_size, v_off + 
	//				px_size); 
	//			glVertex2f(h_off + px_size, v_off); 
	//			glEnd(); 
	//		} 
	//	} 
	//	return; 
	//} 
	//
	//static void text(float* lay, int x, int y) 
	//{ 

	//	for(int i=0;i<y;i++) 
	//	{ 
	//		for(int j=0;j<x;j++) 
	//		{ 
	//			printf("[%f]",lay[i*x + j]); 
	//		} 
	//		printf("\n"); 
	//	} 
	//	return; 
	//} 


} 

#endif 