#include "GL/freeglut.h" 
#include "GL/gl.h" 

//#include <histo.h> 

#include "../inc/connection.h"
//void Connection::histogramW() 
//{ 
//	Color tmp; 
//	tmp.r = 1.0; 
//	tmp.g = 0.0; 
//	tmp.b = 0.0; 
//
//
//	cudaMemcpy(_weight, d_weight, _w_size*sizeof(float), 
//		cudaMemcpyDeviceToHost); 
//	Histo h(_weight,_w_size,2,tmp, 2.0); 
//
//	h.drawStatic(); 
//
//} 
//
//void Connection::histogramDw() 
//{ 
//	Color tmp; 
//	tmp.r = 0.7; 
//	tmp.g = 0.3; 
//	tmp.b = 0.3; 
//
//
//	cudaMemcpy(_dw, d_dw, _w_size*sizeof(float), 
//		cudaMemcpyDeviceToHost); 
//	Histo h(_dw,_w_size,7,tmp,0.01); 
//
//	h.drawStatic(); 
//
//} 
//
//void Connection::histogramA() 
//{ 
//	Color tmp; 
//	tmp.r = 0.0; 
//	tmp.g = 1.0; 
//	tmp.b = 0.0; 
//
//	cudaMemcpy(_a, d_a, _v_size*sizeof(float), 
//		cudaMemcpyDeviceToHost); 
//	Histo h(_a,_v_size,1,tmp,12.0); 
//
//	h.drawStatic(); 
//
//} 
//
//void Connection::histogramB() 
//{ 
//	Color tmp; 
//	tmp.r = 0.0; 
//	tmp.g = 0.0; 
//	tmp.b = 1.0; 
//
//	cudaMemcpy(_b, d_b, _h_size*sizeof(float), 
//		cudaMemcpyDeviceToHost); 
//	Histo h(_b,_h_size,2,tmp,3.0); 
//
//	h.drawStatic(); 
//
//} 