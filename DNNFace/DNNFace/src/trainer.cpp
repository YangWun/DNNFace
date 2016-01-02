#include "../inc/trainer.h" 

/* --------------------------------------------------------------- 
*    CONSTRUCTOR 
* mini_batch_size  | int | number of examples per batch 
* gibbs_samples | int | number of fantasy particles 
* num_epochs    | int | total epochs to train over 
* learn_rate    | float | the learning rate being used 
------------------------------------------------------------------*/ 
Trainer::Trainer(int mini_batch_size, int gibbs_samples, int 
	num_epochs, float learn_rate, int num_classes, float momentum, float 
	lr_beta) 
{ 
	_mini_batch_size = mini_batch_size; 
	_gibbs_samples = gibbs_samples; 
	_num_epochs = num_epochs; 
	_cur_epoch = 0; 
	_cur_batch = 0; 
	_next_batch = 0; 
	_n = 0; 
	_learn_rate = learn_rate; 
	_num_classes = num_classes; 
	_using_labels = false; 
	_momentum = momentum; 
	_lr_beta = lr_beta; 

	_input_size = 1; 
	_input_dim_x = 1; 
	_input_dim_y = 1; 
} 

/* --------------------------------------------------------------- 
*    LOAD TRAINING DATA 
* Loads in the training examples from file 
* 
* data_file | char* | pointer to name of file 
------------------------------------------------------------------*/ 
int Trainer::loadTrainingData(char* data_file) 
{ 
	ifstream i_img_train; 
	i_img_train.open(data_file, ios::binary); 

	if(i_img_train.is_open()) 
	{ 
		printf("Loading Images...\n"); 
		//MAGIC NUMBER 
		char c_magic_number[4]; 
		int i_magic_number; 
		i_img_train.read(c_magic_number,4); 
		swap4(c_magic_number); 
		memcpy(&i_magic_number,c_magic_number,4); 
		printf("magic number: %d\n",i_magic_number); 

		//# IMAGES 
		char c_images[4]; 
		int i_images; 
		i_img_train.seekg(4); 
		i_img_train.read(c_images,4); 
		swap4(c_images); 
		memcpy(&i_images,c_images,4); 
		printf("images: %d\n",i_images); 
		_train_size = i_images; //Set trianing size 
		update_valid(); 

		//ROWS 
		char c_rows[4]; 
		int i_rows; 
		i_img_train.seekg(8); 
		i_img_train.read(c_rows,4); 
		swap4(c_rows); 
		memcpy(&i_rows,c_rows,4); 
		printf("rows: %d\n",i_rows); 
		_input_dim_x = i_rows; 

		//COLUMNS 
		char c_cols[4]; 
		int i_cols; 
		i_img_train.seekg(12); 
		i_img_train.read(c_cols,4); 
		swap4(c_cols); 
		memcpy(&i_cols,c_cols,4); 
		printf("columns: %d\n",i_cols); 
		_input_dim_y = i_cols; 

		//SET INPUT SIZE 
		_input_size = _input_dim_y * _input_dim_x; 
		printf("Data items are size:%d\n",_input_size); 

		//Allocate memory 
		_mini_batch_data = (float*)malloc(_input_size * _train_size * sizeof(float)); 
		dev_alloc(&d_mini_batch_data, _input_size * _mini_batch_size * sizeof(float)); 
	} 
	else 
	{ 
		printf("Couldn't open file. Exiting..."); 
		return -1; 
	} 
	//Grab Images 
	int input_ptr = 16; 
	char* c_img; 
	c_img = (char*)malloc(_input_size*sizeof(char)); 

	//Allocate Training Data Space 
	_train_data = (float*)malloc(_train_size * _input_size * sizeof(float)); 
	printf("Allocated host memory for training data\n"); 

	//Image # 
	for(int n=0;n<_train_size;n++) 
	{ 
		i_img_train.seekg(input_ptr); 
		i_img_train.read(c_img,_input_size); 
		input_ptr+=_input_size; 
		//Down Column 
		//printf("read in"); 
		for(int i=0;i<_input_size;i++) 
		{ 
			//convert to 'binary' 
			/*(if (c_img[i] != 0x00) 
			c_img[i] = 0x01;*/ 
			//_train_data[ n*_v_size + i] = (float)c_img[i]; 
			unsigned char tmp; 
			memcpy(&tmp,&c_img[i],1); 

			//memcpy(&_train_data[ n*_v_size + i*28 + j ],&c_img[i*28 + j],1); 
			//printf("[%d]",tmp); 


			/*//memcpy(&data[ n*V_SIZE + i*28 + j 
			],&c_img[i*28 + j],1); 
			if(tmp > 20) 
			_train_data[ n*_input_size + i] = 1.0f; 
			else 
			_train_data[ n*_input_size + i] = 
			0.0f;*/ 
			
			_train_data[n*_input_size + i] = (float)tmp/255;
			

		} 
		//printf("copied"); 
		//TEST PRINT 
		//print_img(&data[n*V_SIZE], 28, 28); 
	} 

	i_img_train.close();
	//free mem 
	free(c_img); 
	printf("Data Load Complete!\n"); 
	return 0; 
} 

int Trainer::loadTrainingDataMAT(char* data_file) 
{ 

	ifstream i_img_train; 
	i_img_train.open(data_file, ios::binary); 

	if(i_img_train.is_open()) 
	{ 
		printf("Loading Images...\n"); 
		//MAGIC NUMBER 
		char c_magic_number[4]; 
		int i_magic_number; 
		i_img_train.read(c_magic_number,4); 
		//swap4(c_magic_number); 
		memcpy(&i_magic_number,c_magic_number,4); 
		printf("magic number: %d\n",i_magic_number); 

		//# NUMBER OF DIMENSIONS 
		char c_ndim[4]; 
		int i_ndim; 
		i_img_train.seekg(4); 
		i_img_train.read(c_ndim,4); 
		//swap4(c_ndim); 
		memcpy(&i_ndim,c_ndim,4); 
		printf("ndim: %d\n",i_ndim); 

		//# IMAGES 
		char c_images[4]; 
		int i_images; 
		i_img_train.seekg(8); 
		i_img_train.read(c_images,4); 
		//swap4(c_images); 
		memcpy(&i_images,c_images,4); 
		printf("images: %d\n",i_images); 
		_train_size = i_images; //Set trianing size 
		update_valid(); 

		//IMAGES 
		char c_number[4]; 
		int i_number; 
		i_img_train.seekg(12); 
		i_img_train.read(c_number,4); 
		//swap4(c_dim); 
		memcpy(&i_number,c_number,4); 
		printf("views per example: %d\n",i_number); 

		//ROWS 
		char c_rows[4]; 
		int i_rows; 
		i_img_train.seekg(16); 
		i_img_train.read(c_rows,4); 
		//swap4(c_rows); 
		memcpy(&i_rows,c_rows,4); 
		printf("rows: %d\n",i_rows); 
		_input_dim_x = i_rows; 

		//COLUMNS 
		char c_cols[4]; 
		int i_cols; 
		i_img_train.seekg(20); 
		i_img_train.read(c_cols,4); 
		//swap4(c_cols); 
		memcpy(&i_cols,c_cols,4); 
		printf("columns: %d\n",i_cols); 
		_input_dim_y = i_cols; 

		//SET INPUT SIZE 
		_input_size = _input_dim_y * _input_dim_x; 
		printf("Data items are size:%d\n",_input_size); 

		//Allocate memory 
		_mini_batch_data = (float*)malloc(_input_size * _train_size * sizeof(float)); 
		dev_alloc(&d_mini_batch_data, _input_size * _mini_batch_size * sizeof(float)); 
	} 
	else 
	{ 
		printf("Couldn't open file. Exiting..."); 
		return -1; 
	} 
	//Grab Images 
	int input_ptr = 24; 
	char* c_img; 
	c_img = (char*)malloc(_input_size*sizeof(char)); 

	//Allocate Training Data Space 
	_train_data = (float*)malloc(_train_size * _input_size * sizeof(float));
	if(!_train_data)//如果malloc失败，可以得到一些log
	{
		perror("malloc");
	}

	printf("Allocated host memory for training data\n"); 

	//Image # 
	for(int n=0;n<_train_size;n++) 
	{ 
		i_img_train.seekg(input_ptr); 
		i_img_train.read(c_img,_input_size); 
		input_ptr+=_input_size;//input_ptr+=_input_size*2;
		//Down Column 
		//printf("read in"); 
		for(int i=0;i<_input_size;i++) 
		{ 
			//convert to 'binary' 
			/*(if (c_img[i] != 0x00) 
			c_img[i] = 0x01;*/ 
			//_train_data[ n*_v_size + i] = (float)c_img[i]; 
			unsigned char tmp;//int tmp; 
			//memcpy(&tmp,&c_img[i],1); 
			memcpy(&tmp,&c_img[i],sizeof(char));
			//memcpy(&_train_data[ n*_v_size + i*28 + j], &c_img[i*28 + j],1); 
			//printf("[%d]",tmp); 


			/*//memcpy(&data[ n*V_SIZE + i*28 + j ],&c_img[i*28 
			+ j],1); 
			if(tmp > 20) 
			_train_data[ n*_input_size + i] = 1.0f; 
			else 
			_train_data[ n*_input_size + i] = 0.0f;*/ 

			_train_data[n*_input_size + i] = (float)tmp/255; 
		} 
		//printf("copied"); 
		//TEST PRINT 
		//print_img(&data[n*V_SIZE], 28, 28); 
	} 

	i_img_train.close(); 


	//free mem 
	free(c_img); 

	printf("Data Load Complete!\n"); 
	return 0; 

} 

int Trainer::loadTrainingLabels(char* label_file) 
{ 
	ifstream i_lbl_train; 
	i_lbl_train.open(label_file, ios::binary); 

	if(i_lbl_train.is_open()) 
	{ 
		printf("Loading Labels...\n"); 
		//MAGIC NUMBER 
		char c_magic_number[4];
		int i_magic_number; 
		i_lbl_train.read(c_magic_number,4); 
		swap4(c_magic_number); 
		memcpy(&i_magic_number,c_magic_number,4); 
		printf("magic number: %d\n",i_magic_number); 

		//# LABELS 
		char c_labels[4]; 
		int i_labels; 
		i_lbl_train.seekg(4); 
		i_lbl_train.read(c_labels,4); 
		swap4(c_labels); 
		memcpy(&i_labels,c_labels,4); 
		printf("labels: %d\n",i_labels); 
		if(_train_size != i_labels) 
		{ 
			printf("Warning! Label size differs from Data size\n"); 
				//return -1; 
		} 

		//Allocate memory 
		_mini_batch_labels = (float*)malloc(_train_size * _num_classes * sizeof(float)); 
		_mini_batch_label_vals = (int*)malloc(_mini_batch_size * sizeof(int)); 
		dev_alloc(&d_mini_batch_labels, _mini_batch_size * _num_classes * sizeof(float)); 

	} 
	else 
	{ 
		printf("Couldn't open file. Exiting..."); 
		return -1; 
	} 


	//Grab Images 
	int input_ptr = 8; 
	char* c_lbl; 
	c_lbl = (char*)malloc(sizeof(char)); 

	//Allocate Training Data Space 
	_train_label_vals = (int*)malloc(_train_size * sizeof(int)); 
	_train_labels = (float*)malloc(_train_size * _num_classes * sizeof(float)); 
	printf("Allocated host memory for training labels\n"); 

	//Image # 
	for(int n=0;n<_train_size;n++) 
	{ 
		i_lbl_train.seekg(input_ptr); 
		i_lbl_train.read(c_lbl,1); 
		input_ptr++; 

		unsigned char tmp; 
		memcpy(&tmp,c_lbl,1); 

		_train_label_vals[n] = tmp; 

		for(int c=0;c<_num_classes;c++) 
		{ 
			if (c == tmp) 
				_train_labels[n*_num_classes + c] = 1.0; 
			else 
				_train_labels[n*_num_classes + c] = 0.0; 
		} 

		//printf("Label[%d]=%d\n",n, _train_labels[n]); 
	} 

	i_lbl_train.close(); 


	//free mem 
	free(c_lbl); 

	printf("Label Load Complete!\n"); 
	_using_labels = true; 
	return 0; 
} 


int Trainer::loadTrainingLabelsMAT(char* label_file) 
{ 
	ifstream i_lbl_train; 
	i_lbl_train.open(label_file, ios::binary); 

	if(i_lbl_train.is_open()) 
	{ 
		printf("Loading Labels...\n"); 
		//MAGIC NUMBER 
		char c_magic_number[4]; 
		int i_magic_number; 
		i_lbl_train.read(c_magic_number,4); 
		//swap4(c_magic_number); 
		memcpy(&i_magic_number,c_magic_number,4); 
		printf("magic number: %d\n",i_magic_number); 

		//# NUMBER OF DIMENSIONS 
		char c_ndim[4]; 
		int i_ndim; 
		i_lbl_train.seekg(4); 
		i_lbl_train.read(c_ndim,4); 
		//swap4(c_ndim); 
		memcpy(&i_ndim,c_ndim,4); 
		printf("ndim: %d\n",i_ndim); 

		//# LABELS 
		char c_labels[4]; 
		int i_labels; 
		i_lbl_train.seekg(8); 
		i_lbl_train.read(c_labels,4); 
		//swap4(c_labels); 
		memcpy(&i_labels,c_labels,4); 
		printf("labels: %d\n",i_labels); 
		if(_train_size != i_labels) 
		{ 
			printf("Warning! Label size differs from Data size\n"); 
				return -1; 
		} 



		//ROWS 
		char c_rows[4]; 
		int i_rows; 
		i_lbl_train.seekg(16);//i_lbl_train.seekg(12); 
		i_lbl_train.read(c_rows,4); 
		//swap4(c_rows); 
		memcpy(&i_rows,c_rows,4); 
		printf("ignore: %d\n",i_rows); 

		//COLUMNS 
		char c_cols[4]; 
		int i_cols; 
		i_lbl_train.seekg(20);//i_lbl_train.seekg(16); 
		i_lbl_train.read(c_cols,4); 
		//swap4(c_cols); 
		memcpy(&i_cols,c_cols,4); 
		printf("ignore: %d\n",i_cols); 

		//Allocate memory 
		_mini_batch_labels = (float*)malloc(_train_size * 
			_num_classes * sizeof(float)); 
		_mini_batch_label_vals = 
			(int*)malloc(_mini_batch_size * sizeof(int)); 
		dev_alloc(&d_mini_batch_labels, _mini_batch_size * 
			_num_classes * sizeof(float)); 

	} 
	else 
	{ 
		printf("Couldn't open file. Exiting..."); 
		return -1; 
	} 
	//Grab Images 
	int input_ptr = 24; //int input_ptr = 20; 
	char* c_lbl; 
	c_lbl = (char*)malloc(sizeof(char)); 

	//Allocate Training Data Space 
	_train_label_vals = (int*)malloc(_train_size * sizeof(int)); 
	_train_labels = (float*)malloc(_train_size * _num_classes * sizeof(float)); 
	printf("Allocated host memory for training labels\n"); 

	//Image # 
	for(int n=0;n<_train_size;n++) 
	{ 
		i_lbl_train.seekg(input_ptr); 
		//i_lbl_train.read(c_lbl,4); 
		i_lbl_train.read(c_lbl,1); 
		input_ptr++;//input_ptr+=4; 

		//cout << c_lbl[0] << c_lbl[1] << c_lbl[2] << c_lbl[3] << endl; 

		unsigned char tmp;//int tmp; 
		memcpy(&tmp,c_lbl,1);//memcpy(&tmp,c_lbl,4); 

		_train_label_vals[n] = tmp; //_train_label_vals[n] = tmp; 

		for(int c=0;c<_num_classes;c++) 
		{ 
			if (c == tmp) 
				_train_labels[n*_num_classes + c] = 1.0; 
			else 
				_train_labels[n*_num_classes + c] = 0.0; 
		} 
		//cout << "int:" << c_lbl << endl; 
		//printf("Label[%d]=%d\n",n, _train_label_vals[n]); 


	} 

	i_lbl_train.close(); 


	//free mem 
	free(c_lbl); 

	printf("Label Load Complete!\n"); 
	_using_labels = true; 
	return 0; 
} 
/* --------------------------------------------------------------- 
*    LOAD CONVERT TRAINING DATA 
* Loads in the probability outputs, driven by the training data, 
* from previous layer from file 
* 
* data_file | char* | pointer to name of file 
------------------------------------------------------------------*/ 
int Trainer::loadConvertTrainingData(char* data_file) 
{ 
	ifstream i_img_train; 
	i_img_train.open(data_file, ios::binary); 

	if(i_img_train.is_open()) 
	{ 
		printf("Loading Images...\n"); 

		//# IMAGES 
		char c_images[4]; 
		int i_images; 
		i_img_train.seekg(0); 
		i_img_train.read(c_images,4); 
		memcpy(&i_images,c_images,4); 
		printf("images: %d\n",i_images); 
		_train_size = i_images; //Set trianing size 
		update_valid(); 

		//ROWS 
		char c_rows[4]; 
		int i_rows; 
		i_img_train.seekg(4); 
		i_img_train.read(c_rows,4); 
		memcpy(&i_rows,c_rows,4); 
		printf("rows: %d\n",i_rows); 
		_input_dim_x = i_rows; 

		//COLUMNS 
		char c_cols[4]; 
		int i_cols; 
		i_img_train.seekg(8); 
		i_img_train.read(c_cols,4); 
		memcpy(&i_cols,c_cols,4); 
		printf("columns: %d\n",i_cols); 
		_input_dim_y = i_cols; 

		//SET INPUT SIZE 
		_input_size = _input_dim_y * _input_dim_x; 
		printf("Data items are size:%d\n",_input_size); 


		//Allocate memory 
		_mini_batch_data = (float*)malloc(_input_size * _train_size * sizeof(float)); 
		dev_alloc(&d_mini_batch_data, _input_size * _mini_batch_size * sizeof(float)); 


		//Grab Images 
		int input_ptr = 12; 
		char* c_img; 
		c_img = (char*)malloc(_input_size*sizeof(float)); 

		//Allocate Training Data Space 
		_train_data = (float*)malloc(_train_size * 
			_input_size * sizeof(float)); 
		printf("Allocated host memory for training data\n"); 

			//Image # 
			for(int n=0;n<_train_size;n++) 
			{ 
				i_img_train.seekg(input_ptr); 

				i_img_train.read(c_img,_input_size*sizeof(float)); 
				input_ptr+=_input_size*sizeof(float); 
				memcpy(&_train_data[n*_input_size],c_img,_input_size*sizeof(float)); 
			} 

			i_img_train.close(); 

			//free mem 
			free(c_img); 

			printf("Data Load Complete!\n"); 
			return 0; 
	} 
	else 
	{ 
		printf("Couldn't open file. Exiting..."); 
		return -1; 
	} 

} 

/* --------------------------------------------------------------- 
*    SET V 
* Pushes a specific training example into a spot in the mini_batch 
* Copies mini_batch to device 
* 
* n | int | training example to use 
* batch | int | location in mini_batch 
------------------------------------------------------------------*/ 
void Trainer::setV(int n, int batch) 
{ 
	memcpy(&_mini_batch_data[batch*_input_size], &_train_data[n*_input_size], _input_size*sizeof(float)); 
	cudaMemcpy(d_mini_batch_data, _mini_batch_data, _input_size*_mini_batch_size*sizeof(float), cudaMemcpyHostToDevice); 
	if(_using_labels) 
	{ 
		memcpy(&_mini_batch_labels[n * _num_classes],&_train_labels[n * _num_classes],_num_classes * sizeof(float)); 
		_mini_batch_label_vals[n] = _train_label_vals[n]; 

		cudaMemcpy(d_mini_batch_labels,_mini_batch_labels,_mini_batch_size*_num_classes*sizeof(float),cudaMemcpyHostToDevice); 
	} 
	return; 

} 

/* --------------------------------------------------------------- 
*    RAND BATCH V 
* Pushes a batch of training data from the host to V0 on device 
------------------------------------------------------------------*/ 
void Trainer::randBatchV() 
{ 
	for(int b=0;b<_mini_batch_size;b++) 
	{ 
		//setV(rand() % (_train_size - _valid_size),b); 
		int n = rand() % (_train_size - _valid_size); 

		memcpy(&_mini_batch_data[b*_input_size],&_train_data[n*_input_size],_input_size*sizeof(float)); 
		if(_using_labels) 
		{ 
			memcpy(&_mini_batch_labels[b * _num_classes],&_train_labels[n * _num_classes],_num_classes * sizeof(float)); 
			_mini_batch_label_vals[b] = _train_label_vals[n]; 
		} 
	} 
	//>--Pushes a batch of training data from the host to V0 on device

	cudaMemcpy(d_mini_batch_data,_mini_batch_data,_input_size*_mini_batch_size*sizeof(float),cudaMemcpyHostToDevice); 
	if(_using_labels) 
		cudaMemcpy(d_mini_batch_labels,_mini_batch_labels,_mini_batch_size*_num_classes*sizeof(float),cudaMemcpyHostToDevice); 

	return; 
} 
//returns batch size 
int Trainer::nextBatchTrain() 
{ 
	_cur_batch = _next_batch; 
	//check if we have enough examples left for full mini_batch 
	int partial_batch_size = _mini_batch_size -((_train_size - _valid_size)-_cur_batch);
	//>--not enough examples left
	if(partial_batch_size>0) 
	{ 
		//>----是否应该将partial_batch_size改为(_train_size - _valid_size)-_cur_batch
		memcpy(&_mini_batch_data[0],&_train_data[_cur_batch*_input_size],partial_batch_size*_input_size*sizeof(float)); 

		cudaMemcpy(d_mini_batch_data, _mini_batch_data, _input_size*partial_batch_size*sizeof(float),cudaMemcpyHostToDevice); 
		_next_batch = 0; //we've reached the end if we need to load partial 
			return partial_batch_size; 
	} 
	else 
	{ 

		memcpy(&_mini_batch_data[0],&_train_data[_cur_batch*_input_size],_mini_batch_size*_input_size*sizeof(float)); 

		cudaMemcpy(d_mini_batch_data,_mini_batch_data,_input_size*_mini_batch_size*sizeof(float),cudaMemcpyHostToDevice); 
		//update _cur_batch 
		_next_batch+=_mini_batch_size; 
		if (_next_batch>=(_train_size - _valid_size)) 
			_next_batch=0; 

		return _mini_batch_size; 
	} 
} 

//returns batch size 
int Trainer::nextBatchValid() 
{ 
	if(_next_batch < (_train_size - _valid_size)) 
		_next_batch = (_train_size - _valid_size); //set to start of validation data 

		_cur_batch = _next_batch; 

	//check if we have enough examples left for full mini_batch 
	int partial_batch_size = _mini_batch_size -(_train_size-_cur_batch); 
	if(partial_batch_size>0) 
	{ 
		//>--是否应该将partial_batch_size改为_train_size - _cur_batch
		memcpy(&_mini_batch_data[0],&_train_data[_cur_batch*_input_size],partial_batch_size*_input_size*sizeof(float)); 

		cudaMemcpy(d_mini_batch_data,_mini_batch_data,_input_size*partial_batch_size*sizeof(float),cudaMemcpyHostToDevice); 
		_next_batch = 0; //we've reached the end if we need to load partial 
			return partial_batch_size; 
	} 
	else 
	{ 

		memcpy(&_mini_batch_data[0],&_train_data[_cur_batch*_input_size],_mini_batch_size*_input_size*sizeof(float)); 

		cudaMemcpy(d_mini_batch_data,_mini_batch_data,_input_size*_mini_batch_size*sizeof(float),cudaMemcpyHostToDevice); 
		//update _cur_batch 
		_next_batch+=_mini_batch_size; 
		if (_next_batch >= _train_size ) 
			_next_batch=0; 

		return _mini_batch_size; 
	} 
} 

//>--返回样本分类错误的个数
int Trainer::batchClassification(float* d_top_prob, int batch_size) 
{ 
	//int* answers = (int*)malloc(batch_size * sizeof(int)); 
	int incorrect = 0; 
	float* probs = (float*)malloc(_num_classes * batch_size * sizeof(float)); 
	cudaMemcpy(probs, d_top_prob, batch_size*_num_classes*sizeof(float), cudaMemcpyDeviceToHost); 

	float max; 
	int max_loc; 
	for(int n=0;n<batch_size;n++) 
	{ 
		max = 0; 
		max_loc = 0; 
		for(int k=0;k<_num_classes;k++) 
		{ 
			//cout<<probs[n*_num_classes + k]<<" ";
			if(probs[n*_num_classes + k] > max) 
			{ 
				max = probs[n*_num_classes + k]; 
				max_loc = k; 
			} 
		} 
		//answers[n] = max_loc; 
		printf("Test[%d]: Guess[%d] || Answer[%d]\n",_cur_batch+n,max_loc,_train_label_vals[_cur_batch+n]); 
		if (max_loc != _train_label_vals[_cur_batch + n]) 
			incorrect++; 
	} 

	free(probs); 
	return incorrect; 
	//return answers; 
} 

float Trainer::batchError(float* d_top_prob, int batch_size) 
{ 
	//int* answers = (int*)malloc(batch_size * sizeof(int)); 
	int incorrect = 0; 
	float* probs = (float*)malloc(_num_classes * batch_size * sizeof(float)); 
	cudaMemcpy(probs, d_top_prob, batch_size*_num_classes*sizeof(float), cudaMemcpyDeviceToHost); 

	float sum=0; 

	for(int n=0;n<batch_size;n++) 
	{ 
		for(int k=0;k<_num_classes;k++) 
		{ 
			//cout<<_train_labels[(_cur_batch+n)*_num_classes + k]<<" ";
			sum += (pow(probs[n*_num_classes + k] - _train_labels[(_cur_batch+n)*_num_classes + k],2) / 2); 
		} 
	} 

	free(probs); 
	return sum; 
	//return answers; 
} 

/* --------------------------------------------------------------- 
*    SHOW TRAINING 
* Draws the selected training item 
* n | int | training data number to display 
------------------------------------------------------------------*/ 
//void Trainer::showTraining(int n) 
//{ 
//	if(n>=_train_size || n<0) 
//	{ 
//		printf("Selected training data does not exist\n"); 
//		return; 
//	} 
//	else 
//	{ 
//		return show(&_train_data[n*_input_size], _input_dim_x , _input_dim_y); 
//	} 
//} 
//
//void Trainer::showCurrent(int b) 
//{ 
//	if(b > _mini_batch_size) 
//	{ 
//		printf("Selected training batch item does not exist\n"); 
//		return; 
//	} 
//	else 
//	{ 
//		return show(&_mini_batch_data[b*_input_size], 
//			_input_dim_x , _input_dim_y); 
//	} 
//} 

/* --------------------------------------------------------------- 
*    INC N 
* Increment counter for the number of training examples seen 
* this epoch 
------------------------------------------------------------------*/ 
void Trainer::incN() 
{ 
	_n += _mini_batch_size; 
} 

/* --------------------------------------------------------------- 
*    EPOCH COMPLETE 
* Check if all the training examples have been seen for this epoch 
* 
* return  true if all examples seen 
*     false otherwise 
------------------------------------------------------------------*/ 
bool Trainer::epochComplete() 
{ 
	if( _n >= _train_size - _valid_size) 
	{ 
		_n = 0; 
		_cur_epoch++; 
		_learn_rate = _learn_rate / (1 + (float)_cur_epoch * _lr_beta); 
		if(_momentum < 0.9) 
		{ 
			_momentum += 0.1; 
			printf("updating momentum to: %f\n", _momentum); 
		} 
		printf("Current learning rate: %f\n", _learn_rate); 
		return true; 
	} 
	else 
		return false; 
} 


/* --------------------------------------------------------------- 
*    PIXEL PROB 
* Calculates p(v) over the training data 
* 
* index | int | pixel index to calculates 
------------------------------------------------------------------*/ 
float Trainer::pixelProb(int index) 
{ 
	//Visible Bias = log[Pi/(1 - Pi)] 
	// Pi calculated from training examples 
	if(index<_input_size) 
	{ 
		float sum=0.;//int sum = 0; 
		//Calc Pi 
		for (int n=0;n<_train_size;n++) 
		{ 
			//cout<<_train_data[n*_input_size + index]<<endl;
			sum += _train_data[n*_input_size + index]; 
		} 
		//avoid log(0) = -inf 
		if(sum == 0) 
			sum = 1; 
		float Pi = (float)sum / _train_size; 
		return log(Pi / (1-Pi)); 
		//printf("Pi=%f | ai[%d])=%f\n",Pi,i,log(Pi / (1-Pi))); 
	} 
	else 
	{ 
		printf("Pixel Probability requested for index out of bounds.\n"); 
			return 0; 
	} 
}