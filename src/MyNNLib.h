#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <string.h>
#define max(a,b) (((a)>(b)) ? (a) : (b))


//EXAMPLE USAGE
/*
	#define LAYER_COUNT 4
	#define TRAIN_SIZE 2700
	#define INPUT_SIZE 128
	#define OUTPUT_SIZE 4
	int main(){
		srand(time(NULL));
		int i;
		double bias[LAYER_COUNT]={0.1,0.1,0.1,0};
		int matrix[LAYER_COUNT]={INPUT_SIZE,256,256,OUTPUT_SIZE};
		double(*act[LAYER_COUNT-1])(double)={relu,relu,sigmoid};
		TRAINING_SET* trainSets[TRAIN_SIZE]=init_sample_set(TRAIN_SIZE,INPUT_SIZE,OUTPUT_SIZE);
		
		//you must assign samples' input and output values to trainSets array
		
		//initilize network
		TOPOLOGY* network=create_network(matrix,LAYER_COUNT,bias,act);
		for(i=0;i<ITERATION_COUNT;i++){
	
			train(network,trainSets,TRAIN_SIZE,"backpropogation",0.001,50,0.9,32);
			shuffle(trainSets,TRAIN_SIZE); //You can shuffle or renew trainSets to improve learning
		
			//You can add some functions to test is training enough or to learn how it is going
		}
		//You can add functions to use trained neural network
		return 0;
	}
*/ 
//Training Set struct
typedef struct{
	double* input;
	double* output;
}TRAINING_SET;

//Neuron struct
typedef struct{
	double data;
	double derived_Data;
}NEURON;

//Network LAYER struct
typedef struct{
	int neuronsize;
	NEURON* neurons;
}LAYER;

//Nueral Network architecture topology struct
typedef struct {
	int layerSize;
	LAYER* layers;
	double** weights;
	double* bias;
	double error;
	double (**actfunc)(double);
	short int USE_SOFTMAX; //If 1, disables last layer's activation functions and uses Softmax instead of that
	double REG_PARAMETER; //Regularization parameter for L1 regularization, If 0, disables L1 regularization
}TOPOLOGY;

//Activation functions and activation derivation functions
double tanh_der(double X){
	return 1-pow(tanh(X),2);
}
double relu(double X){
	return max(X,X);
}
double sigmoid(double x){
     return 1 / (1 + exp(-x));
}
double relu_der(double X){
	if(X>0)
		return 1;
	else if(X==0)
		return 1;
	else 
		return 0;
}
double sigmoid_der(double X){
	     double a= sigmoid(X);
	     return a*(1-a);
}
void softmax(double *input, int input_len) {
  double m = -RAND_MAX;
  int i;
  for (i = 0; i < input_len; i++) {
    if (input[i] > m) {
      m = input[i];
    }
  }

  double sum = 0.0;
  for (i = 0; i < input_len; i++) {
    sum += expf(input[i] - m);
  }

  double offset = m + logf(sum);
  for (i = 0; i < input_len; i++) {
    input[i] = expf(input[i] - offset);
  }
}
void softmax_der(double* input,int len){
	int i;
	softmax(input,len);
	for(i=0;i<len;i++){
		input[i]=input[i]*(1-input[i]);
	}
}


void Normalize(double* , int ); //Normalizes given array
TOPOLOGY* create_network(int*,int,double*,double (**)(double)); //Inıtializes Neural network
void print_weights(TOPOLOGY*); //Prints all weights
void feed_forward(TOPOLOGY*,double*,double**,int); //FeedForward function of Neural network
double totalerror(TOPOLOGY*,double*); //Calculate total error
double** backpropogation(TOPOLOGY*,double*,double*,double,float,double**); //Backpropogation function for neural network
void train(TOPOLOGY*,TRAINING_SET**,int,char*,float,int,float,int); //Train Neural network
TRAINING_SET** init_sample_set(int,int,int); //Inıtialize sample set
void shuffle(TRAINING_SET**, int); //Shuffle array
void printLayer(TOPOLOGY*,int); //Print 
double randn(double,double); //Function for generating random double numbers using normal distribution function (mu,sigma)
int save_network(TOPOLOGY*,char*); //Saves network by binary format to a file 
void load_network(TOPOLOGY*,char*); //Loads network from file from binary format
TOPOLOGY* create_network(int* matrix, int n,double* bias,double(**actfunc)(double)){

	int i=0,col,j;
	TOPOLOGY* tmp=(TOPOLOGY*)malloc(sizeof(TOPOLOGY)); //Allocation for toğology
	tmp->layerSize=n;
	tmp->layers=(LAYER*)malloc(n*sizeof(LAYER)); //Allocation of Layers in Topology
	tmp->weights=(double**)malloc((n-1)*sizeof(double*)); //Allocation of weights
	tmp->bias =(double*) malloc((n-1)*sizeof(double)); //Allocation for biases
	tmp->actfunc=(double(**)(double))malloc(sizeof(double(*)(double))*(tmp->layerSize-1));//Allocation for activation functions
	for(i=0;i<n;i++){
		tmp->bias[i]=bias[i];
		col=matrix[i];
		tmp->layers[i].neuronsize=col; 
		tmp->layers[i].neurons=(NEURON*)malloc(col*sizeof(NEURON));//Topolojinin satırlarındaki nöronların sayısının atanması
	}    
	for(i=0;i<n-1;i++){ //Giving weights their first values
		tmp->actfunc[i]=actfunc[i];	
		
		tmp->weights[i]=(double*)malloc(matrix[i]*matrix[i+1]*sizeof(double));
		for(j=0;j<matrix[i]*matrix[i+1];j++){
			
			do{
				//Giving weights' first values with XAVIER INITILIZATION
				tmp->weights[i][j]=randn(0,sqrt((double)2/((double)matrix[i]+(double)matrix[i+1])));
			}while(tmp->weights[i][j]<=0);
		}

	}
	tmp->USE_SOFTMAX=0;
	tmp->REG_PARAMETER=0;
	return tmp;
}

//Parameters struct for thread function of feedforward function
typedef struct{
	TOPOLOGY* network;
	double** weights;
	int** dropped;
	int startIndex;
	int nls;
	int neuronsize;
	int i;
}PARAM;

//Thread function for feedforward function
void *threadfeed(void* args){
	int j,k,X;
	int startIndex = ((PARAM*)args)->startIndex;
	int nls = ((PARAM*)args)->nls;
	TOPOLOGY* network=((PARAM*)args)->network;
	int** dropped=((PARAM*)args)->dropped;
	double** weights=((PARAM*)args)->weights;
	int neuronsize = ((PARAM*)args)->neuronsize;
	int i = ((PARAM*)args)->i;
		for(j=startIndex;j<nls;j++){
			X=0;
			for(k=0;k<neuronsize;k++){ 
					X += network->layers[i].neurons[k].data*weights[i][j+k*nls];
			}
			X+=network->bias[i];
			if(i==network->layerSize-2 && network->USE_SOFTMAX == 1)
				network->layers[i+1].neurons[j].data=X;
			else
				network->layers[i+1].neurons[j].data=network->actfunc[i](X);
			if(i>0 && i<network->layerSize-1 && dropped[i][j]==1)
				network->layers[i+1].neurons[j].data=0;	

		}
}
void feed_forward(TOPOLOGY* network,double* input,double** weights,int dropout_rate){ 
	int i,j,k,nls,neuronsize;
	double* softinput;
	double X;
	int	input_size =network->layers[0].neuronsize;
	int target_size = network->layers[network->layerSize-1].neuronsize;
	softinput=(double*)malloc(sizeof(double)*network->layers[network->layerSize-1].neuronsize);

	//inputların verilmesi
	for(i=0;i<input_size;i++)
		network->layers[0].neurons[i].data=input[i];
	int** dropped;
	dropped=(int**)malloc(sizeof(int*)*network->layerSize-1);
	//dropout implementation
	for(i=0;i<network->layerSize-1;i++){
		dropped[i]=(int*)malloc(sizeof(int)*network->layers[i].neuronsize);
		for(j=0;j<network->layers[i].neuronsize;j++){
			
			if(dropout_rate>rand()%100){
				dropped[i][j]=1;
			}
			else
				dropped[i][j]=0;
		}
	}
	for(i=0;i<network->layerSize-1;i++){ 
		neuronsize=network->layers[i].neuronsize; //Neuron count of i th Layer
		nls= network->layers[i+1].neuronsize; //Neuron count of next layer
		//parameters for Thread function
		pthread_t Thread;
		PARAM* parametreler=(PARAM*)malloc(sizeof(PARAM));
		parametreler->dropped=dropped;
		parametreler->i=i;
		parametreler->network=network;
		parametreler->neuronsize=neuronsize;
		parametreler->nls=nls;
		parametreler->startIndex=nls/2;
		parametreler->weights=weights;
		pthread_create(&Thread, NULL, threadfeed, parametreler);
		for(j=0;j<nls/2;j++){
			X=0;
			for(k=0;k<neuronsize;k++){
					X += network->layers[i].neurons[k].data*weights[i][j+k*nls];
			}
			X+=network->bias[i];
			if(i==network->layerSize-2 && network->USE_SOFTMAX == 1)
				network->layers[i+1].neurons[j].data=X;
			else
				network->layers[i+1].neurons[j].data=network->actfunc[i](X); //Passing X to activation function
			if(i>0 && i<network->layerSize-1 && dropped[i][j]==1)
				network->layers[i+1].neurons[j].data=0;	

		}
		pthread_join(Thread,NULL);
		
	}
	if(network->USE_SOFTMAX==1){
		
		for(i=0;i<network->layers[network->layerSize-1].neuronsize;i++)
			softinput[i]=network->layers[network->layerSize-1].neurons[i].data;
		
		softmax(softinput,network->layers[network->layerSize-1].neuronsize);
		for(i=0;i<network->layers[network->layerSize-1].neuronsize;i++)
			network->layers[network->layerSize-1].neurons[i].data=softinput[i];
		
	}

}

double totalerror(TOPOLOGY* network, double* target){
	
	double error=0;
	int i,j;
	double regX=0;
	for(i=0;i<network->layerSize-1;i++){
		for(j=0;j<network->layers[i].neuronsize;j++){
			regX+=fabs(network->weights[i][j]);
		}
	}
	regX*=network->USE_SOFTMAX/2*network->layers[0].neuronsize;
	for(i=0;i<network->layers[network->layerSize-1].neuronsize;i++){
		error+=pow(network->layers[network->layerSize-1].neurons[i].data-target[i],2);

	}
	
	return error/2 + regX;
}

//Parameters struct for backpropogation thread function
typedef struct{
	double** deltaWeights;
	LAYER* next;
	LAYER* back;
	double learning_rate;
	double (**actderivfunc)(double);
	int startIndex;
	TOPOLOGY* network;
	int biasIndex;
	int currLayer;
	int i;
}PARAMb;
// Thread function for backpropogation
void *threadBack(void* args){
	int i,k,j;
	i=((PARAMb*)args)->i;
	k=((PARAMb*)args)->currLayer;
	double** deltaWeights=((PARAMb*)args)->deltaWeights;
	LAYER* next=((PARAMb*)args)->next;
	LAYER* back=((PARAMb*)args)->back;
	double learning_rate=((PARAMb*)args)->learning_rate;
	double (**actderivfunc)(double)=((PARAMb*)args)->actderivfunc;
	int startIndex=((PARAMb*)args)->startIndex;
	int biasIndex=((PARAMb*)args)->biasIndex;
	TOPOLOGY* network= ((PARAMb*)args)->network;
	
	for(j=startIndex;j<next->neuronsize;j++){
		deltaWeights[k-1][i*next->neuronsize+j]= 
		(learning_rate)*((next->neurons[j].derived_Data)*(back->neurons[i].data)+network->USE_SOFTMAX*(network->weights[k-1][i*next->neuronsize+j]==0 ? 0: network->weights[k-1][i*next->neuronsize+j]/fabs(network->weights[k-1][i*next->neuronsize+j])));
		back->neurons[i].derived_Data+= next->neurons[j].derived_Data*network->weights[k-1][i*next->neuronsize+j];
		//Last index of deltaWeights are Bias Deltas
		deltaWeights[k-1][biasIndex]+=next->neurons[j].derived_Data;
	}
}

double** backpropogation(TOPOLOGY* network,double* input,double* target, double learning_rate,float MOMENTUM,double** prev_delta_weights){
	int i,j,k;
	PARAMb* parameters=(PARAMb*)malloc(sizeof(PARAMb)); //parameters for passing required values for thread function

	double (**actderivfunc)(double)=(double(**)(double))malloc(sizeof(double(*)(double))*(network->layerSize-1));
	//Determine derivative functions
	for(i=0;i<network->layerSize-1;i++){
		if(network->actfunc[i]==relu){
			actderivfunc[i]=&relu_der;
		}
		else if(network->actfunc[i]==sigmoid){
			actderivfunc[i]=&sigmoid_der;
		}
		else if(network->actfunc[i]==tanh)
			actderivfunc[i]=&tanh_der;		
	}
	double** deltaWeights=(double**)malloc((network->layerSize-1)*sizeof(double*));
	for(i=0;i<network->layerSize-1;i++){
		deltaWeights[i]=(double*)malloc((1+network->layers[i].neuronsize*network->layers[i+1].neuronsize)*sizeof(double));
	}
	
	double* sinput;
	sinput=(double*)malloc(sizeof(double)*network->layers[network->layerSize-1].neuronsize);

	double** finput=(double**)malloc(sizeof(double)*network->layerSize);
		
	LAYER* output = &network->layers[network->layerSize-1];
	
	//DETERMİNE DERİVED DATA OF OUTPUT LAYER 
	for(i=0;i<output->neuronsize;i++){
		if(network->USE_SOFTMAX==1)
			sinput[i]=output->neurons[i].data;
		else
			output->neurons[i].derived_Data= (output->neurons[i].data-target[i])*actderivfunc[network->layerSize-2](output->neurons[i].data);	
	}
			
	if(network->USE_SOFTMAX==1){
		softmax_der(sinput,network->layers[network->layerSize-1].neuronsize);
		for(i=0;i<output->neuronsize;i++){
			output->neurons[i].derived_Data= (output->neurons[i].data-target[i]);
		}
	}
	
	for(k=network->layerSize-1;k>0;k--){
		LAYER* next = &network->layers[k];
		LAYER* back = &network->layers[k-1];
		finput[k]=(double*)malloc(sizeof(double)*back->neuronsize);	

			parameters->currLayer=k;
		for(i=0;i<back->neuronsize/2;i++){
			int biasIndex= back->neuronsize*next->neuronsize;
			deltaWeights[k-1][biasIndex]=0;
			back->neurons[i].derived_Data=0;
			pthread_t Thread;
			parameters->i=i;
			parameters->actderivfunc=actderivfunc;
			parameters->back=back;
			parameters->deltaWeights=deltaWeights;
			parameters->learning_rate=learning_rate;
			parameters->network=network;
			parameters->next=next;
			parameters->startIndex=next->neuronsize/2;
			parameters->biasIndex=biasIndex;
			pthread_create(&Thread, NULL, threadBack, parameters);
			for(j=0;j<next->neuronsize/2;j++){
			//Weight Format: network->weights[LAYER NUMBER][(NEXT LAYER)+(CURRENT LAYER)*(NEXT LAYER NEURON SIZE)]

				deltaWeights[k-1][i*next->neuronsize+j]= 
				(learning_rate)*((next->neurons[j].derived_Data)*(back->neurons[i].data)+network->USE_SOFTMAX*(network->weights[k-1][i*next->neuronsize+j]==0 ? 0: network->weights[k-1][i*next->neuronsize+j]/fabs(network->weights[k-1][i*next->neuronsize+j])));
				back->neurons[i].derived_Data+= next->neurons[j].derived_Data*network->weights[k-1][i*next->neuronsize+j];
				//Last index of deltaWeights are Bias Deltas
				deltaWeights[k-1][biasIndex]+=next->neurons[j].derived_Data;
			}
			//Wait until thread finishes
			pthread_join(Thread,NULL);

			//multiply with acivation derivative function 
			if(k>1)
				back->neurons[i].derived_Data*=actderivfunc[k-2](back->neurons[i].data);
		}
	}
		//return new weights
		return deltaWeights;
}

void train(TOPOLOGY* network,TRAINING_SET** samples,int sample_size ,char* train_func, float learning_rate,int dropout_rate,float MOMENTUM, int batch_size){
	if(strcmp(train_func,"backpropogation")==0){
			int weight_count;
			int i,j,k,f;
			double*** delta_weights=(double***)malloc(batch_size*sizeof(double**)); 
			double** zero=(double**)malloc((network->layerSize-1)*sizeof(double*));
			for(j=0;j<network->layerSize-1;j++){
				zero[j]=(double*)malloc(network->layers[j].neuronsize*network->layers[j+1].neuronsize*sizeof(double));
				for(k=0;k<network->layers[k].neuronsize*network->layers[k+1].neuronsize;k++)
					zero[j][k]=0;
			}	
			for(i=0;i<batch_size;i++){
				delta_weights[i]=(double**)malloc((network->layerSize-1)*sizeof(double*));				
				for(j=0;j<network->layerSize-1;j++){
					delta_weights[i][j]=(double*)malloc(network->layers[j].neuronsize*network->layers[j+1].neuronsize*sizeof(double));
				}			
			}
			//If total sample_size 100 and batch_Size is 50 so epoch is 2 
			int itr= sample_size/batch_size;
			
			
			//Get delta weights from 1 batch
			for(f=0;f<itr;f++){
				for(j=0;j<batch_size;j++){
					int index=f*batch_size+j;
					//Feed forward for samples 
					feed_forward(network,samples[index]->input,network->weights,dropout_rate);
					//calculate totalerror, Cross-Entropy for now
					totalerror(network,samples[index]->output);
					//Save delta weights from backpropogation
					delta_weights[j]=backpropogation(network,samples[index]->input,samples[index]->output, learning_rate,MOMENTUM,zero);
				}
				//Update weights when batch finishes
				for(j=0;j<batch_size;j++){
					for(k=0;k<network->layerSize-1;k++){
						weight_count = network->layers[k].neuronsize*network->layers[k+1].neuronsize;
						for(i=0;i<weight_count;i++){
							//Update Weights at kth Layer and ith index, use previous delta_Weights at (j-1)th batch, kth layer and ith index
							if(j>0)
								network->weights[k][i]-=delta_weights[j][k][i]+MOMENTUM*delta_weights[j-1][k][i];
							else
								network->weights[k][i]-=delta_weights[j][k][i];
						}
						//Update biases, i is at biasIndex now
						network->bias[k]-=learning_rate*delta_weights[j][k][i];
					}
				}
				//Decrease learning rate
				//learning_rate=LEARNING_DECREASE(firsRate,sayac); //learning rate ve prevdeltaweights, daha genel fonksiyonda tanımlanmalı
			}
			
			
	}
		
		
}

void printLayer(TOPOLOGY* network, int index){
	int i,j;
	int r = network->layers[index].neuronsize;
	printf("\nPrintLayerTest\n");
	for(i=0;i<r;i++)
		printf("%f ",network->layers[index].neurons[i].data);
	printf("\n");
}

void print_weights(TOPOLOGY* network){
	int i,j,k;
	for(i=0;i<network->layerSize-1;i++){
		printf("{");
		for(j=0;j<network->layers[i].neuronsize*network->layers[i+1].neuronsize;j++){
			printf("%.5lf,",network->weights[i][j]);
		}
		printf("\b}\n");
	}
	printf("\n");
}
void Normalize(double* array , int n){
	double min=array[0];
	double max=array[0];
	int i;
	for(i=1;i<n;i++){
		if(min>array[i])
			min=array[i];
		if(max<array[i])
			max=array[i];
	}
	max-=min;
	for(i=0;i<n;i++){
		array[i]-=min;
		array[i]/=max;
		
	}
	
	
}
TRAINING_SET** init_sample_set(int sampleCount,int input_size,int target_size){
	TRAINING_SET** set=(TRAINING_SET**)malloc(sizeof(TRAINING_SET*)*sampleCount);
	int i;
	for(i=0;i<sampleCount;i++){
		set[i]=(TRAINING_SET*)malloc(sizeof(TRAINING_SET));
		set[i]->input=(double*)malloc(sizeof(double)*input_size);
		set[i]->output=(double*)malloc(sizeof(double)*target_size);
	}
	return set;
}
void shuffle(TRAINING_SET **array, int n){
    if (n > 1) 
    {
        int i;
        for (i = 0; i < n - 1; i++) 
        {
          int j = i + rand() / (RAND_MAX / (n - i) + 1);
          TRAINING_SET* t = array[j];
          array[j] = array[i];
          array[i] = t;
        }
    }
}
double randn (double mu, double sigma){
  double U1, U2, W, mult;
  static double X1, X2;
  static int call = 0;
 
  if (call == 1)
    {
      call = !call;
      return (mu + sigma * (double) X2);
    }
 
  do
    {
      U1 = -1 + ((double) rand () / RAND_MAX) * 2;
      U2 = -1 + ((double) rand () / RAND_MAX) * 2;
      W = pow (U1, 2) + pow (U2, 2);
    }
  while (W >= 1 || W == 0);
 
  mult = sqrt ((-2 * log (W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult;
 
  call = !call;
 
  return (mu + sigma * (double) X1);
}
int save_network(TOPOLOGY* network, char* fileName){
	int i;
	FILE* file=fopen(fileName,"w+b");
	fwrite(&(network->error),sizeof(double),1,file);
	fwrite(&(network->layerSize),sizeof(int),1,file);
	fwrite((network->layers),network->layerSize*sizeof(LAYER),1,file);	
	for(i=0;i<network->layerSize;i++)
		fwrite(network->layers[i].neurons,network->layers[i].neuronsize*sizeof(NEURON),1,file);
	fwrite(&(network->REG_PARAMETER),sizeof(double),1,file);
	fwrite(&(network->USE_SOFTMAX),sizeof(short int),1,file);
	
	fwrite((network->bias),(network->layerSize-1)*sizeof(double),1,file);
	
	for(i=0;i<network->layerSize-1;i++)
		fwrite(network->weights[i],network->layers[i].neuronsize*network->layers[i+1].neuronsize*sizeof(double),1,file);
	
	fwrite((network->actfunc),(network->layerSize-1)*sizeof(double(*)(double)),1,file);
	fclose(file);
	return 0;
}
void load_network(TOPOLOGY* network,char* fileName){
	int i;
	network->layers=(LAYER*)malloc(sizeof(LAYER));
	network->bias=(double*)malloc(sizeof(double));
	network->weights=(double**)malloc(sizeof(double*));
	FILE* file=fopen(fileName,"rb");
	
	fread(&(network->error),sizeof(double),1,file);
	fread(&(network->layerSize),sizeof(int),1,file);
	fread((network->layers),network->layerSize*sizeof(LAYER),1,file);	
	for(i=0;i<network->layerSize;i++){
		network->layers[i].neurons=(NEURON*)malloc(network->layers[i].neuronsize*sizeof(NEURON));
		fread(network->layers[i].neurons,network->layers[i].neuronsize*sizeof(NEURON),1,file);	
	}		
	fread(&(network->REG_PARAMETER),sizeof(double),1,file);
	fread(&(network->USE_SOFTMAX),sizeof(short int),1,file);

	fread((network->bias),(network->layerSize-1)*sizeof(double),1,file);
	
	for(i=0;i<network->layerSize-1;i++){
		network->weights[i]=(double*)malloc(network->layers[i].neuronsize*network->layers[i+1].neuronsize*sizeof(double));
		fread(network->weights[i],network->layers[i].neuronsize*network->layers[i+1].neuronsize*sizeof(double),1,file);
	}
	network->actfunc=(double(**)(double))malloc(sizeof(double(*)(double))*(network->layerSize-1));//Allocation for activation functions	
	fread((network->actfunc),(network->layerSize-1)*sizeof(double(*)(double)),1,file);
	fclose(file);
}
