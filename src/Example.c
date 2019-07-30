#include "MyNNLib.h"
#include <stdint.h>

#define DATA_SIZE 320
#define ITERATION_SIZE 3
#define INPUT_SIZE 2
#define OUTPUT_SIZE 2
#define LAYER_SIZE 3
int main(){
	srand(time(NULL));
	int i;
	//Determine topology matrix, first bias values and activation functions according to neural network layers
	int matrix[LAYER_SIZE]={INPUT_SIZE,4,2};
	double bias[LAYER_SIZE-1]={1,0};
	double(*act[LAYER_SIZE-1])(double)={relu,sigmoid};
	//AllCreate Network
	TOPOLOGY* network=create_network(matrix,3,bias,act);
	int j;
	network->USE_SOFTMAX=1; //Softmax enabled, network will use Softmax instead of sigmoid in last layer as activation function
	double input7[2]={6,12}; //1
	double input2[2]={3,6}; //1
	double input8[2]={100,2}; //0
	double zero[2]={0,1};
	double one[2]={1,0};
	TRAINING_SET** set=init_sample_set(DATA_SIZE,INPUT_SIZE,OUTPUT_SIZE);
	
	//Create training samples.
	//That neural network will determine if second input is bigger than first input or not.
	for(i=0;i<DATA_SIZE;i++){
		if(rand()%100<50){
			set[i]->input[0]=rand()%1000+1;
			do{
				set[i]->input[1]=rand()%(int)(set[i]->input[0]);		
			}while(set[i]->input[1]>=set[i]->input[0]);
			set[i]->output[0]=0;
			set[i]->output[1]=1;
		}
		else{
				set[i]->input[0]=rand()%1000;
			do{
				set[i]->input[1]=set[i]->input[0]+rand()%(int)(1000-set[i]->input[0])+1;	
			}while(set[i]->input[1]<set[i]->input[0]);
			//If second input is bigger than first one, output is 1 and 0 (One hot Encoding)
			set[i]->output[0]=1;
			set[i]->output[1]=0;
		}
	}
	
	
		for(i=0;i<ITERATION_SIZE;i++){
			//Train neural network with samples and shuffle sample array 
			train(network,set,DATA_SIZE,"backpropogation",0.001,0,0.9,1);
			shuffle(set,DATA_SIZE);
		}
		

	system("cls");
	feed_forward(network,input7,network->weights,0); //1
	printf("Result: %f %f Expected: 1 0\n",network->layers[2].neurons[0].data,network->layers[2].neurons[1].data);
	network->layers[0].neuronsize;
	feed_forward(network,input7,network->weights,0); //1
	printf("Result: %f %f Expected: 1 0\n",network->layers[2].neurons[0].data,network->layers[2].neurons[1].data);
	feed_forward(network,input2,network->weights,0); //1
	printf("Result: %f %f Expected: 1 0\n",network->layers[2].neurons[0].data,network->layers[2].neurons[1].data);
	feed_forward(network,input8,network->weights,0); //0
	printf("Result: %f %f Expected: 0 1\n ",network->layers[2].neurons[0].data,network->layers[2].neurons[1].data);
	input8[0]=45;
	input8[1]=22;
	feed_forward(network,input8,network->weights,0); //0
	printf("Result: %f %f Expected: 0 1\n",network->layers[2].neurons[0].data,network->layers[2].neurons[1].data);
	save_network(network,"save.dat");
	return 0;
}
