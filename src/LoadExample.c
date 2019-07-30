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
	//AllCreate Network
	TOPOLOGY* network=(TOPOLOGY*)malloc(sizeof(TOPOLOGY));
	int j;
	double input7[2]={6,12}; //1
	double input2[2]={3,6}; //1
	double input8[2]={100,2}; //0	
	system("cls");
	load_network(network,"save.dat");
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
	return 0;
}
