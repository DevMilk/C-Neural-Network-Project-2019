#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define max(a,b) (((a)>(b)) ? (a) : (b))


#define LAYER_SIZE 3
#define INPUT_SIZE 3
#define TARGET_SIZE 3
#define RANDOM (double)rand()/RAND_MAX*1.0-1.0;  //-1 ile 1 arasýnda
#define POPULATION_SIZE 50
#define ITERATION_COUNT 300
#define PARENT_SIZE 2
#define MUTATION_RATE 400 //binde 5
int dosya=0;
//!!Popülasyondaki DNA'larýn hatalarýný sýralayan bir fonksiyon yazman gerek.
//MEVCUT SORUN: Þu anlýk bir sorun yok fakat genetik algoritmanýn garanti bir þekilde errorleri azalttýðý söylenemez.
//Üyeler
typedef struct{
	double** DNA; //aðýrlýklara göre malloc et
	double Error;
}MEMBER;

//Popülasyon
typedef struct{
	MEMBER members[POPULATION_SIZE];
}POPULATION;

//NÖRON
typedef struct{
	double data;
	double derived_Data;
}NEURON;

//LAYER
typedef struct{
	int neuronsize;
	NEURON* neurons;
}LAYER;

//NETWORK MÝMARÝSÝ
typedef struct {
	int layerSize;
	LAYER* layers;
	double** weights;
	double* bias;
	double error;
}TOPOLOGY;

//AKTÝVASYON FONKSÝYONLARI
double relu(double X){
	return max(0,X);
//return log(1+exp(X)); //RELU
}
double sigmoid(double x){
     /*double exp_value;
     double return_value;
*/
     /*** Exponential calculation ***/
     /*exp_value = exp((double) -x);

     /*** Final sigmoid value ***/
    /* return_value = 1 / (1 + exp_value);

     return return_value;*/
     return 1 / (1 + exp(-x));
}
double relu_der(double X){
	if(X>0)
		return 1;
	else 
		return 0;
}
double sigmoid_der(double X){
	     double a= 1 / (1 + exp(-X));
	     return a*(1-a);
}
//Sinir Aðý constructor
TOPOLOGY create_network(int* matrix,double* bias){
	int i=0,col,j;
	TOPOLOGY tmp;
//	int row = sizeof(matrix) / sizeof(matrix[i]);
	tmp.layerSize=LAYER_SIZE;
	tmp.layers=(LAYER*)malloc(LAYER_SIZE*sizeof(LAYER)); //Topolojinin satýrlarýnýn atanmasý
	tmp.weights=malloc((LAYER_SIZE-1)*sizeof(int*)); //Aðýrlýklarýn yer açýlmasý
	tmp.bias = malloc(LAYER_SIZE*sizeof(int)); //biaslara yer açýlmasý
	for(i=0;i<LAYER_SIZE;i++){
		tmp.bias[i]=bias[i];
		col=matrix[i];
		tmp.layers[i].neuronsize=col; 
		tmp.layers[i].neurons=(NEURON*)malloc(col*sizeof(NEURON));//Topolojinin satýrlarýndaki nöronlarýn sayýsýnýn atanmasý
	}
	for(i=0;i<LAYER_SIZE-1;i++){ //aðýrlýklara rastgele deðerler verilmesi
		tmp.weights[i]=malloc(matrix[i]*matrix[i+1]*sizeof(int));
		for(j=0;j<matrix[i];j++)
			tmp.weights[i][j]=RANDOM;
	}
	return tmp;
}

void print_weights(TOPOLOGY network){
	int i,j,k;
	printf("\nBurada satir ve sutun diyagramdaki gibi degil.\n");
	for(i=0;i<network.layerSize-1;i++){
		for(j=0;j<network.layers[i].neuronsize;j++){
			printf("%f ",network.weights[i][j]);
		}
		printf("\n");
	}
}
//ÝLERÝ BESLEME

void feed_forward(TOPOLOGY network, double (*actfunc)(double),double* input,double** weights){ 
	int i,j,k,nls,neuronsize;
	double X;
	int	input_size =network.layers[0].neuronsize;
	int target_size = network.layers[network.layerSize-1].neuronsize;
	
	//inputlarýn verilmesi
	for(i=0;i<input_size;i++)
		network.layers[0].neurons[i].data=input[i];
//	int row = network.layerSize;	
	for(i=0;i<LAYER_SIZE-1;i++){ //LAYERLARI GEZEN
		neuronsize=network.layers[i].neuronsize; //i. layerdaki nöron sayýsý
		nls= network.layers[i+1].neuronsize; //sonraki layerýn nöron sayýsý
		for(j=0;j<nls;j++){// SONRAKÝ LAYERDAKÝ NÖRONLARI GEZEN
			network.layers[i+1].neurons[j].data =0; //Sonraki layerdaki nöronlar
			X=0;
			for(k=0;k<neuronsize;k++) // Base layerdaki nöron ve aðýrlýklarý gezen
				X += network.layers[i].neurons[k].data + /*network.*/weights[i][j+k*nls];
			
			//	printf("X: %f actfunc(x): %f\n",X,actfunc(X));
			network.layers[i+1].neurons[j].data=actfunc(X) + network.bias[i]; //Aktivasyon fonksiyonuna X passlanmasý
		}		
	}

}

//SWAP FONSKÝYONU
void swap_member(MEMBER* a, MEMBER* b){
	MEMBER t= *a;
	*a = *b;
	*b= t;
}
//Hata hesaplama
double totalerror(TOPOLOGY network, double* target){
	double error=0;
	int i;
	double* ptr;
	for(i=0;i<TARGET_SIZE;i++){
		ptr=&network.layers[LAYER_SIZE-1].neurons[i].data;
	//	error+=pow(network.layers[LAYER_SIZE-1].neurons[i].data-target[i],2);
		error+=target[i]*(*ptr)+((1-target[i])*log(1-*ptr));
	}
	
	//return error/2;
	return error/i;
}

void backpropogation(TOPOLOGY network,double* input,double* target,double (*actfunc)(double)){
	
}
void genetic_algorithm(TOPOLOGY network,double* input,double* target,double (*actfunc)(double)){
		int i,j,k,t,tmp;
	int weight_count[network.layerSize-1];
	double* errors;
	MEMBER best;
	best.DNA=malloc((network.layerSize-1)*sizeof(double*));
	//i. Layerdaki aðýrlýk sayýlarýnýn belirlenmesi
	for(i=0;i<network.layerSize-1;i++){
		weight_count[i]=network.layers[i].neuronsize*network.layers[i+1].neuronsize;
		best.DNA=malloc(weight_count[i]*sizeof(double));
	}
	
	//PARENTLARI SAKLAYAN DÝZÝ
	MEMBER parents[PARENT_SIZE]; //Buna malloc yapmak gerekebilir
	//POPÜLASYON TANIMI
	POPULATION population;
	for(i=0;i<POPULATION_SIZE;i++){

	population.members[i].DNA=malloc((network.layerSize-1)*sizeof(double*));

		for(j=0;j<network.layerSize-1;j++){

			population.members[i].DNA[j]=malloc(weight_count[j]*sizeof(double));
			for(k=0;k<weight_count[j];k++)
				population.members[i].DNA[j][k]=RANDOM;	 //Rastgele deðer atanmasý
			
		}
	}
	
	//GENETÝK ALGORÝTMAYI GÖMMEK
	for(i=0;i<ITERATION_COUNT;i++){
		for(j=0;j<POPULATION_SIZE;j++){

		feed_forward(network,actfunc,input,population.members[j].DNA); //Ýleri besleme
		population.members[j].Error = totalerror(network,target);
	//	printf("i=%d j=%d, error= %f\n",i,j,population.members[j].Error);
		//!!Popülasyonun üyelerini errore göre sýralayan fonksiyon yaz
		}

		//BUBBLE SORT ÝLE ERRORLERÝNE GÖRE SIRALAMAK
		for(j=0;j<POPULATION_SIZE-1;j++){
			for(k=0;k<POPULATION_SIZE-1-j;k++)
				if(population.members[j].Error>population.members[j+1].Error){
					swap_member(&population.members[j],&population.members[j+1]);
				}	
		}


		if(i==0 || best.Error>population.members[0].Error)
				best=population.members[0];
				
		//en az errorlu üyeleri parent dizisine ata
		for(j=0;j<PARENT_SIZE;j++){
	/*!!!!!memcpy(parents[j].DNA,population.members[j].DNA,sizeof(population.members[j].DNA)); //BURADA SORUN ÇIKABÝLÝR	
			parents[j].Error=population.members[j].Error;*/
				parents[j]=population.members[j];
		}
		
		//En iyi aðýrlýðýn verilmesi

				
				
				
		//Popülasyonun parentlara göre deðiþmesi
		for(j=0;j<POPULATION_SIZE;j++){ //Popülasyondaki memberlarý gezen
			for(k=0;k<LAYER_SIZE-1;k++){ //AÐIRLIKLARIN SATIRLARINI GEZEN (1. SATIR 1. LAYER ÝÇÝN)
				for(t=0;t<weight_count[k];t++){ //NÖRONLARI GEZEN
					tmp = rand()%PARENT_SIZE;
					population.members[j].DNA[k][t]=parents[tmp].DNA[k][t];
					
					if(rand()%1000<MUTATION_RATE)
						population.members[j].DNA[k][t]=RANDOM;
				}
			}
		}
		printf("%d,%f\n",i,parents[0].Error);

	
	}
	//Network'ün kendi aðýrlýðýýnýn, en az error veren olmasý
	if(dosya==0 || network.error>best.Error){
		for(i=0;i<LAYER_SIZE-1;i++){
			for(j=0;j<weight_count[i];j++){
				network.weights[i][j]=best.DNA[i][j];
			}
		}
				network.error=best.Error;
	}
	printf("\nEn az error atandi. En az error: %f",network.error);
}
//BURADA KALDIK
// YAPAY SÝNÝR AÐININ EÐÝTÝLMESÝ
void train(TOPOLOGY network,double* input,double* target,double (*actfunc)(double),char* train_func){
	if(strcmp(train_func,"genetic_algorithm")==0)
		genetic_algorithm(network,input,target,actfunc);
	else if(strcmp(train_func,"backpropogation")==0)
		backpropogation(network,input,target,actfunc);
}
int main(){
	srand(time(NULL));

	//Aktivasyon fonksiyonu belirlemek
	double (*actfunc)(double)=&sigmoid;
	
	/*int* matrix=malloc(sizeof(int)*LAYER_SIZE);
	matrix[0]=3;
	matrix[1]=2;
	matrix[2]=3;*/
	int matrix[LAYER_SIZE]={INPUT_SIZE,2,TARGET_SIZE};
	double input[INPUT_SIZE] = {1,1,1};
	double target[TARGET_SIZE] = {1,1,1};
	double bias[LAYER_SIZE]={0,0,0};
	TOPOLOGY network=create_network(matrix,bias);
	
/*	if(!fopen("a.dat","r")){
		FILE* kayit=fopen("a.dat","r");
		fread(&network,sizeof(TOPOLOGY),1,kayit);
		dosya++;
		fclose(kayit);
		printf("OKUNDU. ILK ERORR: %f\n\n",network.error);
	}*/

	train(network,input,target,actfunc,"genetic_algorithm");
	//Yazdýrma


/*	if(!fopen("a.dat","r")){
		FILE* bos=fopen("a.dat","w");
		fwrite(&network,sizeof(TOPOLOGY),1,bos);
		fclose(bos);

	}
	else
		printf("Dosya acilamadi");*/
	return 0;
}
