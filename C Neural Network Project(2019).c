#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define max(a,b) (((a)>(b)) ? (a) : (b))


#define LAYER_SIZE 3
#define INPUT_SIZE 3
#define TARGET_SIZE 3
#define RANDOM ((double)rand()/RAND_MAX)*2.0-1.0  //-1 ve 1 arasında
#define POPULATION_SIZE 50
#define ITERATION_COUNT 300
#define ITERATION_COUNT_BP 300
#define PARENT_SIZE 10
#define MUTATION_RATE 500 //bindelik değer
#define LEARNING_RATE 0.000001 
#define DROPOUT_RATE 50 //yüzde cinsinden
#define MOMENTUM 1
int dosya=0;
//!!Popülasyondaki DNA'ların hatalarını sıralayan bir fonksiyon yazman gerek.
//MEVCUT SORUN: Uğur'u Seyir Defteri 28.03.2019 saat 00:15...
//... Backpropogation'da errorler yar sürekli azalıyor ya da sürekli artıyor. 5 denemenin 4'ünde azalma var fakat bu doğru çalıştığını göstermez



// Genetik Algoritma Backpropogation'a göre daha verimli. Genetik algoritma errory 3000 iterasyonda 121'den 70'e getirmeyi başarmışken Backpropogation erroru 112'dem 75'e çekti
//... Ayrıca aralığı -2,5 ve 2,5 arasına sabitleyince Genetik algoritma 89 olan erroru 3000 iterasyonla 0.5 yaptı. 
//EK BİLGİ: Relu fonksiyonu kullanırsak, x>0 değeri için 1, x<0 değeri için 0 verir fakat x=0 için bir şey yapılamaz...
//... Onun yerine x=0 olduğunda 0,0.5 ya da 1 olarak 3 değerden birini verebilirsin.


//Üyeler
typedef struct{
	double** DNA; //ağırlıklara göre malloc et
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

//NETWORK MİMARİSİ
typedef struct {
	int layerSize;
	LAYER* layers;
	double** weights;
	double* bias;
	double error;
}TOPOLOGY;

//AKTİVASYON FONKSİYONLARI
double relu(double X){
	return max(0,X);
}
double sigmoid(double x){
     return 1 / (1 + exp(-x));
}
double relu_der(double X){
	if(X>0)
		return 1;
	else if(X==0)
		return 0.5;
	else 
		return 0;
}
double sigmoid_der(double X){
	     double a= 1 / (1 + exp(-X));
	     return a*(1-a);
}
//Sinir Ağı constructor
TOPOLOGY create_network(int* matrix,double* bias){

	int i=0,col,j;
	TOPOLOGY tmp;
//	int row = sizeof(matrix) / sizeof(matrix[i]);
	tmp.layerSize=LAYER_SIZE;
	tmp.layers=(LAYER*)malloc(LAYER_SIZE*sizeof(LAYER)); //Topolojinin satırlarının atanması
	tmp.weights=malloc((LAYER_SIZE-1)*sizeof(double*)); //Ağırlıkların yer açılması
	tmp.bias = malloc((LAYER_SIZE-1)*sizeof(double)); //biaslara yer açılması
	
	for(i=0;i<LAYER_SIZE;i++){
		tmp.bias[i]=bias[i];
		col=matrix[i];
		tmp.layers[i].neuronsize=col; 
		tmp.layers[i].neurons=(NEURON*)malloc(col*sizeof(NEURON));//Topolojinin satırlarındaki nöronların sayısının atanması
	}

	for(i=0;i<LAYER_SIZE-1;i++){ //ağırlıklara rastgele değerler verilmesi
		tmp.weights[i]=malloc(matrix[i]*matrix[i+1]*sizeof(double));
		for(j=0;j<matrix[i]*matrix[i+1];j++){
			tmp.weights[i][j]=RANDOM;
		}

	}
	return tmp;
}

void print_weights(TOPOLOGY network){
	int i,j,k;
	for(i=0;i<network.layerSize-1;i++){
		for(j=0;j<network.layers[i].neuronsize*network.layers[i+1].neuronsize;j++){
			printf("%.5lf ",network.weights[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}
//İLERİ BESLEME

void feed_forward(TOPOLOGY network, double (*actfunc)(double),double* input,double** weights /*,DROPOUT MATRİXİ*/){ 
	int i,j,k,nls,neuronsize;
	double X;
	int	input_size =network.layers[0].neuronsize;
	int target_size = network.layers[network.layerSize-1].neuronsize;
	//inputların verilmesi
	for(i=0;i<input_size;i++)
		network.layers[0].neurons[i].data=input[i];
//	int row = network.layerSize;	

	for(i=0;i<LAYER_SIZE-1;i++){ //LAYERLARI GEZEN
		neuronsize=network.layers[i].neuronsize; //i. layerdaki nöron sayısı
		nls= network.layers[i+1].neuronsize; //sonraki layerın nöron sayısı
		for(j=0;j<nls;j++){// SONRAKİ LAYERDAKİ NÖRONLARI GEZEN
			X=0;
			for(k=0;k<neuronsize;k++){ // Base layerdaki nöron ve ağırlıkları gezen
				X += network.layers[i].neurons[k].data*weights[i][j+k*nls];
			}
			network.layers[i+1].neurons[j].data=actfunc(X) + network.bias[i]; //Aktivasyon fonksiyonuna X passlanması
		}		
	}
}

//SWAP FONSKİYONU
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
		error+=pow(network.layers[LAYER_SIZE-1].neurons[i].data-target[i],2);
	}
	
	return error/2;
}

void backpropogation(TOPOLOGY network,double* input,double* target,double (*actfunc)(double)){
	int i,j,k;
	double (*actderivfunc)(double);
	if(actfunc==relu){
		actderivfunc=&relu_der;
	}
	else if(actfunc==sigmoid){
		actderivfunc=&sigmoid_der;
	}
	//Yeni ağırlıkları taşıyacak matrix
	double** newWeights=malloc((network.layerSize-1)*sizeof(double*));
	for(i=0;i<network.layerSize-1;i++)
		newWeights[i]=malloc(network.layers[i].neuronsize*network.layers[i+1].neuronsize*sizeof(double));
	
	
	LAYER* output = &network.layers[network.layerSize-1];
	
	//OUTPUT'UN DERİVED DATASININ BELİRLENMESİ
	for(i=0;i<output->neuronsize;i++){
		//Her errorün hesaplanıp outputun derivlerine aktarılması
		output->neurons[i].derived_Data= (output->neurons[i].data-target[i])*actderivfunc(output->neurons[i].data);
	}

	feed_forward(network,actfunc,input,network.weights);
		printf("\n");
	for(i=0;i<network.layerSize;i++){
		for(j=0;j<network.layers[i].neuronsize;j++)
			printf("%f ",network.layers[i].neurons[j].data);
		printf("\n");
	}
	
	printf("\n");
	int sayac=0;
	double _error = totalerror(network,target);
	int weight_count;
	int d,p;
	
	while((sayac++)!=ITERATION_COUNT_BP){
		for(k=network.layerSize-1;k>0;k--){
			LAYER* next = &network.layers[k];
			LAYER* back = &network.layers[k-1];
			
		/*
			DROPOUT İMPLEMENTASYONU		
			for(d=0;d<network.layerSize;d++){
				for(k=0;k<network.layers[d].neuronsize;k++){
					if(rand()%101<DROPOUT_RATE)
						network.layers[d].neurons[k].
				}
			}*/
			
			for(i=0;i<back->neuronsize;i++){
				
				back->neurons[i].derived_Data=0;
				for(j=0;j<next->neuronsize;j++){
					//Weight Formatım: network.weights[LAYER NUMARASI][(BAĞLI OLDUĞU OUTPUT)+(BAĞLI OLDUĞU INPUT)*(BAĞLI OLDUĞU OUTPUT NÖRON SAYISI)]
										
					newWeights[k-1][i+j*back->neuronsize]= 
					MOMENTUM*network.weights[k-1][i+j*back->neuronsize]-(LEARNING_RATE)*(next->neurons[j].derived_Data)*(back->neurons[i].data);
					
					back->neurons[i].derived_Data-= next->neurons[j].derived_Data*network.weights[k-1][i+j*back->neuronsize];

				}
			//En son aktivasyon fonksiyonunun deriviyle çarpılamsı
				back->neurons[i].derived_Data*=actderivfunc(back->neurons[i].data);
			
			}
		}
		
		//Yeni ağırlıkların verilmesi
		int h=0;
		for(k=0;k<network.layerSize-1;k++){
			weight_count = network.layers[k].neuronsize*network.layers[k+1].neuronsize;
			for(i=0;i<weight_count;i++){

				network.weights[k][i]=newWeights[k][i];
			}
		}
		feed_forward(network,actfunc,input,network.weights);
		printf("%d,%lf\n",sayac,totalerror(network,target));
		//print_weights(network);
	}

}
void genetic_algorithm(TOPOLOGY network,double* input,double* target,double (*actfunc)(double)){
	int i,j,k,t,tmp,f;
	int weight_count[network.layerSize-1];
	double* errors;
	MEMBER best;
	best.DNA=malloc((network.layerSize-1)*sizeof(double*));
	
	
	//i. Layerdaki ağırlık sayılarının belirlenmesi
	for(i=0;i<network.layerSize-1;i++){
		weight_count[i]=network.layers[i].neuronsize*network.layers[i+1].neuronsize;
		best.DNA[i]=malloc(weight_count[i]*sizeof(double));
	}
	
	//PARENTLARI SAKLAYAN DİZİ
	MEMBER parents[PARENT_SIZE]; //Buna malloc yapmak gerekebilir
	//POPÜLASYON TANIMI
	POPULATION population;
	for(i=0;i<POPULATION_SIZE;i++){

	population.members[i].DNA=malloc((LAYER_SIZE-1)*sizeof(double*));

		for(j=0;j<LAYER_SIZE-1;j++){

			population.members[i].DNA[j]=malloc(weight_count[j]*sizeof(double));
			for(k=0;k<weight_count[j];k++)
				population.members[i].DNA[j][k]=RANDOM;	 //Rastgele değer atanması
			
		}
	}
	
	
	
	
	//GENETİK ALGORİTMAYI GÖMMEK
	for(i=0;i<ITERATION_COUNT;i++){
		for(j=0;j<POPULATION_SIZE;j++){

		feed_forward(network,actfunc,input,population.members[j].DNA); //İleri besleme
		population.members[j].Error = totalerror(network,target);
		}

		//BUBBLE SORT İLE ERRORLERİNE GÖRE SIRALAMAK
		for(j=0;j<POPULATION_SIZE-1;j++){
			for(k=0;k<POPULATION_SIZE-1-j;k++)
				if(population.members[j].Error>population.members[j+1].Error){
					swap_member(&population.members[j],&population.members[j+1]);
				}	
		}

		if(i==0 || best.Error>population.members[0].Error){
			for(j=0;j<LAYER_SIZE-1;j++){
				for(k=0;k<weight_count[j];k++){
					best.DNA[j][k]=population.members[0].DNA[j][k];
				}
			}
			feed_forward(network,actfunc,input,best.DNA);
			best.Error=totalerror(network,target);

		for(j=0;j<PARENT_SIZE;j++){
				parents[j]=population.members[j];
		}
	}
		//En iyi ağırlığın verilmesi

							printf("%d,%.10lf\n",i,parents[0].Error);

		//Popülasyonun parentlara göre değişmesi
		for(j=0;j<POPULATION_SIZE;j++){ //Popülasyondaki memberları gezen

			for(k=0;k<LAYER_SIZE-1;k++){ //AĞIRLIKLARIN SATIRLARINI GEZEN (1. SATIR 1. LAYER İÇİN)
				for(t=0;t<weight_count[k];t++){ //NÖRONLARI GEZEN
					tmp = rand()%PARENT_SIZE;
					population.members[j].DNA[k][t]=parents[tmp].DNA[k][t];
					
					if(rand()%1000<MUTATION_RATE)
						population.members[j].DNA[k][t]=RANDOM;
				}
			}
		}

	}
	//Network'ün kendi ağırlığıının, en az error veren olması
		network.weights=malloc((LAYER_SIZE-1)*sizeof(double));
		for(i=0;i<LAYER_SIZE-1;i++){
			network.weights[i]=malloc(weight_count[i]*weight_count[i+1]*sizeof(double));
			for(j=0;j<weight_count[i];j++){
				network.weights[i][j]=best.DNA[i][j];
			}
		}

	printf("Son Dna: %.10lf\n",best.DNA[0][0]);

	feed_forward(network,actfunc,input,best.DNA);
	network.error=totalerror(network,target);
	printf("\nEn az error atandi. En az error: %.10lf\n",network.error);
	for(i=0;i<TARGET_SIZE;i++)
		printf("%.10lf ",network.layers[LAYER_SIZE-1].neurons[i].data);
}
//BURADA KALDIK
// YAPAY SİNİR AĞININ EĞİTİLMESİ
void train(TOPOLOGY network,double* input,double* target,double (*actfunc)(double),char* train_func){

	if(strcmp(train_func,"genetic_algorithm")==0)
		genetic_algorithm(network,input,target,actfunc);
	else if(strcmp(train_func,"backpropogation")==0)
		backpropogation(network,input,target,actfunc);
}
int main(){
	srand(time(NULL));
	int i;
	//Aktivasyon fonksiyonu belirlemek
	double (*actfunc)(double)=&relu; //RELU OLURSA NEGATİF AĞIRLIKLAR DİREK 0 SAYILIYOR
	

	int matrix[LAYER_SIZE]={INPUT_SIZE,2,TARGET_SIZE};
	double input[INPUT_SIZE] = {1,1,1};
	double target[TARGET_SIZE] = {10,9,8};
	double bias[LAYER_SIZE-1]={0.05,0.05};

	//Networku oluştur
	TOPOLOGY network=create_network(matrix,bias);
	train(network,input,target,actfunc,"backpropogation");
	printf("\n\n");

	return 0;
}
