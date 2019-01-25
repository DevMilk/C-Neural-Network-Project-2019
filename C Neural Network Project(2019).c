#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define max(a,b) (((a)>(b)) ? (a) : (b))


#define LAYER_SIZE 4
#define INPUT_SIZE 3
#define TARGET_SIZE 3
#define RANDOM (double)rand()/RAND_MAX*30.0-15.0  //-15 ile 15 arasında
#define POPULATION_SIZE 50
#define ITERATION_COUNT 300
#define PARENT_SIZE 10
#define MUTATION_RATE 500 //binde 5
int dosya=0;
//!!Popülasyondaki DNA'ların hatalarını sıralayan bir fonksiyon yazman gerek.
//MEVCUT SORUN: Şu anlık bir sorun yok fakat genetik algoritmanın garanti bir şekilde errorleri azalttığı söylenemez.
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
//Sinir Ağı constructor
TOPOLOGY create_network(int* matrix,double* bias){

	int i=0,col,j;
	TOPOLOGY tmp;
//	int row = sizeof(matrix) / sizeof(matrix[i]);
	tmp.layerSize=LAYER_SIZE;
	tmp.layers=(LAYER*)malloc(LAYER_SIZE*sizeof(LAYER)); //Topolojinin satırlarının atanması
	tmp.weights=malloc((LAYER_SIZE-1)*sizeof(int*)); //Ağırlıkların yer açılması
	tmp.bias = malloc(LAYER_SIZE*sizeof(int)); //biaslara yer açılması

	for(i=0;i<LAYER_SIZE;i++){
	printf("r\n");
		tmp.bias[i]=bias[i];
		col=matrix[i];
		tmp.layers[i].neuronsize=col; 
		tmp.layers[i].neurons=(NEURON*)malloc(col*sizeof(NEURON));//Topolojinin satırlarındaki nöronların sayısının atanması
	}
			printf("a\n");

	for(i=0;i<LAYER_SIZE-1;i++){ //ağırlıklara rastgele değerler verilmesi
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
			printf("%lf ",network.weights[i][j]);
		}
		printf("\n");
	}
}
//İLERİ BESLEME

void feed_forward(TOPOLOGY network, double (*actfunc)(double),double* input,double** weights){ 
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
			network.layers[i+1].neurons[j].data =0; //Sonraki layerdaki nöronlar
			X=0;
			for(k=0;k<neuronsize;k++){ // Base layerdaki nöron ve ağırlıkları gezen
				X += network.layers[i].neurons[k].data + /*network.*/weights[i][j+k*nls];
			
		}
			//	printf("X: %f actfunc(x): %f\n",X,actfunc(X));
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
		//error+=target[i]*(*ptr)+((1-target[i])*log(1-*ptr));
	}
	
	return error;
//	return error/i;
}

void backpropogation(TOPOLOGY network,double* input,double* target,double (*actfunc)(double)){
	
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
	//	printf("%lf\n",population.members[j].Error);
	//	printf("i=%d j=%d, error= %f\n",i,j,population.members[j].Error);
		//!!Popülasyonun üyelerini errore göre sıralayan fonksiyon yaz
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

		//	printf("Error: %.10lf\n",best.Error);
		/*	for(j=0;j<TARGET_SIZE;j++)
				printf("%.10lf ",network.layers[LAYER_SIZE-1].neurons[j].data);*/
		//	printf("\n\n");
			//	printf("%d,%lf\n",i,best.Error);
		//en az errorlu üyeleri parent dizisine ata
//	}
	

		for(j=0;j<PARENT_SIZE;j++){
	/*!!!!!memcpy(parents[j].DNA,population.members[j].DNA,sizeof(population.members[j].DNA)); //BURADA SORUN ÇIKABİLİR	
			parents[j].Error=population.members[j].Error;*/
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
//	if(dosya==0 || network.error>best.Error){
		network.weights=malloc((LAYER_SIZE-1)*sizeof(double));
		for(i=0;i<LAYER_SIZE-1;i++){
			network.weights[i]=malloc(weight_count[i]*weight_count[i+1]*sizeof(double));
			for(j=0;j<weight_count[i];j++){
				network.weights[i][j]=best.DNA[i][j];
			}
		}
	//	printf("\nson best: %.10lf\n",best.Error);
		/*feed_forward(network,actfunc,input,best.DNA);
		network.error= best.Error;*/
//	}
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
	
	/*int* matrix=malloc(sizeof(int)*LAYER_SIZE);
	matrix[0]=3;
	matrix[1]=2;
	matrix[2]=3;*/
	int matrix[LAYER_SIZE]={INPUT_SIZE,2,5,TARGET_SIZE};
	double input[INPUT_SIZE] = {1,1,1};
	double target[TARGET_SIZE] = {100,150,106};
	double bias[LAYER_SIZE]={0,0,0,0};
		printf("a");

	TOPOLOGY network=create_network(matrix,bias);

/*	if(!fopen("a.dat","r")){0
		FILE* kayit=fopen("a.dat","r");
		fread(&network,sizeof(TOPOLOGY),1,kayit);
		dosya++;
		fclose(kayit);
		printf("OKUNDU. ILK ERORR: %f\n\n",network.error);
	}*/

	train(network,input,target,actfunc,"genetic_algorithm");
	//Yazdırma


/*	if(!fopen("a.dat","r")){
		FILE* bos=fopen("a.dat","w");
		fwrite(&network,sizeof(TOPOLOGY),1,bos);
		fclose(bos);

	}
	else
		printf("Dosya acilamadi");*/
		printf("\n");
		feed_forward(network,actfunc,input,network.weights);
		printf("\n");
	/*	for(i=0;i<TARGET_SIZE;i++)
			printf("%lf ",network.layers[LAYER_SIZE-1].neurons[i].data);
			
		printf("\n\n");
		print_weights(network);*/
	return 0;
}
