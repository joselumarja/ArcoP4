/*
Jose Luis Mira Serrano
Ruben Marquez Villalta

Las opciones de ejecucion seleccionables estan en el rango de 1-4 aplicando los algoritmos de aplicacion de los vectores vertical y horizontal de la siguiente forma:
1-secuencial secuencial
2-paralelo paralelo
3-secuencial paralelo
4-paralelo secuencial
*/

#include <QtGui/QApplication>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsPixmapItem>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 5
const int M=N/2;
int matrizGauss[N][N] =	{{ 1, 4, 6, 4,1 },
						 { 4,16,24,16,4 },
						 { 6,24,36,24,6 },
						 { 4,16,24,16,4 },
						 { 1, 4, 6, 4,1}};
int vectorGauss[N] = { 1, 4, 6, 4, 1 };

int alto, ancho, opt;
#define max(num1, num2) (num1>num2 ? num1 : num2)
#define min(num1, num2) (num1<num2 ? num1 : num2)

double naive_matriz(QImage* image, QImage* result) {
  
  int h, w, i, j;
  int red, green, blue;
  int mini, minj, supi, supj;
  QRgb aux;  
  double start_time = omp_get_wtime();

  for (h = 0; h < alto; h++)
    
    for (w = 0; w < ancho ; w++) {

	  /* Aplicamos el kernel (matrizGauss dividida entre 256) al entorno de cada pixel de coordenadas (w,h)
		 Corresponde multiplicar la componente matrizGauss[i,j] por el pixel de coordenadas (w-M+j, h-M+i)
		 Pero ha de cumplirse 0<=w-M+j<ancho y 0<=h-M+i<alto. Por tanto: M-w<=j<ancho+M-w y M-h<=i<alto+M-h
		 Además, los índices i y j de la matriz deben cumplir 0<=j<N, 0<=i<N
		 Se deduce:  máx(M-w,0) <= j < mín(ancho+M-w,N); máx(M-h,0) <= i < mín(alto+M-h,N)*/
	  red=green=blue=0;
	  mini = max((M-h),0); minj = max((M-w),0);					// Ver comentario anterior
	  supi = min((alto+M-h),N); supj = min((ancho+M-w),N);	    // Íd.
	  for (i=mini; i<supi; i++)
		for (j=minj; j<supj; j++)	{
			aux = image->pixel(w-M+j, h-M+i);
			red += matrizGauss[i][j]*qRed(aux);
			green += matrizGauss[i][j]*qGreen(aux);
			blue += matrizGauss[i][j]*qBlue(aux);
		};

	  red /= 256; green /= 256; blue /= 256;
	  result->setPixel(w,h, QColor(red, green, blue).rgba());
      
	}
  
  return omp_get_wtime() - start_time;    
}	// Fin naive_matriz

void aplicar_vect_vertical(int *ResultadoParcialRojo,int *ResultadoParcialVerde,int *ResultadoParcialAzul, const int *VectorRed, const int *VectorGreen, const int *VectorBlue,int h, int w, int mini, int minj, int supi, int supj)
{
	int i,j,PixelPosition;
	
	for (i=mini; i<supi; i++)
	{
		for (j=minj; j<supj; j++)	
		{
			PixelPosition=((h-M+i)*ancho)+w+j-M;
			ResultadoParcialRojo[(i*N)+j]=VectorRed[PixelPosition]*vectorGauss[i];
			ResultadoParcialVerde[(i*N)+j]=VectorGreen[PixelPosition]*vectorGauss[i];
			ResultadoParcialAzul[(i*N)+j]=VectorBlue[PixelPosition]*vectorGauss[i];
		}
	}
}

void vect_vertical_paral(int *ResultadoParcialRojo,int *ResultadoParcialVerde,int *ResultadoParcialAzul, const int *VectorRed, const int *VectorGreen, const int *VectorBlue, int h, int w, int mini, int minj, int supi, int supj)
{
	int i,j,PixelPosition;
	
	#pragma omp parallel for schedule(static,6) private(i,j,PixelPosition) 
  	//#pragma omp parallel for schedule(dynamic,6) private(i,j,PixelPosition)
  	//#pragma omp parallel for schedule(static,supj-minj/omp_get_num_procs()) private(i,j,PixelPosition)
	//#pragma omp parallel for schedule(dynamic,supi-mini/omp_get_num_procs()) private(i,j,PixelPosition)*/
	
	for (i=mini; i<supi; i++)
	{
		for (j=minj; j<supj; j++)	
		{
			PixelPosition=((h-M+i)*ancho)+w+j-M;
			ResultadoParcialRojo[(i*N)+j]=VectorRed[PixelPosition]*vectorGauss[i];
			ResultadoParcialVerde[(i*N)+j]=VectorGreen[PixelPosition]*vectorGauss[i];
			ResultadoParcialAzul[(i*N)+j]=VectorBlue[PixelPosition]*vectorGauss[i];
		}
	}
}

void aplicar_vect_horizontal(int *ResultadoParcialRojo,int *ResultadoParcialVerde,int *ResultadoParcialAzul, int mini,int minj,int supi,int supj)
{
	int i,j;
	
	for (j=minj; j<supj; j++)
	{
		for (i=mini; i<supi; i++)	
		{
			ResultadoParcialRojo[(i*N)+j]*=vectorGauss[j];
			ResultadoParcialVerde[(i*N)+j]*=vectorGauss[j];
			ResultadoParcialAzul[(i*N)+j]*=vectorGauss[j];
		}
	}
}

void vect_horizontal_paral(int *ResultadoParcialRojo,int *ResultadoParcialVerde,int *ResultadoParcialAzul, int mini,int minj,int supi,int supj)
{
	int i,j;
	
	#pragma omp parallel for schedule(static,6) private(i,j)
  	//#pragma omp parallel for schedule(dynamic,6) private(i,j)
  	//#pragma omp parallel for schedule(static,supj-minj/omp_get_num_procs()) private(i,j)
	//#pragma omp parallel for schedule(dynamic,supj-minj/omp_get_num_procs()) private(i,j)
	
	for (j=minj; j<supj; j++)
	{
		for (i=mini; i<supi; i++)	
		{
			ResultadoParcialRojo[(i*N)+j]*=vectorGauss[j];
			ResultadoParcialVerde[(i*N)+j]*=vectorGauss[j];
			ResultadoParcialAzul[(i*N)+j]*=vectorGauss[j];
		}
	}
}

/*Inicializamos los vectores Rojo Verde Y Azul desde la imagen*/
void InitializeVectors(QImage* image, int *VectorRed,int *VectorGreen,int *VectorBlue)
{
	int i,j,PixelPosition;
	QRgb aux;
	
	for(i=0;i<alto;i++)
	{
		for(j=0;j<ancho;j++)
		{
			aux = image->pixel(j, i);
			PixelPosition=(i*ancho)+j;
			VectorRed[PixelPosition]=qRed(aux);
			VectorGreen[PixelPosition]=qGreen(aux);
			VectorBlue[PixelPosition]=qBlue(aux);
		}
	}
}

void InitializeVectorsParallel(QImage* image, int *VectorRed,int *VectorGreen,int *VectorBlue)
{
	int i,j,PixelPosition;
	QRgb aux;
	
	#pragma omp parallel for schedule(static,6) private(aux,i,j,PixelPosition) 
  	//#pragma omp parallel for schedule(dynamic,6) private(aux,i,j,PixelPosition)
  	//#pragma omp parallel for schedule(static,alto/omp_get_num_procs()) private(aux,i,j,PixelPosition)
	//#pragma omp parallel for schedule(dynamic,alto/omp_get_num_procs()) private(aux,i,j,PixelPosition)
	for(i=0;i<alto;i++)
	{
		for(j=0;j<ancho;j++)
		{
			aux = image->pixel(j, i);
			PixelPosition=(i*ancho)+j;
			VectorRed[PixelPosition]=qRed(aux);
			VectorGreen[PixelPosition]=qGreen(aux);
			VectorBlue[PixelPosition]=qBlue(aux);
			
		}
	}
}

double separa_vectores(QImage* image, QImage* result)
{

  	double start_time = omp_get_wtime();
  	
	int h, w, i, j;
	/*Matrices con la imagen dividida en RGB*/
	int *VectorRed=(int*) malloc(sizeof(int)*ancho*alto); 
	int *VectorGreen=(int*) malloc(sizeof(int)*ancho*alto);
	int *VectorBlue=(int*) malloc(sizeof(int)*ancho*alto);
	
  	int mini, minj, supi, supj;
  	int red, green, blue;
  	/*Matrices locales para realizar los calculos sobre los colores Rojo Verde y Azul*/
  	int *ResultadoParcialRojo=(int*) malloc(sizeof(int)*N*N);
  	int *ResultadoParcialVerde=(int*) malloc(sizeof(int)*N*N);
  	int *ResultadoParcialAzul=(int*) malloc(sizeof(int)*N*N);
  	  	
  	InitializeVectorsParallel(image,VectorRed, VectorGreen, VectorBlue);
  	
  	switch(opt)
	{
	  	case 1:
	  			
	  		printf("Ejecutando: aplicar_vect_vertical() (seq) aplicar_vect_horizontal() (seq)\n");
			break;
					
	  	case 2:
	  			
	  		printf("Ejecutando: vect_vertical_paral() (paral) vect_horizontal_paral() (paral)\n");
			break;
					
	  	case 3:
	  			
	  		printf("Ejecutando: aplicar_vect_vertical() (seq) vect_horizontal_paral() (paral)\n");
			break;
					
	  	case 4:
	  			
	  		printf("Ejecutando: vect_vertical_paral() (paral) aplicar_vect_horizontal() (seq)\n");
			break;
					
	  	default:
	  	
	  		printf("Opcion de ejecucuion incorrecta\n");
	  		printf("Ejecutando por defecto: aplicar_vect_vertical() (seq) aplicar_vect_horizontal() (seq)\n"); 
			break;
	}
  	
  	
  	for (h = 0; h < alto; h++)
  	{
    
    	for (w = 0; w < ancho ; w++) 
    	{  	
  			mini = max((M-h),0); minj = max((M-w),0);					
			supi = min((alto+M-h),N); supj = min((ancho+M-w),N);
    		red=green=blue=0;
	  		
	  		switch(opt)
	  		{
	  			case 1:
	  			
	  				aplicar_vect_vertical(ResultadoParcialRojo, ResultadoParcialVerde, ResultadoParcialAzul, VectorRed, VectorGreen, VectorBlue, h, w, mini, minj, supi, supj);
					aplicar_vect_horizontal(ResultadoParcialRojo, ResultadoParcialVerde, ResultadoParcialAzul, mini, minj, supi, supj);
					break;
					
	  			case 2:
	  			
	  				vect_vertical_paral(ResultadoParcialRojo, ResultadoParcialVerde, ResultadoParcialAzul, VectorRed, VectorGreen, VectorBlue, h, w, mini, minj, supi, supj);
					vect_horizontal_paral(ResultadoParcialRojo, ResultadoParcialVerde, ResultadoParcialAzul, mini, minj, supi, supj);
					break;
					
	  			case 3:
	  			
	  				aplicar_vect_vertical(ResultadoParcialRojo, ResultadoParcialVerde, ResultadoParcialAzul, VectorRed, VectorGreen, VectorBlue, h, w, mini, minj, supi, supj);
					vect_horizontal_paral(ResultadoParcialRojo, ResultadoParcialVerde, ResultadoParcialAzul, mini, minj, supi, supj);
					break;
					
	  			case 4:
	  			
	  				vect_vertical_paral(ResultadoParcialRojo, ResultadoParcialVerde, ResultadoParcialAzul, VectorRed, VectorGreen, VectorBlue, h, w, mini, minj, supi, supj);
					aplicar_vect_horizontal(ResultadoParcialRojo, ResultadoParcialVerde, ResultadoParcialAzul, mini, minj, supi, supj);
					break;
					
	  			default:
	  			
					aplicar_vect_vertical(ResultadoParcialRojo, ResultadoParcialVerde, ResultadoParcialAzul, VectorRed, VectorGreen, VectorBlue, h, w, mini, minj, supi, supj);
					aplicar_vect_horizontal(ResultadoParcialRojo, ResultadoParcialVerde, ResultadoParcialAzul, mini, minj, supi, supj);
					break;
			}
			
			for(i=mini;i<supi;i++)
			{
				for(j=minj;j<supj;j++)
				{
					/*Suma de los resultados de las matrices locales para obtener el pixel con el filtro*/
					red+=ResultadoParcialRojo[(i*N)+j];
					green+=ResultadoParcialVerde[(i*N)+j];
					blue+=ResultadoParcialAzul[(i*N)+j];

				}
			}

			red/=256; green/=256; blue/=256;
	  		result->setPixel(w,h, QColor(red, green, blue).rgba());
		}
	}
	
	/*Liberacion de recursos*/
	free(ResultadoParcialRojo);
	free(ResultadoParcialVerde);
	free(ResultadoParcialAzul);
	free(VectorRed);
	free(VectorGreen);
	free(VectorBlue);
  
  return omp_get_wtime() - start_time;    
}

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QGraphicsScene scene;
    QGraphicsView view(&scene);
	
    if (argc != 3) {printf("Vuelva a ejecutar. Uso: <ejecutable> <archivo imagen> <Opciones de ejecucion> (1-4)\n"); return -1;} 
    QPixmap qp = QPixmap(argv[1]);
    if(qp.isNull()) { printf("image not found\n"); return -1;}
	
	opt=atoi(argv[2]);
	
    QImage image = qp.toImage();
    
    alto = image.height(); ancho = image.width();
    
    QImage matrGaussImage(image);
    
    double computeTime = naive_matriz(&image, &matrGaussImage);
    printf("naive_matriz time: %0.9f seconds\n", computeTime);
    
    QImage vectGaussImage(image);
    
    computeTime = separa_vectores(&image, &vectGaussImage);
    printf("separa_vectores time: %0.9f seconds\n", computeTime);
    
    if(matrGaussImage==vectGaussImage)	printf("naive_matriz y separa_vectores dan la misma imagen\n");
    else	printf("naive_matriz y separa_vectores no dan la misma imagen\n");
    
    QPixmap pixmap = pixmap.fromImage(matrGaussImage);
    QGraphicsPixmapItem *item = new QGraphicsPixmapItem(pixmap);
    scene.addItem(item);

    view.show();
    a.exec();
    
    pixmap = pixmap.fromImage(vectGaussImage);
    item = new QGraphicsPixmapItem(pixmap);
    scene.addItem(item);

    view.show();
    return a.exec();
}
