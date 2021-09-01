#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define STB_IMAGE_IMPLEMENTATION
#include "C:\workspace\vcpkg\installed\x64-windows\include\stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "C:\workspace\vcpkg\installed\x64-windows\include\stb_image_write.h"
// stbi_write_png("A.png", input_image.w, input_image.h, input_image.c, stb_image, 0);

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "C:\workspace\vcpkg\installed\x64-windows\include\stb_image_resize.h"

#define LAYERS 9

/// M, H, W, C
/**
    The kernels weights are indexed by M(filters), H(altura), W(largura), C(canais)
*/


typedef struct Layer{
    int id;
    int type; // 0 conv; 1 fc; 2 pool; 3 flatt
    int M; // filtros
    int C; // canais
    int H; // altura
    int W; // largura
    int stride;
    int padding;
    int * weights;
    int * bias;
    int * input; // armazenar ponteiro para entrada de cada camada visando liberacao de memoria
} Layer_t;

typedef struct {
    int num_layers;
    Layer_t layers[LAYERS];
} Network;

typedef struct {
    int w;
    int h;
    int c;
    int *data;
} Image;


void weight_reader(const char* file_path, Network * network){
    FILE *fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;

    fp = fopen(file_path, "r");

    if (fp == NULL)
        exit(EXIT_FAILURE);

    //Caractere que separa as informações
    const char delimiter[] = ";";

    int layer_cnt = 0;
    char *ch;
    while ((read = getline(&line, &len, fp)) != -1) {

        // network->layers[layer_cnt].id = layer_cnt;

        // numero de pesos
        int num_weights = atoi(line);
        network->layers[layer_cnt].weights = (int*) malloc(sizeof(int) * num_weights);

        // shape line
        if ((read = getline(&line, &len, fp)) == -1){
            printf("erro\n"); break;
        }
        // shape param
        ch = strtok(line, delimiter);
        network->layers[layer_cnt].M = atoi(ch);
        // shape param
        ch = strtok(NULL, delimiter);
        network->layers[layer_cnt].W = atoi(ch);
        // shape param
        ch = strtok(NULL, delimiter);
        network->layers[layer_cnt].H = atoi(ch);
        // shape param
        ch = strtok(NULL, delimiter);
        network->layers[layer_cnt].C = atoi(ch);
        /////////////////////////////

        // kernel weights
        if ((read = getline(&line, &len, fp)) == -1){
            printf("erro\n"); break;
        }
        // weights
        ch = strtok(line, delimiter);
        network->layers[layer_cnt].weights[0] = atoi(ch);
        for(int i = 1; i < num_weights; i++){
            ch = strtok(NULL, delimiter);
            network->layers[layer_cnt].weights[i] = atoi(ch);
        }

        // numero de bias
        if ((read = getline(&line, &len, fp)) == -1){
            printf("erro\n"); break;
        }

        // numero de bias
        int num_bias = atoi(line);
        network->layers[layer_cnt].bias = (int*) malloc(sizeof(int) * num_bias);


        // bias line
        if ((read = getline(&line, &len, fp)) == -1){
            printf("erro\n"); break;
        }

        // bias
        ch = strtok(line, delimiter);
        network->layers[layer_cnt].bias[0] = atoi(ch);
        for(int i = 1; i < num_bias; i++){
            ch = strtok(NULL, delimiter);
            network->layers[layer_cnt].bias[i] = atoi(ch);
        }


        if (layer_cnt == 0 || layer_cnt == 2 || layer_cnt == 4){
            network->layers[layer_cnt].type = 0;
            network->layers[layer_cnt].stride = 1;
            network->layers[layer_cnt].padding = 1;

            layer_cnt++; // pool layer
            network->layers[layer_cnt].type = 2;
            network->layers[layer_cnt].W = 2;
            network->layers[layer_cnt].H = 2;
            network->layers[layer_cnt].C = -1;
            network->layers[layer_cnt].M = -1;

            network->layers[layer_cnt].stride = 2;
            network->layers[layer_cnt].padding = 0;
        }
        else if (layer_cnt == 6){
            network->layers[layer_cnt].type = 0;
            network->layers[layer_cnt].stride = 1;
            network->layers[layer_cnt].padding = 0; // no padding

            layer_cnt++; // flatten
            network->layers[layer_cnt].type = 3;
            network->layers[layer_cnt].C = -1;
            network->layers[layer_cnt].M = -1;
            network->layers[layer_cnt].W = -1;
            network->layers[layer_cnt].H = -1;
            network->layers[layer_cnt].stride = 0;
            network->layers[layer_cnt].padding = 0;
        } else {
            network->layers[layer_cnt].type = 1;
        }
        layer_cnt++;
    }

}


int get_pixel (Image im, int x, int y, int c){

    if(x >= im.w) return 0;
    if(y >= im.h) return 0;
    if(x < 0) return 0;
    if(y < 0) return 0;

    assert(c >= 0);
    assert(c < im.c);

    int o_index = (x*im.h*im.c) + (y*im.c) + c;
    return im.data[o_index];
}

void set_pixel (Image im, int x, int y, int c, int val){
    assert(c >= 0);
    assert(c < im.c);

    if(x >= 0 && x < im.w && y >= 0 && y < im.h){
        int o_index = (x*im.h*im.c) + (y*im.c) + c;
        im.data[o_index] = val;
    } else {
        printf("index out of range\n");
    }
}


/// ==================================== OPERATIONS
Image forward_conv (Layer_t l, Image input){
    Image out;
    out.c = l.M;    // # filters

    if (l.padding == 1){ // same
        out.w = input.w; // padding
        out.h = input.h; // padding
    } else if (l.padding == 0){ // no padding
        out.w = input.w-2; // padding (-2 porque todos os filtros tem tamanho 3)
        out.h = input.h-2; // padding (-2 porque todos os filtros tem tamanho 3)
    } else {
        printf("config padd desconhecida.\n");
    }

    out.data = (int *) calloc ((out.c*out.h*out.w), sizeof(int));

    int w_index = 0;

    for(int m = 0; m < out.c; m++){                     // # filters (or output channels)
        for (int x = 0; x < out.w; x++){                // colunas Ofmap
            for (int y = 0; y < out.h; y++){            // linhas Ofmap
                int conv_o = 0;
                for(int h = 0; h < l.H; h++){           // linhas kernel
                    for(int w = 0; w < l.W; w++){       // colunas kernel
                        for(int k = 0; k < l.C; k++){   // canais kernel (or input chanel)
                            w_index = (m*l.H*l.W*l.C)+(h*l.W*l.C)+(w*l.C)+ k;
                            // printf("%d * %d;", l.weights[w_index],get_pixel(input, x+w-1, y+h-1, k));
                            int ix = x+w-l.padding;
                            int iy = y+h-l.padding;
                            conv_o += get_pixel(input, x+w-l.padding, y+h-l.padding, k) * l.weights[w_index];
                        }
                    }
                }
                int o = conv_o + l.bias[m]; // add bias
                o = (o < 0) ? 0 : o;        // RELU
                set_pixel(out, x, y, m, (char)o);
            }
        }
    }

    printf("conv\n");
    return out;
}

Image forward_pool (Layer_t l, Image input){
    // calcular dimensões da imagem de saída
    Image out;
    out.c = input.c;    // # filters
    out.w = input.w / 2;
    out.h = input.h / 2;
    out.data = (int *) malloc (sizeof(int) * (out.c*out.h*out.w));

    int o_index = 0;
    int w_index = 0;
    int i_index = 0;

    int max_val=0;

    for (int x = 0; x < out.h; x++){            // linhas Ofmap
        for (int y = 0; y < out.w; y++){        // colunas Ofmap
            for(int k = 0; k < out.c; k++){       // canais
                o_index = (x*out.w*out.c) + (y*out.c) + k;
                i_index = (x*out.w*out.c*l.stride) + (y*out.c*l.stride) + k;
                max_val = input.data[i_index];

                i_index = (x+1*out.w*out.c) + (y*out.c) + k;
                max_val = max_val < input.data[i_index] ? input.data[i_index] : max_val;

                i_index = (x*out.w*out.c) + (y+1*out.c) + k;
                max_val = max_val < input.data[i_index] ? input.data[i_index] : max_val;

                i_index = (x+1*out.w*out.c) + (y*out.c) + k;
                max_val = max_val < input.data[i_index] ? input.data[i_index] : max_val;

                out.data[o_index] = max_val;
            }
        }
    }

    printf("pool\n");

    return out;

}

int * forward_flatten (Layer_t l, Image input_image){
    printf("flatten ");
    printf("%d %d %d = %d\n", input_image.w,input_image.h, input_image.c, (input_image.w*input_image.h*input_image.c));
    return input_image.data;
}

int * forward_fc (Layer_t l, int * input_vector){
    int * out;
    out = (int *) calloc (l.M, sizeof(int));

    for (int m = 0; m < l.M; m++){
        for (int w = 0; w < l.W; w++){
            out[m] += input_vector[w] * l.weights[w];
        }
        out[m] += l.bias[m];

        out[m] = (int)(char) out[m];
    }

    printf("fc\n");
    return out;
}

float * softmax(int* input, int size) {
	int i;
	double m, sum, constant;
	float * result = (float*) calloc (size, sizeof(float));


	m = input[0];
	for (i = 0; i < size; ++i) {
		if (m < input[i]) {
			m = input[i];
		}
	}

	sum = 0.0;
	for (i = 0; i < size; ++i) {
		sum += exp((double)input[i] - m);
	}

	constant = m + log(sum);
	for (i = 0; i < size; ++i) {
		result[i] = exp(((double)input[i]) - constant);
	}
	return result;
}



int * forward_propagation (Network net, Image input_tensor){
    int * output_tensor;

    Image fmap = input_tensor;
    int * out_flatten;
    int * out_fc;

    for (int i = 0; i < net.num_layers; i++){
        Layer_t l = net.layers[i];

        if (l.type == 0){
            fmap = forward_conv(l, fmap);
        } else if (l.type == 1){
            out_fc = forward_fc(l, out_flatten);
            return out_fc;
            // output_tensor = softmax(out_fc, l.M);
        } else if (l.type == 2){
            fmap = forward_pool(l, fmap);
        } else if (l.type == 3){
            out_flatten = forward_flatten(l, fmap);
        }
    }

    return output_tensor;
}


int main()
{
    Network net;
    net.num_layers = LAYERS;
    // net.layers = (Layer_t*) malloc(sizeof(Layer_t) * LAYERS);

    // leitura dos pesos
    weight_reader("../pesos_1.txt", &net);
    for (int i = 0 ; i < LAYERS; i++){
        printf("type %d \t shape %d %d %d %d\n", net.layers[i].type, net.layers[i].M, net.layers[i].C, net.layers[i].H, net.layers[i].W);
    }

    Image input_image;
    unsigned char * stb_image = stbi_load("A_21472.png", &input_image.w, &input_image.h, &input_image.c, 3);
    input_image.c = 3;
    if(input_image.w != 24 || input_image.h != 32){
        printf("Incorrect image size!");
        return 1;
    }

    // aloca memória para converter imagem para inteiro
    input_image.data = (int *) calloc (input_image.w * input_image.h * input_image.c, sizeof(int));

    // converte para int32
    for (int x = 0; x < input_image.w; x++){
        for (int y = 0; y < input_image.h; y++){
            for (int c = 0; c < input_image.c; c++){
                set_pixel(input_image, x, y, c, (int) stb_image[(x*input_image.h*input_image.c)+(y*input_image.c)+c]);
                // set_pixel(input_image, x, y, c, 255);
                // stb_image[(x*input_image.h*input_image.c)+(y*input_image.c)+c] = (unsigned char) get_pixel(input_image, x, y, c);
            }
        }
    }
    // stbi_write_jpg("copia.jpg", input_image.w, input_image.h, input_image.c, stb_image, 100);


    float * result;
    result = forward_propagation(net, input_image);

    /*
    printf("[");
    for (int x = 0; x < input_image.w; x++){
        printf("[");
        for (int y = 0; y < input_image.h; y++){
            printf("[");
            for (int c = 0; c < 6; c++){
                printf("%d ", result[(x*input_image.h*input_image.c)+(y*input_image.c)+c]);
            }
            printf("]\n");
        }
        printf("]\n");
    }
    printf("]\n");
    */

    for (int i = 0; i < 35; i++){
        printf("o[%d] = %f\n", i, result[i]);
    }

    // libera imagem
    free(input_image.data);
    free(stb_image);
    return 0;
}
