#include <stdio.h>
#include <stdlib.h>

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
            network->layers[layer_cnt].padding = 0;

            layer_cnt++; // pool layer
            network->layers[layer_cnt].type = 2;
            network->layers[layer_cnt].W = 2;
            network->layers[layer_cnt].H = 2;
            network->layers[layer_cnt].C = -1;
            network->layers[layer_cnt].M = -1;

            network->layers[layer_cnt].stride = 2;
        }
        else if (layer_cnt == 6){
            network->layers[layer_cnt].type = 0;
            network->layers[layer_cnt].stride = 1;
            network->layers[layer_cnt].padding = 0;

            layer_cnt++; // flatten
            network->layers[layer_cnt].type = 3;
            network->layers[layer_cnt].C = -1;
            network->layers[layer_cnt].M = -1;
            network->layers[layer_cnt].W = -1;
            network->layers[layer_cnt].H = -1;
        } else {
            network->layers[layer_cnt].type = 1;
        }
        layer_cnt++;
    }

}



/// ==================================== OPERATIONS
Image forward_conv (Layer_t l, Image input){
    l.input = input.data;

    // calcular dimensões da imagem de saída
    Image out;
    out.c = l.M;    // # filters
    out.w = input.w;
    out.h = input.h;
    out.data = (int *) malloc (sizeof(int) * (l.M*input.h*input.w));

    for(int m = 0; m < out.c; m++){                   // # filters (or output channels)
        for (int x = 0; x < out.h; x++){            // linhas Ofmap
            for (int y = 0; y < out.w; y++){        // colunas Ofmap
                for(int h = 0; h < l.H; h++){       // linhas
                    for(int w = 0; w < l.W; w++){   // colunas
                        for(int k = 0; k < l.C; k++){
                            out.data[(m*out.h*out.w) + (x*out.w) + y] += l.weights[(m*l.H*l.W*l.C)+(h*l.W*l.C)+(w*l.C)+ k] * input.data[((x+h)*l.W*l.C)+((y+w)*l.C)+ k];
                        }
                    }
                }
                int o = out.data[(m*out.h*out.w) + (x*out.w) + y] + l.bias[m]; // add bias
                out.data[(m*out.h*out.w) + (x*out.w) + y] = (o < 0) ? 0 : o; // RELU
            }
        }
    }

    printf("conv\n");
    return out;
}

Image forward_pool (Layer_t l, Image input_image){
    printf("pool\n");
}

int * forward_flatten (Layer_t l, Image input_image){
    printf("flatten\n");
}

int * forward_fc (Layer_t l, int * input_image){
    printf("fc\n");
}

int * forward_propagation (Network net, Image input_tensor){
    int * output_tensor;

    Image fmap = input_tensor;
    int * out_flatten;

    for (int i = 0; i < net.num_layers; i++){
        Layer_t l = net.layers[i];

        if (l.type == 0){
            fmap = forward_conv(l, fmap);
        } else if (l.type == 1){
            output_tensor = forward_fc(l, out_flatten);
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
    input_image.data = (int *) malloc (sizeof(int) * (input_image.w * input_image.h * input_image.c));

    // converte para int32
    for (int i = 0; i < (input_image.w * input_image.h * input_image.c); i++)
        input_image.data[i] = (int) stb_image[i];


    int * result;
    result = forward_propagation(net, input_image);


    // libera imagem
    free(input_image.data);
    return 0;
}
