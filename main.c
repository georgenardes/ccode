#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "cnn_lib.h"



int main___(){
Image input_image;

    unsigned char * stb_image = stbi_load("Y_74978.png", &input_image.w, &input_image.h, &input_image.c, 3);
    input_image.c = 3;
    if(input_image.w != 24 || input_image.h != 32){
        printf("Incorrect image size!");
        return 1;
    }

    // aloca memória para converter imagem para inteiro
    input_image.data = (int *) calloc (input_image.w * input_image.h * input_image.c, sizeof(int));

    // converte para int32
    printf("[");
    for (int y = 0; y < input_image.h; y++){
        printf("[");
            for (int x = 0; x < input_image.w; x++){
            printf("[");
            for (int c = 0; c < input_image.c; c++){
                int index = (y*input_image.w*input_image.c) + (x*input_image.c) + c;
                set_pixel(input_image, x, y, c, (int) stb_image[index]);
                printf("%d ", (int)stb_image[index]);
            }
            printf("]\n");
        }
        printf("]\n");
    }
}


/*
Main principal
*/
int main()
{
    Network net;
    net.num_layers = LAYERS;

    // leitura dos pesos
    weight_reader("../pesos_com_scale.txt", &net);
    for (int i = 0 ; i < LAYERS; i++){
        printf("type %d \t shape %d %d %d %d\n", net.layers[i].type, net.layers[i].M, net.layers[i].C, net.layers[i].H, net.layers[i].W);
    }

    // carrega imagem
    Image input_image = load_image("Y_74978.png");

    /**
    int * result;
    result = forward_propagation(net, input_image);
    for (int i = 0; i < 35; i++)
    {
        printf("%d \n", result[i]);

    }
    */

    Image im_result = forward_conv(net.layers[0], input_image);
    int * result;
    result = im_result.data;
    printf("[");
    for (int y = 0; y < input_image.h; y++){
        printf("[");
        for (int x = 0; x < input_image.w; x++){
            printf("[");
            for (int c = 0; c < 6; c++){
                int index = (y*input_image.w*input_image.c) + (x*input_image.c) + c;
                printf("%d ", result[index]);
            }
            printf("]\n");
        }
        printf("]\n");
    }
    printf("]\n");


    // libera imagem
    free(input_image.data);
    return 0;
}

/*

MAIN TESTE PESOS FIXOS

    0b00000000 => 0
    0b10000000 => -128
    0b01111111 => 127
    0b11111111 => -127
*/
int main_(){
    // int input [] = {1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0};
    // int weights[] = {1,2,1,0,0,0,-1,-2,-1};
    int input [] = {-1,-1,-1,-128,-128,-128,-128,-128,-128,127,127,-1,-128,-128,-128,-128,-128,-128};
    int weights[] = {64,127,64,0,0,0,-64,-127,-64};
    int bias[] = {111};

    Layer_t l;
    l.M = 1;
    l.C = 1;
    l.W = 3;
    l.H = 3;
    l.stride = 1;
    l.padding = 1;
    l.type = 0;
    l.bias = bias;
    l.weights = weights;
    l.input_scale = 0.250980406999588;
    l.input_zero = -128;
    float weight_scale[] = {0.015748031437397003};
    l.weight_scale = weight_scale;
    l.output_scale = 0.8735899925231934;
    l.output_zero = -128;
    float scale[] = {l.input_scale * l.weight_scale[0] / l.output_scale};
    l.scale = scale;

    Image im;
    im.h = 6;
    im.w = 3;
    im.c = 1;
    im.data = input;

    printf("Imagem de entrada\n");
    for(int h = 0; h < im.h; h++){          // linhas im
        for(int w = 0; w < im.w; w++){      // colunas im
            for(int k = 0; k < im.c; k++){  // canais im (or input chanel)
                printf("%d ", get_pixel(im, w, h, k));
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("====================================\n");

    // executa rede
    Image out = forward_conv(l, im);

    printf("%d ", out.h);
    printf("%d ", out.w);
    printf("%d\n", out.c);

    for(int h = 0; h < out.h; h++){          // linhas out
        for(int w = 0; w < out.w; w++){      // colunas out
            for(int k = 0; k < out.c; k++){  // canais out (or input chanel)
                printf("%d ", get_pixel(out, w, h, k));
            }
            printf("\n");
        }
        printf("\n");
    }
}


