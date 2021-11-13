#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include "cnn_lib.h"






float * processa_imagem(const char *weights_path, const char *img_path)
{
    Network net;
    net.num_layers = LAYERS;

    // leitura dos pesos
    weight_reader(weights_path, &net);
    printf("pesos carregados!\n");

    // carrega imagem
    Image input_image = load_image(img_path);
    printf("imagem carregada!\n");

    int * result;
    printf("iniciando processamento...\n");
    clock_t begin = clock();
    result = forward_propagation(net, input_image);
    float * out_softmax = softmax(result, 35);
    clock_t end = clock();
    double tempo_processamento = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("fim processamento!\n");
    printf("tempo de processamento: %f segundos\n", tempo_processamento);

    // libera imagem
    free(input_image.data);
    free(result);

    return out_softmax;
}

/*
Forward individual
*/
int funcao_teste(const char *weights_path, const char *img_path){
    Network net;
    net.num_layers = LAYERS;

    // leitura dos pesos
    weight_reader(weights_path, &net);
    printf("pesos carregados!\n");

    // carrega imagem
    Image input_image = load_image(img_path);
    printf("imagem carregada!\n");

    // processa individualmente cada camada
    Image im_result = forward_conv(net.layers[0], input_image);
    save_img_channels("output_files/conv1", im_result);
    im_result = forward_pool(net.layers[1], im_result);
    save_img_channels("output_files/pool1", im_result);
    im_result = forward_conv(net.layers[2], im_result);
    save_img_channels("output_files/conv2", im_result);
    im_result = forward_pool(net.layers[3], im_result);
    save_img_channels("output_files/pool2", im_result);
    im_result = forward_conv(net.layers[4], im_result);
    save_img_channels("output_files/conv3", im_result);
    im_result = forward_pool(net.layers[5], im_result);
    save_img_channels("output_files/pool3", im_result);
    im_result = forward_conv(net.layers[6], im_result);
    save_img_channels("output_files/conv4", im_result);

    // system("cls");
    printf("\n\nConv4 output:\n");

    // cria arquivo para salvar resultado contido em "im_result"
    FILE *of;
    of = fopen("output_files/Conv4_out.txt", "w");

    /*
    for (int c = 0; c < im_result.c; c++){
        fprintf(of, "%d\t", c);
    }
    fprintf(of, "\n");
    for (int c = 0; c < im_result.c; c++){
        fprintf(of, "---\t", c);
    }
    fprintf(of, "\n");
    */

    int * result;
    result = im_result.data;
    printf("[");
    for (int c = 0; c < im_result.c; c++){
        printf("[");
        for (int y = 0; y < im_result.h; y++){
            printf("[");
            for (int x = 0; x < im_result.w; x++){
                int pixel = get_pixel(im_result, x, y, c) + 128;

                fprintf(of, "%d %d %d\n", c, y, pixel);
                printf("%d ", pixel);
            }
            // fprintf(of, "\n");
            printf("]\n");
        }
        // fprintf(of, "\n");
        printf("]\n");
    }
    printf("]\n");

    fclose(of);

    // libera imagem
    free(input_image.data);
    return 0;


}


int main () {


    float * out = processa_imagem("pesos_com_scale_v3.txt", "test_images/Y_74978.png"); // "test_images/A_21472.png"
    for (int i = 0; i < 35; i++)
    {
        printf("[%f] ", out[i]);
    }


    // funcao_teste("pesos_com_scale_v3.txt", "test_images/Y_74978.png");




    return 0;
}
