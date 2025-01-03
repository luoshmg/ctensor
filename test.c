/*
gcc -o bin/test test.c ctensor.c -lm
./bin/test
*/

// NOTE: 需要先把 model.txt 中的 '[' 和 ']' 去掉才能正确加载参数  # 20241231


#include <time.h>
#include "train_and_test.h"


int main() {
    srand(time(NULL));


    int shape_W1[] = {IMAGE_SIZE, 80};
    int shape_b1[] = {1, 80};
    Tensor *W1 = create_tensor_zeros(shape_W1, 2);
    Tensor *b1 = create_tensor_zeros(shape_b1, 2);

    int shape_W2[] = {80, NUM_CLASSES};
    int shape_b2[] = {1, NUM_CLASSES};
    Tensor *W2 = create_tensor_zeros(shape_W2, 2);
    Tensor *b2 = create_tensor_zeros(shape_b2, 2);


    // load parameters
    const char *save_model_file = "models/model.txt";
    FILE *fp = fopen(save_model_file, "r");
    if (!fp) {
        perror(FILE_OPEN_ERR);
        exit(1);
    }
    printf("\n");
    int steps[] = {W1->data->size, 
                   W1->data->size + b1->data->size, 
                   W1->data->size + b1->data->size + W2->data->size, 
                   W1->data->size + b1->data->size + W2->data->size + b2->data->size};
    int j;
    for (int i = 0; i < steps[3]; i++) {
        if (0 <= i && i < steps[0]) {
            j = i;
            fscanf(fp, "%f", &W1->data->values[j]);
        } else if (steps[0] <= i && i < steps[1]) {
            j = i - steps[0];
            fscanf(fp, "%f", &b1->data->values[j]);
        } else if (steps[1] <= i && i < steps[2]) {
            j = i - steps[1];
            fscanf(fp, "%f", &W2->data->values[j]);
        } else if (steps[2] <= i && i < steps[3]) {
            j = i - steps[2];
            fscanf(fp, "%f", &b2->data->values[j]);
        }
    }
    fclose(fp);
    printf("done loading parameters from %s\n", save_model_file);

    printf("b2->data:\n");
    print_arr(b2->data);


    // ========================================================
    // test  # 20241231 10:00
    // ========================================================
    
    const int batch_size = 10;

    int shape_inputs[] = {batch_size, IMAGE_SIZE};
    Tensor *inputs = create_tensor_zeros(shape_inputs, 2);      // (batch_size, IMAGE_SIZE)
    inputs->need_grad = 0;
    int shape_targets[] = {batch_size, NUM_CLASSES};
    Tensor *targets = create_tensor_zeros(shape_targets, 2);    // (batch_size, NUM_CLASSES)
    targets->need_grad = 0;

    printf("===== test =====\n");
    const char *test_set_path = "/mnt/d/data/MNIST/test_set.txt";
    const int test_num_lines = line_count(test_set_path);       // test_set: 60184
    char (*test_imagepaths)[MAX_LINE_LENGTH] = (char (*) [MAX_LINE_LENGTH]) calloc(test_num_lines, MAX_LINE_LENGTH);
    if (!test_imagepaths) {
        perror(MEM_ALLOC_ERR);
        exit(1);
    }
    int test_labels[test_num_lines];
    read_imagepaths_and_labels(test_set_path, test_imagepaths, test_labels);
    int idx = test_num_lines-1;
    printf("read %s done\n", test_set_path);
    printf("test_imagepaths[%d]: %s, test_labels[%d]: %d\n", idx, test_imagepaths[idx], idx, test_labels[idx]);

    int num_batches = test_num_lines / batch_size;
    int batch;
    int total_correct = 0;
    int correct = 0;
    int start_line;
    for (batch = 0; batch < num_batches; batch++) {
        // minibatch
        start_line = batch * batch_size;
        randomly_load_data(test_imagepaths, test_labels, test_num_lines, inputs, targets);

        // forward
        fb_once(0, 0, inputs, targets, W1, b1, W2, b2, NULL, &correct);
        total_correct += correct;

        break;
    }
    free(test_imagepaths);
    printf("test %d, correct %d, acc %lf\n", (batch + 1) * batch_size, total_correct, total_correct / ((float) (batch + 1) * batch_size));


    // ========================================================



    free_tensor(inputs);
    free_tensor(targets);

    free_tensor(W1);
    free_tensor(b1);
    free_tensor(W2);
    free_tensor(b2);
    

    // const char *test_save = "test_save.txt";

    return 0;
}
