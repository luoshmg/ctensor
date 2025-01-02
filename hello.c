/*
gcc -o hello hello.c ctensor.c -lm
./hello
*/


#include <time.h>
#include "train_and_test.h"


int main() {
    srand(time(NULL));
    
    // 只使用二维数据，暂不需要考虑更高维数组。或许使用卷积的时候才需要
    const int ndim_mat = 2;

    // hyper parameters
    dtype lr = 0.005;                   // 0.001 太小了  # 20250102 10:00
    int total_epoches = 5;
    const int batch_size = 200;         // 注意: batch_size 不能小于3, 否则 batch normalization 没有意义

    const int N_layer1 = 80;

    // parameters
    int shape_W1[] = {IMAGE_SIZE, N_layer1};
    int shape_b1[] = {1, N_layer1};
    Tensor *W1 = create_tensor_zeros(shape_W1, ndim_mat);
    Tensor *b1 = create_tensor_zeros(shape_b1, ndim_mat);

    int shape_W2[] = {N_layer1, NUM_CLASSES};
    int shape_b2[] = {1, NUM_CLASSES};
    Tensor *W2 = create_tensor_zeros(shape_W2, ndim_mat);
    Tensor *b2 = create_tensor_zeros(shape_b2, ndim_mat);

    // random initialization
    // NOTE: 不要用“标准正态分布”初始化参数，否则 loss 很容易爆炸. 这里使用“He 初始值”  # 20250102
    random_init(W1->data, 0, 2.0 / sqrt(IMAGE_SIZE));
    random_init(b1->data, 0, 2.0 / sqrt(IMAGE_SIZE));
    random_init(W2->data, 0, 2.0 / sqrt(N_layer1));
    random_init(b2->data, 0, 2.0 / sqrt(N_layer1));
    printf("Randomly intialize parameters done\n");
    // printf("b2->data:\n");
    // print_arr(b2->data);


    // ========================================================
    // train
    // ========================================================
    printf("===== train =====\n");

    // data
    const char *train_set_path = "/mnt/d/data/MNIST/train_set.txt";
    const int num_lines = line_count(train_set_path);       // train_set: 60184
    char (*imagepaths)[MAX_LINE_LENGTH] = (char (*) [MAX_LINE_LENGTH]) calloc(num_lines, MAX_LINE_LENGTH);
    if (!imagepaths) {
        perror(MEM_ALLOC_ERR);
        exit(1);
    }
    int labels[num_lines];
    read_imagepaths_and_labels(train_set_path, imagepaths, labels);
    printf("read %s done\n", train_set_path);
    int idx = num_lines-1;
    printf("imagepaths[%d]: %s, labels[%d]: %d\n", idx, imagepaths[idx], idx, labels[idx]);

    int need_backward = 1;
    int shape_inputs[] = {batch_size, IMAGE_SIZE};
    Tensor *inputs = create_tensor_zeros(shape_inputs, ndim_mat);      // (batch_size, IMAGE_SIZE)
    inputs->need_grad = 0;
    int shape_targets[] = {batch_size, NUM_CLASSES};
    Tensor *targets = create_tensor_zeros(shape_targets, ndim_mat);    // (batch_size, NUM_CLASSES)
    targets->need_grad = 0;

    int num_batches = num_lines / batch_size;
    time_t total_start_time = time(NULL);
    for (int epoch = 0; epoch < total_epoches; epoch++) {
        time_t start_time, end_time;
        int print_times = 3;
        int correct = 0;
        dtype loss_return = 0;
        for (int batch = 0; batch < num_batches; batch++) {
            start_time = time(NULL);
            // minibatch
            randomly_load_data(imagepaths, labels, num_lines, inputs, targets);

            // forward, backward, gradient descend
            fb_once(need_backward, lr, inputs, targets, W1, b1, W2, b2, &loss_return, &correct);
            
            // print info
            end_time = time(NULL);
            printf("epoch %d/%d, batch %d/%d, loss=%f, acc=%.3f, time consumming=%lfs \n", epoch + 1, total_epoches, batch + 1, num_batches, loss_return, correct / (float) batch_size, difftime(end_time, start_time));
        }
    }
    free(imagepaths);
    time_t total_end_time = time(NULL);
    printf("total time consumming: %lfs\n", difftime(total_end_time, total_start_time));

    // ========================================================
    // test  # 20241231 10:00
    // ========================================================
    printf("===== test =====\n");
    need_backward = 0;
    const char *test_set_path = "/mnt/d/data/MNIST/test_set.txt";
    const int test_num_lines = line_count(test_set_path);       // test_set: 60184
    char (*test_imagepaths)[MAX_LINE_LENGTH] = (char (*) [MAX_LINE_LENGTH]) calloc(test_num_lines, MAX_LINE_LENGTH);
    if (!test_imagepaths) {
        perror(MEM_ALLOC_ERR);
        exit(1);
    }
    int test_labels[test_num_lines];
    read_imagepaths_and_labels(test_set_path, test_imagepaths, test_labels);
    idx = test_num_lines-1;
    printf("read %s done\n", test_set_path);
    printf("test_imagepaths[%d]: %s, test_labels[%d]: %d\n", idx, test_imagepaths[idx], idx, test_labels[idx]);

    num_batches = test_num_lines / batch_size;
    int batch;
    int total_correct = 0;
    int correct = 0;
    int start_line;
    for (batch = 0; batch < num_batches; batch++) {
        // minibatch
        start_line = batch * batch_size;
        sequentially_load_data(start_line, test_imagepaths, test_labels, test_num_lines, inputs, targets);

        // forward
        fb_once(need_backward, lr, inputs, targets, W1, b1, W2, b2, NULL, &correct);
        total_correct += correct;
    }
    free(test_imagepaths);
    printf("test %d, correct %d, acc %lf\n", batch * batch_size, total_correct, total_correct / ((float) batch * batch_size));


    // ========================================================


    free_tensor(inputs);
    free_tensor(targets);

    // save parameters
    const char *save_model_file = "model.txt";
    sava_arr(save_model_file, W1->data);
    sava_arr(save_model_file, b1->data);
    sava_arr(save_model_file, W2->data);
    sava_arr(save_model_file, b2->data);
    printf("model saved: %s\n", save_model_file);
    

    // free parameters
    free_tensor(W1);
    free_tensor(b1);
    free_tensor(W2);
    free_tensor(b2);
    return 0;
}
