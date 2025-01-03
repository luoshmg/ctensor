#include "ctensor.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"              // load image


#define FILE_OPEN_ERR "Error openning file"
#define MAX_LINE_LENGTH 128         // NOTE：the length of image path (e.g. "/mnt/d/data/MNIST/test/9/01008.png") must be less than MAX_LINE_LENGTH

#define IMAGE_SIZE 784      // 28 * 28 * 1
#define NUM_CLASSES 10      // 10 classes


// Tensor *open_image(const char *imagepath);
void randomly_load_data(char (*imagepaths)[MAX_LINE_LENGTH], int *labels, int num_lines, Tensor *inputs, Tensor *targets);
void sequentially_load_data(int start_line, char (*imagepaths)[MAX_LINE_LENGTH], int *labels, int num_lines, Tensor *inputs, Tensor *targets);

int line_count(const char *filename);
int read_imagepaths_and_labels(const char *dataset_filename, char (*imagepaths)[MAX_LINE_LENGTH], int *labels);

void fb_once(int need_backward, dtype lr, Tensor *inputs, Tensor *targets, Tensor *W1, Tensor *b1, Tensor *W2, Tensor *b2, dtype *loss_return, int *correct);


// MLP forward, (backward and gradient descend if need_backward)
// return loss and correct via loss_return and correct
void fb_once(int need_backward, dtype lr, Tensor *inputs, Tensor *targets, Tensor *W1, Tensor *b1, Tensor *W2, Tensor *b2, dtype *loss_return, int *correct) {
    // 1. forward
    Tensor *x1 = matmultiply(inputs, W1);
    Tensor *x2 = matadd(x1, b1);
    Tensor *x3 = batch_norm(x2, 0);
    Tensor *x4 = relu(x2);
    Tensor *x5 = matmultiply(x4, W2);
    Tensor *x6 = matadd(x5, b2);
    Tensor *x7 = softmaxloss(x6, targets);
    Tensor *loss = mean_all(x7);

    Tensor *inter_vars[] = {x1, x2, x3, x4, x5, x6, x7, loss};
    int num_inter_vars =  sizeof(inter_vars) / sizeof(inter_vars[0]);

    // 2. backward
    if (need_backward) {
        set_grad_to_1s(loss);       // take the derivative with respect to loss itself (dloss/dloss = 1). unless this was done in the loss function, but if so, step 4 can not go before calling backward().

        // TODO: 按照 generation_idx 排序 num_inter_vars, 再按相反的顺序 backward
        for (int i = num_inter_vars - 1; i >= 0; i--) {
            inter_vars[i]->backward(inter_vars[i]);
        }

        // 3. gradient descend
        gradient_descend(W1, lr);
        gradient_descend(b1, lr);
        gradient_descend(W2, lr);
        gradient_descend(b2, lr);

        // 4. return the grad of parameters to 0, because of the "+=" mechanism in _backward
        set_grad_to_0s(W1);
        set_grad_to_0s(b1);
        set_grad_to_0s(W2);
        set_grad_to_0s(b2);
    }

    if (correct != NULL) *correct = argmax_match_count(x6, targets);
    if (loss_return != NULL) *loss_return = loss->data->values[0];

    // free intermediate variables
    for (int i = num_inter_vars - 1; i >= 0; i--) {
        free_tensor(inter_vars[i]);
    }
}


void sequentially_load_data(int start_line, char (*imagepaths)[MAX_LINE_LENGTH], int *labels, int num_lines, Tensor *inputs, Tensor *targets) {
    assert(inputs->data->ndim == 2 && targets->data->ndim == 2);
    int batch_size = inputs->data->shape[0];
    assert(batch_size == targets->data->shape[0]);      // batch_size
    assert(start_line + batch_size <= num_lines);

    set_data_to_0s(targets);
    
    int width, height, channels;
    for (int b_idx = 0; b_idx < batch_size; b_idx++) {
        // inputs
        unsigned char *image_data = stbi_load(imagepaths[start_line + b_idx], &width, &height, &channels, 0);
        if (!image_data) {
            printf("Failed to load image: %s\n", imagepaths[start_line + b_idx]);
            exit(1);
        }
        assert(width * height * channels == inputs->data->shape[1]);

        for (int i = 0; i < width * height * channels; i++) {
            inputs->data->values[b_idx * inputs->data->shape[1] + i] = image_data[i] / 255.0;       // NOTE: 这里不能 memcpy! 因为数据类型不一样！
        }
        stbi_image_free(image_data);    // Do not forget to free image_data

        // targets
        targets->data->values[b_idx * NUM_CLASSES + labels[start_line + b_idx]] = 1;
    }
}

// Randomly load images and labels, fill in inputs and targets
void randomly_load_data(char (*imagepaths)[MAX_LINE_LENGTH], int *labels, int num_lines, Tensor *inputs, Tensor *targets) {
    assert(inputs->data->ndim == 2 && targets->data->ndim == 2);
    int batch_size = inputs->data->shape[0];
    assert(batch_size == targets->data->shape[0]);      // batch_size

    set_data_to_0s(targets);

    int width, height, channels;
    for (int b_idx = 0; b_idx < batch_size; b_idx++) {
        int rand_num = rand() % num_lines;

        // inputs
        unsigned char *image_data = stbi_load(imagepaths[rand_num], &width, &height, &channels, 0);
        if (!image_data) {
            printf("Failed to load image: %s\n", imagepaths[rand_num]);
            exit(1);
        }
        assert(width * height * channels == inputs->data->shape[1]);

        for (int i = 0; i < inputs->data->shape[1]; i++) {
            inputs->data->values[b_idx * inputs->data->shape[1] + i] = image_data[i] / 255.0;       // NOTE: 这里不能 memcpy! 因为数据类型不一样！
        }
        stbi_image_free(image_data);    // Do not forget to free image_data

        // targets
        targets->data->values[b_idx * targets->data->shape[1] + labels[rand_num]] = 1;
    }
}


// count lines of file
int line_count(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror(FILE_OPEN_ERR);
        exit(1);
    }

    char line[MAX_LINE_LENGTH];
    int count = 0;
    while (fgets(line, MAX_LINE_LENGTH, fp) != NULL) {
        count++;
    }
    fclose(fp);
    return count;
}

// Read imagepaths and labels from dataset_filename, save to imagepaths and labels, and return number of lines read.
int read_imagepaths_and_labels(const char *dataset_filename, char (*imagepaths)[MAX_LINE_LENGTH], int *labels) {
    FILE *fp = fopen(dataset_filename, "r");
    if (!fp) {
        perror(FILE_OPEN_ERR);
        exit(1);
    }

    char buffer[MAX_LINE_LENGTH];
    int i = 0;
    while(fgets(buffer, MAX_LINE_LENGTH, fp)) {
        sscanf(buffer, "%s %d\n", &imagepaths[i][0], &labels[i]);       // 注意格式要匹配
        i++;
    }
    fclose(fp);
    return i;
}
