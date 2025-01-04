// TODO:  # 20250102
// 3. parallel computing for matrix, e.g. OpenMP
// 4. better saver and loader for Tensor/Arr


// ============================================================================

// write a simple C tensor library  # 20241224

// References:
// [1] https://github.com/apoorvnandan/tensor.h
// [2] 《深度学习入门：自制框架》
// [3] 《深度学习入门：基于 Python 的理论与实现》

// ============================================================================


#include <stdio.h>
#include <stdlib.h>
#include <string.h>     // memcmp
#include <math.h>       // exp, log
#include <assert.h>     // assert

#define MEM_ALLOC_ERR "Menmory allocation failed"


// ============================================================================
// generating operators
// ============================================================================

#define MAX_GENNERATING_OPERANDS 2     // 最多二元运算，不使用更多元的运算

#define NO_GENERATING_OPERATOR -1
#define ADD 0
#define MINUS 1
#define MATMULTIPLY 2

#define MATNEGATE 3
#define MATTRANSPOSE 4
#define RELU 5
#define BATCH_NORM 6
#define SOFTMAXLOSS 7       // softmax 运算没有 backward, 不需要 id

#define MAX 8
#define SUM 9
#define MEAN 10
#define STD_DEVIATION 11
#define MAX_ALL 12
#define SUM_ALL 13
#define MEAN_ALL 14


// ============================================================================
// Tensor data type
// ============================================================================
// NOTE: 不能只用 Tensor 不用 Arr, 否则不能指定该使用 Tensor 的 data 还是 grad 来做计算！  # 20241230 21:40
// NOTE: 什么是自动反向传播？ --自动反向传播指的是，从inputs到loss之间的每个中间变量 x_i 包括loss，但不包括inputs）都要求导，但是我们希望求导的方式是一致的，都是 x_i->backward(x_i), 不用指定张量运算的类型. 这是可以做到的，只要有 generation_idx 来帮助指定求导的顺序就行  # 20250103 20:00

typedef float dtype;

typedef struct Arr{
    dtype *values;  // 多维数组本质上还是一维数组！
    int *shape;     // e.g. {2, 3, 4, 5}
    int ndim;       // length of shape, e.g. 4

    int size;       // e.g. 2*3*4*5
    int *strides;   // e.g. {3*4*5, 4*5, 5, 1}
} Arr;

typedef struct Tensor {
    Arr *data;

    Arr *grad;
    int need_grad;
    int the_generating_operator;   // operator used to generate this tensor
    struct Tensor *generating_operands[MAX_GENNERATING_OPERANDS];    // tensors used to generate this tensor
    int num_generating_operands;

    void (* backward)(struct Tensor *);
    int generation_idx;         // 自动反向传播的顺序不是随意的，需要根据张量的 generation_idx 来定, generation_idx 表示这个张量是第几代张量，定义为通过运算生成这个张量的所有张量的最大 generation_idx 再加 1
} Tensor;



// ============================================================================
// Tensor creators
// ============================================================================

Arr *create_arr_zeros(int *shape, int ndim);
Arr *create_arr_row(int size);
Arr *create_arr_col(int size);
Arr *create_arr_(dtype *values, int *shape, int ndim);
Arr *create_arr(Arr *arr);
void free_arr(Arr *arr);

Tensor *create_tensor_zeros(int *shape, int ndim);
Tensor *create_tensor_onehot(int size, int hot);
Tensor *create_tensor_row_(int size);
Tensor *create_tensor_row(Arr *arr);
Tensor *create_tensor_col_(int size);
Tensor *create_tensor_col(Arr *arr);
Tensor *create_tensor_(dtype *values, int *shape, int ndim);
Tensor *create_tensor(Arr *arr);
void free_tensor(Tensor *tensor);

void set_grad_to_1s(Tensor *tensor);
void set_grad_to_0s(Tensor *tensor);
void set_data_to_0s(Tensor *tensor);

// ============================================================================
// Arr operators and Tensor operators
// ============================================================================

// NOTE: 反向传播修改 grad 时，一律使用增量式 +=, 而不是直接赋值 =. 否则会错，例如在 y = x + x 的情况下！

// binary operators

Arr *matmultiply_(Arr *a, Arr *b);          // 使用这种函数需要分配两次内存，会慢的吧？有必要对 Arr 进行二元运算吗？ --有必要！否则反向传播会很复杂 # 20241226 15:30
Tensor *matmultiply(Tensor *a, Tensor *b);
void matmultiply_backward(Tensor *out); 

Arr *matadd_(Arr *a, Arr *b);
Tensor *matadd(Tensor *a, Tensor *b);
void matadd_backward(Tensor *out);

Arr *matminus_(Arr *a, Arr *b);
Tensor *matminus(Tensor *a, Tensor *b);
void matminus_backward(Tensor *out);

// unary operators

Arr *matnegate_(Arr *a);
Tensor *matnegate(Tensor *a);
void matnegate_backward(Tensor *out);

Arr *mattranspose_(Arr *a);
Tensor *mattranspose(Tensor *a);
void mattranspose_backward(Tensor *out);

Arr *relu_(Arr *a);
Tensor *relu(Tensor *a);
void relu_backward(Tensor *out);

Arr *softmax_(Arr *y);
Tensor *softmax(Tensor *y);
// void softmax_backward(Tensor *out);

Arr *softmaxloss_(Arr *y, Arr *label);
Tensor *softmaxloss(Tensor *y, Tensor *label);
void softmaxloss_backward(Tensor *out);

Arr *argmax_(Arr *a, int axis);
// Tensor *argmax_(Tensor *a, int axis);
// void argmax_backward(Tensor *out);

Arr *max_(Arr *a, int axis);
Tensor *max(Tensor *a, int axis);
void max_backward(Tensor *out);

Arr *sum_(Arr *a, int axis);
Tensor *sum(Tensor *a, int axis);
void sum_backward(Tensor *out);

Arr *mean_(Arr *a, int axis);
Tensor *mean(Tensor *a, int axis);
void mean_backward(Tensor *out);

Arr *std_deviation_(Arr *a, int axis);
Tensor *std_deviation(Tensor *a, int axis);
// void std_deviation_backward(Tensor *out);

// NOTE: 没有归一化是不行的，矩阵会爆炸     # 20241230 17:00
Arr *batch_norm_(Arr *a, int axis);
Tensor *batch_norm(Tensor *a, int axis);
void batch_norm_backward(Tensor *a);

Arr *argmax_all_(Arr *a);
// Tensor *argmax_all(Tensor *a);
// void argmax_all_backward(Tensor *out);

Arr *max_all_(Arr *a);
Tensor *max_all(Tensor *a);
void max_all_backward(Tensor *out);

Tensor *sum_all(Tensor *a);
Arr *sum_all_(Arr *a);
void sum_all_backward(Tensor *out);

Tensor *mean_all(Tensor *a);
Arr *mean_all_(Arr *a);
void mean_all_backward(Tensor *out);

// Arr *std_deviation_all_(Arr *a);



// ============================================================================
// other
// ============================================================================

void gradient_descend(Tensor *tensor, dtype lr);

void fprint_arr(Arr *a, FILE *stream);
void print_arr(Arr *a);
void fprint_arr_shape(Arr *a, FILE *stream);
void print_arr_shape(Arr *a);
void sava_arr(const char *filename, Arr *a);

double box_muller(double mu, double sigma);
void random_init(Arr *a, double mu, double sigma);

int argmax_match_count(Tensor *ys, Tensor *targets);

int compare_tensor(const void *a, const void *b);
