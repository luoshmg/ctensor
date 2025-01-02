#include "ctensor.h"

// ----------------------------------------------------------------------------
// Arr creators
// ----------------------------------------------------------------------------

Arr *create_arr_zeros(int *shape, int ndim) {
    assert(ndim > 0);

    Arr *arr = (Arr *) malloc(sizeof(Arr));
    if (!arr) {
        perror(MEM_ALLOC_ERR);        // 一旦动态分配内存失败，就应该直接打印错误，并退出程序。  # 20241226 12:45
        exit(1);
    }

    // size and strides
    arr->strides = (int *) malloc(ndim * sizeof(int));
    if (!arr->strides) {
        perror(MEM_ALLOC_ERR);
        exit(1);
    }
    arr->size = 1;
    for (int axis = ndim - 1; axis >= 0; axis--) {
        assert(shape[axis] > 0);
        arr->strides[axis] = arr->size;
        arr->size *= shape[axis];
    }

    // shape
    arr->shape = (int *) malloc(ndim * sizeof(int));
    if (!arr->shape) {
        perror(MEM_ALLOC_ERR);
        exit(1);
    }
    memcpy(arr->shape, shape, ndim * sizeof(int));

    // ndim
    arr->ndim = ndim;
    
    // values
    arr->values = (dtype *) calloc(arr->size, sizeof(dtype));
    if (!arr->values) {
        perror(MEM_ALLOC_ERR);
        exit(1);
    }

    return arr;
}

Arr *create_arr_(dtype *values, int *shape, int ndim) {
    Arr *arr = create_arr_zeros(shape, ndim);
    memcpy(arr->values, values, arr->size * sizeof(dtype));
    return arr;
}

// create a new array with the values
Arr *create_arr(Arr *arr) {
    Arr *new_arr = create_arr_(arr->values, arr->shape, arr->ndim);
    return new_arr;
}

// create a new array of shape (1, size)
Arr *create_arr_row(int size) {
    assert(size > 0);

    int shape[] = {1, size};
    Arr *arr = create_arr_zeros(shape, 2);
    return arr;
}


// create a new array of shape (size, 1)
Arr *create_arr_col(int size) {
    assert(size > 0);

    int shape[] = {size, 1};
    Arr *arr = create_arr_zeros(shape, 2);
    return arr;
}

// free memory
void free_arr(Arr *arr) {
    free(arr->values);
    free(arr->strides);
    free(arr->shape);
    free(arr);
}

// ----------------------------------------------------------------------------
// Tensor creators
// ----------------------------------------------------------------------------

Tensor *create_tensor_zeros(int *shape, int ndim) {
    Tensor *tensor = (Tensor *) malloc(sizeof(Tensor));
    if (!tensor) {
        perror(MEM_ALLOC_ERR);
        exit(1);
    }

    tensor->data = create_arr_zeros(shape, ndim);
    tensor->grad = create_arr_zeros(shape, ndim);
    tensor->need_grad = 1;
    tensor->the_generating_operator = NO_GENERATING_OPERATOR;
    tensor->num_generating_operands = 0;
    memset(tensor->generating_operands, 0, sizeof(tensor->generating_operands));
    return tensor;
}

Tensor *create_tensor_(dtype *values, int *shape, int ndim) {
    Tensor *tensor = create_tensor_zeros(shape, ndim);
    memcpy(tensor->data->values, values, tensor->data->size * sizeof(dtype));
    return tensor;
}

// Just copy arr->values.
Tensor *create_tensor(Arr *arr) {
    Tensor *tensor = create_tensor_(arr->values, arr->shape, arr->ndim);
    return tensor;
}

// shape is (1, size)
Tensor *create_tensor_onehot(int size, int hot) {
    assert(0 <= hot && hot <= size);

    Tensor *tensor = create_tensor_row_(size);
    tensor->data->values[hot] = 1;
    return tensor;
}

// shape is (1, size)
Tensor *create_tensor_row_(int size) {
    assert(size > 0);

    int shape[] = {1, size};
    Tensor *tensor = create_tensor_zeros(shape, 1);
    return tensor;
}

// create a tensor of shape (1, size) from array
Tensor *create_tensor_row(Arr *arr) {
    int shape[] = {1, arr->size};
    Tensor *tensor = create_tensor_zeros(shape, 2);
    memcpy(tensor->data->values, arr->values, arr->size * sizeof(dtype));
    return tensor;
}

// shape is (size, 1)
Tensor *create_tensor_col_(int size) {
    assert(size > 0);

    int shape[] = {size, 1};
    Tensor *tensor = create_tensor_zeros(shape, 1);
    return tensor;
}

// create a tensor of shape (size, 1) from array
Tensor *create_tensor_col(Arr *arr) {
    int shape[] = {arr->size, 1};
    Tensor *tensor = create_tensor_zeros(shape, 2);
    memcpy(tensor->data->values, arr->values, arr->size * sizeof(dtype));
    return tensor;
}

// free memory
void free_tensor(Tensor *tensor) {
    free_arr(tensor->grad);     // 注意！不能直接 free. 本质上，任何包含指针变量的“结构体”都需要写一个专门的 free_ 函数!  # 20241224 21:00
    free_arr(tensor->data);
    free(tensor);
}

void set_grad_to_1s(Tensor *tensor) {
    for (int i = 0; i < tensor->grad->size; i++) {
        tensor->grad->values[i] = 1;
    }
}

void set_grad_to_0s(Tensor *tensor) {
    for (int i = 0; i < tensor->grad->size; i++) {
        tensor->grad->values[i] = 0;
    }
}

void set_data_to_0s(Tensor *tensor) {
    for (int i = 0; i < tensor->data->size; i++) {
        tensor->data->values[i] = 0;
    }
}


// ----------------------------------------------------------------------------
// Tensor operators
// ----------------------------------------------------------------------------

// multiply

// multiply 2 2D arrays
Tensor *matmultiply(Tensor *a, Tensor *b) {
    // a x b -> out
    // (m, n) x (n, k) -> (m, k)

    Arr *out_tmp = matmultiply_(a->data, b->data);
    Tensor *out = create_tensor(out_tmp);
    free_arr(out_tmp);

    out->the_generating_operator = MATMULTIPLY;
    out->num_generating_operands = 2;
    out->generating_operands[0] = a;
    out->generating_operands[1] = b;
    return out;
}


void matmultiply_backward(Tensor *out) {
    // a x b -> out
    // (m, n) x (n, k) -> (m, k)

    Tensor *a = out->generating_operands[0];
    Tensor *b = out->generating_operands[1];
    assert (a && b);

    if (a->need_grad) {
        Arr *b_T_tmp = mattranspose_(b->data);
        Arr *a_grad_tmp = matmultiply_(out->grad, b_T_tmp);   // d/da += d/dout x b.T
        free_arr(b_T_tmp);
        for (int i = 0; i < a->grad->size; i++) {
            a->grad->values[i] += a_grad_tmp->values[i];    // 注意不是直接 = 而是 +=
        }
        free_arr(a_grad_tmp);
    }
    if (b->need_grad) {
        Arr *a_T_tmp = mattranspose_(a->data);
        Arr *b_grad_tmp = matmultiply_(a_T_tmp, out->grad);   // d/db += a.T x d/dout
        free_arr(a_T_tmp);
        for (int i = 0; i < b->grad->size; i++) {
            b->grad->values[i] += b_grad_tmp->values[i];    // 注意不是直接 = 而是 +=
        }
        free_arr(b_grad_tmp);
    }
}

// multiply 2 2D arrays
Arr *matmultiply_(Arr *a, Arr *b) {
    // a x b -> out
    // (m, n) x (n, k) -> (m, k)

    assert((a->ndim == 2) && (b->ndim == 2));   // 要求张量 a 和 b 都是2维的！  # 20241224 21:30

    int m = a->shape[0];
    int n = a->shape[1];
    assert(b->shape[0] == n);
    int k = b->shape[1];

    int shape[] = {m, k};
    Arr *out = create_arr_zeros(shape, 2);

    // out->values
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            for (int l = 0; l < n; l++) {
                out->values[i * k + j] += a->values[i * n + l] * b->values[l * k + j];
            }
        }
    }
    return out;
}

// ----------------------------------------------------------------------------
//  add

// add 2 arrays with shapes of (a0, a1, a2, a3) and (1, a1, a2, a3)
Tensor *matadd(Tensor *a, Tensor *b) {
    Arr *out_tmp = matadd_(a->data, b->data);
    Tensor *out = create_tensor(out_tmp);
    free_arr(out_tmp);

    out->the_generating_operator = ADD;
    out->num_generating_operands = 2;
    out->generating_operands[0] = a;
    out->generating_operands[1] = b;
    return out;
}

void matadd_backward(Tensor *out) {
    Tensor *a = out->generating_operands[0];
    Tensor *b = out->generating_operands[1];
    assert (a && b);
    
    // 使用 % 求余就不用确定 large_arr 和 small_arr 了！ # 20241228
    if (a->need_grad) {
        for (int i = 0; i < out->grad->size; i++) {
            a->grad->values[i % a->grad->size] += out->grad->values[i];       // d/da += sum(d/dout)
        }
    }
    if (b->need_grad) {
        for (int i = 0; i < out->grad->size; i++) {
            b->grad->values[i % b->grad->size] += out->grad->values[i];       // d/db += sum(d/dout)
        }
    }
}

// add 2 arrays with shapes of (a0, a1, a2, a3) and (1, a1, a2, a3)
Arr *matadd_(Arr *a, Arr *b) {
    int ndim = a->ndim;
    assert(ndim == b->ndim);
    
    Arr *small_arr = a;
    Arr *large_arr = b;
    int small_size = a->size;
    if (b->size < a->size) {
        small_arr = b;
        large_arr = a;
        small_size = b->size;
    }
    assert (small_arr->shape[0] == 1);
    for (int axis = 1; axis < ndim; axis++) {
        assert(small_arr->shape[axis] == large_arr->shape[axis]);
    }

    Arr *out = create_arr(large_arr);

    // out->values
    for (int pos = 0; pos < out->size; pos++) {
        out->values[pos] += small_arr->values[pos % small_size];
    }
    return out;
}

// ----------------------------------------------------------------------------
//  minus

// minus 2 arrays with shapes of (a0, a1, a2, a3) and (1, a1, a2, a3)
Tensor *matminus(Tensor *a, Tensor *b) {
    Arr *out_tmp = matminus_(a->data, b->data);
    Tensor *out = create_tensor(out_tmp);
    free_arr(out_tmp);

    out->the_generating_operator = MINUS;
    out->num_generating_operands = 2;
    out->generating_operands[0] = a;
    out->generating_operands[1] = b;
    return out;
}

void matminus_backward(Tensor *out) {
    Tensor *a = out->generating_operands[0];
    Tensor *b = out->generating_operands[1];
    assert(a && b);

    if (a->need_grad) {
        for (int i = 0; i < out->grad->size; i++) {
            a->grad->values[i % a->grad->size] += out->grad->values[i];       // d/da += sum(d/dout)
        }
    }
    if (b->need_grad) {
        for (int i = 0; i < out->grad->size; i++) {
            b->grad->values[i % b->grad->size] += -out->grad->values[i];       // d/db += -sum(d/dout)
        }
    }
}

// minus 2 arrays with shapes of (a0, a1, a2, a3) and (1, a1, a2, a3)
Arr *matminus_(Arr *a, Arr *b) {
    Arr *b_negetive = matnegate_(b);
    Arr *out = matadd_(a, b_negetive);

    free_arr(b_negetive);
    return out;
}

// ----------------------------------------------------------------------------
// transpose

// transpose a 2D Tensor.
Tensor *mattranspose(Tensor *a) {
    assert(a->data->ndim == 2);
    
    int shape_T[] = {a->data->shape[1], a->data->shape[0]};
    Tensor *out = create_tensor_zeros(shape_T, a->data->ndim);

    // data
    for (int i = 0; i < shape_T[0]; i++) {
        for (int j = 0; j < shape_T[1]; j++) {
            out->data->values[i * shape_T[1] + j] = a->data->values[j * shape_T[0] + i];
        }
    }

    out->the_generating_operator = MATTRANSPOSE;
    out->num_generating_operands = 1;
    out->generating_operands[0] = a;
    return out;
}

void mattranspose_backward(Tensor *out) {
    Tensor *a = out->generating_operands[0];
    assert (a);

    if (a->need_grad) {
        Arr *a_grad_tmp = mattranspose_(out->grad);
        for (int i = 0; i < a->grad->size; i++) {
            a->grad->values[i] += a_grad_tmp->values[i];    // d/da += d/dout.T
        }
        free_arr(a_grad_tmp);
    }
}


// transpose a 2D Tensor.
Arr *mattranspose_(Arr *a) {
    assert(a->ndim == 2);
    
    int shape_T[] = {a->shape[1], a->shape[0]};
    Arr *out = create_arr_zeros(shape_T, a->ndim);

    // out->values
    for (int i = 0; i < shape_T[0]; i++) {
        for (int j = 0; j < shape_T[1]; j++) {
            out->values[i * shape_T[1] + j] = a->values[j * shape_T[0] + i];
        }
    }
    return out;
}

// ----------------------------------------------------------------------------
// negate

Tensor *matnegate(Tensor *a) {
    Arr *out_tmp = matnegate_(a->data);
    Tensor *out = create_tensor(out_tmp);
    free_arr(out_tmp);

    out->the_generating_operator = MATNEGATE;
    out->num_generating_operands = 1;
    out->generating_operands[0] = a;
    return out;
}

void matnegate_backward(Tensor *out) {
    Tensor *a = out->generating_operands[0];
    assert(a);

    if (a->need_grad) {
        for (int i = 0; i < a->grad->size; i++) {
            a->grad->values[i] += -out->grad->values[i];    // d/da += -d/dout
        }
    }
}

Arr *matnegate_(Arr *a) {
    Arr *out = create_arr_zeros(a->shape, a->ndim);

    // out->values
    for (int i = 0; i < a->size; i++) {
        out->values[i] = -a->values[i];
    }
    return out;
}

// ----------------------------------------------------------------------------
// softmax

// input must be a 2D array, and output a 2D array with the same shape in which every row is a probability distribution
Tensor *softmax(Tensor *y) {
    Arr *out_tmp = softmax_(y->data);
    Tensor *out = create_tensor(out_tmp);
    free_arr(out_tmp);

    // out->the_generating_operator = SOFTMAX;
    // out->num_generating_operands = 1;
    // out->generating_operands[0] = y;
    return out;
}

// input must be a 2D array, output a 2D array with the same shape in which every row is a probability distribution
Arr *softmax_(Arr *y) {
    assert(y->ndim == 2);

    Arr *out = create_arr(y);

    // out->values
    Arr *y_max = max_(y, 1);
    int pos;
    for (int b_idx = 0; b_idx < out->shape[0]; b_idx ++) {
        // 先找 y_max, 并且让 y 的每个项都减去 y_max, 以免指数过大

        dtype denominator = 0;
        dtype numerator;
        for (int j = 0; j < out->shape[1]; j++) {
            pos = b_idx * y->shape[1] + j;
            out->values[pos] -= y_max->values[b_idx];
            numerator = exp(out->values[pos]);
            out->values[pos] = numerator;
            denominator += numerator;
        }

        for (int j = 0; j < out->shape[1]; j++) {
            pos = b_idx * y->shape[1] + j;
            out->values[pos] /= denominator;
        }
    }
    free_arr(y_max);
    return out;
}

// ----------------------------------------------------------------------------
// softmaxloss

// return a coloum vector (bsize, 1)
Tensor *softmaxloss(Tensor *y, Tensor *label) {
    Arr *out_tmp = softmaxloss_(y->data, label->data);
    Tensor *out = create_tensor(out_tmp);
    free_arr(out_tmp);

    out->the_generating_operator = SOFTMAXLOSS;
    out->num_generating_operands = 2;
    out->generating_operands[0] = y;
    out->generating_operands[1] = label;
    return out;
}

// only y needs backward, label does not need backward
void softmaxloss_backward(Tensor *out) {
    assert(out->data->ndim == 2);
    assert(out->data->shape[1] == 1);

    Tensor *y = out->generating_operands[0];
    Tensor *label = out->generating_operands[1];
    assert (y && label);

    if (y->need_grad) {
        Tensor *probabilities = softmax(y);         // TODO: 把 softmaxloss 写成一个类, 不然反向传播时还要再算一次 probabilities  # 20241226 17:30
        Arr *int_labels = argmax_(label->data, 1);
        for (int b_idx = 0; b_idx < y->grad->shape[0]; b_idx++) {
            int argmax_pos = b_idx * y->grad->shape[1] + (int) int_labels->values[b_idx];
            for (int j = 0; j < y->grad->shape[1]; j++) {
                int pos = b_idx * y->grad->shape[1] + j;
                if (pos == argmax_pos) {
                    y->grad->values[pos] += (probabilities->data->values[pos] - 1) * out->grad->values[b_idx];          // d/dy_label += (probabilities-1) * d/dout     (1)     // backward 必须有 dout 
                } else {
                    y->grad->values[pos] += probabilities->data->values[pos] * out->grad->values[b_idx];              // d/dy_nonlabel += probabilities * d/dout        (2)
                }
            }
        }
        free_arr(int_labels);
        free_tensor(probabilities);
    }
}

// return a coloum vector (bsize, 1)
// TODO: 把 softmaxloss 写成一个类, 这样就可以直接输出形如 (1, 1) 的 loss 了    # 20241230 15:00
Arr *softmaxloss_(Arr *y, Arr *label) {
    assert(y->ndim == 2 && label->ndim == 2);
    assert(y->shape[0] == label->shape[0] && y->shape[1] == label->shape[1]);

    // out->values
    Arr *out = create_arr_col(y->shape[0]);
    Arr *probabilities = softmax_(y);
    Arr *arg = argmax_(label, 1);
    for (int b_idx = 0; b_idx < y->shape[0]; b_idx++) {
        int argmax_pos = b_idx * y->shape[1] + (int) arg->values[b_idx];
        out->values[b_idx] = -log(probabilities->values[argmax_pos]);
    }
    free_arr(arg);
    free_arr(probabilities);
    return out;
}


// ----------------------------------------------------------------------------
// RELU

Tensor *relu(Tensor *a) {
    Arr *out_tmp = relu_(a->data);
    Tensor *out = create_tensor(out_tmp);
    free_arr(out_tmp);

    out->the_generating_operator = RELU;
    out->num_generating_operands = 1;
    out->generating_operands[0] = a;
    return out;
}

void relu_backward(Tensor *out) {
    Tensor *a = out->generating_operands[0];
    assert(a);

    if (a->need_grad) {
        for (int i = 0; i < a->grad->size; i++) {
            if (a->data->values[i] > 0) {
                a->grad->values[i] += out->grad->values[i];     // if a > 0, d/da += d/dout; else d/da += 0
            }
        }
    }
}

Arr *relu_(Arr *a) {
    Arr *out = create_arr(a);

    // out->values
    for (int i = 0; i < out->size; i++) {
        out->values[i] = (out->values[i] > 0 ? out->values[i] : 0);
    }
    return out;
}

// ----------------------------------------------------------------------------
// sum

// if axis is 1, then shape (a0, a1, a2) -> shape (a0, 1, a2)
Tensor *sum(Tensor *a, int axis) {
    Arr *out_tmp = sum_(a->data, axis);
    Tensor *out = create_tensor(out_tmp);
    free_arr(out_tmp);

    out->the_generating_operator = SUM;
    out->num_generating_operands = 1;
    out->generating_operands[0] = a;
    return out;
}

void sum_backward(Tensor *out) {
    Tensor *a = out->generating_operands[0];
    assert(a);

    int the_axis = 0;       // 如果 sum 没有改变形状，则当作是沿着第 0 个轴求和？
    for (int axis = 0; axis < a->data->ndim; axis ++) {
        if(a->data->shape[axis] > out->data->shape[axis]) {
            the_axis = axis;
            break;
        }
    }
    assert(out->data->shape[the_axis] == 1);

    if (a->need_grad) {
        int j;
        int vector_root_pos;
        for (int pos_out = 0; pos_out < out->grad->size; pos_out++) {
            j = pos_out % a->data->strides[the_axis];
            vector_root_pos = (pos_out - j) * a->data->shape[the_axis] + j;
            for (int idx = 0; idx < a->data->shape[the_axis]; idx++) {
                a->grad->values[vector_root_pos + idx * a->data->strides[the_axis]] += out->grad->values[pos_out];      // d/da += (d/dout)
            }
        }
    }
}

// if axis is 1, then shape (a0, a1, a2) -> shape (a0, 1, a2)
Arr *sum_(Arr *a, int axis) {
    assert(0 <= axis && axis <= a->ndim);

    int ndim = a->ndim;
    int *shape_out = (int *) malloc(ndim * sizeof(int));
    if (!shape_out) {
        perror(MEM_ALLOC_ERR);
        exit(1);
    }
    memcpy(shape_out, a->shape, ndim * sizeof(int));
    shape_out[axis] = 1;

    Arr *out = create_arr_zeros(shape_out, ndim);
    free(shape_out);

    // out->values
    int vector_root_pos;                // 先找到该 vector 的起点在数组 a 中的位置
    dtype summation;
    for (int i = 0; i * a->strides[axis] * a->shape[axis] < a->size; i++) {
        for (int j = 0; j < a->strides[axis]; j++) {
            vector_root_pos = i * a->strides[axis] * a->shape[axis] + j;
            summation = 0;
            for (int idx = 0; idx < a->shape[axis]; idx++) {
                summation += a->values[vector_root_pos + idx * a->strides[axis]];
            }

            out->values[i * a->strides[axis] + j] = summation;
        }
    }
    return out;
}

// ----------------------------------------------------------------------------
// sum_all

// return shape of (1, 1)
Tensor *sum_all(Tensor *a) {
    Arr *out_tmp = sum_all_(a->data);
    Tensor *out = create_tensor(out_tmp);
    free_arr(out_tmp);

    out->the_generating_operator = SUM_ALL;
    out->num_generating_operands = 1;
    out->generating_operands[0] = a;
    return out;
}

void sum_all_backward(Tensor *out) {
    Tensor *a = out->generating_operands[0];
    assert(a);

    if (a->need_grad) {
        for (int i = 0; i < a->grad->size; i++) {
            a->grad->values[i] += out->grad->values[0];     // d/da += d/dout
        }
    }
}

// return shape of (1, 1)
Arr *sum_all_(Arr *a) {
    Arr *out = create_arr_row(1);

    // out->values
    for (int i = 0; i < a->size; i++) {
        out->values[0] += a->values[i];
    }
    return out;
}

// ----------------------------------------------------------------------------
// mean

// if axis is 1, then shape (a0, a1, a2) -> shape (a0, 1, a2)
Tensor *mean(Tensor *a, int axis) {
    Arr *out_tmp = mean_(a->data, axis);
    Tensor *out = create_tensor(out_tmp);
    free_arr(out_tmp);

    out->the_generating_operator = MEAN;
    out->num_generating_operands = 1;
    out->generating_operands[0] = a;
    return out;
}

void mean_backward(Tensor *out) {
    Tensor *a = out->generating_operands[0];
    assert(a);

    int the_axis = 0;       // 如果没有改变形状，则当作是沿着第 0 个轴求均值？
    for (int axis = 0; axis < a->data->ndim; axis ++) {
        if(a->data->shape[axis] > out->data->shape[axis]) {
            the_axis = axis;
            break;
        }
    }
    assert(out->data->shape[the_axis] == 1);

    if (a->need_grad) {
        int j;
        int vector_root_pos;
        for (int pos_out = 0; pos_out < out->grad->size; pos_out++) {
            j = pos_out % a->data->strides[the_axis];
            vector_root_pos = (pos_out - j) * a->data->shape[the_axis] + j;
            for (int idx = 0; idx < a->data->shape[the_axis]; idx++) {
                a->grad->values[vector_root_pos + idx * a->data->strides[the_axis]] += out->grad->values[pos_out] / a->data->shape[the_axis];      // d/da += (d/dout) / n
            }
        }
    }
}

// if axis is 1, then shape (a0, a1, a2) -> shape (a0, 1, a2)
Arr *mean_(Arr *a, int axis) {
    Arr *out = sum_(a, axis);

    // out->values
    for (int pos_out = 0; pos_out < out->size; pos_out++) {
        out->values[pos_out] /= a->shape[axis];
    }
    return out;
}

// ----------------------------------------------------------------------------
// std_deviation

// if axis is 1, then shape (a0, a1, a2) -> shape (a0, 1, a2)
// 暂时只考虑 batch normalization, 即 axis = 0 的情况  # 20241230
// need check
Tensor *std_deviation(Tensor *a, int axis) {
    Arr *out_tmp = std_deviation_(a->data, 0);
    Tensor *out = create_tensor(out_tmp);
    free_arr(out_tmp);

    out->the_generating_operator = STD_DEVIATION;
    out->num_generating_operands = 0;
    out->generating_operands[0] = a;
    return out;
}

// if axis is 1, then shape (a0, a1, a2) -> shape (a0, 1, a2)
// 暂时只考虑 batch normalization, 即 axis = 0 的情况  # 20241230
Arr *std_deviation_(Arr *a, int axis) {
    assert(axis == 0);      // 暂时只考虑 batch normalization  # 20241230

    // out->values
    // 1. mean a
    Arr *a_mean = mean_(a, axis);
    // 2. translate a
    Arr *a_translate = matminus_(a, a_mean);
    // 2. square the translated a
    for (int i = 0; i < a_translate->size; i++) {
        a_translate->values[i] *= a_translate->values[i];
    }
    // 3. mean the squared translated a
    Arr *out = mean_(a_translate, axis);
    // 4. sqrt the mean
    for (int i = 0; i < out->size; i++) {
        out->values[i] = sqrt(out->values[i]);
    }

    free_arr(a_translate);
    free_arr(a_mean);
    return out;
}

// ----------------------------------------------------------------------------
// batch normalization

// 沿着 axis 轴归一化（平移和缩放）
// 暂时只考虑 batch normalization, 即 axis = 0 的情况  # 20241230
// need check
Tensor *batch_norm(Tensor *a, int axis) {
    Arr *out_tmp = batch_norm_(a->data, axis);
    Tensor *out = create_tensor(out_tmp);
    free_arr(out_tmp);

    out->the_generating_operator = BATCH_NORM;
    out->num_generating_operands = 1;
    out->generating_operands[0] = a;
    return out;
}

void batch_norm_backward(Tensor *out) {
    Tensor *a = out->generating_operands[0];
    assert(a);

    // 20250101 15:45
    // d/da += Jacobian * d/dout    （列向量形式） ✓, 因为 batch normalization 默认是沿着 0 号轴求 mean 和 std_deviation
    // d/da += d/dout * Jacobian.T  （行向量形式） ✗
    if (a->need_grad) {
        int axis = 0;       // batch normalization 默认是沿着 0 号轴求 mean 和 std_deviation. TODO: 写成类，把 axis 记住
        Arr *a_mean = mean_(a->data, axis);
        Arr *a_std_deviation = std_deviation_(a->data, axis);
        
        // 计算 batch normalization 的输入矩阵的每一列的需要的 Jacobian 是不一样的！
        for (int k = 0; k < a->data->shape[1]; k++) {
            // Jacobian （n x n 的方阵）
            int n = a->data->shape[axis];
            int shape_Jacobian[] = {n, n};
            Arr *Jacobian = create_arr_zeros(shape_Jacobian, 2);
            dtype mean_tmp = a_mean->values[k];
            dtype std_deviation_tmp = a_std_deviation->values[k];

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    int pos_a_i = i * a->data->shape[1] + k;
                    int pos_a_j = j * a->data->shape[1] + k;

                    int pos_J = i * n + j;
                    // partial f_j / partial x_i
                    int dirac = (i == j ? 1 : 0);
                    Jacobian->values[pos_J] = ((dirac - 1.0 / n) * std_deviation_tmp - (a->data->values[pos_a_i] - mean_tmp) * (a->data->values[pos_a_j] - mean_tmp) / (std_deviation_tmp * n)) / (std_deviation_tmp * std_deviation_tmp);
                }
            }

            // TODO: 对 out 切片，取出第 k 列，然后用 Jacobian 去乘。  # 20250102 01:30

            free_arr(Jacobian);
        }

        // int j;
        // int vector_root_pos;
        // int n = a_std_deviation->size;
        // for (int pos_mean = 0; pos_mean < n; pos_mean++) {
        //     j = pos_mean % a->data->strides[axis];
        //     vector_root_pos = (pos_mean - j) * a->data->shape[axis] + j;
        //     for (int idx = 0; idx < a->data->shape[axis]; idx++) {
        //         dtype out_tmp = out->data->values[pos_mean];
        //         dtype std_tmp = a_std_deviation->values[pos_mean];
        //         a->grad->values[vector_root_pos + idx * a->data->strides[axis]] += out->grad->values[pos_mean] * (n - 1) * (n - (out_tmp * std_tmp) * (out_tmp * std_tmp)) / (n * n * std_tmp);    // da += dout * (n- 1) * (n - (out * std_deviation)**n) / (n**2 * std_deviation)
        //     }
        // }

        free_arr(a_std_deviation);
        free_arr(a_mean);
    }
}

// 沿着 axis 轴归一化（平移和缩放）
// 暂时只考虑 batch normalization, 即 axis = 0 的情况  # 20241230
// need check
Arr *batch_norm_(Arr *a, int axis) {
    assert(axis == 0);      // 暂时只考虑 batch normalization  # 20241230

    // 1. translate by mean
    Arr *a_mean = mean_(a, axis);
    Arr *out = matminus_(a, a_mean);

    // 2. scale by std_deviation
    Arr *a_std_deviation = std_deviation_(a, axis);
    int j;
    int vector_root_pos;
    for (int pos_mean = 0; pos_mean < a_std_deviation->size; pos_mean++) {
        j = pos_mean % a->strides[axis];
        vector_root_pos = (pos_mean - j) * a->shape[axis] + j;
        for (int idx = 0; idx < a->shape[axis]; idx++) {
            int pos = vector_root_pos + idx * a->strides[axis];
            out->values[pos] /= a_std_deviation->values[pos_mean];
        }
    }

    free_arr(a_std_deviation);
    free_arr(a_mean);
    return out;
}

// ----------------------------------------------------------------------------
// mean_all

// return shape of (1, 1)
Tensor *mean_all(Tensor *a) {
    Arr *out_tmp = mean_all_(a->data);
    Tensor *out = create_tensor(out_tmp);
    free_arr(out_tmp);

    out->the_generating_operator = MEAN_ALL;
    out->num_generating_operands = 1;
    out->generating_operands[0] = a;
    return out;
}

void mean_all_backward(Tensor *out) {
    Tensor *a = out->generating_operands[0];
    assert(a);

    if (a->need_grad) {
        for (int i = 0; i < a->grad->size; i++) {
            a->grad->values[i] += out->grad->values[0] / a->grad->size;     // d/da += d/dout / n
        }
    }
}

// return shape of (1, 1)
Arr *mean_all_(Arr *a) {
    Arr *out = create_arr_row(1);

    // out->values
    for (int i = 0; i < a->size; i++) {
        out->values[0] += a->values[i];
    }
    out->values[0] /= a->size;
    return out;
}

// ----------------------------------------------------------------------------
// argmax_

// if axis is 1, then shape (a0, a1, a2) -> shape (a0, 1, a2)
// NOTE: the "arg" is not in type int
Arr *argmax_(Arr *a, int axis) {
    assert(0 <= axis && axis <= a->ndim);

    int ndim = a->ndim;
    int *shape_out = (int *) malloc(ndim * sizeof(int));
    if (!shape_out) {
        perror(MEM_ALLOC_ERR);
        exit(1);
    }
    memcpy(shape_out, a->shape, ndim * sizeof(int));
    shape_out[axis] = 1;

    Arr *out = create_arr_zeros(shape_out, ndim);
    free(shape_out);

    // out->values
    int vector_root_pos;                // 先找到该 vector 的起点在数组 a 中的位置
    int arg;
    for (int i = 0; i * a->strides[axis] * a->shape[axis] < a->size; i++) {
        for (int j = 0; j < a->strides[axis]; j++) {
            vector_root_pos = i * a->strides[axis] * a->shape[axis] + j;
            arg = 0;
            for (int idx = 0; idx < a->shape[axis]; idx++) {
                if (a->values[vector_root_pos + idx * a->strides[axis]] > a->values[vector_root_pos + arg * a->strides[axis]]) {
                    arg = idx;
                }
            }

            out->values[i * a->strides[axis] + j] = arg;
        }
    }
    return out;
}

// ----------------------------------------------------------------------------
// max

// if axis is 1, then shape (a0, a1, a2) -> shape (a0, 1, a2)
Tensor *max(Tensor *a, int axis) {
    Arr *out_tmp = max_(a->data, axis);
    Tensor *out = create_tensor(out_tmp);
    free_arr(out_tmp);

    out->the_generating_operator = MAX;
    out->num_generating_operands = 1;
    out->generating_operands[0] = a;
    return out;
}

void max_backward(Tensor *out) {
    Tensor *a = out->generating_operands[0];
    assert(a);
    
    int the_axis = 0;       // 如果没有改变形状，则当作是沿着第 0 个轴求均值？
    for (int axis = 0; axis < a->data->ndim; axis ++) {
        if(a->data->shape[axis] > out->data->shape[axis]) {
            the_axis = axis;
            break;
        }
    }
    assert(out->data->shape[the_axis] == 1);

    if (a->need_grad) {
        Arr *arg = argmax_(a->data, the_axis);                  // TODO: 把 max 写成层, 记住 argmax, 不然不好 backward 时还要再算一次 argmax  # 20241229 15:00
        int j;
        int vector_root_pos;
        for (int pos_out = 0; pos_out < arg->size; pos_out++) {
            j = pos_out % a->data->strides[the_axis];
            vector_root_pos = (pos_out - j) * a->data->shape[the_axis] + j;
            a->grad->values[vector_root_pos + (int) arg->values[pos_out] * a->data->strides[the_axis]] += out->grad->values[pos_out];               // d/da[argmax] += d/dout
        }
        free_arr(arg);
    }
}

// if axis is 1, then shape (a0, a1, a2) -> shape (a0, 1, a2)
Arr *max_(Arr *a, int axis) {
    Arr *arg = argmax_(a, axis);

    // out->values
    int j;
    int vector_root_pos;                // 先找到该 vector 的起点在数组 a 中的位置
    for (int pos_out = 0; pos_out < arg->size; pos_out++) {
        j = pos_out % a->strides[axis];
        vector_root_pos = (pos_out - j) * a->shape[axis] + j;
        arg->values[pos_out] = a->values[vector_root_pos + (int) arg->values[pos_out] * a->strides[axis]];
    }
    return arg;
}

// ----------------------------------------------------------------------------
// argmax_all_

// return shape of (1, 1)
// NOTE: the "arg" is not in type int
Arr *argmax_all_(Arr *a) {
    Arr *out = create_arr_row(1);

    // out->values
    out->values[0] = 0;
    for (int i = 0; i < a->size; i++) {
        if (a->values[i] > out->values[0]) {
            out->values[0] = i;
        }
    }
    return out;
}

// ----------------------------------------------------------------------------
// max_all

Tensor *max_all(Tensor *a) {
    Arr *out_tmp = max_all_(a->data);
    Tensor *out = create_tensor(out_tmp);
    free_arr(out_tmp);

    out->the_generating_operator = MAX_ALL;
    out->num_generating_operands = 1;
    out->generating_operands[0] = a;
}

void max_all_backward(Tensor *out) {
    Tensor *a = out->generating_operands[0];
    assert(a);

    if (a->need_grad) {
        Arr *arg = argmax_all_(a->data);
        a->grad->values[(int) arg->values[0]] += out->grad->values[0];              // d/da[argmax] += d/dout
        free_arr(arg);
    }
}

// return shape of (1, 1)
Arr *max_all_(Arr *a) {
    Arr *out = create_arr_row(1);

    // out->values
    out->values[0] = a->values[0];
    for (int i = 0; i < a->size; i++) {
        if (a->values[i] > out->values[0]) {
            out->values[0] = a->values[i];
        }
    }
    return out;
}


// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

void gradient_descend(Tensor *tensor, dtype lr) {
    for (int i = 0; i < tensor->data->size; i++) {
        tensor->data->values[i] -= lr * tensor->grad->values[i];
    }
}

// 怎么打印一个多维数组? --一行一行打印！
// generally, stream is stdout
void fprint_arr(Arr *a, FILE *stream) {
    int num_col = a->shape[a->ndim - 1];
    for (int pos = 0; pos < a->size; pos += num_col) {
        int min_start_axis = a->ndim - 1;    // 用来表示以当前 位置 为开头的轴中 最小 是几号
        int min_stop_axis = a->ndim - 1;
        for (int axis = a->ndim - 1; axis >= 0; axis--) {       // a->strides[a->ndim - 1] 是 1, 不用考虑
            if (pos % (a->shape[axis] * a->strides[axis]) == 0) {
                min_start_axis = axis;
            }
            if ((pos + num_col) % (a->shape[axis] * a->strides[axis]) == 0) {
                min_stop_axis = axis;
            }
        }

        // print '[' or ' '
        for (int axis = 0; axis < a->ndim - 1; axis++) {
            if (axis < min_start_axis) {
                putc(' ', stream);
            } else {
                putc('[', stream);
            }
        }

        // print one row of values with '[' and ']', e.g. [1, 2, 3]
        putc('[', stream);
        int col;
        for (col = 0; col < num_col - 1; col++) {
            fprintf(stream, "%f ", a->values[pos + col]);       // 不要逗号了  # 20241231 13:45
        }
        fprintf(stream, "%f", a->values[pos + col]);
        putc(']', stream);

        // print ']'
        for (int axis = a->ndim - 1; axis > min_stop_axis; axis--) {
            putc(']', stream);
        }

        putc('\n', stream);
    }
    fprint_arr_shape(a, stream);
}

// equivalent to fprint_arr(a, stdout)
void print_arr(Arr *a) {
    fprint_arr(a, stdout);
}

// generally, stream is stdout
void fprint_arr_shape(Arr *a, FILE *stream) {
    fprintf(stream, "shape: (");
    for (int axis = 0; axis < a->ndim - 1; axis++) {
        fprintf(stream, "%d, ", a->shape[axis]);
    }
    fprintf(stream, "%d)\n", a->shape[a->ndim - 1]);
}

// equivalent to fprint_arr_shape(a, stdout)
void print_arr_shape(Arr *a) {
    fprint_arr_shape(a, stdout);
}

// append array into file
void sava_arr(const char *filename, Arr *a) {
    FILE *fp = fopen(filename, "a");
    fprint_arr(a, fp);
    fclose(fp);
}

// use for computing the accuracy
int argmax_match_count(Tensor *ys, Tensor *targets) {
    assert(ys->data->ndim == 2 && targets->data->ndim == 2);
    assert(ys->data->shape[0] == targets->data->shape[0] && ys->data->shape[1] == targets->data->shape[1]);

    Arr *ys_argmax = argmax_(ys->data, 1);
    Arr *targets_argmax = argmax_(targets->data, 1);

    int correct = 0;
    for (int i = 0; i < ys_argmax->size; i++) {
        if (((int) ys_argmax->values[i]) == ((int) targets_argmax->values[i])) {
            correct++;
        }
    }

    free_arr(ys_argmax);
    free_arr(targets_argmax);
    return correct;
}

// 根据正态分布 N(mu, sigma^2) 随机初始化一个张量
void random_init(Arr *a, double mu, double sigma) {
    for (int i = 0; i < a->size; i++) {
        a->values[i] = box_muller(mu, sigma);
    }
}

// 使用 Box-Muller tranformations 生成服从正态分布 N(mu, sigma^2) 的伪随机数
double box_muller(double mu, double sigma) {
    while (1) {
        double u = 2.0 * (rand() + 1.0) / (RAND_MAX + 1.0) - 1.0;           // U(-1, 1)
        double v = 2.0 * (rand() + 1.0) / (RAND_MAX + 1.0) - 1.0;           // U(-1, 1)
        double r = u * u + v * v;
        if (0.0 < r && r <= 1.0) {
            double x = u * sqrt((-2.0 * log(r)) / r);
            return mu + sigma * x;      // N(mu, sigma^2)
        }
    }
}
