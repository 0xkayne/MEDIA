#include "dnet_sgx_utils.h"
#include "network.h"
#include "image.h"
#include "utils.h"
#include "blas.h"

#include "crop_layer.h"
#include "connected_layer.h"
#include "gru_layer.h"
#include "rnn_layer.h"
#include "crnn_layer.h"
#include "local_layer.h"
#include "convolutional_layer.h"
#include "activation_layer.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "yolo_layer.h"
#include "normalization_layer.h"
#include "batchnorm_layer.h"
#include "maxpool_layer.h"
#include "reorg_layer.h"
#include "avgpool_layer.h"
#include "cost_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"
#include "route_layer.h"
#include "upsample_layer.h"
#include "shortcut_layer.h"
//#include "parser.h"
//#include "data.h"

load_args get_base_args(network *net)
{
    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.size = net->w;

    args.min = net->min_crop;
    args.max = net->max_crop;
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.center = net->center;
    args.saturation = net->saturation;
    args.hue = net->hue;
    return args;
}



network *load_network(list *sections, char *weights, int clear)
{
    printf("create network..\n");
    network *net = create_net_in(sections);
    if(weights && weights[0] != 0){
        load_weights(net, weights);
    }
    if(clear) (*net->seen) = 0;
    return net;
} 

size_t get_current_batch(network *net)
{
    size_t batch_num = (*net->seen) / (net->batch * net->subdivisions);
    return batch_num;
}

void reset_network_state(network *net, int b)
{
    int i;
    for (i = 0; i < net->n; ++i)
    {
        //gpu not allowed
    }
}

void reset_rnn(network *net)
{
    reset_network_state(net, 0);
}

float get_current_rate(network *net)
{
    size_t batch_num = get_current_batch(net);
    int i;
    float rate;
    if (batch_num < net->burn_in)
        return net->learning_rate * pow((float)batch_num / net->burn_in, net->power);
    switch (net->policy)
    {
    case CONSTANT:
        return net->learning_rate;
    case STEP:
        return net->learning_rate * pow(net->scale, batch_num / net->step);
    case STEPS:
        rate = net->learning_rate;
        for (i = 0; i < net->num_steps; ++i)
        {
            if (net->steps[i] > batch_num)
                return rate;
            rate *= net->scales[i];
        }
        return rate;
    case EXP:
        return net->learning_rate * pow(net->gamma, batch_num);
    case POLY:
        return net->learning_rate * pow(1 - (float)batch_num / net->max_batches, net->power);
    case RANDOM:
        return net->learning_rate * pow(rand_uniform(0, 1), net->power);
    case SIG:
        return net->learning_rate * (1. / (1. + exp(net->gamma * (batch_num - net->step))));
    default:
#ifdef DNET_SGX_DEBUG
        printf("Policy is weird!\n");
#endif
        return net->learning_rate;
    }
}

char *get_layer_string(LAYER_TYPE a)
{
    switch (a)
    {
    case CONVOLUTIONAL:
        return "convolutional";
    case ACTIVE:
        return "activation";
    case LOCAL:
        return "local";
    case DECONVOLUTIONAL:
        return "deconvolutional";
    case CONNECTED:
        return "connected";
    case RNN:
        return "rnn";
    case GRU:
        return "gru";
    case LSTM:
        return "lstm";
    case CRNN:
        return "crnn";
    case MAXPOOL:
        return "maxpool";
    case REORG:
        return "reorg";
    case AVGPOOL:
        return "avgpool";
    case SOFTMAX:
        return "softmax";
    case DETECTION:
        return "detection";
    case REGION:
        return "region";
    case YOLO:
        return "yolo";
    case DROPOUT:
        return "dropout";
    case CROP:
        return "crop";
    case COST:
        return "cost";
    case ROUTE:
        return "route";
    case SHORTCUT:
        return "shortcut";
    case NORMALIZATION:
        return "normalization";
    case BATCHNORM:
        return "batchnorm";
    default:
        break;
    }
    return "none";
}

network *make_network(int n)
{
    network *net = calloc(1, sizeof(network));
    net->n = n;
    net->layers = calloc(net->n, sizeof(layer));
    net->seen = calloc(1, sizeof(size_t));
    net->t = calloc(1, sizeof(int));
    net->cost = calloc(1, sizeof(float));
    return net;
}

void forward_network(network *netp)
{

    network net = *netp;
    int i;
    for (i = 0; i < net.n; ++i)
    {
        net.index = i;
        layer l = net.layers[i];
        if (l.delta)
        {
            fill_cpu(l.outputs * l.batch, 0, l.delta, 1);
        }
        l.forward(l, net);
        net.input = l.output;
        if (l.truth)
        {
            net.truth = l.output;
        }
    }
    calc_network_cost(netp);
}

void update_network(network *netp)
{

    network net = *netp;
    int i;
    update_args a = {0};
    a.batch = net.batch * net.subdivisions;
    a.learning_rate = get_current_rate(netp);
    a.momentum = net.momentum;
    a.decay = net.decay;
    a.adam = net.adam;
    a.B1 = net.B1;
    a.B2 = net.B2;
    a.eps = net.eps;
    ++*net.t;
    a.t = *net.t;

    for (i = 0; i < net.n; ++i)
    {
        layer l = net.layers[i];
        if (l.update)
        {
            l.update(l, a);
        }
    }
}

void calc_network_cost(network *netp)
{
    network net = *netp;
    int i;
    float sum = 0;
    int count = 0;
    for (i = 0; i < net.n; ++i)
    {
        if (net.layers[i].cost)
        {
            sum += net.layers[i].cost[0];
            ++count;
        }
    }
    *net.cost = sum / count;
}

int get_predicted_class_network(network *net)
{
    return max_index(net->output, net->outputs);
}

void backward_network(network *netp)
{

    network net = *netp;
    int i;
    network orig = net;
    for (i = net.n - 1; i >= 0; --i)
    {
        layer l = net.layers[i];
        if (l.stopbackward)
            break;
        if (i == 0)
        {
            net = orig;
        }
        else
        {
            layer prev = net.layers[i - 1];
            net.input = prev.output;
            net.delta = prev.delta;
        }
        net.index = i;
        l.backward(l, net);
    }
}

float train_network_datum(network *net)
{
    *net->seen += net->batch;
    net->train = 1;
    forward_network(net);
    backward_network(net);
    float error = *net->cost;
    if (((*net->seen) / net->batch) % net->subdivisions == 0)
        update_network(net);
    return error;
}

float train_network_sgd(network *net, data d, int n)
{
    int batch = net->batch;

    int i;
    float sum = 0;
    for (i = 0; i < n; ++i)
    {
        get_random_batch(d, batch, net->input, net->truth);
        float err = train_network_datum(net);
        sum += err;
    }
    return (float)sum / (n * batch);
}

float train_network(network *net, data d)
{
    assert(d.X.rows % net->batch == 0);
    int batch = net->batch;
    int n = d.X.rows / batch;

    int i;
    float sum = 0;
    for (i = 0; i < n; ++i)
    {
        get_next_batch(d, batch, i * batch, net->input, net->truth);
        float err = train_network_datum(net);
        sum += err;
    }
    return (float)sum / (n * batch);
}

void set_temp_network(network *net, float t)
{
    int i;
    for (i = 0; i < net->n; ++i)
    {
        net->layers[i].temperature = t;
    }
}

void set_batch_network(network *net, int b)
{
    net->batch = b;
    int i;
    for (i = 0; i < net->n; ++i)
    {
        net->layers[i].batch = b;
    }
}

int resize_network(network *net, int w, int h)
{

    int i;
    //if(w == net->w && h == net->h) return 0;
    net->w = w;
    net->h = h;
    int inputs = 0;
    size_t workspace_size = 0;
    //fprintf(stderr, "Resizing to %d x %d...\n", w, h);
    //fflush(stderr);
    for (i = 0; i < net->n; ++i)
    {
        layer l = net->layers[i];
        if (l.type == CONVOLUTIONAL)
        {
            resize_convolutional_layer(&l, w, h);
        }
        else if (l.type == CROP)
        {
            resize_crop_layer(&l, w, h);
        }
        else if (l.type == MAXPOOL)
        {
            resize_maxpool_layer(&l, w, h);
        }
        else if (l.type == REGION)
        {
            resize_region_layer(&l, w, h);
        }
        else if (l.type == YOLO)
        {
            resize_yolo_layer(&l, w, h);
        }
        else if (l.type == ROUTE)
        {
            resize_route_layer(&l, net);
        }
        else if (l.type == SHORTCUT)
        {
            resize_shortcut_layer(&l, w, h);
        }
        else if (l.type == UPSAMPLE)
        {
            resize_upsample_layer(&l, w, h);
        }
        else if (l.type == REORG)
        {
            resize_reorg_layer(&l, w, h);
        }
        else if (l.type == AVGPOOL)
        {
            resize_avgpool_layer(&l, w, h);
        }
        else if (l.type == NORMALIZATION)
        {
            resize_normalization_layer(&l, w, h);
        }
        else if (l.type == COST)
        {
            resize_cost_layer(&l, inputs);
        }
        else
        {
            error("Cannot resize this type of layer");
        }
        if (l.workspace_size > workspace_size)
            workspace_size = l.workspace_size;
        if (l.workspace_size > 2000000000)
            assert(0);
        inputs = l.outputs;
        net->layers[i] = l;
        w = l.out_w;
        h = l.out_h;
        if (l.type == AVGPOOL)
            break;
    }
    layer out = get_network_output_layer(net);
    net->inputs = net->layers[0].inputs;
    net->outputs = out.outputs;
    net->truths = out.outputs;
    if (net->layers[net->n - 1].truths)
        net->truths = net->layers[net->n - 1].truths;
    net->output = out.output;
    free(net->input);
    free(net->truth);
    net->input = calloc(net->inputs * net->batch, sizeof(float));
    net->truth = calloc(net->truths * net->batch, sizeof(float));

    free(net->workspace);
    net->workspace = calloc(1, workspace_size);

    //fprintf(stderr, " Done!\n");
    return 0;
}

layer get_network_detection_layer(network *net)
{
    int i;
    for (i = 0; i < net->n; ++i)
    {
        if (net->layers[i].type == DETECTION)
        {
            return net->layers[i];
        }
    }
#ifdef DNET_SGX_DEBUG
    printf("Detection layer not found!!\n");
#endif

    layer l = {0};
    return l;
}

image get_network_image_layer(network *net, int i)
{
    layer l = net->layers[i];

    if (l.out_w && l.out_h && l.out_c)
    {
        return float_to_image(l.out_w, l.out_h, l.out_c, l.output);
    }
    image def = {0};
    return def;
}

image get_network_image(network *net)
{
    int i;
    for (i = net->n - 1; i >= 0; --i)
    {
        image m = get_network_image_layer(net, i);
        if (m.h != 0)
            return m;
    }
    image def = {0};
    return def;
}

/* void visualize_network(network *net)
{
    image *prev = 0;
    int i;
    char buff[256];
    for(i = 0; i < net->n; ++i){
        sprintf(buff, "Layer %d", i);
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            prev = visualize_convolutional_layer(l, buff, prev);
        }
    } 
} */

void top_predictions(network *net, int k, int *index)
{
    top_k(net->output, net->outputs, k, index);
}

float *network_predict(network *net, float *input)
{
    network orig = *net;
    net->input = input;
    net->truth = 0;
    net->train = 0;
    net->delta = 0;
    forward_network(net);
    float *out = net->output;
    *net = orig;
    return out;
}

int num_detections(network *net, float thresh)
{
    int i;
    int s = 0;
    for (i = 0; i < net->n; ++i)
    {
        layer l = net->layers[i];
        if (l.type == YOLO)
        {
            s += yolo_num_detections(l, thresh);
        }
        if (l.type == DETECTION || l.type == REGION)
        {
            s += l.w * l.h * l.n;
        }
    }
    return s;
}

detection *make_network_boxes(network *net, float thresh, int *num)
{
    layer l = net->layers[net->n - 1];
    int i;
    int nboxes = num_detections(net, thresh);
    if (num)
        *num = nboxes;
    detection *dets = calloc(nboxes, sizeof(detection));
    for (i = 0; i < nboxes; ++i)
    {
        dets[i].prob = calloc(l.classes, sizeof(float));
        if (l.coords > 4)
        {
            dets[i].mask = calloc(l.coords - 4, sizeof(float));
        }
    }
    return dets;
}

void fill_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, detection *dets)
{
    int j;
    for (j = 0; j < net->n; ++j)
    {
        layer l = net->layers[j];
        if (l.type == YOLO)
        {
            int count = get_yolo_detections(l, w, h, net->w, net->h, thresh, map, relative, dets);
            dets += count;
        }
        if (l.type == REGION)
        {
            get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets);
            dets += l.w * l.h * l.n;
        }
        if (l.type == DETECTION)
        {
            get_detection_detections(l, w, h, thresh, dets);
            dets += l.w * l.h * l.n;
        }
    }
}

detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num)
{
    detection *dets = make_network_boxes(net, thresh, num);
    fill_network_boxes(net, w, h, thresh, hier, map, relative, dets);
    return dets;
}

void free_detections(detection *dets, int n)
{
    int i;
    for (i = 0; i < n; ++i)
    {
        free(dets[i].prob);
        if (dets[i].mask)
            free(dets[i].mask);
    }
    free(dets);
}

float *network_predict_image(network *net, image im)
{
    image imr = letterbox_image(im, net->w, net->h);
    set_batch_network(net, 1);
    float *p = network_predict(net, imr.data);
    free_image(imr);
    return p;
}

int network_width(network *net) { return net->w; }
int network_height(network *net) { return net->h; }

matrix network_predict_data_multi(network *net, data test, int n)
{
    int i, j, b, m;
    int k = net->outputs;
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net->batch * test.X.rows, sizeof(float));
    for (i = 0; i < test.X.rows; i += net->batch)
    {
        for (b = 0; b < net->batch; ++b)
        {
            if (i + b == test.X.rows)
                break;
            memcpy(X + b * test.X.cols, test.X.vals[i + b], test.X.cols * sizeof(float));
        }
        for (m = 0; m < n; ++m)
        {
            float *out = network_predict(net, X);
            for (b = 0; b < net->batch; ++b)
            {
                if (i + b == test.X.rows)
                    break;
                for (j = 0; j < k; ++j)
                {
                    pred.vals[i + b][j] += out[j + b * k] / n;
                }
            }
        }
    }
    free(X);
    return pred;
}

matrix network_predict_data(network *net, data test)
{
    int i, j, b;
    int k = net->outputs;
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net->batch * test.X.cols, sizeof(float));
    for (i = 0; i < net->batch; i += net->batch)
    {
        for (b = 0; b < net->batch; ++b)
        {
            if (i + b == test.X.rows)
                break;
            memcpy(X + b * test.X.cols, test.X.vals[i + b], test.X.cols * sizeof(float));
        }
        float *out = network_predict(net, X);
        for (b = 0; b < net->batch; ++b)
        {
            if (i + b == test.X.rows)
                break;
            for (j = 0; j < k; ++j)
            {
                pred.vals[i + b][j] = out[j + b * k];
            }
        }
    }
    free(X);
    return pred;
}

/* void print_network(network *net)
{
    int i,j;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        float *output = l.output;
        int n = l.outputs;
        float mean = mean_array(output, n);
        float vari = variance_array(output, n);
        fprintf(stderr, "Layer %d - Mean: %f, Variance: %f\n",i,mean, vari);
        if(n > 100) n = 100;
        for(j = 0; j < n; ++j) fprintf(stderr, "%f, ", output[j]);
        if(n == 100)fprintf(stderr,".....\n");
        fprintf(stderr, "\n");
    }
} */

void compare_networks(network *n1, network *n2, data test)
{
    matrix g1 = network_predict_data(n1, test);
    matrix g2 = network_predict_data(n2, test);
    int i;
    int a, b, c, d;
    a = b = c = d = 0;
    for (i = 0; i < g1.rows; ++i)
    {
        int truth = max_index(test.y.vals[i], test.y.cols);
        int p1 = max_index(g1.vals[i], g1.cols);
        int p2 = max_index(g2.vals[i], g2.cols);
        if (p1 == truth)
        {
            if (p2 == truth)
                ++d;
            else
                ++c;
        }
        else
        {
            if (p2 == truth)
                ++b;
            else
                ++a;
        }
    }
    printf("%5d %5d\n%5d %5d\n", a, b, c, d);
    float num = pow((abs(b - c) - 1.), 2.);
    float den = b + c;
    printf("%f\n", num / den);
}

float network_accuracy(network *net, data d)
{
    matrix guess = network_predict_data(net, d);
    float acc = matrix_topk_accuracy(d.y, guess, 1);
    free_matrix(guess);
    return acc;
}

// 新增：部分网络推理（只在enclave内执行前split_layer层）
float *network_predict_partial_inside(network *net, float *input, int split_layer)
{
    network orig = *net;
    net->input = input;
    net->truth = 0;
    net->train = 0;
    net->delta = 0;
    
    // 只执行前split_layer层
    for (int i = 0; i < split_layer && i < net->n; ++i) {
        net->index = i;
        layer l = net->layers[i];
        if (l.delta) {
            fill_cpu(l.outputs * l.batch, 0, l.delta, 1);
        }
        l.forward(l, *net);
        net->input = l.output;
        if (l.truth) {
            net->truth = l.output;
        }
    }
    
    // 返回中间结果
    float *intermediate = net->layers[split_layer-1].output;
    *net = orig;
    return intermediate;
}

// 新增：分层网络推理
matrix network_predict_data_split(network *net, data test, int split_layer)
{
    int i, j, b;
    int k = net->outputs;
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net->batch * test.X.cols, sizeof(float));
    
    for (i = 0; i < test.X.rows; i += net->batch) {
        for (b = 0; b < net->batch; ++b) {
            if (i + b == test.X.rows) break;
            memcpy(X + b * test.X.cols, test.X.vals[i + b], test.X.cols * sizeof(float));
        }
        
        // 在enclave内执行前split_layer层
        float *intermediate = network_predict_partial_inside(net, X, split_layer);
        int intermediate_size = net->layers[split_layer-1].outputs * net->batch;
        
        printf("Inference in enclave successfully\n");
        printf("intermediate_size: %d\n", intermediate_size);
        // 通过ocall在enclave外完成剩余推理
        printf("Calling ocall_network_predict_remaining\n");
        float *final_output = calloc(k * net->batch, sizeof(float));
        ocall_network_predict_remaining(intermediate, intermediate_size, split_layer, 
                                      net->batch, final_output, k * net->batch);
        
        // 保存最终结果
        printf("final_output: %f\n", final_output[0]);
        for (b = 0; b < net->batch; ++b) {
            if (i + b == test.X.rows) break;
            for (j = 0; j < k; ++j) {
                pred.vals[i + b][j] = final_output[j + b * k];
            }
        }
        
        free(final_output);
    }
    free(X);
    return pred;
}

float *network_accuracies(network *net, data d, int n)
{
    static float acc[2];
    matrix guess = network_predict_data(net, d);
    acc[0] = matrix_topk_accuracy(d.y, guess, 1);
    acc[1] = matrix_topk_accuracy(d.y, guess, n);
    free_matrix(guess);
    return acc;
}

// 新增：支持分层推理的准确率计算
float *network_accuracies_split(network *net, data d, int n, int split_layer)
{
    static float acc[2];
    matrix guess;
    
    if (split_layer > 0 && split_layer < net->n) {
        guess = network_predict_data_split(net, d, split_layer);
    } else {
        guess = network_predict_data(net, d);
    }
    
    acc[0] = matrix_topk_accuracy(d.y, guess, 1);
    acc[1] = matrix_topk_accuracy(d.y, guess, n);
    free_matrix(guess);
    return acc;
}

layer get_network_output_layer(network *net)
{
    int i;
    for (i = net->n - 1; i >= 0; --i)
    {
        if (net->layers[i].type != COST)
            break;
    }
    return net->layers[i];
}

float network_accuracy_multi(network *net, data d, int n)
{
    matrix guess = network_predict_data_multi(net, d, n);
    float acc = matrix_topk_accuracy(d.y, guess, 1);
    free_matrix(guess);
    return acc;
}

void free_network(network *net)
{
    int i;
    for (i = 0; i < net->n; ++i)
    {
        free_layer(net->layers[i]);
    }
    free(net->layers);
    if (net->input)
        free(net->input);
    if (net->truth)
        free(net->truth);

    free(net);
}

layer network_output_layer(network *net)
{
    int i;
    for (i = net->n - 1; i >= 0; --i)
    {
        if (net->layers[i].type != COST)
            break;
    }
    return net->layers[i];
}

int network_inputs(network *net)
{
    return net->layers[0].inputs;
}

int network_outputs(network *net)
{
    return network_output_layer(net).outputs;
}

float *network_output(network *net)
{
    return network_output_layer(net).output;
}
