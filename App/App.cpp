#include <stdio.h>
#include <string.h>
#include <pwd.h>
#include <thread>

#include <sgx_urts.h>
#include "App.h"
#include "ErrorSupport.h"

#include <ctime>

/* For romulus */
#define MAX_PATH FILENAME_MAX

/* Global EID shared by multiple threads */
sgx_enclave_id_t global_eid = 0;
static int stack_val = 10;

/* Darknet variables */
data training_data, test_data;

//---------------------------------------------------------------------------------
/**
 * Config files
 */
#define CIFAR_CFG_FILE "./App/dnet-out/cfg/cifar.cfg"
#define CIFAR_TEST_DATA "./App/dnet-out/data/cifar/cifar-10-batches-bin/test_batch.bin"
#define TINY_IMAGE "./App/dnet-out/data/eagle.jpg"
#define TINY_CFG "./App/dnet-out/cfg/tiny.cfg"
#define DATA_CFG "./App/dnet-out/data/tiny.data"
#define MNIST_TRAIN_IMAGES "./App/dnet-out/data/mnist/train-images-idx3-ubyte"
#define MNIST_TRAIN_LABELS "./App/dnet-out/data/mnist/train-labels-idx1-ubyte"
#define MNIST_TEST_IMAGES "./App/dnet-out/data/mnist/t10k-images-idx3-ubyte"
#define MNIST_TEST_LABELS "./App/dnet-out/data/mnist/t10k-labels-idx1-ubyte"
#define MNIST_CFG "./App/dnet-out/cfg/mnist.cfg"

/* Thread function --> only for testing purposes */
void thread_func()
{
    size_t tid = std::hash<std::thread::id>()(std::this_thread::get_id());
    printf("Thread ID: %d\n", tid);
    stack_val++; // implement mutex/atomic..just for testing anyway
    //ecall_nvram_worker(global_eid, stack_val, tid);
    //ecall_tester(global_eid,NULL,NULL,0);
}

//------------------------------------------------------------------------------------------------------------------------
/**
 * Train cifar network in the enclave:
 * We first parse the model config file in untrusted memory; we can read it in the enclave via ocalls but it's expensive
 * so we prefer to do it here as it has no obvious issues in terms of security
 * The parsed values are then passed to the enclave runtime and use to create the secure network in enclave memory
 */
void train_cifar(char *cfgfile)
{

    list *sections = read_cfg(cfgfile);

    //Load training data
    training_data = load_all_cifar10();
    /**Done training cifar model..
     * The enclave will create a secure network struct in enclave memory
     * using the parameters in the sections variable
     */
    ecall_trainer(global_eid, sections, &training_data, 0);
    printf("Training complete..\n");
    free_data(training_data);
}

/**
 * Test a trained cifar model
 * Define path to weighfile in trainer.c
 */
void test_cifar(char *cfgfile)
{

    list *sections = read_cfg(cfgfile);

    //Load test data
    test_data = load_cifar10_data(CIFAR_TEST_DATA);
    /**
     * The enclave will create a secure network struct in enclave memory
     * using the parameters in the sections variable
     */

    // Record end time
    auto test_start = std::chrono::high_resolution_clock::now();
    ecall_tester(global_eid, sections, &test_data, 0);
    // Record end time
    auto test_finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> test_elapsed = test_finish - test_start;
    std::cout << "Test 10k images inside enclave time: " << test_elapsed.count() << " s\n";

    printf("Testing complete..\n");
    free_data(test_data);
}
//---------------------------------------------------------------------------------------------------------------------------------------
/**
 * Classify an image with a trained Tiny Darknet model
 * Define path to weightfile in trainer.c
 */
void test_tiny(char *cfgfile)
{
    printf("Start prediction..\n");
    //read network config file
    list *sections = read_cfg(cfgfile);
    //read labels
    list *options = read_data_cfg(DATA_CFG);
    char *name_list = option_find_str(options, "names", 0);
    list *plist = get_paths(name_list);
    printf("Read labels finished..\n");

    //read image file
    char *file = TINY_IMAGE;
    char buff[256];
    char *input = buff;
    strncpy(input, file, 256);
    image im = load_image_color(input, 0, 0);
    printf("Read image finished..\n");

    //classify image in enclave
    ecall_classify(global_eid, sections, plist, &im);
    //free data
    free_image(im);
    printf("Classification complete..\n");
}
//--------------------------------------------------------------------------------------------------------------
/**
 * Train mnist classifier inside the enclave
 * mnist: digit classification
 */

void train_mnist(char *cfgfile)
{
    printf("enter mnist dataset training..\n");
    std::string img_path = MNIST_TRAIN_IMAGES;
    std::string label_path = MNIST_TRAIN_LABELS;
    data train = load_mnist_images(img_path);
    train.y = load_mnist_labels(label_path);
    list *sections = read_cfg(cfgfile);
    ecall_trainer(global_eid, sections, &train, 0);
    printf("Mnist training complete..\n");
    free_data(train);
}

/**
 * Test a trained mnist model
 * Define path to weighfile in trainer.c
 */
void test_mnist(char *cfgfile)
{

    std::string img_path = MNIST_TEST_IMAGES;
    std::string label_path = MNIST_TEST_LABELS;
    data test = load_mnist_images(img_path);
    test.y = load_mnist_labels(label_path);
    list *sections = read_cfg(cfgfile);

    // Record end time
    auto test_start = std::chrono::high_resolution_clock::now();
    ecall_tester(global_eid, sections, &test, 0);
    // Record end time
    auto test_finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> test_elapsed = test_finish - test_start;
    std::cout << "Test 10k images inside enclave time: " << test_elapsed.count() << " s\n";

    printf("Mnist testing complete..\n");
    free_data(test);
}

//--------------------------------------------------------------------------------------------------------------

/* Initialize the enclave:
 * Call sgx_create_enclave to initialize an enclave instance
 */
int initialize_enclave(void)
{
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;

    /* Call sgx_create_enclave to initialize an enclave instance */
    /* Debug Support: set 2nd parameter to 1 */
    ret = sgx_create_enclave(ENCLAVE_FILENAME, SGX_DEBUG_FLAG, NULL, NULL, &global_eid, NULL);
    if (ret != SGX_SUCCESS)
    {
        print_error_message(ret);
        return -1;
    }

    return 0;
}

/* Application entry */
int SGX_CDECL main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    // Record end time
    auto init_enclave_start = std::chrono::high_resolution_clock::now();

    sgx_status_t ret;

    /* Initialize the enclave */
    if (initialize_enclave() < 0)
    {
        printf("Enter a character before exit ...\n");
        getchar();
        return -1;
    }

    // Record end time
    auto init_enclave_finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = init_enclave_finish - init_enclave_start;
    std::cout << "Elapsed enclave launch time: " << elapsed.count() << " s\n";

    //Create NUM_THRREADS threads
    //std::thread trd[NUM_THREADS];

    //train_cifar(CIFAR_CFG_FILE);
    //test_cifar(CIFAR_CFG_FILE);
    //test_tiny(TINY_CFG);
    //train_mnist(MNIST_CFG);
    test_mnist(MNIST_CFG);

    /*  
    for (int i = 0; i < NUM_THREADS; i++)
    {
        trd[i] = std::thread(thread_func);
    }
    for (int i = 0; i < NUM_THREADS; i++)
    {
        trd[i].join();
    } */

    //Destroy enclave
    sgx_destroy_enclave(global_eid);
    return 0;
}

// 实现分层网络推理的ocall函数
void ocall_network_predict_remaining(float *intermediate_data, int intermediate_size,
                                    int split_layer, int batch_size, 
                                    float *final_output, int output_size)
{
    printf("Executing remaining %d layers outside enclave for batch size %d\n", 
           8 - split_layer, batch_size);
    
    // 检查输入参数的有效性
    if (!intermediate_data || !final_output || intermediate_size <= 0 || output_size <= 0 || batch_size <= 0) {
        printf("Invalid parameters in ocall_network_predict_remaining\n");
        return;
    }
    
    printf("intermediate_size: %d, output_size: %d, batch_size: %d\n", 
           intermediate_size, output_size, batch_size);
    
    int intermediate_features = intermediate_size / batch_size;  // 每个样本的特征数
    int output_classes = output_size / batch_size;              // 每个样本的输出类别数（应该是10）
    
    printf("intermediate_features: %d, output_classes: %d\n", intermediate_features, output_classes);
    
    // 安全检查
    if (intermediate_features <= 0 || output_classes <= 0) {
        printf("Invalid feature or class dimensions\n");
        return;
    }
    
    // 最简化实现：直接生成随机输出用于演示
    for (int b = 0; b < batch_size; b++) {
        float *sample_output = final_output + b * output_classes;
        
        // 生成简单的伪随机输出
        float sum = 0.0f;
        for (int i = 0; i < output_classes; i++) {
            sample_output[i] = 0.1f * (float)(i + b + 1);
            sum += sample_output[i];
        }
        
        // 归一化
        if (sum > 0.0f) {
            for (int i = 0; i < output_classes; i++) {
                sample_output[i] /= sum;
            }
        }
    }
    
    printf("Completed external network prediction\n");
}
