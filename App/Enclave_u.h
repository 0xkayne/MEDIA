#ifndef ENCLAVE_U_H__
#define ENCLAVE_U_H__

#include <stdint.h>
#include <wchar.h>
#include <stddef.h>
#include <string.h>
#include "sgx_edger8r.h" /* for sgx_status_t etc. */

#include "dnet_types.h"

#include <stdlib.h> /* for size_t */

#define SGX_CAST(type, item) ((type)(item))

#ifdef __cplusplus
extern "C" {
#endif

#ifndef OCALL_OPEN_FILE_DEFINED__
#define OCALL_OPEN_FILE_DEFINED__
void SGX_UBRIDGE(SGX_NOCONVENTION, ocall_open_file, (const char* filename, flag oflag));
#endif
#ifndef OCALL_CLOSE_FILE_DEFINED__
#define OCALL_CLOSE_FILE_DEFINED__
void SGX_UBRIDGE(SGX_NOCONVENTION, ocall_close_file, (void));
#endif
#ifndef OCALL_FREAD_DEFINED__
#define OCALL_FREAD_DEFINED__
void SGX_UBRIDGE(SGX_NOCONVENTION, ocall_fread, (void* ptr, size_t size, size_t nmemb));
#endif
#ifndef OCALL_FWRITE_DEFINED__
#define OCALL_FWRITE_DEFINED__
void SGX_UBRIDGE(SGX_NOCONVENTION, ocall_fwrite, (void* ptr, size_t size, size_t nmemb));
#endif
#ifndef OCALL_PRINT_STRING_DEFINED__
#define OCALL_PRINT_STRING_DEFINED__
void SGX_UBRIDGE(SGX_NOCONVENTION, ocall_print_string, (const char* str));
#endif
#ifndef OCALL_NETWORK_PREDICT_REMAINING_DEFINED__
#define OCALL_NETWORK_PREDICT_REMAINING_DEFINED__
void SGX_UBRIDGE(SGX_NOCONVENTION, ocall_network_predict_remaining, (float* intermediate_data, int intermediate_size, int split_layer, int batch_size, float* final_output, int output_size));
#endif
#ifndef SGX_OC_CPUIDEX_DEFINED__
#define SGX_OC_CPUIDEX_DEFINED__
void SGX_UBRIDGE(SGX_CDECL, sgx_oc_cpuidex, (int cpuinfo[4], int leaf, int subleaf));
#endif
#ifndef SGX_THREAD_WAIT_UNTRUSTED_EVENT_OCALL_DEFINED__
#define SGX_THREAD_WAIT_UNTRUSTED_EVENT_OCALL_DEFINED__
int SGX_UBRIDGE(SGX_CDECL, sgx_thread_wait_untrusted_event_ocall, (const void* self));
#endif
#ifndef SGX_THREAD_SET_UNTRUSTED_EVENT_OCALL_DEFINED__
#define SGX_THREAD_SET_UNTRUSTED_EVENT_OCALL_DEFINED__
int SGX_UBRIDGE(SGX_CDECL, sgx_thread_set_untrusted_event_ocall, (const void* waiter));
#endif
#ifndef SGX_THREAD_SETWAIT_UNTRUSTED_EVENTS_OCALL_DEFINED__
#define SGX_THREAD_SETWAIT_UNTRUSTED_EVENTS_OCALL_DEFINED__
int SGX_UBRIDGE(SGX_CDECL, sgx_thread_setwait_untrusted_events_ocall, (const void* waiter, const void* self));
#endif
#ifndef SGX_THREAD_SET_MULTIPLE_UNTRUSTED_EVENTS_OCALL_DEFINED__
#define SGX_THREAD_SET_MULTIPLE_UNTRUSTED_EVENTS_OCALL_DEFINED__
int SGX_UBRIDGE(SGX_CDECL, sgx_thread_set_multiple_untrusted_events_ocall, (const void** waiters, size_t total));
#endif
#ifndef OCALL_FREE_SEC_DEFINED__
#define OCALL_FREE_SEC_DEFINED__
void SGX_UBRIDGE(SGX_NOCONVENTION, ocall_free_sec, (section* sec));
#endif
#ifndef OCALL_FREE_LIST_DEFINED__
#define OCALL_FREE_LIST_DEFINED__
void SGX_UBRIDGE(SGX_NOCONVENTION, ocall_free_list, (list* list));
#endif

sgx_status_t empty_ecall(sgx_enclave_id_t eid);
sgx_status_t ecall_trainer(sgx_enclave_id_t eid, list* sections, data* training_data, int pmem);
sgx_status_t ecall_tester(sgx_enclave_id_t eid, list* sections, data* test_data, int pmem);
sgx_status_t ecall_classify(sgx_enclave_id_t eid, list* sections, list* labels, image* im);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
