/* Bounded VIPLite buffer allocation probe; not an OOM/max-VRAM test.
 * Build: cc -O2 -Wall -Wextra memory_probe.c -lNBGlinker -lVIPhal -o memory_probe
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <vip_lite.h>

int main(void) {
    const unsigned sizes_mib[] = {16, 64, 256};
    vip_status_e status = vip_init();
    if (status != VIP_SUCCESS) { fprintf(stderr, "vip_init=%d\n", status); return 1; }
    int failed = 0;
    for (unsigned i = 0; i < sizeof(sizes_mib) / sizeof(sizes_mib[0]); ++i) {
        vip_buffer_create_params_t params;
        memset(&params, 0, sizeof(params));
        params.num_of_dims = 3;
        params.sizes[0] = 1024;
        params.sizes[1] = 1024;
        params.sizes[2] = sizes_mib[i];
        params.data_format = VIP_BUFFER_FORMAT_UINT8;
        params.quant_format = VIP_BUFFER_QUANTIZE_NONE;
        vip_buffer buffer = NULL;
        status = vip_create_buffer(&params, sizeof(params), &buffer);
        printf("requested_mib=%u create_status=%d", sizes_mib[i], status);
        if (status == VIP_SUCCESS) {
            unsigned bytes = vip_get_buffer_size(buffer);
            volatile unsigned char *mapped = vip_map_buffer(buffer);
            if (!mapped) { failed = 1; printf(" map=FAIL"); }
            else {
                for (size_t offset = 0; offset < bytes; offset += 4096) mapped[offset] = 0x5a;
                unsigned errors = 0;
                for (size_t offset = 0; offset < bytes; offset += 4096) errors += mapped[offset] != 0x5a;
                status = vip_flush_buffer(buffer, VIP_BUFFER_OPER_TYPE_FLUSH);
                printf(" allocated_bytes=%u page_check_errors=%u flush_status=%d", bytes, errors, status);
                failed |= (errors != 0 || status != VIP_SUCCESS);
                failed |= vip_unmap_buffer(buffer) != VIP_SUCCESS;
            }
            failed |= vip_destroy_buffer(buffer) != VIP_SUCCESS;
        } else failed = 1;
        printf("\n");
        if (failed) break;
    }
    failed |= vip_destroy() != VIP_SUCCESS;
    return failed;
}
