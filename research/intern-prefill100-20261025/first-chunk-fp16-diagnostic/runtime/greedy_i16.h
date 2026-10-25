#ifndef OPI_GREEDY_I16_H
#define OPI_GREEDY_I16_H
#include <stdint.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
/* Count must be positive. Return the earliest maximum, including all-negative
 * vectors and ties crossing vector blocks. DFP scale must be shared per shard. */
static unsigned greedy_i16(const int16_t *values,unsigned count) {
    int16_t maximum=INT16_MIN;unsigned i=0;
#if defined(__aarch64__)
    int16x8_t largest=vdupq_n_s16(INT16_MIN);
    for(;i+8<=count;i+=8)largest=vmaxq_s16(largest,vld1q_s16(values+i));
    maximum=vmaxvq_s16(largest);
#endif
    for(;i<count;i++)if(values[i]>maximum)maximum=values[i];
    for(i=0;i<count;i++)if(values[i]==maximum)return i;
    return 0;
}
#endif
