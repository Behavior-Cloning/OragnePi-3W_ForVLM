#define _POSIX_C_SOURCE 200809L
#define __fp16 uint16_t /* Host only: opaque16bit storage; put emulates two exact IEEE constants. */
#include <vip_lite.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
typedef struct { vip_buffer b; vip_buffer_create_params_t p; void *data; unsigned count; } Tensor;
static void put(Tensor *t,unsigned i,float x) {
    if(i>=t->count) {fprintf(stderr,"tensor index overflow\n");exit(2);}
    if(t->p.data_format==VIP_BUFFER_FORMAT_FP16) {if(x!=0 && x!=-10000)exit(9);uint16_t bits=x==0?0:0xf0e2;memcpy((unsigned char*)t->data+2*i,&bits,2);return;}
    if(t->p.data_format==VIP_BUFFER_FORMAT_FP32) {((float *)t->data)[i]=x;return;}
    if(t->p.data_format==VIP_BUFFER_FORMAT_INT8 && t->p.quant_format==VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) {
        float y=roundf(ldexpf(x,t->p.quant_data.dfp.fixed_point_pos));
        ((int8_t *)t->data)[i]=(int8_t)fmaxf(-128,fminf(127,y));return;
    }
    if(t->p.data_format==VIP_BUFFER_FORMAT_INT16 && t->p.quant_format==VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) {
        float y=roundf(ldexpf(x,t->p.quant_data.dfp.fixed_point_pos));
        ((int16_t *)t->data)[i]=(int16_t)fmaxf(-32768,fminf(32767,y));return;
    }
    fprintf(stderr,"unsupported input dtype/quant\n");exit(2);
}
static int same_tensor_contract(const Tensor *a,const Tensor *b) {
    if(a->count!=b->count||a->p.num_of_dims!=b->p.num_of_dims||
       a->p.data_format!=b->p.data_format||a->p.quant_format!=b->p.quant_format||
       memcmp(a->p.sizes,b->p.sizes,a->p.num_of_dims*sizeof(a->p.sizes[0])))return 0;
    if(a->p.quant_format==VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT)
        return a->p.quant_data.dfp.fixed_point_pos==b->p.quant_data.dfp.fixed_point_pos;
    return a->p.quant_format==VIP_BUFFER_QUANTIZE_NONE;
}

#include "mask_templates.h"
static unsigned cases=0;
static void exercise(int format,int fl,int c,int n,int repeat){
 Tensor a={0},b={0};a.p.data_format=format;a.p.quant_format=(format==VIP_BUFFER_FORMAT_INT8||format==VIP_BUFFER_FORMAT_INT16)?VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT:VIP_BUFFER_QUANTIZE_NONE;a.p.quant_data.dfp.fixed_point_pos=fl;a.count=n?repeat*n*(c+n):c+1;a.p.num_of_dims=4;a.p.sizes[0]=n?c+n:c+1;a.p.sizes[1]=n?n:1;a.p.sizes[2]=n?repeat:1;a.p.sizes[3]=1;b=a;
 size_t bytes=a.count*mask_width(&a);unsigned char *buf=malloc(bytes+32);a.data=buf+16;b.data=malloc(bytes);if(!buf||!b.data)exit(2);
 int keeps[]={0,1,7,80,80,86,94,c-1,c,0,80};
 for(unsigned ki=0;ki<sizeof(keeps)/sizeof(keeps[0]);ki++){
  int keep=keeps[ki];if(keep>c)continue;
  memset(buf,0xa5,bytes+32);
  if(!n){for(int j=0;j<c;j++)put(&b,j,j<keep?0:-10000);put(&b,c,0);}
  else for(int r=0;r<repeat;r++)for(int t=0;t<n;t++)for(int j=0;j<c+n;j++)put(&b,(r*n+t)*(c+n)+j,(j<keep||(j>=c&&j<=c+t))?0:-10000);
  restore_mask_template(&a,c,n,repeat,keep);
  if(memcmp(a.data,b.data,bytes)){fprintf(stderr,"mismatch format%d fl%d n%d keep%d\n",format,fl,n,keep);exit(3);}
  for(int j=0;j<16;j++)if(buf[j]!=0xa5||buf[16+bytes+j]!=0xa5)exit(4);
  cases++;
 }
 free(buf);free(b.data);
}
int main(void){
 int formats[]={VIP_BUFFER_FORMAT_FP16,VIP_BUFFER_FORMAT_FP32,VIP_BUFFER_FORMAT_INT8,VIP_BUFFER_FORMAT_INT16};int scales[]={-12,-4,0,3,15};
 for(unsigned i=0;i<4;i++)for(unsigned j=0;j<(i<2?1:5);j++){
  exercise(formats[i],scales[j],8,0,1);exercise(formats[i],scales[j],8,4,3);exercise(formats[i],scales[j],1280,0,1);exercise(formats[i],scales[j],1280,16,7);
 }
 printf("{\"cases\":%u,\"mismatches\":0,\"guard_bytes_intact\":true,\"allocated_template_bytes\":%zu,\"validation_mode\":%s}\n",cases,mask_template_bytes,getenv("OPI_MASK_TEMPLATE_VALIDATE")?"true":"false");free_mask_templates();if(mask_templates||mask_template_bytes)exit(5);return 0;
}
