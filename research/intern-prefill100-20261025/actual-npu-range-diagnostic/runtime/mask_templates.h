/* Host-only exact raw mask templates. No graph/KV/flush/partition changes. */
#ifndef OPI_MASK_TEMPLATES_H
#define OPI_MASK_TEMPLATES_H
#include <stddef.h>
typedef struct MaskTemplate {
    Tensor contract;
    int capacity,tokens,repeat,keep;
    size_t bytes,width;
    unsigned char *raw;
    struct MaskTemplate *next;
} MaskTemplate;
static MaskTemplate *mask_templates=NULL;
static size_t mask_template_bytes=0;
static size_t mask_width(const Tensor *m) {
    switch(m->p.data_format){
    case VIP_BUFFER_FORMAT_INT8:return 1;
    case VIP_BUFFER_FORMAT_INT16:case VIP_BUFFER_FORMAT_FP16:return 2;
    case VIP_BUFFER_FORMAT_FP32:return 4;
    default:fprintf(stderr,"unsupported mask format\n");exit(2);
    }
}
/* Encode with exactly the old put(), including clamping and DFP rounding. */
static void mask_encode(const Tensor *m,float x,unsigned char *out) {
    union {uint32_t word;float f;__fp16 h;int16_t i16;int8_t i8;unsigned char bytes[4];} storage={0};
    Tensor one=*m;one.data=&storage;one.count=1;put(&one,0,x);
    memcpy(out,&storage,mask_width(m));
}
static void mask_fill(unsigned char *dst,size_t n,size_t width,const unsigned char *bits) {
    if(width==1){memset(dst,bits[0],n);return;}
    if(width==2){uint16_t x;memcpy(&x,bits,2);uint16_t *p=(uint16_t*)dst;for(size_t i=0;i<n;i++)p[i]=x;return;}
    if(width==4){uint32_t x;memcpy(&x,bits,4);uint32_t *p=(uint32_t*)dst;for(size_t i=0;i<n;i++)p[i]=x;return;}
    exit(2);
}
static void mask_template_build(MaskTemplate *entry,const Tensor *m,int keep) {
    int c=entry->capacity,n=entry->tokens;
    unsigned char zero[4]={0},blocked[4]={0};mask_encode(m,0,zero);mask_encode(m,-10000,blocked);
    if(!n){
        mask_fill(entry->raw,c+1,entry->width,blocked);
        mask_fill(entry->raw,keep,entry->width,zero);
        mask_fill(entry->raw+(size_t)c*entry->width,1,entry->width,zero);
    }else{
        size_t stride=(size_t)(c+n)*entry->width,block=(size_t)n*stride;
        for(int t=0;t<n;t++){
            unsigned char *row=entry->raw+(size_t)t*stride;
            mask_fill(row,c+n,entry->width,blocked);
            mask_fill(row,keep,entry->width,zero);
            mask_fill(row+(size_t)c*entry->width,t+1,entry->width,zero);
        }
        for(int r=1;r<entry->repeat;r++)memcpy(entry->raw+(size_t)r*block,entry->raw,block);
    }
    entry->keep=keep;
    if(getenv("OPI_MASK_TEMPLATE_VALIDATE")){
        Tensor reference=*m;reference.data=malloc(entry->bytes);if(!reference.data)exit(2);
        if(!n){for(int j=0;j<c;j++)put(&reference,j,j<keep?0:-10000);put(&reference,c,0);}
        else for(int r=0;r<entry->repeat;r++)for(int t=0;t<n;t++)for(int j=0;j<c+n;j++)
            put(&reference,(r*n+t)*(c+n)+j,(j<keep||(j>=c&&j<=c+t))?0:-10000);
        int diff=memcmp(entry->raw,reference.data,entry->bytes);free(reference.data);
        if(diff){fprintf(stderr,"mask template differs from original put loop\n");exit(3);}
    }
}
static void restore_mask_template(Tensor *m,int capacity,int tokens,int repeat,int keep) {
    size_t count=tokens?(size_t)repeat*tokens*(capacity+tokens):(size_t)capacity+1;
    if(keep<0||keep>capacity||repeat<1||tokens<0||count!=m->count){fprintf(stderr,"mask template contract mismatch\n");exit(2);}
    MaskTemplate *entry=mask_templates;
    while(entry){
        if(entry->capacity==capacity&&entry->tokens==tokens&&entry->repeat==repeat&&same_tensor_contract(&entry->contract,m))break;
        entry=entry->next;
    }
    if(!entry){
        entry=calloc(1,sizeof(*entry));if(!entry)exit(2);
        entry->contract=*m;entry->contract.data=NULL;entry->capacity=capacity;entry->tokens=tokens;entry->repeat=repeat;entry->keep=-1;entry->width=mask_width(m);entry->bytes=count*entry->width;
        entry->raw=malloc(entry->bytes);if(!entry->raw)exit(2);
        entry->next=mask_templates;mask_templates=entry;mask_template_bytes+=entry->bytes;
    }
    if(entry->keep!=keep)mask_template_build(entry,m,keep);
    memcpy(m->data,entry->raw,entry->bytes);
}
static void free_mask_templates(void) {
    while(mask_templates){MaskTemplate *next=mask_templates->next;free(mask_templates->raw);free(mask_templates);mask_templates=next;}
    mask_template_bytes=0;
}
#endif
