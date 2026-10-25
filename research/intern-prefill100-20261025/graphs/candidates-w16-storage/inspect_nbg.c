#include <stdio.h>
#include <stdlib.h>
#include <VX/vx.h>
#include <VX/vx_khr_import_kernel.h>
#define OK(call) do{vx_status rc=(call);if(rc!=VX_SUCCESS){fprintf(stderr,"QUERY_FAIL line%d status%d\n",__LINE__,(int)rc);return 3;}}while(0)
int main(int argc,char **argv){
 if(argc!=2)return 2;
 vx_context ctx=vxCreateContext();OK(vxGetStatus((vx_reference)ctx));
 vx_kernel k=vxImportKernelFromURL(ctx,VX_VIVANTE_IMPORT_KERNEL_FROM_FILE,argv[1]);OK(vxGetStatus((vx_reference)k));
 vx_uint32 n=0;OK(vxQueryKernel(k,VX_KERNEL_PARAMETERS,&n,sizeof(n)));
 printf("{\"count\":%u,\"ports\":[",n);
 for(vx_uint32 i=0;i<n;i++){
  vx_parameter p=vxGetKernelParameterByIndex(k,i);OK(vxGetStatus((vx_reference)p));
  vx_enum dir=0,type=0,dtype=0,quant=0;vx_meta_format m=NULL;vx_size nd=0,dims[8]={0};vx_int8 fl=0;vx_int32 zp=0;vx_float32 scale=0;
  OK(vxQueryParameter(p,VX_PARAMETER_DIRECTION,&dir,sizeof(dir)));OK(vxQueryParameter(p,VX_PARAMETER_TYPE,&type,sizeof(type)));OK(vxQueryParameter(p,VX_PARAMETER_META_FORMAT,&m,sizeof(m)));
  OK(vxQueryMetaFormatAttribute(m,VX_TENSOR_NUMBER_OF_DIMS,&nd,sizeof(nd)));if(nd>8)return 4;
  OK(vxQueryMetaFormatAttribute(m,VX_TENSOR_DIMS,dims,nd*sizeof(vx_size)));
  OK(vxQueryMetaFormatAttribute(m,VX_TENSOR_DATA_TYPE,&dtype,sizeof(dtype)));
  vx_status q=vxQueryMetaFormatAttribute(m,VX_TENSOR_QUANT_FORMAT,&quant,sizeof(quant));
  vx_status f=vxQueryMetaFormatAttribute(m,VX_TENSOR_FIXED_POINT_POSITION,&fl,sizeof(fl));
  vx_status z=vxQueryMetaFormatAttribute(m,VX_TENSOR_ZERO_POINT,&zp,sizeof(zp));
  vx_status s=vxQueryMetaFormatAttribute(m,VX_TENSOR_SCALE,&scale,sizeof(scale));
  printf("%s{\"index\":%u,\"direction\":%d,\"type\":%d,\"dtype\":%d,\"dims\":[",i?",":"",i,(int)dir,(int)type,(int)dtype);
  for(vx_size j=0;j<nd;j++)printf("%s%zu",j?",":"",(size_t)dims[j]);
  printf("],\"quant_status\":%d,\"quant_type\":%d,\"fl_status\":%d,\"fl\":%d,\"zp_status\":%d,\"zp\":%d,\"scale_status\":%d,\"scale\":%.17g}",(int)q,(int)quant,(int)f,(int)fl,(int)z,(int)zp,(int)s,(double)scale);
  vxReleaseParameter(&p);
 }
 printf("]}\n");vxReleaseKernel(&k);vxReleaseContext(&ctx);return 0;
}
