/* Close-only ownership barrier: all groups, all networks, then owned buffers.
 * Aliased inputs do not transfer ownership. No network may outlive any user
 * buffer it has been bound to. This is not an inference-path change. */
#ifndef OPI_SESSION_RELEASE_H
#define OPI_SESSION_RELEASE_H
static void release_group_once(vip_group *group) {
    if(*group){check(vip_destroy_group(*group),"release session group");*group=NULL;}
}
static void release_network_once(Layer *layer) {
    if(layer->net){check(vip_destroy_network(layer->net),"release session network");layer->net=NULL;}
}
static void release_owned_tensor_once(Tensor *t) {
    if(t->b){
        check(vip_unmap_buffer(t->b),"unmap owned session tensor");
        check(vip_destroy_buffer(t->b),"release owned session tensor");
        t->b=NULL;t->data=NULL;
    }
}
static void release_owned_layer_tensors(Layer *layer,unsigned ni,unsigned no) {
    for(unsigned i=0;i<ni;i++)release_owned_tensor_once(&layer->in[i]);
    for(unsigned i=0;i<no;i++)release_owned_tensor_once(&layer->out[i]);
}
static void release_session_networks_and_buffers(Session *s) {
    BatchPrefill *banks[3]={&s->tail8,&s->batch,&s->compact};
    fprintf(stderr,"CLOSE phase=groups\n");
    release_group_once(&s->decoder_group);release_group_once(&s->head_group);
    for(unsigned b=0;b<3;b++){
        if(banks[b]->grouped)release_group_once(&banks[b]->group);
        banks[b]->grouped=0;
    }
    fprintf(stderr,"CLOSE phase=networks\n");
    /* Keep every user buffer alive until every possible alias consumer is gone. */
    for(unsigned b=0;b<3;b++){
        BatchPrefill *bank=banks[b];
        for(int i=bank->blocks-1;i>=0;i--)release_network_once(&bank->layers[i]);
        if(bank->scalar)release_network_once(bank->scalar);
    }
    for(int i=HEAD_SHARDS-1;i>=0;i--)release_network_once(&s->head[i]);
    for(int i=s->blocks-1;i>=0;i--)release_network_once(&s->decoder[i]);
    fprintf(stderr,"CLOSE phase=owned_buffers\n");
    for(unsigned b=0;b<3;b++){
        BatchPrefill *bank=banks[b];
        for(int i=0;i<bank->blocks;i++)release_owned_layer_tensors(&bank->layers[i],TENSOR_MAX_INPUTS,TENSOR_MAX_OUTPUTS);
        if(bank->scalar){release_owned_layer_tensors(bank->scalar,TENSOR_MAX_INPUTS,TENSOR_MAX_OUTPUTS);free(bank->scalar);bank->scalar=NULL;}
        free(bank->layers);bank->layers=NULL;bank->blocks=0;bank->tokens=0;
    }
    for(int i=0;i<s->blocks;i++)release_owned_layer_tensors(&s->decoder[i],TENSOR_MAX_INPUTS,TENSOR_MAX_OUTPUTS);
    for(int i=0;i<HEAD_SHARDS;i++)release_owned_layer_tensors(&s->head[i],1,1);
    fprintf(stderr,"CLOSE phase=owned_buffers_done\n");
}
#endif
