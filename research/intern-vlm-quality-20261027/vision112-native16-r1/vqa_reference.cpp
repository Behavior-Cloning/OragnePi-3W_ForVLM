#include "llama.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>
using json=nlohmann::ordered_json;
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
static std::vector<llama_token> tok(const llama_vocab *v,const std::string &s){std::vector<llama_token> t(s.size()+32);int n=llama_tokenize(v,s.data(),s.size(),t.data(),t.size(),false,true);if(n<0){t.resize(-n);n=llama_tokenize(v,s.data(),s.size(),t.data(),t.size(),false,true);}if(n<0)throw std::runtime_error("tokenize");t.resize(n);return t;}
int main(int argc,char **argv){
 if(argc!=2)return 2;ggml_backend_load_all();llama_backend_init();auto mp=llama_model_default_params();mp.n_gpu_layers=0;auto model=llama_model_load_from_file(argv[1],mp);if(!model)return 3;
 auto cp=llama_context_default_params();cp.n_ctx=1024;cp.n_batch=512;cp.n_ubatch=256;cp.n_threads=6;cp.n_threads_batch=6;
 auto ctx=llama_init_from_model(model,cp);if(!ctx)return 4;auto vocab=llama_model_get_vocab(model);int dim=llama_model_n_embd(model);int nv=llama_vocab_n_tokens(vocab);if(dim!=896)return 5;
 std::cout<<json({{"event","ready"},{"scope","Windows WSL CPU Q8 language reference; not board/NPU latency"}}).dump()<<std::endl;
 std::string line;
 while(std::getline(std::cin,line))try{
  auto q=json::parse(line);if(q.value("op","")=="close")break;int count=q.at("image_tokens");if(count!=16&&count!=64&&count!=256)throw std::runtime_error("image token count");
  std::vector<float> emb(size_t(count)*dim);std::ifstream f(q.at("embedding").get<std::string>(),std::ios::binary);f.read((char*)emb.data(),emb.size()*4);if(!f||f.peek()!=EOF)throw std::runtime_error("embedding bytes");for(float x:emb)if(!std::isfinite(x))throw std::runtime_error("embedding finite");
  if(q.value("zero_features",false))std::fill(emb.begin(),emb.end(),0.f);
  std::string sys=q.at("system"),question=q.at("question");std::string sp="<|im_start|>system\n"+sys+"<|im_end|>\n";
  auto st=tok(vocab,sp),stbody=tok(vocab,sys),qt=tok(vocab,question),pre=tok(vocab,sp+"<|im_start|>user\n<img>"),tail=tok(vocab,"</img>\n"+question+"<|im_end|>\n<|im_start|>assistant\n");
  auto now=Clock::now();llama_memory_clear(llama_get_memory(ctx),true);
  if(llama_decode(ctx,llama_batch_get_one(pre.data(),pre.size())))throw std::runtime_error("prefix decode");
  auto b=llama_batch_init(count,dim,1);b.n_tokens=count;std::copy(emb.begin(),emb.end(),b.embd);for(int i=0;i<count;i++){b.pos[i]=pre.size()+i;b.n_seq_id[i]=1;b.seq_id[i][0]=0;b.logits[i]=0;}
  int rc=llama_decode(ctx,b);llama_batch_free(b);if(rc)throw std::runtime_error("vision embedding decode");
  if(llama_decode(ctx,llama_batch_get_one(tail.data(),tail.size())))throw std::runtime_error("tail decode");
  std::string answer;std::vector<int> ids;double ttft=0;bool eos=false;int maxout=q.value("max_output",32);
  for(int i=0;i<maxout;i++){
   float *logits=llama_get_logits_ith(ctx,-1);if(!logits)throw std::runtime_error("no logits");llama_token id=std::max_element(logits,logits+nv)-logits;
   if(i==0)ttft=ms(now);if(llama_vocab_is_eog(vocab,id)){eos=true;break;}ids.push_back(id);
   char buf[256];int n=llama_token_to_piece(vocab,id,buf,sizeof(buf),0,false);if(n<0)throw std::runtime_error("piece length");answer.append(buf,n);
   if(i+1<maxout&&llama_decode(ctx,llama_batch_get_one(&id,1)))throw std::runtime_error("decode");
  }
  json out={{"event","answer"},{"id",q.at("id")},{"question",question},{"answer",answer},{"output_ids",ids},{"eos",eos},{"system_body_tokens",stbody.size()},{"system_template_tokens",st.size()-stbody.size()},{"system_total_tokens",st.size()},{"dynamic_text_tokens",qt.size()},{"image_tokens",count},{"dynamic_template_and_boundaries",pre.size()+tail.size()-st.size()-qt.size()},{"total_input_tokens",pre.size()+count+tail.size()},{"reused_tokens",0},{"reference_cpu_ttft_ms",ttft},{"reference_cpu_total_ms",ms(now)}};
  std::cout<<out.dump(-1,' ',false,json::error_handler_t::replace)<<std::endl;
 }catch(const std::exception&e){std::cout<<json({{"event","error"},{"error",e.what()}}).dump()<<std::endl;}
 llama_free(ctx);llama_model_free(model);llama_backend_free();return 0;
}
