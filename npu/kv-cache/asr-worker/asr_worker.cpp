// Resident CPU ASR. The pinned 16 kHz mono PCM recording is decoded repeatedly;
// model/audio loading happens once, outside the timed inference calls.
#include "whisper.h"
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static uint32_t u32(const unsigned char *p) {
    return uint32_t(p[0]) | uint32_t(p[1])<<8 | uint32_t(p[2])<<16 | uint32_t(p[3])<<24;
}
static uint16_t u16(const unsigned char *p) { return uint16_t(p[0]) | uint16_t(p[1])<<8; }
static std::vector<float> read_pcm(const char *path) {
    std::ifstream file(path,std::ios::binary);
    if(!file) throw std::runtime_error("audio open failed");
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(file)),{});
    if(bytes.size()<12 || std::memcmp(bytes.data(),"RIFF",4) || std::memcmp(bytes.data()+8,"WAVE",4))
        throw std::runtime_error("invalid WAV");
    bool format=false;
    std::vector<float> pcm;
    for(size_t pos=12;pos+8<=bytes.size();) {
        size_t length=u32(bytes.data()+pos+4),start=pos+8;
        if(length>bytes.size()-start) throw std::runtime_error("truncated WAV");
        if(!std::memcmp(bytes.data()+pos,"fmt ",4)) {
            if(length<16 || u16(bytes.data()+start)!=1 || u16(bytes.data()+start+2)!=1 ||
               u32(bytes.data()+start+4)!=16000 || u16(bytes.data()+start+14)!=16)
                throw std::runtime_error("expected mono 16 kHz PCM16");
            format=true;
        }
        if(!std::memcmp(bytes.data()+pos,"data",4)) {
            if(!format || length%2) throw std::runtime_error("invalid PCM chunk");
            for(size_t i=0;i<length;i+=2) pcm.push_back(int16_t(u16(bytes.data()+start+i))/32768.f);
        }
        pos=start+length+(length%2);
    }
    if(pcm.empty()) throw std::runtime_error("no PCM data");
    return pcm;
}
static std::string quote(const std::string &s) {
    std::string out="\"";
    const char *hex="0123456789abcdef";
    for(unsigned char c:s) {
        if(c=='"' || c=='\\') {out+='\\';out+=char(c);}
        else if(c<32) {out+="\\u00";out+=hex[c>>4];out+=hex[c&15];}
        else out+=char(c);
    }
    return out+'"';
}
int main(int argc,char **argv) {
    if(argc!=4) return 2;
    whisper_context *ctx=nullptr;
    try {
        int threads=std::stoi(argv[3]);
        if(threads<1 || threads>8) throw std::runtime_error("thread count");
        auto pcm=read_pcm(argv[2]);
        auto cp=whisper_context_default_params();cp.use_gpu=false;
        ctx=whisper_init_from_file_with_params(argv[1],cp);
        if(!ctx) throw std::runtime_error("model load failed");
        auto params=whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        params.n_threads=threads;params.language="en";params.no_context=true;
        params.print_progress=false;params.print_realtime=false;
        params.print_timestamps=false;params.print_special=false;
        params.no_timestamps=true;params.temperature=0;params.temperature_inc=0;
        params.greedy.best_of=1;
        std::cout<<"{\"event\":\"ready\",\"audio_seconds\":"<<pcm.size()/16000.0<<"}"<<std::endl;
        std::string input;
        while(std::getline(std::cin,input)) {
            if(input=="close") break;
            if(input!="infer") throw std::runtime_error("unknown request");
            auto start=std::chrono::steady_clock::now();
            if(whisper_full(ctx,params,pcm.data(),int(pcm.size()))) throw std::runtime_error("ASR inference failed");
            auto elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            std::string text;
            for(int i=0;i<whisper_full_n_segments(ctx);i++) text+=whisper_full_get_segment_text(ctx,i);
            std::cout<<"{\"event\":\"done\",\"inference_seconds\":"<<elapsed
                     <<",\"rtf\":"<<elapsed/(pcm.size()/16000.0)<<",\"text\":"<<quote(text)<<"}"<<std::endl;
        }
        whisper_free(ctx);ctx=nullptr;
        std::cout<<"{\"event\":\"closed\"}"<<std::endl;
        return 0;
    } catch(const std::exception &e) {
        if(ctx) whisper_free(ctx);
        std::cerr<<e.what()<<std::endl;
        return 2;
    }
}
