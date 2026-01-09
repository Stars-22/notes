# CosyVoice 详细架构文档

## 1. 项目概述

CosyVoice 是一个基于大型语言模型（LLM）的先进文本到语音（TTS）系统，支持多语言、多方言的语音合成。它实现了零样本语音克隆、发音修复、文本标准化等功能，并支持流式传输。

## 2. 系统架构图

```mermaid
graph TB
    subgraph "输入层"
        A[文本输入] --> B[前端处理]
        C[语音提示] --> B
        D[说话人ID] --> B
        E[指令文本] --> B
    end

    subgraph "前端处理"
        B --> F[文本标准化]
        B --> G[语音标记提取]
        B --> H[说话人嵌入提取]
        B --> I[语音特征提取]
    end

    subgraph "模型层"
        subgraph "LLM (语言模型)"
            J[文本编码器]
            K[LLM解码器]
            L[采样方法]
        end

        subgraph "Flow (流模型)"
            M[编码器]
            N[长度调节器]
            O[解码器]
        end

        subgraph "HiFiGAN (声码器)"
            P[源模块]
            Q[逆STFT]
            R[残差块]
        end
    end

    subgraph "推理流程"
        S[文本标记] --> J
        T[提示文本标记] --> K
        U[提示语音标记] --> K
        V[说话人嵌入] --> K
        W[LLM输出] --> M
        X[提示语音特征] --> O
        Y[Flow输出] --> P
        Z[HiFiGAN输出] --> AA[音频输出]
    end

    F --> S
    G --> U
    H --> V
    I --> X
    J --> W
    K --> W
    M --> W
    N --> Y
    O --> Y
    P --> Z
    Q --> Z
    R --> Z

    subgraph "模型类型"
        BB[CosyVoice]
        CC[CosyVoice2]
        DD[CosyVoice3]
    end

    BB --> J
    BB --> M
    BB --> P
    CC --> K
    CC --> O
    CC --> Q
    DD --> L
    DD --> R
```

## 3. 类图

### 3.1 主模型类图

```mermaid
classDiagram
    class CosyVoice {
        +String model_dir
        +Bool fp16
        +CosyVoiceFrontEnd frontend
        +CosyVoiceModel model
        +Int sample_rate
        +list_available_spks() String[]
        +add_zero_shot_spk(String prompt_text, String prompt_wav, String zero_shot_spk_id) Bool
        +save_spkinfo() void
        +inference_sft(String tts_text, String spk_id, Bool stream, Float speed, Bool text_frontend) Generator
        +inference_zero_shot(String tts_text, String prompt_text, String prompt_wav, String zero_shot_spk_id, Bool stream, Float speed, Bool text_frontend) Generator
        +inference_cross_lingual(String tts_text, String prompt_wav, String zero_shot_spk_id, Bool stream, Float speed, Bool text_frontend) Generator
        +inference_instruct(String tts_text, String spk_id, String instruct_text, Bool stream, Float speed, Bool text_frontend) Generator
        +inference_vc(String source_wav, String prompt_wav, Bool stream, Float speed) Generator
    }

    class CosyVoice2 {
        +inference_instruct2(String tts_text, String instruct_text, String prompt_wav, String zero_shot_spk_id, Bool stream, Float speed, Bool text_frontend) Generator
    }

    CosyVoice <|-- CosyVoice2
    CosyVoice2 <|-- CosyVoice3
```

### 3.2 前端处理类图

```mermaid
classDiagram
    class CosyVoiceFrontEnd {
        +Tokenizer tokenizer
        +FeatureExtractor feat_extractor
        +ONNXRuntimeSession campplus_session
        +ONNXRuntimeSession speech_tokenizer_session
        +Dict spk2info
        +String allowed_special
        +InflectParser inflect_parser
        +String text_frontend
        +_extract_text_token(String text) Tuple
        +_extract_speech_token(String prompt_wav) Tuple
        +_extract_spk_embedding(String prompt_wav) Tensor
        +_extract_speech_feat(String prompt_wav) Tuple
        +text_normalize(String text, Bool split, Bool text_frontend) List
        +frontend_sft(String tts_text, String spk_id) Dict
        +frontend_zero_shot(String tts_text, String prompt_text, String prompt_wav, Int resample_rate, String zero_shot_spk_id) Dict
        +frontend_cross_lingual(String tts_text, String prompt_wav, Int resample_rate, String zero_shot_spk_id) Dict
        +frontend_instruct(String tts_text, String spk_id, String instruct_text) Dict
        +frontend_instruct2(String tts_text, String instruct_text, String prompt_wav, Int resample_rate, String zero_shot_spk_id) Dict
        +frontend_vc(String source_speech_16k, String prompt_wav, Int resample_rate) Dict
    }
```

### 3.3 模型层类图

```mermaid
classDiagram
    class CosyVoiceModel {
        +Device device
        +LLM llm
        +Flow flow
        +HiFT hift
        +Bool fp16
        +Int token_min_hop_len
        +Int token_max_hop_len
        +Int token_overlap_len
        +Int mel_overlap_len
        +Int mel_cache_len
        +Int source_cache_len
        +Float[] mel_window
        +Float[] speech_window
        +Float stream_scale_factor
        +Context llm_context
        +Lock lock
        +Dict tts_speech_token_dict
        +Dict llm_end_dict
        +Dict mel_overlap_dict
        +Dict flow_cache_dict
        +Dict hift_cache_dict
        +List silent_tokens
        +load(String llm_model, String flow_model, String hift_model) void
        +load_jit(String llm_text_encoder_model, String llm_llm_model, String flow_encoder_model) void
        +load_trt(String flow_decoder_estimator_model, String flow_decoder_onnx_model, Int trt_concurrent, Bool fp16) void
        +get_trt_kwargs() Dict
        +llm_job(Tensor text, Tensor prompt_text, Tensor llm_prompt_speech_token, Tensor llm_embedding, String uuid) void
        +vc_job(Tensor source_speech_token, String uuid) void
        +token2wav(Tensor token, Tensor prompt_token, Tensor prompt_feat, Tensor embedding, String uuid, Bool finalize, Float speed) Tensor
        +tts(Tensor text, Tensor flow_embedding, Tensor llm_embedding, Tensor prompt_text, Tensor llm_prompt_speech_token, Tensor flow_prompt_speech_token, Tensor prompt_speech_feat, Tensor source_speech_token, Bool stream, Float speed) Generator
    }

    class CosyVoice2Model {
        +Int token_hop_len
        +load_jit(String flow_encoder_model) void
        +load_vllm(String model_dir) void
        +token2wav(Tensor token, Tensor prompt_token, Tensor prompt_feat, Tensor embedding, Int token_offset, String uuid, Bool stream, Bool finalize, Float speed) Tensor
    }

    class CosyVoice3Model {
        +Int token_hop_len
        +load_vllm(String model_dir) void
    }

    CosyVoiceModel <|-- CosyVoice2Model
    CosyVoice2Model <|-- CosyVoice3Model
```

### 3.4 语言模型类图

```mermaid
classDiagram
    class TransformerLM {
        +Int llm_input_size
        +Int llm_output_size
        +Int speech_token_size
        +Int sos
        +Int task_id
        +Int eos_token
        +Embedding text_embedding
        +TextEncoder text_encoder
        +Linear text_encoder_affine_layer
        +Embedding llm_embedding
        +LLM llm
        +Linear llm_decoder
        +LabelSmoothingLoss criterion_ce
        +Embedding speech_embedding
        +Linear spk_embed_affine_layer
        +Callable sampling
        +encode(Tensor text, Tensor text_lengths) Tuple
        +pad_unpad_sequence(Tensor sos_emb, Tensor embedding, Tensor text_token, Tensor text_token_len, Tensor task_id_emb, Tensor speech_token, Tensor speech_token_len) Tuple
        +forward(Dict batch, Device device) Dict
        +sampling_ids(Tensor weighted_scores, List decoded_tokens, Int sampling, Bool ignore_eos) Int
        +inference(Tensor text, Tensor text_len, Tensor prompt_text, Tensor prompt_text_len, Tensor prompt_speech_token, Tensor prompt_speech_token_len, Tensor embedding, Int sampling, Float max_token_text_ratio, Float min_token_text_ratio, String uuid) Generator
    }

    class Qwen2LM {
        +Int fill_token
        +List stop_token_ids
        +Dict vllm_output_queue
        +List mix_ratio
        +prepare_lm_input_target(Tensor sos_emb, Tensor text_token, Tensor text_token_emb, Tensor text_token_len, Tensor task_id_emb, Tensor speech_token, Tensor speech_token_emb, Tensor speech_token_len, Tensor instruct_token, Tensor instruct_token_emb, Tensor instruct_token_len) Tuple
        +forward_dpo(Dict batch, Device device) Dict
        +inference_wrapper(Tensor lm_input, Int sampling, Int min_len, Int max_len, String uuid) Generator
    }

    TransformerLM <|-- Qwen2LM
```

### 3.5 流模型类图

```mermaid
classDiagram
    class MaskedDiffWithXvec {
        +Int input_size
        +Int output_size
        +Int vocab_size
        +String output_type
        +Int input_frame_rate
        +Embedding input_embedding
        +Linear spk_embed_affine_layer
        +Encoder encoder
        +Linear encoder_proj
        +Decoder decoder
        +LengthRegulator length_regulator
        +Bool only_mask_loss
        +forward(Dict batch, Device device) Dict
        +inference(Tensor token, Tensor token_len, Tensor prompt_token, Tensor prompt_token_len, Tensor prompt_feat, Tensor prompt_feat_len, Tensor embedding, Tensor flow_cache) Tuple
    }

    class CausalMaskedDiffWithXvec {
        +Int token_mel_ratio
        +Int pre_lookahead_len
        +forward(Dict batch, Device device) Dict
        +inference(Tensor token, Tensor token_len, Tensor prompt_token, Tensor prompt_token_len, Tensor prompt_feat, Tensor prompt_feat_len, Tensor embedding, Bool streaming, Bool finalize) Tuple
    }

    MaskedDiffWithXvec <|-- CausalMaskedDiffWithXvec
```

### 3.6 声码器类图

```mermaid
classDiagram
    class HiFTGenerator {
        +Int out_channels
        +Int nb_harmonics
        +Int sampling_rate
        +Dict istft_params
        +Float lrelu_slope
        +Float audio_limit
        +Int num_kernels
        +Int num_upsamples
        +SourceModuleHnNSF m_source
        +Upsample f0_upsamp
        +Conv1d conv_pre
        +ModuleList ups
        +ModuleList source_downs
        +ModuleList source_resblocks
        +ModuleList resblocks
        +Conv1d conv_post
        +ReflectionPad1d reflection_pad
        +Tensor stft_window
        +F0Predictor f0_predictor
        +remove_weight_norm() void
        +_stft(Tensor x) Tuple
        +_istft(Tensor magnitude, Tensor phase) Tensor
        +decode(Tensor x, Tensor s) Tensor
        +forward(Dict batch, Device device) Tuple
        +inference(Tensor speech_feat, Tensor cache_source) Tuple
    }

    class CausalHiFTGenerator {
        +List upsample_rates
        +Int conv_pre_look_right
    }

    HiFTGenerator <|-- CausalHiFTGenerator
```

### 3.7 分词器类图

```mermaid
classDiagram
    class CosyVoiceTokenizer {
        +Dict special_tokens
        +AutoTokenizer tokenizer
        +Bool skip_special_tokens
        +encode(String text) List
        +decode(List tokens) String
    }

    class CosyVoice2Tokenizer {
        +encode(String text) List
        +decode(List tokens) String
    }

    class CosyVoice3Tokenizer {
        +encode(String text) List
        +decode(List tokens) String
    }

    CosyVoiceTokenizer <|-- CosyVoice2Tokenizer
    CosyVoice2Tokenizer <|-- CosyVoice3Tokenizer
```

## 4. 组件图

```mermaid
graph LR
    subgraph Input ["输入组件"]
        TI["文本输入"]
        SP["语音提示"]
        SID["说话人ID"]
        IT["指令文本"]
    end

    subgraph Frontend ["前端处理组件"]
        FE["前端处理"]
        TN["文本标准化"]
        ST["语音标记提取"]
        SE["说话人嵌入提取"]
        SF["语音特征提取"]
    end

    subgraph LLM ["LLM组件"]
        LLMComp["LLM"]
        TE["文本编码器"]
        LD["LLM解码器"]
        SM["采样方法"]
    end

    subgraph Flow ["Flow组件"]
        FLOW["Flow"]
        ENC["编码器"]
        LR["长度调节器"]
        DEC["解码器"]
    end

    subgraph HiFiGAN ["HiFiGAN组件"]
        HIFIGAN["HiFiGAN"]
        SMOD["源模块"]
        ISTFT["逆STFT"]
        RB["残差块"]
    end

    subgraph Output ["输出组件"]
        AO["音频输出"]
    end

    TI --> FE
    SP --> FE
    SID --> FE
    IT --> FE

    FE --> TN
    FE --> ST
    FE --> SE
    FE --> SF

    TN --> TE
    ST --> LD
    SE --> LD
    SF --> DEC

    TE --> LLMComp
    LD --> LLMComp
    SM --> LLMComp

    LLMComp --> ENC
    ENC --> FLOW
    LR --> FLOW
    DEC --> FLOW

    FLOW --> SMOD
    SMOD --> HIFIGAN
    ISTFT --> HIFIGAN
    RB --> HIFIGAN

    HIFIGAN --> AO
```

## 5. 数据流图

```mermaid
graph TD
    subgraph "输入阶段"
        A1[原始文本] --> A2[标准化文本]
        A3[语音提示音频] --> A4[语音特征]
        A5[语音提示音频] --> A6[说话人嵌入]
        A7[语音提示音频] --> A8[语音标记]
    end

    subgraph "LLM阶段"
        A2 --> B1[文本编码]
        A4 --> B2[提示文本编码]
        A8 --> B3[提示语音标记编码]
        A6 --> B4[说话人嵌入投影]
        B1 --> B5[LLM解码]
        B2 --> B5
        B3 --> B5
        B4 --> B5
        B5 --> B6[语音标记序列]
    end

    subgraph "Flow阶段"
        B6 --> C1[语音标记编码]
        A8 --> C2[提示语音标记编码]
        A4 --> C3[提示语音特征]
        C1 --> C4[编码器]
        C2 --> C4
        C3 --> C5[长度调节]
        C4 --> C5
        C5 --> C6[解码器]
        C6 --> C7[梅尔频谱]
    end

    subgraph "HiFiGAN阶段"
        C7 --> D1[频谱到波形]
        D1 --> D2[音频输出]
    end

    subgraph "缓存管理"
        E1[LLM缓存] --> B5
        E2[Flow缓存] --> C6
        E3[HiFiGAN缓存] --> D1
    end
```

## 6. 模块详解

### 6.1 前端处理 (Frontend)

前端处理模块负责将输入的文本、语音提示和说话人信息转换为模型可以处理的格式。

- **文本标准化**: 使用 `ttsfrd` 或 `wetext` 对输入文本进行标准化处理。
- **语音标记提取**: 从语音提示中提取语音标记。
- **说话人嵌入提取**: 从语音提示中提取说话人嵌入向量。
- **语音特征提取**: 从语音提示中提取声学特征。

### 6.2 语言模型 (LLM)

语言模型负责将文本和上下文信息转换为语音标记序列。

- **文本编码器**: 将文本转换为向量表示。
- **LLM解码器**: 基于文本、上下文和说话人信息生成语音标记。
- **采样方法**: 用于从模型输出的概率分布中采样语音标记。

### 6.3 流模型 (Flow)

流模型负责将语音标记转换为声学特征。

- **编码器**: 将语音标记转换为中间表示。
- **长度调节器**: 调节序列长度以匹配声学特征长度。
- **解码器**: 将中间表示转换为声学特征。

### 6.4 声码器 (HiFiGAN)

声码器负责将声学特征转换为音频波形。

- **源模块**: 生成激励信号。
- **逆STFT**: 将频域特征转换为时域信号。
- **残差块**: 对音频进行精细化处理。

## 7. 模型类型

### 7.1 CosyVoice

基础模型，包含 LLM、Flow 和 HiFiGAN 三个模块。

### 7.2 CosyVoice2

在 CosyVoice 基础上进行了优化，支持流式处理和 vLLM 加速。

### 7.3 CosyVoice3

最新版本，进一步优化了模型结构和性能，支持更复杂的任务。

## 8. 推理流程

1. **输入处理**: 将文本、语音提示和说话人信息转换为模型输入格式。
2. **LLM 推理**: 生成语音标记序列。
3. **Flow 推理**: 将语音标记转换为声学特征。
4. **HiFiGAN 推理**: 将声学特征转换为音频波形。
5. **输出**: 生成最终的音频输出。

## 9. 关键特性

- **多语言支持**: 支持中文、英文、日文、韩文等多种语言。
- **多方言支持**: 支持多种中文方言和口音。
- **零样本语音克隆**: 仅需几秒语音提示即可克隆说话人声音。
- **流式处理**: 支持实时流式语音合成。
- **指令控制**: 支持通过指令控制语音的情感、语速、音量等属性。