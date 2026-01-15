# CosyVoice项目详细文档

## 项目概述

CosyVoice是一个基于大型语言模型（LLM）的先进文本转语音（TTS）系统，由阿里巴巴通义实验室开发。该项目支持多语言、多方言和零样本语音克隆，具有高度的自然性和可控性。

### 主要版本
- **Fun-CosyVoice 3.0**：最新版本，超越了前代在内容一致性、说话人相似度和韵律自然性方面的表现
- **CosyVoice 2.0**：支持流式语音合成的版本
- **CosyVoice 1.0**：基础版本

## 项目结构

```
CosyVoice/
├── CODE_OF_CONDUCT.md
├── downloadModel.py
├── example.py
├── FAQ.md
├── LICENSE
├── README.md
├── requirements.txt
├── vllm_example.py
├── webui.py
├── __pycache__/
├── .git/
├── .github/
├── .idea/
├── asset/
├── cosyvoice/
├── demo/
├── docker/
├── examples/
├── pretrained_models/
├── runtime/
├── third_party/
└── tools/
```

### 核心模块结构

```
cosyvoice/
├── __init__.py
├── bin/
│   ├── train.py          # 训练脚本
│   ├── export_jit.py     # JIT导出脚本
│   ├── export_onnx.py    # ONNX导出脚本
│   └── average_model.py  # 模型平均脚本
├── cli/
│   ├── __init__.py
│   ├── cosyvoice.py      # 主要的CosyVoice接口类
│   ├── frontend.py       # 前端处理模块
│   └── model.py          # 模型定义和推理逻辑
├── dataset/
│   ├── __init__.py
│   ├── dataset.py        # 数据集处理
│   └── processor.py      # 数据处理器
├── flow/
│   ├── decoder.py        # 流模型解码器
│   └── flow.py           # 流模型实现
├── hifigan/
│   ├── hifigan.py        # HiFiGAN声码器
│   ├── generator.py      # 生成器
│   ├── discriminator.py  # 判别器
│   └── f0_predictor.py   # F0预测器
├── llm/
│   └── llm.py            # 大语言模型实现
├── tokenizer/
│   └── tokenizer.py      # 分词器
├── transformer/
│   ├── encoder.py        # Transformer编码器
│   ├── decoder.py        # Transformer解码器
│   ├── attention.py      # 注意力机制
│   ├── convolution.py    # 卷积模块
│   ├── activation.py     # 激活函数
│   ├── embedding.py      # 嵌入层
│   ├── positionwise_feed_forward.py  # 前馈网络
│   ├── subsampling.py    # 子采样
│   ├── encoder_layer.py  # 编码器层
│   ├── decoder_layer.py  # 解码器层
│   ├── label_smoothing_loss.py  # 标签平滑损失
│   └── upsample_encoder.py  # 上采样编码器
├── utils/
│   ├── common.py         # 通用工具函数
│   ├── file_utils.py     # 文件处理工具
│   ├── mask.py           # 掩码工具
│   ├── losses.py         # 损失函数
│   ├── train_utils.py    # 训练工具
│   ├── executor.py       # 执行器
│   └── class_utils.py    # 类工具
└── vllm/
```

## 核心功能

### 1. 预训练音色（SFT）
使用预训练的说话人音色进行语音合成，适用于标准的文本转语音任务。

### 2. 零样本语音克隆（Zero-shot）
通过3秒的音频样本，克隆任意说话人的音色，实现个性化语音合成。

### 3. 跨语种语音克隆（Cross-lingual）
使用一种语言的音频样本，合成另一种语言的语音，实现跨语言语音克隆。

### 4. 自然语言控制（Instruct）
通过自然语言指令控制语音合成，如调整情感、语速、音量等。

### 5. 语音转换（Voice Conversion）
将一个说话人的语音转换为另一个说话人的语音。

## 技术架构

### 模型架构
CosyVoice采用三阶段架构：
1. **LLM（Large Language Model）**：负责将文本转换为语音token序列
2. **Flow Model**：将语音token转换为梅尔频谱图
3. **HiFi-GAN**：将梅尔频谱图转换为波形音频

### 前端处理
- **文本标准化**：支持中英文数字、符号的标准化
- **语音token提取**：从音频中提取离散语音token
- **说话人嵌入**：提取说话人特征向量

### 流式推理
支持流式语音合成，实现低延迟的实时语音生成。

## 安装与配置

### 环境要求
- Python 3.10
- PyTorch (支持CUDA)
- 依赖库详见requirements.txt

### 安装步骤
```bash
# 克隆项目
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
git submodule update --init --recursive

# 创建conda环境
conda create -n cosyvoice -y python=3.10
conda activate cosyvoice

# 安装依赖
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# 安装sox（如有兼容性问题）
# ubuntu
sudo apt-get install sox libsox-dev
# centos
sudo yum install sox sox-devel
```

### 模型下载
```python
# 通过ModelScope下载
from modelscope import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('iic/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')
snapshot_download('iic/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT')
snapshot_download('iic/CosyVoice-300M-Instruct', local_dir='pretrained_models/CosyVoice-300M-Instruct')
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
```

## 使用方法

### 基本使用
```python
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

# 初始化模型
cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice-300M-SFT')

# 预训练音色合成
for i, j in enumerate(cosyvoice.inference_sft('你好，我是通义生成式语音大模型', '中文女', stream=False)):
    torchaudio.save('sft_output.wav', j['tts_speech'], cosyvoice.sample_rate)

# 零样本语音克隆
for i, j in enumerate(cosyvoice.inference_zero_shot('这是零样本语音克隆示例', '提示文本', './asset/prompt.wav')):
    torchaudio.save('zero_shot_output.wav', j['tts_speech'], cosyvoice.sample_rate)
```

### WebUI使用
```bash
python3 webui.py --port 50000 --model_dir pretrained_models/CosyVoice-300M
```

### vLLM加速
```bash
# 安装vLLM
pip install vllm==0.11.0 transformers==4.57.1 numpy==1.26.4

# 使用vLLM加速
python vllm_example.py
```

## API接口

### CosyVoice类
- `inference_sft(tts_text, spk_id, stream=False)`：预训练音色合成
- `inference_zero_shot(tts_text, prompt_text, prompt_wav, stream=False)`：零样本语音克隆
- `inference_cross_lingual(tts_text, prompt_wav, stream=False)`：跨语种语音克隆
- `inference_instruct(tts_text, spk_id, instruct_text, stream=False)`：自然语言控制
- `inference_vc(source_wav, prompt_wav, stream=False)`：语音转换
- `list_available_spks()`：列出可用的预训练音色
- `add_zero_shot_spk(prompt_text, prompt_wav, zero_shot_spk_id)`：添加零样本音色
- `save_spkinfo()`：保存音色信息

### CosyVoice2类
- `inference_instruct2(tts_text, instruct_text, prompt_wav, stream=False)`：增强版自然语言控制

### CosyVoice3类
- 继承自CosyVoice2，支持更多高级功能

## 配置文件

### 模型配置
- `cosyvoice.yaml`：CosyVoice 1.0配置
- `cosyvoice2.yaml`：CosyVoice 2.0配置
- `cosyvoice3.yaml`：CosyVoice 3.0配置

### 前端配置
- 支持ttsfrd和wetext两种文本前端
- 自动检测并使用可用的前端工具

## 训练流程

### 模型组成
CosyVoice模型由三个主要组件构成：
1. **LLM（Large Language Model）**：负责将文本和语音token序列映射到语音token序列
2. **Flow Model**：将语音token转换为梅尔频谱图
3. **HiFi-GAN**：将梅尔频谱图转换为波形音频

### 训练数据准备
使用`tools/make_parquet_list.py`脚本来准备训练数据：
```bash
python tools/make_parquet_list.py --src_dir data/src --des_dir data/des --num_utts_per_parquet 1000
```

数据准备流程包括：
1. **音频文件**：WAV格式音频文件
2. **文本标注**：对应音频的文本内容
3. **说话人信息**：说话人ID映射
4. **说话人嵌入**：使用`tools/extract_embedding.py`提取说话人嵌入向量
5. **语音token**：使用`tools/extract_speech_token.py`提取离散语音token

### 模型训练
使用`cosyvoice/bin/train.py`进行模型训练：
```bash
# 训练LLM模型
python cosyvoice/bin/train.py --model llm --config config.yaml --train_data train.list --cv_data cv.list --model_dir exp/model

# 训练Flow模型
python cosyvoice/bin/train.py --model flow --config config.yaml --train_data train.list --cv_data cv.list --model_dir exp/model

# 训练HiFi-GAN模型
python cosyvoice/bin/train.py --model hifigan --config config.yaml --train_data train.list --cv_data cv.list --model_dir exp/model
```

### 训练配置
训练配置文件定义了模型架构和训练参数：

**CosyVoice 1.0配置** (`examples/libritts/cosyvoice/conf/cosyvoice.yaml`)：
- LLM使用TransformerEncoder架构，14层，16头注意力
- Flow使用ConformerEncoder架构，6层，8头注意力
- HiFi-GAN使用HiFTGenerator架构

**CosyVoice 2.0配置** (`examples/libritts/cosyvoice2/conf/cosyvoice2.yaml`)：
- LLM使用Qwen2架构
- Flow使用因果流模型，支持流式推理
- 采样率24kHz，流式推理块大小25个token

### 特殊训练模式

**DPO（Direct Preference Optimization）训练**：
- 仅支持LLM模型
- 需要正负样本对
- 使用`--dpo`参数启用
- 需要指定参考模型`--ref_model`

**SFT（Supervised Fine-Tuning）训练**：
- 使用预训练模型进行微调
- 需要设置`use_spk_embedding: True`
- 调整学习率和调度器参数

### 模型导出
- JIT导出：`python cosyvoice/bin/export_jit.py --model_dir path/to/model`
- ONNX导出：`python cosyvoice/bin/export_onnx.py --model_dir path/to/model`

## 部署方式

### 本地部署
```bash
# 直接运行Python脚本
python example.py
```

### Web服务
```bash
# 启动WebUI
python webui.py --port 50000 --model_dir pretrained_models/CosyVoice-300M
```

### GRPC服务
```bash
# 启动GRPC服务器
cd runtime/python/grpc
python server.py --port 50000 --model_dir pretrained_models/CosyVoice-300M

# 客户端调用
python client.py --mode sft --tts_text "你好世界" --tts_wav output.wav
```

### FastAPI服务
```bash
# 启动FastAPI服务器
cd runtime/python/fastapi
python server.py --port 50000 --model_dir pretrained_models/CosyVoice-300M

# 客户端调用
curl -X POST "http://localhost:50000/inference_sft" -d "tts_text=你好世界" -d "spk_id=中文女"
```

### Docker部署
```bash
# 构建Docker镜像
cd runtime/python
docker build -t cosyvoice:v1.0 .

# 运行Docker容器
docker run -d --runtime=nvidia -p 50000:50000 cosyvoice:v1.0 /bin/bash -c "cd /opt/CosyVoice/CosyVoice/runtime/python/fastapi && python3 server.py --port 50000 --model_dir iic/CosyVoice-300M && sleep infinity"
```

### TensorRT-LLM加速
```bash
cd runtime/triton_trtllm
docker compose up -d
```

## 性能优化

### 推理加速
- JIT编译：支持PyTorch JIT加速
- TensorRT：支持NVIDIA TensorRT推理加速
- vLLM：支持vLLM库加速LLM推理

### 内存优化
- 流式推理：减少内存占用
- 模型量化：支持FP16推理
- 缓存机制：优化重复推理

## 工具脚本

### 数据处理工具
- `tools/extract_embedding.py`：提取说话人嵌入向量
- `tools/extract_speech_token.py`：提取语音token
- `tools/make_parquet_list.py`：制作parquet数据列表

### 模型工具
- `cosyvoice/bin/average_model.py`：模型平均
- `cosyvoice/bin/export_jit.py`：JIT导出
- `cosyvoice/bin/export_onnx.py`：ONNX导出
- `cosyvoice/bin/train.py`：模型训练

## 评估指标

根据README中的评估表格，CosyVoice在多个指标上表现优异：
- **CER（Character Error Rate）**：中文字符错误率
- **SS（Speaker Similarity）**：说话人相似度
- **WER（Word Error Rate）**：英文单词错误率

## 开发指南

### 代码结构
- `cli/`：命令行接口和主要推理逻辑
- `flow/`：流模型相关组件
- `llm/`：大语言模型相关组件
- `hifigan/`：声码器相关组件
- `utils/`：工具函数和辅助功能
- `dataset/`：数据集处理
- `tokenizer/`：分词器
- `transformer/`：Transformer相关组件

### 扩展功能
- 可以通过添加新的`inference_*`方法扩展功能
- 支持自定义前端处理逻辑
- 支持模型微调和训练

## 常见问题

### 模型下载问题
- 确保网络连接正常
- 可以使用ModelScope或HuggingFace下载模型
- 检查磁盘空间是否充足

### 推理问题
- 确保音频格式正确（WAV格式，采样率≥16kHz）
- 检查文本格式是否符合要求
- 确保模型文件完整

### 性能问题
- 使用GPU加速推理
- 启用vLLM或TensorRT加速
- 调整流式推理参数

## 贡献与支持

### 社区支持
- GitHub Issues：提交问题和功能请求
- 钉钉群：加入官方交流群

### 代码贡献
- 遵循Apache 2.0许可证
- 提交PR前确保代码质量
- 包含适当的测试用例

## 引用信息

```bibtex
@article{du2024cosyvoice,
  title={Cosyvoice: A scalable multilingual zero-shot text-to-speech synthesizer based on supervised semantic tokens},
  author={Du, Zhihao and Chen, Qian and Zhang, Shiliang and Hu, Kai and Lu, Heng and Yang, Yexin and Hu, Hangrui and Zheng, Siqi and Gu, Yue and Ma, Ziyang and others},
  journal={arXiv preprint arXiv:2407.05407},
  year={2024}
}

@article{du2024cosyvoice,
  title={Cosyvoice 2: Scalable streaming speech synthesis with large language models},
  author={Du, Zhihao and Wang, Yuxuan and Chen, Qian and Shi, Xian and Lv, Xiang and Zhao, Tianyu and Gao, Zhifu and Yang, Yexin and Gao, Changfeng and Wang, Hui and others},
  journal={arXiv preprint arXiv:2412.10117},
  year={2024}
}

@article{du2025cosyvoice,
  title={CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training},
  author={Du, Zhihao and Gao, Changfeng and Wang, Yuxuan and Yu, Fan and Zhao, Tianyu and Wang, Hao and Lv, Xiang and Wang, Hui and Shi, Xian and An, Keyu and others},
  journal={arXiv preprint arXiv:2505.17589},
  year={2025}
}
```

## 致谢

CosyVoice项目借鉴了多个开源项目：
- [FunASR](https://github.com/modelscope/FunASR)
- [FunCodec](https://github.com/modelscope/FunCodec)
- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS)
- [AcademiCodec](https://github.com/yangdongchao/AcademiCodec)
- [WeNet](https://github.com/wenet-e2e/wenet)

## 免责声明

本项目内容仅供学术研究使用，旨在展示技术能力。部分示例来源于互联网，如有内容侵犯您的权益，请联系我们请求删除。