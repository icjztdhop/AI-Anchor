
<div align="center">
  <img alt="AI Anchor Logo" height="80" src="web/favicon.png"/>
  <h1>AI Anchor</h1>
  <p>
     本地部署 AI 虚拟主播系统<br/>
     LLM + 流式 TTS + Live2D + 情绪驱动
  </p>
</div>

---

## 项目简介

**AI Anchor** 是一个 **完全本地部署** 的 AI 虚拟主播项目，融合了语言模型、语音合成与 Live2D 展示，目标是在 **消费级显卡** 上实现低延迟、高沉浸感的虚拟主播体验。

### 核心组件

-  **LMDeploy**：高性能 LLM 推理服务端  
-  **GPT-SoVITS**：高质量语音合成（支持流式输出）  
-  **LlamaIndex**：主播记忆 / 知识库管理  
-  **Live2D Web**：虚拟主播形象渲染  
-  **Emotion System**：情绪驱动 TTS 与 Live2D 动作  

> 在 **RTX 4070Ti 12GB** 显卡上  
> 使用 `internlm2_5-7b-chat-int4`（INT4 量化）  
> **最快约 1 秒返回首段语音**（启动apiserver第一个输入较慢，后续会正常）

⚠️ 本项目处于实验 / 学习阶段，代码大部分由 AI 编写，存在不完善或 Bug 属正常现象。

---

## 功能特性

-  完全本地部署，无需云服务
-  LLM → TTS 全链路流式输出
-  Live2D + 情绪动作联动
-  支持知识库注入（RAG）
-  消费级显卡可运行

---

##  环境要求

- Windows / Linux
- Python ≥ 3.9
- NVIDIA GPU（推荐 ≥ 8GB 显存）
- 已正确配置 CUDA

---

##  快速开始

###  克隆仓库

```bash
git clone https://github.com/icjztdhop/AI-Anchor.git
cd AI-Anchor
```

###  创建并激活虚拟环境

```bash
python -m venv venv
```

Windows：
```bash
.\venv\Scripts\Activate.ps1
```

Linux：
```bash
source venv/bin/activate
```

###  安装依赖

```bash
pip install -r requirements.txt
```

（可选：清华源）
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

##  准备运行资源

### LLM 模型（LMDeploy）

1. 创建 `model/` 目录
2. 放入 **LMDeploy 支持的模型**
3. 推荐使用 **INT4 / AWQ 量化模型**

参考文档：
- https://lmdeploy.readthedocs.io/zh-cn/v0.5.2/supported_models/
- https://lmdeploy.readthedocs.io/zh-cn/v0.5.2/quantization/

---

###  GPT-SoVITS（TTS）

1. 下载 GPT-SoVITS 离线包  
   https://github.com/RVC-Boss/GPT-SoVITS/releases
2. 解压并重命名为：

```text
GPT-SoVITS/
```

---

###  Live2D 模型

目录结构示例：

```text
Live2D/
 └─ hiyori/
    └─ hiyori.model3.json
```

> 已测试公模 `hiyori_pro_t11` 可正常使用

---

##  配置说明

编辑 `config.txt`：

- 新手用户：**仅需修改模型路径即可启动**
- 进阶用户：可调整 TTS、情绪系统、流式参数等

 完整参数说明建议查看[独立文档](README_config.md)。

---

##  启动项目

Windows：
```text
双击 run.bat
```
或
```bash
run.ps1
```

Linux：
```bash
./run.sh
```

启动成功后会出现 **3 个终端窗口**，均无报错即表示运行成功。

---

##  Web UI 使用

1. 浏览器会自动打开
2. 在右侧填写 Live2D 模型路径，例如：

```text
/Live2D/hiyori/hiyori.model3.json
```

3. 点击 **加载模型**
4. 点击 **连接 WS** 开始对话

---

##  未来计划

- [ ] 接入真实直播平台
- [ ] 增强 TTS 情绪表达
- [ ] Live2D 动作适配优化
- [ ] Web UI 美化
- [ ] 接入互联网搜索
- [ ] AI 操作电脑
- [ ] 训练专用虚拟主播模型
- [ ] 其他

> ~~（学不会）~~ 学编程中 

---

##  联系方式

> ~~联系我也不一定能解决问题~~

- QQ：1792767935  
- Email：icjztdhop@gmail.com  

---

##  致谢

感谢以下优秀的开源项目：

- [LMDeploy](https://github.com/InternLM/lmdeploy/)
- [书生·浦语（InternLM）](https://internlm.intern-ai.org.cn/)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [LlamaIndex](https://github.com/run-llama/llama_index)
- [faiss](https://github.com/facebookresearch/faiss)
- [PixiJS](https://pixijs.com/)
- 以及所有 Python 开源库作者 

---

##  开源许可

本项目基于 **Apache License 2.0**：

-  允许个人 / 企业使用
-  允许修改、分发、商用
-  必须保留版权声明与许可证

```text
Licensed under the Apache License, Version 2.0
```
