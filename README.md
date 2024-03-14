# Multi-modal Auto-regressive Modeling via Visual Words

[[`arXiv`](https://arxiv.org/abs/2403.07720)] [[`BibTeX`](#Citing)]

This is the official repository for the multi-modal large language models: VW-LMM

<div align="left">
  <img src="assets/radar.png" width="500"/>
</div><br/>

## Introduction
We propose VW-LMM, a large multi-modal model (LMM) that successfully performs multi-modal auto-regressive modeling with a unified objective for the first time.
Specifically, we propose the concept of visual words, which maps the visual features to probability distributions over LLM's vocabulary, providing supervision information for visual modelling.
We further explore the distribution of visual features in the semantic space within LMM and the possibility of using text embeddings to represent visual information.
Experimental results and ablation studies on 5 VQA tasks and 4 benchmark toolkits validate the powerful performance of our proposed approach.
For more technical details, please refer to our [paper](https://arxiv.org/abs/2403.07720).

<div align="center">
  <img src="assets/model_structure.png" width="800"/>
</div><br/>

In order to verify whether the visual words learnt by VW-LMM can realistically reflect the image information, we take VW-LMM-Vicuna-7B as an example to explore.
For each patch in the image, we select the token with the highest probability in its corresponding visual words, and compare the region of interest in the image with its visualisation result, visualization is as follows <strong>(Best viewed zoomed-in)</strong>: 

<div align="center">
  <img src="assets/visualization_vp.png">
</div><br/>

## Model Zoo

| Version       | Size | Support pseudo image features | Checkpoint                                                                        |
|-------------------|----------|-----------------------------------|---------------------------------------------------------------------------------------|
| VW-LMM-Vicuna     | 7B       |  False                            | [VW-LMM-Vicuna-7b](https://huggingface.co/MYTH-Lab/VW-LMM-Vicuna-7b)           |
| VW-LMM-Mistral    | 7B       |  False                            | [VW-LMM-Mistral-7b](https://huggingface.co/MYTH-Lab/VW-LMM-Mistral-7b)         |
| VW-LMM-Vicuna-pif | 7B       |  True                             | [VW-LMM-Vicuna-pif-7b](https://huggingface.co/MYTH-Lab/VW-LMM-Vicuna-pif-7b) |


VW-LMM, by constructing visual words to introduce visual supervisory information, achieves the best performance among models of the same scale of 7B, and obtains vision-language understanding capability competitive to or even surpassing that of 13B or even larger scale models.

<table>
    <tr>
        <td>Methods</td>
        <td>LLM</td>
        <td>Res.</td>
        <td>VQA^v2</td>
        <td>GQA</td>
        <td>VisWiz</td>
        <td>SQA^I</td>
        <td>VQA^T</td>
        <td>POPE</td>
        <td>MMB</td>
        <td>MMB^CN</td>
        <td>MM-Vet</td>
    </tr>
    <tr>
        <td colspan=12>*Language Modeling Method*</td>
    </tr>
    <tr>
        <td>IDEFICS-80B</td>
        <td>LLaMA-65B</td>
        <td>224</td>
        <td>60.0</td>
        <td>45.2</td>
        <td>36.0</td>
        <td>--</td>
        <td>30.9</td>
        <td>--</td>
        <td>54.5</td>
        <td>38.1</td>
        <td>--</td>
    </tr>
    <tr>
        <td>InstructBLIP</td>
        <td>Vicuna-13B</td>
        <td>224</td>
        <td>--</td>
        <td>49.5</td>
        <td>33.4</td>
        <td>63.1</td>
        <td>50.7</td>
        <td>78.9</td>
        <td>--</td>
        <td>--</td>
        <td>25.6</td>
    </tr>
    <tr>
        <td>BLIP-2</td>
        <td>Vicuna-13B</td>
        <td>224</td>
        <td>41.0</td>
        <td>41.0</td>
        <td>19.6</td>
        <td>61.0</td>
        <td>42.5</td>
        <td>85.3</td>
        <td>--</td>
        <td>--</td>
        <td>22.4</td>
    </tr>
    <tr>
        <td>LLaVA-v1.5</td>
        <td>Vicuna-13B</td>
        <td>336</td>
        <td>80.0</td>
        <td>63.3</td>
        <td>53.6</td>
        <td>71.6</td>
        <td>61.3</td>
        <td>85.9</td>
        <td>67.7</td>
        <td>63.6</td>
        <td>35.4</td>
    </tr>
    <tr>
        <td>InstructBLIP</td>
        <td>Vicuna-7B</td>
        <td>224</td>
        <td>--</td>
        <td>49.2</td>
        <td>34.5</td>
        <td>60.5</td>
        <td>50.1</td>
        <td>--</td>
        <td>36</td>
        <td>23.7</td>
        <td>26.2</td>
    </tr>
    <tr>
        <td>IDEFICS-9B</td>
        <td>LLaMA-7B</td>
        <td>224</td>
        <td>50.9</td>
        <td>38.4</td>
        <td>35.5</td>
        <td>--</td>
        <td>25.9</td>
        <td>--</td>
        <td>48.2</td>
        <td>25.2</td>
        <td>--</td>
    </tr>
    <tr>
        <td>Qwen-VL</td>
        <td>Qwen-7B</td>
        <td>448</td>
        <td>78.8</td>
        <td>59.3</td>
        <td>35.2</td>
        <td>67.1</td>
        <td>63.8</td>
        <td>--</td>
        <td>38.2</td>
        <td>7.4</td>
        <td>--</td>
    </tr>
    <tr>
        <td>Qwen-VL-Chat</td>
        <td>Qwen-7B</td>
        <td>448</td>
        <td>78.2</td>
        <td>57.5</td>
        <td>38.9</td>
        <td>68.2</td>
        <td>61.5</td>
        <td>--</td>
        <td>60.6</td>
        <td>56.7</td>
        <td>--</td>
    </tr>
    <tr>
        <td>LLaVA-v1.5</td>
        <td>Vicuna-7B</td>
        <td>336</td>
        <td>78.5</td>
        <td>62.0</td>
        <td>50.0</td>
        <td>66.8</td>
        <td>58.2</td>
        <td>85.9</td>
        <td>64.3</td>
        <td>58.3</td>
        <td>30.5</td>
    </tr>
    <tr>
        <td>MoE-LLaVA-2.7B×4-Top2</td>
        <td>Phi-2-2.7B</td>
        <td>336</td>
        <td>77.6</td>
        <td>61.4</td>
        <td>43.9</td>
        <td>68.5</td>
        <td>51.4</td>
        <td>86.3</td>
        <td>65.2</td>
        <td>--</td>
        <td>34.3</td>
    </tr>
    <tr>
        <td colspan=12>*Multi-modal Modeling Method*</td>
    </tr>
    <tr>
        <td>Emu2-Chat</td>
        <td>LLaMA-33B</td>
        <td>448</td>
        <td>84.9</td>
        <td>65.1</td>
        <td>54.9</td>
        <td>65.5</td>
        <td>66.6</td>
        <td>--</td>
        <td>--</td>
        <td>--</td>
        <td>48.5</td>
    </tr>
    <tr>
        <td>Emu-I</td>
        <td>LLaMA-13B</td>
        <td>224</td>
        <td>62.0</td>
        <td>46.0</td>
        <td>38.3</td>
        <td>--</td>
        <td>--</td>
        <td>--</td>
        <td>--</td>
        <td>--</td>
        <td>36.3</td>
    </tr>
    <tr>
        <td>MM-Interleaved-SFT</td>
        <td>Vicuna-13B</td>
        <td>224</td>
        <td>80.2</td>
        <td>60.5</td>
        <td>54.9</td>
        <td>--</td>
        <td>61.0</td>
        <td>--</td>
        <td>--</td>
        <td>--</td>
        <td>--</td>
    </tr>
    <tr>
        <td>Unified-IO 2</td>
        <td>UIO-2-6.8B</td>
        <td>384</td>
        <td>79.4</td>
        <td>--</td>
        <td>--</td>
        <td>86.2</td>
        <td>--</td>
        <td>87.7</td>
        <td>71.5</td>
        <td>--</td>
        <td>--</td>
    </tr>
    <tr>
        <td>DreamLLM</td>
        <td>Vicuna-7B</td>
        <td>224</td>
        <td>56.6</td>
        <td>--</td>
        <td>38.1</td>
        <td>--</td>
        <td>34.9</td>
        <td>--</td>
        <td>--</td>
        <td>--</td>
        <td>--</td>
    </tr>
    <tr>
        <td>VL-GPT-I</td>
        <td>LLaMA-7B</td>
        <td>224</td>
        <td>67.2</td>
        <td>51.5</td>
        <td>38.9</td>
        <td>--</td>
        <td>--</td>
        <td>--</td>
        <td>--</td>
        <td>--</td>
        <td>--</td>
    </tr>
    <tr>
        <td>LaVIT-v2</td>
        <td>LLaMA2-7B</td>
        <td>224</td>
        <td>68.3</td>
        <td>47.9</td>
        <td>41.0</td>
        <td>--</td>
        <td>--</td>
        <td>--</td>
        <td>--</td>
        <td>--</td>
        <td>--</td>
    </tr>
    <tr>
        <td>VW-LMM</td>
        <td>Vicuna-7B</td>
        <td>336</td>
        <td>78.9</td>
        <td>62.7</td>
        <td>48.3</td>
        <td>68.1</td>
        <td>57.6</td>
        <td>85.9</td>
        <td>65.9</td>
        <td>59.8</td>
        <td>31.3</td>
    </tr>
    <tr>
        <td>VW-LMM</td>
        <td>Mistral-7B</td>
        <td>336</td>
        <td>80.8</td>
        <td>65.4</td>
        <td>58.5</td>
        <td>75.9</td>
        <td>63.1</td>
        <td>87.0</td>
        <td>80.6</td>
        <td>79.0</td>
        <td>44.0</td>
    </tr>
</table>

## Setup

### Requirements

```shell
git clone https://github.com/pengts/VW-LMM.git
cd VW-LMM
pip install -r requirements.txt
```

## Multi-modal Inference

### Model Configurations

- VW-LMM-Vicuna
```python
model_path="VW-LMM-Vicuna"
conv_mode="vicuna_v1"
model_base="llama"
device = "cuda"
```
- VW-LMM-Mistral
```python
model_path="VW-LMM-Mistral"
conv_mode="mistral"
model_base="mistral"
device = "cuda"
```

VW-LMM-Vicuna-pif
```python
model_path="VW-LMM-Vicuna-pif"
conv_mode="vicuna_v1"
model_base="llama"
device = "cuda"
```

### Model Initialization
```python
disable_torch_init()
model_path = os.path.expanduser(model_path)
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_name, model_base,device=device)
```

### Input Processing
```python
question="Write an exhaustive depiction of the given image."
image_path="./example.jpg"
qs = question
qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
conv = conv_templates[conv_mode].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

image = Image.open(image_path).convert('RGB')
image_tensor = process_images([image], image_processor, model.config)[0].unsqueeze(0).to(device)
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
```

### Inference
```python
with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=image_tensor.to(dtype=torch.float16, device=device, non_blocking=True),
        do_sample= False,
        temperature=0,
        top_p=None,
        num_beams=1,
        max_new_tokens=128,
        use_cache=True)

input_token_len = input_ids.shape[1]
n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
if n_diff_input_output > 0:
    print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
outputs = outputs.strip()
print(outputs)
```

## Acknowledgement
We are grateful for the following awesome projects when implementing VW-LMM:
* [LLaVA](https://github.com/haotian-liu/LLaVA/): Visual instruction tuning towards large language and vision models with GPT-4 level capabilities.
* [LLaMA](https://github.com/facebookresearch/llama): Open and Efficient Foundation Language Models
* [Vicuna](https://github.com/lm-sys/FastChat): Open-source LLM with amazing language capabilities!
* [Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2): A 7B transformer model, fast-deployed and easily customisable. Small, yet very powerful for a variety of use cases.


## <a name="Citing"></a>Citation
Consider giving this repository a star and cite VW-LMM in your publications if it helps your research.

```
@misc{peng2024multimodal,
      title={Multi-modal Auto-regressive Modeling via Visual Words}, 
      author={Tianshuo Peng and Zuchao Li and Lefei Zhang and Hai Zhao and Ping Wang and Bo Du},
      year={2024},
      eprint={2403.07720},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
