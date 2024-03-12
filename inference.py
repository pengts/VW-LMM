import torch
import os
import json
from tqdm import tqdm
from visual_words.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from visual_words.conversation import conv_templates, SeparatorStyle
from visual_words.model.builder import load_pretrained_model
from visual_words.utils import disable_torch_init
from visual_words.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from PIL import Image



# Configurations

# model_path="path_to_model_weights"
# question="Write an exhaustive depiction of the given image."
# image_path="./example.jpg"
# conv_mode="vicuna_v1"
# model_base="llama"
# device = "cuda"

# model_path="path_to_model_weights"
# question="Write an exhaustive depiction of the given image."
# image_path="./example.jpg"
# conv_mode="mistral"
# model_base="mistral"
# device = "cuda"

model_path="path_to_model_weights"
question="Write an exhaustive depiction of the given image."
image_path="./example.jpg"
conv_mode="vicuna_v1"
model_base="llama"
device = "cuda"

# Model
disable_torch_init()
model_path = os.path.expanduser(model_path)
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_name, model_base,device=device)

# input
qs = question
qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
conv = conv_templates[conv_mode].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()


image = Image.open(image_path).convert('RGB')
image_tensor = process_images([image], image_processor, model.config)[0].unsqueeze(0).to(device)
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)

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


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default="./Text_encoder/model_best", )
    parser.add_argument('--pretrain_model', type=str, default="./Text_encoder/model_best", )
    parser.add_argument('--train_ds', type=str, default="./playground/twitter2015/MASC/train")
    parser.add_argument('--eval_ds', type=str, default="./playground/twitter2015/MASC/dev")
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--itc', type=float, default=1.0)
    parser.add_argument('--itm', type=float, default=1.0)
    parser.add_argument('--epe', type=float, default=1.0)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--log_step', type=int, default=1)
    parser.add_argument('--save_step', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--accumulation_steps', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=200)
    parser.add_argument('--save_path', type=str, default="./checkpoints/MASC_2015")
    args = parser.parse_args()

