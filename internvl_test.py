import torch
from transformers import AutoTokenizer, AutoModel
from transformers.modules.InternVL2_1B.configuration_internvl_chat import InternVLChatConfig


local_path = "/home/caozhe/workspace/InternVL2-1B"


from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq

for class_name in AutoModelForCausalLM._model_mapping.keys():
    print(class_name)
print("---------------------------------")
for class_name in AutoModelForVision2Seq._model_mapping.keys():
    print(class_name)
trust_remote_code = True
actor_model_config = AutoConfig.from_pretrained(
    local_path, trust_remote_code=trust_remote_code, attn_implementation="flash_attention_2"
)
qwen2_5_vl3b_config = AutoConfig.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    trust_remote_code=trust_remote_code,
    attn_implementation="flash_attention_2"
)

actor_module_class = AutoModelForVision2Seq
# actor_module_class = AutoModelForCausalLM

actor_module = actor_module_class.from_pretrained(
    pretrained_model_name_or_path=local_path,
    torch_dtype=torch.bfloat16,
    config=actor_model_config,
    trust_remote_code=trust_remote_code,
).eval().cuda()
model = AutoModel.from_pretrained(
    local_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
print("done")