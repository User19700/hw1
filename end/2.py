from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

prompt = "如果动物会说话，它们最想告诉人类的是"


# 加载预训练模型和分词器
model_name = "uer/gpt2-chinese-cluecorpussmall"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
# 将模型设置为评估模式
model.eval()

# 文本生成配置
generation_config = {
    "max_length": 300,
    "temperature": 0.8,
    "top_p": 0.9,
    "repetition_penalty": 1.2,
    "do_sample": True,
    "num_return_sequences": 1
}

# 生成续写文本
def generate_continuation(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            **generation_config
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 执行生成
generated_text = generate_continuation(prompt)

# 输出结果
print(generated_text)