# 1. 安装必要的库
# !pip install transformers torch

# 2. 导入所需库
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch

# 3. 设置镜像站点（使用国内镜像加速下载）
# 这里使用清华大学镜像站作为示例
MIRROR = "https://mirror.tuna.tsinghua.edu.cn/hugging-face-mirror"

# 4. 加载预训练模型和分词器（通过镜像站点）
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name, mirror=MIRROR)
model = BertForSequenceClassification.from_pretrained(model_name, mirror=MIRROR, num_labels=2)

# 5. 准备微调数据（示例数据）
# 格式: [("文本", 标签), ...] 其中0=负面，1=正面
train_data = [
    ("这部电影太棒了，演员演技在线！", 1),
    ("剧情拖沓，完全浪费时间。", 0),
    ("特效非常震撼，值得一看。", 1),
    ("台词尴尬，逻辑混乱。", 0),
    ("美术设计精美，视觉享受。", 1),
    ("服务态度很差，送餐超时。", 0),
    ("味道正宗，分量十足。", 1),
    ("包装破损，食物都凉了。", 0),
    ("画面精美，音乐动人。", 1),
    ("剧情老套，毫无新意。", 0)
]

# 6. 数据预处理
texts = [item[0] for item in train_data]
labels = [item[1] for item in train_data]

# 对文本进行编码
encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

# 7. 简化版微调（实际应用需要更完整的训练过程）
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):  # 实际训练可能需要更多轮次
    outputs = model(**encodings, labels=torch.tensor(labels))
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# 8. 创建情感分析管道
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# 9. 待分类的文本
reviews = [
    "美术、服装、布景细节丰富，完全是视觉盛宴！",
    "味道非常一般，跟评论区说的完全不一样。"
]

# 10. 进行分类预测
results = classifier(reviews)

# 11. 输出结果
print("\n情感分类结果：")
for review, result in zip(reviews, results):
    label = "正面" if result["label"] == "LABEL_1" else "负面"
    print(f"评论: {review}")
    print(f"情感倾向: {label}, 置信度: {result['score']:.4f}")
    print("-" * 50)