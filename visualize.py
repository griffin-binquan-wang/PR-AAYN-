import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
from model import Transformer  # 正确引用你的模型类
from utils import BertTokenizerAdapter, create_masks

# 设置字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

def run_and_visualize(model_path, test_sentence):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 初始化分词器和模型 (参数必须与训练时一致)
    tokenizer = BertTokenizerAdapter("bert-base-multilingual-cased")
    vocab_size = tokenizer.get_vocab_size()
    
    model = Transformer(
        src_vocab=vocab_size, 
        trg_vocab=vocab_size, 
        d_model=512, 
        num_layers=8,
        num_heads=8, 
        d_ff=2048, 
        dropout=0.1
    ).to(device)

    # 2. 加载权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. 处理输入句子
    src_ids, _ = tokenizer.encode(test_sentence, max_len=32)
    src = src_ids.unsqueeze(0).to(device)
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

    # 4. 推理并截取注意力权重
    with torch.no_grad():
        e_outputs = model.encoder(src, src_mask)
        trg_indexes = [tokenizer.tokenizer.cls_token_id]
        
        # 存储模型翻译出的每个词
        translated_tokens = ["[CLS]"]
        
        for i in range(32):
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
            # 生成测试用的 trg_mask
            size = trg_tensor.size(1)
            l_mask = torch.triu(torch.ones((1, size, size), device=device), diagonal=1).type(torch.uint8) == 0
            trg_mask = (trg_tensor != 0).unsqueeze(1).unsqueeze(3) & l_mask
            
            # 运行 Decoder
            # 注意：由于我们修改了 model.py，这里接收两个返回值
            d_output, attn_weights = model.decoder(trg_tensor, e_outputs, src_mask, trg_mask)
            output = model.out(d_output)
            
            # 取最后一个预测词
            next_token = output.argmax(2)[:,-1].item()
            trg_indexes.append(next_token)
            translated_tokens.append(tokenizer.tokenizer.decode([next_token]))

            if next_token == tokenizer.tokenizer.sep_token_id:
                break
        
        # 5. 准备绘图数据
        # attn_weights 形状通常是 [1, num_heads, trg_len, src_len]
        # 我们取所有头的平均值，并去掉 batch 维度
        combined_attn = attn_weights.squeeze(0).mean(dim=0).cpu().numpy()
        
        # 获取源句子的 Token 用于横轴坐标
        src_tokens = [tokenizer.tokenizer.decode([idx]) for idx in src_ids if idx != 0]
        
        # 裁剪注意力矩阵，使其匹配实际句子长度
        final_attn = combined_attn[:len(translated_tokens), :len(src_tokens)]
        
        # 6. 绘图
        fig, ax = plt.subplots(figsize=(10, 8)) # 建议加上 ax
        sns.heatmap(final_attn, xticklabels=src_tokens, yticklabels=translated_tokens, 
                    cmap='viridis', annot=False, ax=ax)
        
        # --- 核心修改：反转 Y 轴，让句子从上往下读 ---
        ax.invert_yaxis() 

        if not os.path.exists('gallery'): os.makedirs('gallery')
        plt.savefig('gallery/attention_map.png')
        print("可视化完成！图片已保存至 gallery/attention_map.png")
        plt.show()

if __name__ == "__main__":
    # 使用你之前的那个 .pth 文件名
    run_and_visualize("transformer_epoch_15.pth", "The world is beautiful.")