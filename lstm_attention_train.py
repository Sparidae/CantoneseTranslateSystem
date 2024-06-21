import numpy as np
import torch
from data_process import DataProcess
# from metrics import compute_metrics
from lstm_attention import Config,LA_Model
from torch.utils.data import DataLoader
from utils import get_logger
from tqdm import tqdm
logger = get_logger("Train_lstm+attention")

def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 清空梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(input_ids, attention_mask, labels)
        
        # 计算损失
        loss = outputs['loss']
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(data_loader)

def validate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, labels)
            loss = outputs['loss']
            total_loss += loss.item()
    
    return total_loss / len(data_loader)
if __name__ == "__main__":

    # 加载数据
    logger.info("Load Data")
    data = DataProcess()
    dataset = data.get_dataset(test_size=0.2)
    # print(dataset)
    # print(dataset["train"][1])
    train_dataset = dataset['train']
    val_dataset = dataset['test']

    # 模型配置
    config = Config(embedding_dim=256,hidden_size=256, num_layers=1, dropout=0.1, vocab_size=10000)

    # 初始化模型
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LA_Model(config)
    model.to(device)
    # model=model.load_state_dict(torch.load('simple_model_state_dict.pth'))
    # 设置优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 训练模型
    epochs = 10
    max_val_loss=1000
    for epoch in tqdm(range(epochs)):
        train_loss = train_one_epoch(model,train_loader, criterion, optimizer,device)
        val_loss = validate(model, val_loader, criterion, device)
        if val_loss<max_val_loss:
           max_val_loss=val_loss
           torch.save(model, 'model.pth')
        # 保存模型的状态字典
        torch.save(model.state_dict(), 'simple_model_state_dict.pth')
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    torch.save(model, 'model.pth')