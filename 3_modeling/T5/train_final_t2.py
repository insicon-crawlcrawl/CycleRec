import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import AdamW
import pandas as pd
from datasets import Dataset
from tqdm import tqdm

# 데이터 로드
df_train = pd.read_csv("/workspace/T5/t5_train_pair_final_t2.csv")
df_val = pd.read_csv("/workspace/T5/t5_valid_pair_final_t2.csv")

# Dataset으로 변환
dataset_train = Dataset.from_pandas(df_train)
dataset_val = Dataset.from_pandas(df_val)

# Tokenizer & Model
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base").to("cuda:3")

# 전처리 함수
def tokenize_function(example):
    input_text = "predict next item: " + example["input"]
    input_enc = tokenizer(
        input_text,
        padding="max_length",
        truncation=True,
        max_length=300
    )
    target_enc = tokenizer(
        example["target"],
        padding="max_length",
        truncation=True,
        max_length=64
    )
    # input_enc["labels"] = [
    #     (l if l != tokenizer.pad_token_id else -100)
    #     for l in target_enc["input_ids"]
    # ]
    input_enc["labels"] = target_enc["input_ids"]
    return input_enc

# 전처리
tokenized_train = dataset_train.map(tokenize_function)
tokenized_val = dataset_val.map(tokenize_function)

# Collate 함수
def collate_fn(batch):
    input_ids = torch.tensor([item["input_ids"] for item in batch])
    attention_mask = torch.tensor([item["attention_mask"] for item in batch])
    labels = torch.tensor([item["labels"] for item in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# DataLoader
train_loader = DataLoader(tokenized_train, batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(tokenized_val, batch_size=64, shuffle=False, collate_fn=collate_fn)

# Optimizer
optimizer = AdamW(model.parameters(), lr=3e-4)

# 검증 함수
def evaluate(model, val_loader):
    model.eval()
    total_val_loss = 0
    loop = tqdm(val_loader, desc="Validating", leave=False)

    with torch.no_grad():
        for batch in loop:
            input_ids = batch["input_ids"].to("cuda:3")
            attention_mask = batch["attention_mask"].to("cuda:3")
            labels = batch["labels"].to("cuda:3")

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_val_loss += loss.item()
            loop.set_postfix(val_loss=loss.item())

    model.train()
    return total_val_loss / len(val_loader)

# 학습 루프
num_epochs = 5
model.train()

for epoch in range(num_epochs):
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for batch in loop:
        input_ids = batch["input_ids"].to("cuda:3")
        attention_mask = batch["attention_mask"].to("cuda:3")
        labels = batch["labels"].to("cuda:3")

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_loader)
    avg_val_loss = evaluate(model, val_loader)
    print(f"\nEpoch {epoch+1} -> train loss: {avg_train_loss:.4f}, val loss: {avg_val_loss:.4f}")

    save_dir = f"/workspace/T5/t5base_pair_final_t2/epoch{epoch+1}"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"모델 저장 완료: {save_dir}")

print("학습 완료!")


test_path = "/workspace/T5/t5_test_pair_final_t2.csv"
output_path = "/workspace/T5/test_r20_predictions(t5base_pair_final_t2).csv"
device = "cuda:3" if torch.cuda.is_available() else "cpu"
model.eval()

# 테스트 데이터
df_test = pd.read_csv(test_path)

# 파라미터
batch_size = 64
max_input_length = 300
max_output_length = 32
num_beams = 20
num_return_sequences = 20
# num_beams = 150
# num_return_sequences = 150


# 결과 저장용 변수
recall_hits = 0
total = 0
results = []

# 배치 단위로 처리
for i in tqdm(range(0, len(df_test), batch_size)):
    batch_df = df_test.iloc[i:i+batch_size]
    input_texts = ["predict next item: " + text for text in batch_df["input"].tolist()]
    targets = batch_df["target"].tolist()

    # 토크나이즈
    inputs = tokenizer(
        input_texts,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_input_length
    ).to(device)

    # 생성
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_output_length,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            early_stopping=True,
            do_sample=False
        )

    # 배치를 위한 구조 재정렬
    batch_predictions = [
        outputs[j * num_return_sequences: (j + 1) * num_return_sequences]
        for j in range(len(batch_df))
    ]

    # 결과 처리
    for idx, preds in enumerate(batch_predictions):
        decoded_preds = [
            tokenizer.decode(pred, skip_special_tokens=True).strip()
            for pred in preds
        ]
        unique_preds = list(dict.fromkeys(decoded_preds))  # 중복 제거

        target = targets[idx].strip().lower()
        hit = target in [p.lower() for p in unique_preds[:20]]

        if hit:
            recall_hits += 1
        total += 1

        results.append({
            "user_id": batch_df.iloc[idx]["user_id"],  
            "input": batch_df.iloc[idx]["input"],
            "target": batch_df.iloc[idx]["target"],
            "order_date_sequence": batch_df.iloc[idx]["order_date_sequence"],
            "top_20": unique_preds[:20],
            "hit": hit
        })


# R@20 계산 및 출력
recall_at_20 = recall_hits / total
print(f"\nRecall@20: {recall_at_20:.4f}")

# R@20 계산 및 출력
# recall_at_150 = recall_hits / total
# print(f"\nRecall@150: {recall_at_150:.4f}")

# 결과 저장
df_result = pd.DataFrame(results)
df_result.to_csv(output_path, index=False)
print("결과 저장 완료")
