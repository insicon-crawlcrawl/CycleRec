import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
from tqdm import tqdm

# 경로 설정
model_path = "/workspace/T5/t5base_pair_final_nn/epoch5"
test_path = "/workspace/T5/t5_test_pair_final_nn.csv"
ama_path = "/workspace/T5/transaction_categorized_n.csv"
output_path = "/workspace/T5/test_r20_n.csv"

# 디바이스 설정
device = "cuda:2" if torch.cuda.is_available() else "cpu"

# 모델 및 토크나이저 로드
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
model.eval()

# 테스트 데이터
df_test = pd.read_csv(test_path)

# transaction_categorized_n.csv 로드 및 ASIN→카테고리 매핑
df_ama = pd.read_csv(ama_path)
asin_to_category = df_ama.set_index("ASIN/ISBN (Product Code)")["Category"].to_dict()

# 파라미터
batch_size = 64
max_input_length = 256
max_output_length = 32
num_beams = 20
num_return_sequences = 20

# 결과 저장용 변수
recall_hits = 0
total = 0
results = []

# 배치 단위로 처리
for i in tqdm(range(0, len(df_test), batch_size)):
    batch_df = df_test.iloc[i:i+batch_size]
    input_texts = ["predict next item: " + text for text in batch_df["input"].tolist()]
    targets = batch_df["target"].tolist()
    order_date_sequences = batch_df["order_date_sequence"].tolist()  # order_date_sequence 추출

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

        target_code = targets[idx].strip()
        target_category = asin_to_category.get(target_code, None)

        match_count = 0
        pred_categories = []

        for pred_code in unique_preds[:20]:
            pred_category = asin_to_category.get(pred_code, None)
            pred_categories.append(pred_category)
            if pred_category is not None and target_category is not None:
                if pred_category == target_category:
                    match_count += 1

        hit = match_count > 0
        
        total += 1

        if hit:
            recall_hits += 1

        results.append({
            "user_id": batch_df.iloc[idx]["user_id"],
            "input": batch_df.iloc[idx]["input"],
            "target": target_code,
            "target_category": target_category,
            "order_date_sequence": order_date_sequences[idx],  # 그대로 유지
            "top_20_predictions": unique_preds[:20],
            "top_20_categories": pred_categories,
            "match_count": match_count,
            "hit": hit
        })

# recall 계산 및 출력
recall_at_20 = recall_hits / total
print(f"\nCategory-level Recall@20: {recall_at_20:.4f}")

# 결과 저장
df_result = pd.DataFrame(results)
df_result.to_csv(output_path, index=False)
print(f"결과 저장 완료: {output_path}")
