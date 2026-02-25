import os
import json
import urllib.request
from pathlib import Path

def download_and_convert_squad(output_path: str, max_examples: int = 2000):
    url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
    
    print(f"Downloading SQuAD v2.0 dataset from {url}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            squad_data = json.loads(response.read().decode())
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        return

    print("Converting to Alpaca format...")
    alpaca_data = []
    
    # SQuAD format: data -> paragraphs -> context -> qas -> question
    for article in squad_data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                
                # Check if it has an answer or is explicitly unanswerable
                is_impossible = qa.get('is_impossible', False)
                if is_impossible:
                    answer = "I am sorry, but the provided context does not contain the answer."
                else:
                    if not qa['answers']:
                        continue
                    answer = qa['answers'][0]['text']

                # Format as Alpaca JSONL: {instruction, input, output}
                # For RAG, the "context" is the input, pushing the model to answer based on it.
                instruction = f"Answer the following question based on the provided context.\n\nContext:\n{context}\n\nQuestion:\n{question}"
                
                alpaca_data.append({
                    "instruction": instruction,
                    "input": "",
                    "output": answer
                })
                
                if len(alpaca_data) >= max_examples:
                    break
            if len(alpaca_data) >= max_examples:
                break
        if len(alpaca_data) >= max_examples:
            break

    # Save to JSONL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in alpaca_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"Successfully saved {len(alpaca_data)} examples to {output_path}")

if __name__ == "__main__":
    out_dir = Path(__file__).parent.parent / "data" / "training" / "squad_qa.jsonl"
    download_and_convert_squad(str(out_dir), max_examples=2000)
