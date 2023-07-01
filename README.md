# Multilingual Knowledge-Based Question Answering

Mengshi Ma's master's thesis

## Generate train data set

Run the following commands to generate data sets for fine-tuning. 

If linguistic_context is not needed, change it to `--no-linguistic_context`.

If entity_knowledge is not needed, change it to `--no-entity_knowledge`.

lcquad1:
```bash
python3 code/generate_train_csv.py -i datasets/lcquad1/train-data.json -o datasets/lcquad1/train.csv -t lcquad1 --linguistic_context --entity_knowledge
```

lcquad2:
```bash
python3 code/generate_train_csv.py \
-i datasets/lcquad2/train.json \ 
-o datasets/lcquad2/train.csv \
-t lcquad2 \
--linguistic_context \
--entity_knowledge
```

qald dbpedia:
```bash
python3 code/generate_train_csv.py \
-i datasets/qald9plus/dbpedia/qald_9_plus_train_dbpedia.json \
-o datasets/qald9plus/dbpedia/qald_9_plus-train_dbpedia.csv \
-t qald \
-kg DBpedia \
-l all \
--linguistic_context \
--entity_knowledge
```

qald wikidata:
```bash
python3 code/generate_train_csv.py \
-i datasets/qald9plus/wikidata/qald_9_plus_train_wikidata.json \
-o datasets/qald9plus/wikidata/qald_9_plus_train_wikidata.csv \
-t qald \
-kg Wikidata \ 
-l all \
--linguistic_context \
--entity_knowledge
```

## Train on a csv dataset

`train_ds.sh` or `train.sh`

## Evaluation

`eval.sh`

