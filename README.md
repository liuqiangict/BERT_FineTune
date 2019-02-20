### **QueryPairModels**
#### Usage
* Command: python main.py --modeltype (modeltype)
* Parameter Priorty: --(modeltype)\_cfg k:v > config/(modeltype)\_config.json > --other
* Data Reader:
  - Preprocessor: Set in header as "query:(preprocessor\_name)" or "query:(prerpocessor\_name):(preprocessor\_index):(preprocessor\_columnIdx)", will convert using corresponding prerpocessor, inference/score mode will keep origin fields
  - prefetch\_buffer best set to GPU count?
* Adding Model:
  - model/config class inherit model/base\_model
  - three level param
  - add index in main.py
#### Support list
* Model type: cdssm, bert2seq, bertpair2seq, bert\_qk, xletter2seq
* Mode: train (include data reader mode: eval\_auc,eval\_bleu), infer, score
* Preprocessor name: xletter, bertseq, bertpair
#### Features
* --local N (Use N GPU by setting os environ CUDA\_VISIBLE\_DEVICES)
* --timeline\_enable & --timeline\_desc will enable profiling, the profile log will saved in log folder
* --init\_status to print trainable parameter init source
* multi GPU on single machine
  - default setting use parameter sharing strategy
  - Grad sharing between GPUS
    * --grad\_mode:1 enable this mode 
    * --grad\_float16:1 cast grad to float16 before sharing 
#### Models
1 .CDSSM(modeltype=CDSSM): Based on https://www.microsoft.com/en-us/research/publication/a-convolutional-latent-semantic-model-for-web-search/
  * Options 
    * input\_mode: mstf(mstf ops), pyfunc(extract xletter in data reader), pyfunc\_batch(customized ops)
    * maxpooling\_mode: mstf(mstf ops), emb(sparse embedding)
  * Training Speed:  (bs=128, neg=4, 288->64)

| Config        | #GPU          | Trainer Setting  | Speed(e/s) | 
|:-------------:|:-------------:|:----------------:|:-----:|
| input=mstf, maxp=mstf| 1 | GM=0,G16=0 | 13800 |
| input=mstf, maxp=mstf| 1 | GM=1,G16=0 | 13000 |
| input=mstf, maxp=mstf| 1 | GM=1,G16=1 | 12300 |
| input=mstf, maxp=mstf| 2 | GM=0,G16=0 | 6000 |
| input=mstf, maxp=mstf| 2 | GM=1,G16=0 | 27600 |
| input=mstf, maxp=mstf| 2 | GM=1,G16=1 | 28000 |
| input=mstf, maxp=emb| 1 | GM=0,G16=0 | 7060 |
| input=mstf, maxp=emb| 2 | GM=0,G16=0 | 14200 |
| input=mstf, maxp=emb| 2 | GM=0,G16=0 | 14400 |
| input=mstf, maxp=emb| 2 | GM=1,G16=1 | 13800 |
| input=pyfunc, maxp=emb| 2 | GM=1,G16=0 | 1050 |
| input=pyfunc\_batch, maxp=emb| 2 | GM=1,G16=0 | 3000 |
2. Bert2Seq, BertPair2Seq

3. Seq2Seq Encoder: xletter, Decoder: term

4. QDocTreeRetrieve -- Based on the idea of Learning Tree-based Deep Model for Recommender Systems https://arxiv.org/pdf/1801.02294.pdf
