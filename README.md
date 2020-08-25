##  Aplicação "Market Recommender" é um Data App utilizado para fornecer um serviço automatizado que recomenda leads de um mercado dado uma lista atual de clientes (Portfólio).
 
Contextualização

Projeto desenvolvido durante a **Data Science Codenation** 

Algumas empresas gostariam de saber quem são as demais empresas em um determinado mercado (população) que tem maior probabilidade se tornarem seus próximos clientes. Ou seja, a sua solução deve encontrar no mercado quem são os leads mais aderentes dado as características dos clientes presentes no portfólio do usuário.

Além disso, sua solução deve ser agnóstica ao usuário. Qualquer usuário com uma lista de clientes que queira explorar esse mercado pode extrair valor do serviço.

Para o desafio, deverão ser consideradas as seguintes bases:

Mercado: Base com informações sobre as empresas do Mercado a ser considerado
Portfolio 1: Ids dos clientes da empresa 1
Portfolio 2: Ids dos clientes da empresa 2
Portfolio 3: Ids dos clientes da empresa 3

🦸‍♂️ Arquivos disponíveis em: https://drive.google.com/drive/folders/13yoxj9ErdJRo9o6jR-Kwf-urmdg6KCC1?usp=sharing

🦸‍♂️ Vídeo de apresentação da Aplicação: https://www.youtube.com/watch?v=l9-fOsTjXUo&feature=youtu.be

🦸‍♂️ Demo da Aplicação: https://marketrecommender.herokuapp.com/

## 🛠 Algoritmos ML testados:

- [OneClassSVM]

- [KMEANS]


## 🚀 Como rodar este projeto

🦸‍♂️  Nesta solução o DATASET já tratado precisa estar em ./data/df_mkt_enc.csv porém devido ao limite de github eu o deixei compartilhado na pasta citada com os arquivos de exemplo.

🦸‍♂️  O arquivo do Jupyter Notebook utilizado para o tratamento e análise dos dados é o MarketRecommender.ipynb com todas explicações pertinentes.

# 
1 - Instalar Python 3.8 e o pip
# 
2 - pip install -r requirements.txt
# 
3 - streamlit run app.py






