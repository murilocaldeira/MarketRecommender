import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import streamlit as st

def realiza_predicao(pt1):
    # carregando dados do mercado
    df_mkt_cluster  = pd.read_csv('data/df_mkt_enc.csv')
    df_mkt_cluster = df_mkt_cluster.set_index('id')
    df_port1 = df_mkt_cluster[df_mkt_cluster.index.isin(pt1.index)]

    # gera o modelo de KMeans
    kmeans = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=10,
           n_clusters=6, n_init=30, n_jobs=None, precompute_distances='auto',
           random_state=42, tol=0.0001, verbose=0)

    # treina o modelo
    kmeans.fit(df_port1)

    #salva os clusters definidos pelo Kmeans de cada emrpesa do portfólio
    #salva adiciona nos dados do mercado os clusters previstos
    clusters_port1 = kmeans.labels_
    clusters_mkt =  kmeans.predict(df_mkt_cluster)

    #salva as distancias para os clusters dos dados do portfólio
    distancia_port1 = kmeans.transform(df_port1)
    distancia_port1 = pd.DataFrame(distancia_port1)   
    
    #salva as distancias para os clusters dos dados do mercado
    distancia_mkt = kmeans.transform(df_mkt_cluster)
    distancia_mkt = pd.DataFrame(distancia_mkt)

    #salva cluster no dataset do mercado
    df_mkt_cluster['cluster'] = clusters_mkt

    #salvar lista das distancias de cada cluster dos dados do portfólio
    dist_port1_cluster0 = (distancia_port1[0][clusters_port1 == 0]).to_list()
    dist_port1_cluster1 = (distancia_port1[1][clusters_port1 == 1]).to_list()
    dist_port1_cluster2 = (distancia_port1[2][clusters_port1 == 2]).to_list()
    dist_port1_cluster3 = (distancia_port1[3][clusters_port1 == 3]).to_list()
    dist_port1_cluster4 = (distancia_port1[4][clusters_port1 == 4]).to_list()
    dist_port1_cluster5 = (distancia_port1[5][clusters_port1 == 5]).to_list()

    #com base na distribuição de distancias de cada cluster, pegar um limite máximo aceitavel para prever no mercado mais a frente
    if dist_port1_cluster0:
      limite_max_cluster0 = np.quantile(dist_port1_cluster0, 0.95)

    if dist_port1_cluster1:
      limite_max_cluster1 = np.quantile(dist_port1_cluster1, 0.95)

    if dist_port1_cluster2:
      limite_max_cluster2 = np.quantile(dist_port1_cluster2, 0.95)

    if dist_port1_cluster3:
      limite_max_cluster3 = np.quantile(dist_port1_cluster3, 0.95)

    if dist_port1_cluster4:
      limite_max_cluster4 = np.quantile(dist_port1_cluster4, 0.95)

    if dist_port1_cluster5:
      limite_max_cluster5 = np.quantile(dist_port1_cluster5, 0.95)

    df_mkt_cluster.insert(10, 'predito', False)

    # Para cada Cluster:
    # pegar dados mercado.
    # pegaras distancias previstas desses dados. 
    # definir um tataset de predicoes com as emrpesas que tenham distancia dentro so limite estabelecido para o cluster
    # se estiver dentro do limite, setar coluna "predito" no data set principal para TRUE

    mkt_cluster0       = (df_mkt_cluster[:][clusters_mkt == 0])
    dist_mkt_cluster0  = np.array(distancia_mkt[0][clusters_mkt == 0])
    preditos_cluster0 = mkt_cluster0[:][dist_mkt_cluster0 <= limite_max_cluster0]
    df_mkt_cluster['predito'][df_mkt_cluster.index.isin(preditos_cluster0.index)] = True

    mkt_cluster1       = (df_mkt_cluster[:][clusters_mkt == 1])
    dist_mkt_cluster1  = np.array(distancia_mkt[1][clusters_mkt == 1])
    preditos_cluster1 = mkt_cluster1[:][dist_mkt_cluster1 <= limite_max_cluster1]
    df_mkt_cluster['predito'][df_mkt_cluster.index.isin(preditos_cluster1.index)] = True

    mkt_cluster2       = (df_mkt_cluster[:][clusters_mkt == 2])
    dist_mkt_cluster2  = np.array(distancia_mkt[2][clusters_mkt == 2])
    preditos_cluster2 = mkt_cluster2[:][dist_mkt_cluster2 <= limite_max_cluster2]
    df_mkt_cluster['predito'][df_mkt_cluster.index.isin(preditos_cluster2.index)] = True

    mkt_cluster3       = (df_mkt_cluster[:][clusters_mkt == 3])
    dist_mkt_cluster3  = np.array(distancia_mkt[3][clusters_mkt == 3])
    preditos_cluster3 = mkt_cluster3[:][dist_mkt_cluster3 <= limite_max_cluster3]
    df_mkt_cluster['predito'][df_mkt_cluster.index.isin(preditos_cluster3.index)] = True

    mkt_cluster4       = (df_mkt_cluster[:][clusters_mkt == 4])
    dist_mkt_cluster4  = np.array(distancia_mkt[4][clusters_mkt == 4])
    preditos_cluster4 = mkt_cluster4[:][dist_mkt_cluster4 <= limite_max_cluster4]
    df_mkt_cluster['predito'][df_mkt_cluster.index.isin(preditos_cluster4.index)] = True

    mkt_cluster5       = (df_mkt_cluster[:][clusters_mkt == 5])
    dist_mkt_cluster5  = np.array(distancia_mkt[5][clusters_mkt == 5])
    preditos_cluster5 = mkt_cluster5[:][dist_mkt_cluster5 <= limite_max_cluster5]
    df_mkt_cluster['predito'][df_mkt_cluster.index.isin(preditos_cluster5.index)] = True

    cli_atual = df_mkt_cluster.index.isin(pt1.index) 

    df_result = df_mkt_cluster.copy()

    df_result.insert(11, 'already_client', cli_atual)

    df_leads_kmeans = df_result.loc[(df_result['predito']==True) & (df_result['already_client']==False)]

    #Devido limitação do streamlit foi limitado ao maximo de Cento e Cinquenta Mil Leads
    #Aproxmadamente 32,5% do total do mercado
    return df_leads_kmeans[:][:150000]

def plota_graficos(leads_port1, pt1):
    #barra de loading
    bar = st.progress(0)
    latest_iteration = st.empty()
    latest_iteration.text('Analisando Dados')

    #carrega dados de portfólio, leads previstos e mercado
    df_mkt =  pd.read_csv('data/df_mkt_enc.csv', index_col = 0)
    df_port1 = df_mkt[df_mkt.index.isin(pt1.index)]

    clusters_leads_port1 = leads_port1['cluster']

    leads_port1 = df_mkt[df_mkt.index.isin(leads_port1.index)]

    #barra de loading
    bar.progress(10)
    latest_iteration.text('Aplicando Modelo')

    #aplicando kmeans no portfólio para gerar os clusters
    kmeans = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=10,
           n_clusters=6, n_init=30, n_jobs=None, precompute_distances='auto',
           random_state=42, tol=0.0001, verbose=0)

    kmeans.fit(df_port1)

    #buscando clusters previstos para portfólio, leads previstos e mercado
    mkt_predict = kmeans.predict(df_mkt)
    df_port1_predict = kmeans.labels_
    leads_port1_predict = clusters_leads_port1

    #aplicando PCA para reduzir dimenções e utilizar PC1 e PC2 para visualizar os dados
    pca = PCA(0.95)
    pca.fit(df_port1)

    #gerando lista de componentes do PCA para portfólio, leads previstos e mercado
    new_mkt = pca.transform(df_mkt)
    new_port1 = pca.transform(df_port1)
    new_leads_port1 = pca.transform(leads_port1)
    new_centroids = pca.transform(kmeans.cluster_centers_)

    #definindo range de cores, tamanho do gráfico e posição da legenda
    colors = cm.rainbow(np.linspace(0, 1, 18))
    plt.rcParams['figure.figsize'] = (10,10)
    plt.legend(loc='best')

    #barra de loading
    bar.progress(25)
    latest_iteration.text('Por favor, aguarde para visualizar os gráficos...')

    #plotando infomações em gráficos separados----------------------------------------------------

    #Plotando empresas do portfólio
    plt.title('EMPRESAS DE SEU PORTFÓLIO')
    print(plt.scatter(new_port1[df_port1_predict == 0, 0], new_port1[df_port1_predict == 0, 1], s=30, color=colors[0], label='cluster1_port'))
    print(plt.scatter(new_port1[df_port1_predict == 1, 0], new_port1[df_port1_predict == 1, 1], s=30, color=colors[1], label='cluster2_port'))
    print(plt.scatter(new_port1[df_port1_predict == 2, 0], new_port1[df_port1_predict == 2, 1], s=30, color=colors[2], label='cluster3_port'))
    print(plt.scatter(new_port1[df_port1_predict == 3, 0], new_port1[df_port1_predict == 3, 1], s=30, color=colors[3], label='cluster4_port'))
    print(plt.scatter(new_port1[df_port1_predict == 4, 0], new_port1[df_port1_predict == 4, 1], s=30, color=colors[4], label='cluster5_port'))
    print(plt.scatter(new_port1[df_port1_predict == 5, 0], new_port1[df_port1_predict == 5, 1], s=30, color=colors[5], label='cluster6_port'))

    plt.scatter(new_centroids[:, 0], new_centroids[:, 1], s=30, color='black', label='centroid_port')
    plt.legend()
    st.pyplot()

    #Plotando empresas recomendadas
    plt.title('EMPRESAS RECOMENDADAS')
    print(plt.scatter(new_leads_port1[leads_port1_predict == 0, 0], new_leads_port1[leads_port1_predict == 0, 1], s=30, color=colors[6], label='cluster1_leads'))
    print(plt.scatter(new_leads_port1[leads_port1_predict == 1, 0], new_leads_port1[leads_port1_predict == 1, 1], s=30, color=colors[7], label='cluster2_leads'))
    print(plt.scatter(new_leads_port1[leads_port1_predict == 2, 0], new_leads_port1[leads_port1_predict == 2, 1], s=30, color=colors[8], label='cluster3_leads'))
    print(plt.scatter(new_leads_port1[leads_port1_predict == 3, 0], new_leads_port1[leads_port1_predict == 3, 1], s=30, color=colors[9], label='cluster4_leads'))
    print(plt.scatter(new_leads_port1[leads_port1_predict == 4, 0], new_leads_port1[leads_port1_predict == 4, 1], s=30, color=colors[10], label='cluster5_leads'))
    print(plt.scatter(new_leads_port1[leads_port1_predict == 5, 0], new_leads_port1[leads_port1_predict == 5, 1], s=30, color=colors[11], label='cluster6_leads'))

    plt.scatter(new_centroids[:, 0], new_centroids[:, 1], s=30, color='black', label='centroid_port')
    plt.legend()
    st.pyplot()

    #barra de loading
    bar.progress(35)

    #Plotando empresas do mercado
    plt.title('EMPRESAS DO MERCADO')
    print(plt.scatter(new_mkt[mkt_predict == 0, 0], new_mkt[mkt_predict == 0, 1], s=30, color=colors[12], label='cluster1_mkt'))
    print(plt.scatter(new_mkt[mkt_predict == 1, 0], new_mkt[mkt_predict == 1, 1], s=30, color=colors[13], label='cluster2_mkt'))
    print(plt.scatter(new_mkt[mkt_predict == 2, 0], new_mkt[mkt_predict == 2, 1], s=30, color=colors[14], label='cluster3_mkt'))
    print(plt.scatter(new_mkt[mkt_predict == 3, 0], new_mkt[mkt_predict == 3, 1], s=30, color=colors[15], label='cluster4_mkt'))
    print(plt.scatter(new_mkt[mkt_predict == 4, 0], new_mkt[mkt_predict == 4, 1], s=30, color=colors[16], label='cluster5_mkt'))
    print(plt.scatter(new_mkt[mkt_predict == 5, 0], new_mkt[mkt_predict == 5, 1], s=30, color=colors[17], label='cluster6_mkt'))

    #barra de loading
    bar.progress(55)

    plt.scatter(new_centroids[:, 0], new_centroids[:, 1], s=30, color='black', label='centroid_port')
    plt.legend()
    st.pyplot()

    #barra de loading
    bar.progress(60)

    #Plotando tudo em um mesmo gráfico
    plt.title('EMPRESAS DO MERCADO, PORTFÓLIO E RECOMENDADAS')

    #Plotando empresas do mercado
    print(plt.scatter(new_mkt[mkt_predict == 0, 0], new_mkt[mkt_predict == 0, 1], s=30, color=colors[12], label='cluster1_mkt'))
    print(plt.scatter(new_mkt[mkt_predict == 1, 0], new_mkt[mkt_predict == 1, 1], s=30, color=colors[13], label='cluster2_mkt'))
    print(plt.scatter(new_mkt[mkt_predict == 2, 0], new_mkt[mkt_predict == 2, 1], s=30, color=colors[14], label='cluster3_mkt'))
    print(plt.scatter(new_mkt[mkt_predict == 3, 0], new_mkt[mkt_predict == 3, 1], s=30, color=colors[15], label='cluster4_mkt'))
    print(plt.scatter(new_mkt[mkt_predict == 4, 0], new_mkt[mkt_predict == 4, 1], s=30, color=colors[16], label='cluster5_mkt'))
    print(plt.scatter(new_mkt[mkt_predict == 5, 0], new_mkt[mkt_predict == 5, 1], s=30, color=colors[17], label='cluster6_mkt'))

    #barra de loading
    bar.progress(65)

    #Plotando empresas do portfólio
    print(plt.scatter(new_port1[df_port1_predict == 0, 0], new_port1[df_port1_predict == 0, 1], s=30, color=colors[0], label='cluster1_port'))
    print(plt.scatter(new_port1[df_port1_predict == 1, 0], new_port1[df_port1_predict == 1, 1], s=30, color=colors[1], label='cluster2_port'))
    print(plt.scatter(new_port1[df_port1_predict == 2, 0], new_port1[df_port1_predict == 2, 1], s=30, color=colors[2], label='cluster3_port'))
    print(plt.scatter(new_port1[df_port1_predict == 3, 0], new_port1[df_port1_predict == 3, 1], s=30, color=colors[3], label='cluster4_port'))
    print(plt.scatter(new_port1[df_port1_predict == 4, 0], new_port1[df_port1_predict == 4, 1], s=30, color=colors[4], label='cluster5_port'))
    print(plt.scatter(new_port1[df_port1_predict == 5, 0], new_port1[df_port1_predict == 5, 1], s=30, color=colors[5], label='cluster6_port'))

    #barra de loading
    bar.progress(70)
    latest_iteration.text('Por favor, aguarde para visualizar os gráficos, finalizando...')

    #Plotando empresas recomendadas
    print(plt.scatter(new_leads_port1[leads_port1_predict == 0, 0], new_leads_port1[leads_port1_predict == 0, 1], s=30, color=colors[6], label='cluster1_leads'))
    print(plt.scatter(new_leads_port1[leads_port1_predict == 1, 0], new_leads_port1[leads_port1_predict == 1, 1], s=30, color=colors[7], label='cluster2_leads'))
    print(plt.scatter(new_leads_port1[leads_port1_predict == 2, 0], new_leads_port1[leads_port1_predict == 2, 1], s=30, color=colors[8], label='cluster3_leads'))
    print(plt.scatter(new_leads_port1[leads_port1_predict == 3, 0], new_leads_port1[leads_port1_predict == 3, 1], s=30, color=colors[9], label='cluster4_leads'))
    print(plt.scatter(new_leads_port1[leads_port1_predict == 4, 0], new_leads_port1[leads_port1_predict == 4, 1], s=30, color=colors[10], label='cluster5_leads'))
    print(plt.scatter(new_leads_port1[leads_port1_predict == 5, 0], new_leads_port1[leads_port1_predict == 5, 1], s=30, color=colors[11], label='cluster6_leads'))
   
    #barra de loading
    bar.progress(90)

    plt.scatter(new_centroids[:, 0], new_centroids[:, 1], s=30, color='black', label='centroid_port')
    plt.legend()
    st.pyplot()

    #barra de loading
    latest_iteration.text('GRÁFICOS GERADOS!')
    bar.progress(100)