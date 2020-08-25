import numpy as np
import pandas as pd
import streamlit as st
from predic import realiza_predicao, plota_graficos
import base64

def get_table_download_link(df1):
    csv = df1.to_csv()
    #barra de loading
    bar.progress(75)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="LeadsRecomendados.csv">Download Leads Recomendados!!!</a>'
    return href
#-------------------------------------------------------------------------------------
#Implemetação do streamlit app
#-------------------------------------------------------------------------------------
st.set_option('deprecation.showfileUploaderEncoding', False)

# título
st.title("Market Recommender - Prevendo Leads")

# subtítulo
st.markdown("O objetivo deste produto é fornecer um serviço automatizado que recomenda leads de um mercado dado uma lista atual de clientes (Portfólio).")
st.markdown("Os dados do mercado utilizado e de alguns portfólios de exemplo estão disponíveis em:")
href =  f'<a href="https://drive.google.com/drive/folders/13yoxj9ErdJRo9o6jR-Kwf-urmdg6KCC1?usp=sharing">Download Dados Exemplo</a>'
st.markdown(href, unsafe_allow_html=True)
href_git =  f'<a href="https://github.com/murilocaldeira/MarketRecommender.git">Veja no GitHub!</a>'
st.markdown(href_git, unsafe_allow_html=True)  

st.subheader("Faça o upload de sua base de clientes: ")
file_port  = st.file_uploader('', type = 'csv')

if file_port:
    btn_predict= st.button("Realizar predição de Leads")
    if btn_predict:
        #barra de loading
        bar = st.progress(0)
        latest_iteration = st.empty()
        latest_iteration.text('Gerando Leads..')
        
        pt1 = pd.read_csv(file_port)

        #barra de loading
        bar.progress(20)

        pt1 = pt1.set_index('id')
        df_leads = realiza_predicao(pt1)

        #barra de loading
        bar.progress(40)

        st.markdown("Leads Recomendados")
        df_leads.index

        #barra de loading
        bar.progress(50)
        latest_iteration.text('Gerando Arquivo para Download...')

        st.markdown(get_table_download_link(df_leads), unsafe_allow_html=True)

        #barra de loading
        bar.progress(100)
        latest_iteration.text('ARQUIVO DE DOWNLOAD PRONTO!')

        st.subheader("Gráficos Comparativos: portfólios x Previsões x Mercado")

        plota_graficos(df_leads, pt1) 






   

