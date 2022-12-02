import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

st.write("""<h1>Aplikasi Klasifikasi Penderita Anemia</h1>""",unsafe_allow_html=True)

with st.container():
    with st.sidebar:
        selected = option_menu(
        st.write("""<h3 style = "text-align: center;"><img src="https://lh3.googleusercontent.com/a/ALm5wu2PukBXPMX88VuehLVmYvtTCLj1-XFDgkoky1-JBg=s192-c-rg-br100" width="90" height="90"><br> MUHAMMAD HANIF SANTOSO <p>200411100078 hanifsans05@gmail.com</p></h3>""",unsafe_allow_html=True), 
        ["Home", "Description", "Dataset", "Prepocessing", "Modeling", "Implementation"], 
            icons=['house', 'file-earmark-font', 'bar-chart', 'gear', 'arrow-down-square', 'check2-square'], menu_icon="cast", default_index=1,
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "#FF9933", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px"},
                "nav-link-selected": {"background-color": "teal"},
            }
        )

    if selected == "Home":
        st.image('https://cdn-2.tstatic.net/jatim/foto/bank/images/anemia.jpg', use_column_width=False, width=500)

        st.write("""
        Anemia adalah suatu kondisi di mana Anda kekurangan sel darah merah yang sehat untuk membawa oksigen yang cukup ke jaringan tubuh Anda. Penderita anemia, juga disebut hemoglobin rendah, bisa membuat Anda merasa lelah dan lemah.
        """)

    elif selected == "Description":
        st.subheader("""Pengertian""")
        st.write("""
        Dataset ini merupakan data gejala-gejala penderita anemia yang terdapat di website kaggle.com, data ini nantinya di gunakan untuk melakukan prediksi penyakit anemia. Dataset ini sendiri terdiri dari 6 atribut yaitu Gender, Hemoglobin, MCHC, MCV, MCH dan Hasil.
        """)

        st.subheader("""Kegunaan Dataset""")
        st.write("""
        Dataset ini digunakan untuk melakukan klasifikasi penderita penyakit anemia. Setelah dilakukan klasifikasi selanjutnya dilakukan implementasi dengan memprediksi pasien apakah mengidap anemia atau tidak.
        """)

        st.subheader("""Fitur""")
        st.markdown(
            """
            Fitur-fitur yang terdapat pada dataset:
            - Jenis kelamin: 
                - 0 : laki-laki
                - 1 : perempuan.
            - Hemoglobin: Hemoglobin adalah protein dalam sel darah merah Anda yang membawa oksigen ke organ dan jaringan tubuh Anda dan mengangkut karbon dioksida dari organ dan jaringan Anda kembali ke paru-paru.
                - Normal Range
                    - Wanita Dewasa : 12.0 – 15.5 g/dL.
                    - Pria Dewasa : 13.5 – 17.5 g/dL.
                    - Anak - anak : 11.0 - 13,5 g/dL.
                    - Bayi (3 Bulan) : 9.5 - 12.5 g/dL.
                    - Bayi Baru Lahir : 15.0 - 21.0 g/dL.
            - MCH: MCH atau mean corpuscular hemoglobin adalah pengukuran yang menjelaskan jumlah rata-rata hemoglobin dalam satu sel darah merah (eritrosit).
                - Normal Range : 26 – 33 pg.
            - MCHC: MCHC adalah singkatan dari rata-rata konsentrasi hemoglobin corpuscular. Ini adalah ukuran konsentrasi rata-rata hemoglobin di dalam satu sel darah merah.
                - Normal Range untuk orang Dewasa : 32-36%.
                - Normal Range untuk Bayi Baru Lahir : 31-35%.
            - MCV: MCV adalah singkatan dari mean corpuscular volume. Tes darah MCV mengukur ukuran rata-rata sel darah merah Anda.
                - Normal Range
                    - Dewasa : 80 – 100 fL.
                    - Bayi baru lahir : 98 – 122 fL.
                    - Anak usia 1-3 tahun : 73 – 101 fL.
                    - Anak usia 4-5 tahun : 72 – 88 fL.
                    - Anak usia 6-10 tahun : 69 – 93 fL.
            - Hasil: 
                - 0 : Negative Anemia
                - 1 : Positive Anemia.
            """
        )

        st.subheader("""Sumber Dataset""")
        st.write("""
        Sumber data di dapatkan melalui website kaggle.com, Berikut merupakan link untuk mengakses sumber dataset.
        <a href="https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset">Klik disini</a>""", unsafe_allow_html=True)
        
        st.subheader("""Tipe Data""")
        st.write("""
        Tipe data yang di gunakan pada dataset anemia ini adalah NUMERICAL.
        """)

    elif selected == "Dataset":
        st.subheader("""Dataset Anemia""")
        df = pd.read_csv('https://raw.githubusercontent.com/HanifSantoso05/dataset_matkul/main/anemia.csv')
        st.dataframe(df)

    elif selected == "Prepocessing":
        st.subheader("""Normalisasi Data""")
        st.write("""Rumus Normalisasi Data :""")
        st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
        df = pd.read_csv('https://raw.githubusercontent.com/HanifSantoso05/dataset_matkul/main/anemia.csv')
        st.markdown("""
        Dimana :
        - X = data yang akan dinormalisasi atau data asli
        - min = nilai minimum semua data asli
        - max = nilai maksimum semua data asli
        """)
        #Mendefinisikan Varible X dan Y
        X = df.drop(columns=['Result'])
        y = df['Result'].values
        df_min = X.min()
        df_max = X.max()
        
        #NORMALISASI NILAI X
        scaler = MinMaxScaler()
        #scaler.fit(features)
        #scaler.transform(features)
        scaled = scaler.fit_transform(X)
        features_names = X.columns.copy()
        #features_names.remove('label')
        scaled_features = pd.DataFrame(scaled, columns=features_names)

        st.subheader('Hasil Normalisasi Data')
        st.write(scaled_features)

        st.subheader('Target Label')
        dumies = pd.get_dummies(df.Result).columns.values.tolist()
        dumies = np.array(dumies)

        labels = pd.DataFrame({
            'Positive' : [dumies[1]],
            'Negative' : [dumies[0]]
        })

        st.write(labels)

    elif selected == "Modeling":
        #Read Dataset
        df = pd.read_csv('https://raw.githubusercontent.com/HanifSantoso05/dataset_matkul/main/anemia.csv')

        #Preprocessing data
        #Mendefinisikan Varible X dan Y
        X = df.drop(columns=['Result'])
        y = df['Result'].values
        
        #NORMALISASI NILAI X
        scaler = MinMaxScaler()
        #scaler.fit(features)
        #scaler.transform(features)
        scaled = scaler.fit_transform(X)
        features_names = X.columns.copy()
        #features_names.remove('label')
        scaled_features = pd.DataFrame(scaled, columns=features_names)

        #Split Data 
        training, test = train_test_split(scaled_features,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
        training_label, test_label = train_test_split(y, test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing

        with st.form("modeling"):
            st.subheader('Modeling')
            st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
            naive = st.checkbox('Gaussian Naive Bayes')
            k_nn = st.checkbox('K-Nearest Neighboor')
            destree = st.checkbox('Decission Tree')
            submitted = st.form_submit_button("Submit")

            #Gaussian Naive Bayes
            gaussian = GaussianNB()
            gaussian = gaussian.fit(training, training_label)

            probas = gaussian.predict_proba(test)
            probas = probas[:,1]
            probas = probas.round()

            gaussian_akurasi = round(100 * accuracy_score(test_label,probas))

            #KNN
            K=10
            knn=KNeighborsClassifier(n_neighbors=K)
            knn.fit(training,training_label)
            knn_predict=knn.predict(test)

            knn_akurasi = round(100 * accuracy_score(test_label,knn_predict))

            #Decission Tree
            dt = DecisionTreeClassifier()
            dt.fit(training, training_label)
            # prediction
            dt_pred = dt.predict(test)
            #Accuracy
            dt_akurasi = round(100 * accuracy_score(test_label,dt_pred))

            if submitted :
                if naive :
                    st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(gaussian_akurasi))
                if k_nn :
                    st.write("Model KNN accuracy score : {0:0.2f}" . format(knn_akurasi))
                if destree :
                    st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(dt_akurasi))
            
            grafik = st.form_submit_button("Grafik akurasi semua model")
            if grafik:
                data = pd.DataFrame({
                    'Akurasi' : [gaussian_akurasi, knn_akurasi, dt_akurasi],
                    'Model' : ['Gaussian Naive Bayes', 'K-NN', 'Decission Tree'],
                })

                chart = (
                    alt.Chart(data)
                    .mark_bar()
                    .encode(
                        alt.X("Akurasi"),
                        alt.Y("Model"),
                        alt.Color("Akurasi"),
                        alt.Tooltip(["Akurasi", "Model"]),
                    )
                    .interactive()
                )
                st.altair_chart(chart,use_container_width=True)

    elif selected == "Implementation":
        with st.form("my_form"):
            #Read Dataset
            df = pd.read_csv('https://raw.githubusercontent.com/HanifSantoso05/dataset_matkul/main/anemia.csv')

            #Preprocessing data
            #Mendefinisikan Varible X dan Y
            X = df.drop(columns=['Result'])
            y = df['Result'].values
            
            #NORMALISASI NILAI X
            scaler = MinMaxScaler()
            #scaler.fit(features)
            #scaler.transform(features)
            scaled = scaler.fit_transform(X)
            features_names = X.columns.copy()
            #features_names.remove('label')
            scaled_features = pd.DataFrame(scaled, columns=features_names)

            #Split Data 
            training, test = train_test_split(scaled_features,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
            training_label, test_label = train_test_split(y, test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing

            #Gaussian Naive Bayes
            gaussian = GaussianNB()
            gaussian = gaussian.fit(training, training_label)
            probas = gaussian.predict_proba(test)
            probas = probas[:,1]
            probas = probas.round()

            #KNN
            K=10
            knn=KNeighborsClassifier(n_neighbors=K)
            knn.fit(training,training_label)
            knn_predict=knn.predict(test)

            #Decission Tree
            dt = DecisionTreeClassifier()
            dt.fit(training, training_label)
            dt_pred = dt.predict(test)


            st.subheader("Implementasi Prediksi Penyakit Diabetes")
            Gender = st.slider('Jenis Kelamin', 0, 1)
            Hemoglobin = st.number_input('Hemoglobin')
            MCH = st.number_input('MCH (Mean Cell Hemoglobin)')
            MCHC = st.number_input('MCHC (Mean corpuscular hemoglobin concentration)')
            MCV = st.number_input('MCV (Mean Cell Volume)')
            model = st.selectbox('Pilihlah model yang akan anda gunakan untuk melakukan prediksi?',
                    ('Gaussian Naive Bayes', 'K-NN', 'Decision Tree'))

            prediksi = st.form_submit_button("Submit")
            if prediksi:
                inputs = np.array([
                    Gender,
                    Hemoglobin,
                    MCH,
                    MCHC,
                    MCV
                ])
                
                df_min = X.min()
                df_max = X.max()
                input_norm = ((inputs - df_min) / (df_max - df_min))
                input_norm = np.array(input_norm).reshape(1, -1)

                if model == 'Gaussian Naive Bayes':
                    mod = gaussian
                    akurasi = round(100 * accuracy_score(test_label,probas))
                if model == 'K-NN':
                    mod = knn 
                    akurasi = round(100 * accuracy_score(test_label,knn_predict))
                if model == 'Decision Tree':
                    mod = dt
                    akurasi = round(100 * accuracy_score(test_label,dt_pred))

                input_pred = mod.predict(input_norm)

                st.subheader('Hasil Prediksi')
                st.write('Menggunakan Pemodelan :',model)
                st.write('Akurasi: {0:0.2f}'. format(akurasi))
                

                if input_pred == 1:
                    st.error('Positive')
                else:
                    st.success('Negative')