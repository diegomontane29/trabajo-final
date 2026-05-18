import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Bank Marketing Dashboard",
    layout="wide"
)

# =========================================================
# SESSION STATE
# =========================================================
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("📌 Menú")

menu = st.sidebar.radio(
    "Navegación",
    ["🏠 Home", "📂 Cargar Dataset"]
)

# =========================================================
# POO
# =========================================================
class DataAnalyzer:

    def __init__(self, df):

        self.df = df
        self.target = "y"

        self.numeric = df.select_dtypes(
            include=["number"]
        ).columns.tolist()

        self.categorical = [
            col for col in df.columns
            if df[col].dtype == "object"
            or str(df[col].dtype) == "string"
        ]

    def describe(self):

        return self.df.describe(include="all")

    def nulls(self):

        return self.df.isnull().sum()

    def corr(self):

        return self.df.select_dtypes(
            include=["number"]
        ).corr()

    def conversion_rate(self):

        return (
            self.df[self.target]
            .value_counts(normalize=True)
            .get("yes", 0) * 100
        )

    def best_segment(self, col):

        table = pd.crosstab(
            self.df[col],
            self.df[self.target],
            normalize="index"
        ) * 100

        if "yes" in table.columns:

            best = table["yes"].idxmax()
            rate = table["yes"].max()

            return best, rate

        return None, None

    def top_numeric_drivers(self):

        results = {}

        for col in self.numeric:

            if col != self.target:

                diff = (
                    self.df
                    .groupby(self.target)[col]
                    .mean()
                    .diff()
                    .abs()
                    .mean()
                )

                results[col] = diff

        return pd.Series(results).sort_values(
            ascending=False
        )

# =========================================================
# HOME
# =========================================================
if menu == "🏠 Home":

    st.title("📊 Bank Marketing Dashboard")

    st.markdown("""
    ## Objetivo del proyecto
    Mostrar el comportamiento de las principales variables que afectaron el rendimiento de la campaña marketing, el cual permitirá analizar los factores que hayan influenciado en una menor eficiencia obtenida.

    ## Autor
    - Nombre: Diego Eduardo Montané Quintana
    - Curso: Espacialización de Python for Analytics
    - Fecha: Mayo 2026

    ## Dataset
    Datos de campañas de marketing bancario con variables demográficas, financieras y de contacto.

    ## Tecnologías
    Python, Pandas, Streamlit, Seaborn, Matplotlib
    """)

# =========================================================
# CARGAR DATASET
# =========================================================
elif menu == "📂 Cargar Dataset":

    st.title("📂 Carga de Dataset")

    uploaded_file = st.file_uploader(
        " ",
        type=["csv"]
    )

    if uploaded_file is not None:

        if uploaded_file.name.endswith(".csv"):

            try:

                st.session_state.df_raw = pd.read_csv(
                    uploaded_file,
                    sep=";"
                )

                st.success(
                    "✅ Archivo cargado exitosamente"
                )

            except Exception as e:

                st.error(
                    f"❌ Error al leer el archivo: {e}"
                )

        else:

            st.error(
                "❌ Tipo de archivo no compatible. "
                "Por favor sube un archivo .csv"
            )

    # =====================================================
    # VALIDACIÓN GLOBAL
    # =====================================================
    df_raw = st.session_state.df_raw

    if df_raw is None:

        st.info(
            "Carga un dataset para activar el análisis."
        )

    else:

        df_raw = df_raw.convert_dtypes()

        # =================================================
        # DATAFRAME ACTIVO
        # =================================================
        df_filtered = df_raw.copy()

        analyzer = DataAnalyzer(df_filtered)

        target = "y"

        # =================================================
        # KPIs
        # =================================================
        yes_rate = analyzer.conversion_rate()

        col1, col2, col3 = st.columns(3)

        col1.metric("Clientes", len(df_filtered))
        col2.metric("Tasa de aceptación", f"{yes_rate:.2f}%")
        col3.metric("Variables", len(df_filtered.columns))

        # =================================================
        # TABS
        # =================================================
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Información",
            "📈 EDA",
            "🎯 Análisis",
            "💡 Hallazgos"
        ])

        # =================================================
        # TAB 1
        # =================================================
        with tab1:

            st.header("Información del Dataset")

            info_df = pd.DataFrame({
                "Column": df_filtered.columns,
                "Non-Null Count": df_filtered.notnull().sum().values,
                "Dtype": df_filtered.dtypes.astype(str).values
            })

            st.write(f"<class 'pandas.DataFrame'>")
            st.write(
                f"RangeIndex: {len(df_filtered)} entries, "
                f"0 to {len(df_filtered)-1}"
            )

            st.write(
                f"Data columns "
                f"(total {df_filtered.shape[1]} columns):"
            )

            info_df.insert(0, "#", range(len(info_df)))

            st.dataframe(
                info_df,
                use_container_width=True
            )

            st.subheader("Preview")

            st.dataframe(df_filtered.head())

            st.subheader("Variables")

            st.write("Numéricas:", analyzer.numeric)
            st.write("Categóricas:", analyzer.categorical)

            st.subheader("Nulos")

            st.dataframe(analyzer.nulls())

            st.subheader("Estadísticas")

            st.dataframe(analyzer.describe())

        # =================================================
        # TAB 2
        # =================================================
        with tab2:

            st.header("EDA")

            # =====================================================
            # NUM
            # =====================================================
            num_col = st.selectbox(
                "Variable numérica",
                analyzer.numeric
            )

            fig, ax = plt.subplots()

            sns.histplot(
                df_filtered[num_col],
                kde=True,
                ax=ax
            )

            st.pyplot(fig)

            # =====================================================
            # CAT
            # =====================================================
            cat_col = st.selectbox(
                "Variable categórica",
                analyzer.categorical
            )

            fig, ax = plt.subplots()

            sns.countplot(
                data=df_filtered,
                x=cat_col,
                ax=ax
            )

            ax.tick_params(axis='x', rotation=45)

            st.pyplot(fig)

            # =====================================================
            # NUM VS NUM
            # =====================================================
            st.subheader(
                "📊 Relación entre variables numéricas"
            )

            if len(analyzer.numeric) >= 2:

                c1_num = st.selectbox(
                    "variable del eje X",
                    analyzer.numeric,
                    key="c1_num"
                )

                c2_num = st.selectbox(
                    "variable del eje Y",
                    analyzer.numeric,
                    key="c2_num"
                )

                fig, ax = plt.subplots()

                ax.scatter(
                    df_filtered[c1_num],
                    df_filtered[c2_num],
                    alpha=0.4,
                    color="steelblue"
                )

                ax.set_title(f"{c1_num} vs {c2_num}")

                ax.set_xlabel(c1_num)
                ax.set_ylabel(c2_num)

                ax.grid(
                    True,
                    linestyle="--",
                    alpha=0.3
                )

                st.pyplot(fig)

            # =====================================================
            # CAT VS CAT
            # =====================================================
            st.subheader(
                "🔗 Relación entre variables categóricas"
            )

            if len(analyzer.categorical) >= 2:

                c1_cat = st.selectbox(
                    "variable del eje X",
                    analyzer.categorical,
                    key="c1_cat"
                )

                c2_cat = st.selectbox(
                    "variable del eje Y",
                    analyzer.categorical,
                    key="c2_cat"
                )

                if c1_cat != c2_cat:

                    tabla = pd.crosstab(
                        df_filtered[c1_cat],
                        df_filtered[c2_cat],
                        normalize="index"
                    )

                    col1, col2 = st.columns(2)

                    # =================================================
                    # HEATMAP
                    # =================================================
                    with col1:

                        st.markdown("### Heatmap")

                        fig, ax = plt.subplots()

                        sns.heatmap(
                            tabla,
                            cmap="Blues",
                            ax=ax
                        )

                        st.pyplot(fig)

                    # =================================================
                    # STACKED BAR
                    # =================================================
                    with col2:

                        st.markdown("### Stacked Bar")

                        fig, ax = plt.subplots()

                        tabla.plot(
                            kind="bar",
                            stacked=True,
                            ax=ax
                        )

                        ax.tick_params(
                            axis='x',
                            rotation=45
                        )

                        ax.set_ylabel("Proporción")

                        st.pyplot(fig)

            # =====================================================
            # CORRELACIÓN
            # =====================================================
            st.subheader(
                "📈 Correlación entre variables numéricas"
            )

            numeric_df = df_filtered.select_dtypes(
                include=["number"]
            )

            if numeric_df.shape[1] >= 2:

                corr = numeric_df.corr()

                fig, ax = plt.subplots(
                    figsize=(10, 6)
                )

                sns.heatmap(
                    corr,
                    cmap="coolwarm",
                    annot=True,
                    fmt=".2f",
                    ax=ax
                )

                ax.set_title("Matriz de correlación")

                st.pyplot(fig)

        # =================================================
        # TAB 3
        # =================================================
        with tab3:

            st.header("Análisis Bivariado")

            num = st.selectbox(
                "Num vs y",
                analyzer.numeric
            )

            fig, ax = plt.subplots()

            sns.boxplot(
                data=df_filtered,
                x=target,
                y=num,
                ax=ax
            )

            st.pyplot(fig)

            cat = st.selectbox(
                "Cat vs y",
                analyzer.categorical
            )

            tabla = pd.crosstab(
                df_filtered[cat],
                df_filtered[target],
                normalize="index"
            )

            st.dataframe(tabla)

            fig, ax = plt.subplots()

            tabla.plot(
                kind="bar",
                ax=ax
            )

            st.pyplot(fig)

        # =================================================
        # TAB 4
        # =================================================
        with tab4:

            st.header("💡 Hallazgos Automáticos")

            st.metric(
                "Tasa de aceptación",
                f"{yes_rate:.2f}%"
            )

            top = analyzer.top_numeric_drivers()

            st.dataframe(top)

            if len(top) > 0:

                st.success(
                    f"Variable más influyente: "
                    f"{top.index[0]}"
                )

            # =====================================================
            # duration
            # =====================================================
            if "duration" in df_filtered.columns:

                d = df_filtered.groupby(target)[
                    "duration"
                ].mean()

                if "yes" in d.index and "no" in d.index:

                    st.subheader("⏱ Duration")

                    st.write(d)

                    if d["yes"] > d["no"]:

                        st.success(
                            "Mayor duración → "
                            "mayor conversión"
                        )

            # =====================================================
            # campaign
            # =====================================================
            if "campaign" in df_filtered.columns:

                c = df_filtered.groupby(target)[
                    "campaign"
                ].mean()

                if "yes" in c.index and "no" in c.index:

                    st.subheader("📞 Campaign")

                    st.write(c)

                    if c["no"] > c["yes"]:

                        st.warning(
                            "Más contactos "
                            "reducen conversión"
                        )

            # =====================================================
            # job
            # =====================================================
            if "job" in df_filtered.columns:

                best_job, rate = analyzer.best_segment(
                    "job"
                )

                if best_job is not None:

                    st.subheader("👤 Job")

                    st.success(
                        f"{best_job} → "
                        f"{rate:.2f}% conversión"
                    )

            # =====================================================
            # contact
            # =====================================================
            if "contact" in df_filtered.columns:

                best_contact, rate = analyzer.best_segment(
                    "contact"
                )

                if best_contact is not None:

                    st.subheader("📡 Contact")

                    st.success(
                        f"{best_contact} → "
                        f"{rate:.2f}% conversión"
                    )

            # =====================================================
            # marital
            # =====================================================
            if "marital" in df_filtered.columns:

                best_marital, rate = analyzer.best_segment(
                    "marital"
                )

                if best_marital is not None:

                    st.subheader("💍 Marital")

                    st.success(
                        f"{best_marital} → "
                        f"{rate:.2f}% conversión"
                    )

            # =====================================================
            # poutcome
            # =====================================================
            if "poutcome" in df_filtered.columns:

                best_poutcome, rate = analyzer.best_segment(
                    "poutcome"
                )

                if best_poutcome is not None:

                    st.subheader("📊 Poutcome")

                    st.success(
                        f"{best_poutcome} → "
                        f"{rate:.2f}% conversión"
                    )

            # =====================================================
            # CONCLUSIONES
            # =====================================================
            st.subheader("📌 Conclusiones")

            conclusions = []

            if "duration" in df_filtered.columns:

                d = df_filtered.groupby(target)[
                    "duration"
                ].mean()

                if "yes" in d.index and "no" in d.index:

                    if d["yes"] > d["no"]:

                        conclusions.append(
                            "Mayor duración aumenta "
                            "la conversión"
                        )

            if "campaign" in df_filtered.columns:

                c = df_filtered.groupby(target)[
                    "campaign"
                ].mean()

                if "yes" in c.index and "no" in c.index:

                    if c["no"] > c["yes"]:

                        conclusions.append(
                            "Exceso de contactos "
                            "reduce la conversión"
                        )

            conclusions.append(
                f"Conversión global: "
                f"{yes_rate:.2f}%"
            )

            for i, c in enumerate(conclusions, 1):

                st.write(f"{i}. {c}")

