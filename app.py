"""
Aplicación Streamlit: Análisis y Predicción de Presupuesto Público
Análisis de Presupuesto y Gasto de Organismos Públicos Argentinos (2015-2025)
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pickle
from datetime import datetime

# Importar librerías de ML
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Análisis Presupuesto Público",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

@st.cache_data
def load_data():
    """Cargar datos desde CSV"""
    try:
        df = pd.read_csv('presupuesto_gasto_output.csv')
        return df
    except FileNotFoundError:
        st.error("❌ No se encontró el archivo 'presupuesto_gasto_output.csv'")
        st.stop()

@st.cache_data
def prepare_data_for_model(df):
    """Preparar datos con feature engineering (mismo código del notebook)"""
    df = df[(df['TotalPresupuesto'] > 0) & (df['Periodo'] >= 2021)].copy()
    df = df.sort_values(['Organismo', 'PlanDeCuenta', 'Periodo']).copy()

    df['Presupuesto_Lag1'] = df.groupby(['Organismo', 'PlanDeCuenta'])['TotalPresupuesto'].shift(1)
    df['Presupuesto_Lag2'] = df.groupby(['Organismo', 'PlanDeCuenta'])['TotalPresupuesto'].shift(2)
    df['PresupuestoPromedio3Anios'] = df.groupby(
        ['Organismo', 'PlanDeCuenta']
    )['TotalPresupuesto'].transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())

    df['CrecimientoAnterior'] = df.groupby(['Organismo', 'PlanDeCuenta'])['TotalPresupuesto'].pct_change()
    df['AjusteNeto'] = df['Aumento'] - df['Disminucion']

    df['RatioAjuste'] = np.where(
        (df['Aumento'] + df['Disminucion']) > 0,
        df['Aumento'] / (df['Aumento'] + df['Disminucion']),
        0.5
    )

    df['GastoPorEmpleado'] = np.where(
        df['CantidadEmpleados'] > 0,
        df['TotalGastado'] / df['CantidadEmpleados'],
        0
    )

    df['PresupuestoFinal'] = df['TotalPresupuesto'] + df['AjusteNeto']
    df['RatioEjecucion'] = np.where(
        df['PresupuestoFinal'] > 0,
        df['TotalGastado'] / df['PresupuestoFinal'],
        0
    )

    df['TieneEmpleados'] = (df['CantidadEmpleados'] > 0).astype(int)
    df['TieneGasto'] = (df['TotalGastado'] > 0).astype(int)
    df['IndicadorAmpliacion'] = (df['Aumento'] > df['Disminucion']).astype(int)

    df = df.dropna(subset=['Presupuesto_Lag1'])
    return df

@st.cache_resource
def train_model(df):
    df_train = df[df['Periodo'].between(2022, 2024)].copy()
    df_test = df[df['Periodo'] == 2025].copy()

    X_train = df_train.drop(columns=['TotalPresupuesto', 'PresupuestoFinal'])
    y_train = df_train['TotalPresupuesto']

    X_test = df_test.drop(columns=['TotalPresupuesto', 'PresupuestoFinal'])
    y_test = df_test['TotalPresupuesto']

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))
    ])

    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    metrics = {
        'train': {
            'r2': r2_score(y_train, y_train_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'mae': mean_absolute_error(y_train, y_train_pred)
        },
        'test': {
            'r2': r2_score(y_test, y_test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'mae': mean_absolute_error(y_test, y_test_pred)
        }
    }

    return pipeline, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, metrics

def calcular_metricas_prediccion(real, predicho):
    error_abs = abs(real - predicho)
    error_pct = (error_abs / real * 100) if real > 0 else 0
    return error_abs, error_pct

# ============================================================================
# CARGAR DATOS Y MODELO
# ============================================================================

df_raw = load_data()
df_modelo = prepare_data_for_model(df_raw)
modelo, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, metrics = train_model(df_modelo)

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.markdown("# 📊 Navegación")
st.sidebar.markdown("---")

pagina = st.sidebar.radio(
    "Selecciona una página:",
    [
        "📊 Dashboard Principal",
        "📈 Exploración de Datos",
        "📉 Análisis Exploratorio",
        "🤖 Modelo Predictivo",
        "🎯 Hacer Predicciones",
        "📚 Documentación"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Información del Proyecto")
st.sidebar.info(f"""
**Dataset:** Presupuesto Público Argentino  
**Período:** 2015-2025  
**Registros:** {len(df_raw):,}  
**Organismos:** {df_raw['Organismo'].nunique()}  
**R² Test:** {metrics['test']['r2']:.3f}
""")
# ============================================================================
# 📊 PÁGINA: DASHBOARD PRINCIPAL (MODIFICADO SEGÚN TU PEDIDO)
# ============================================================================

if pagina == "📊 Dashboard Principal":
    st.markdown('<h1 class="main-header">📊 Dashboard Principal - Análisis Presupuestario</h1>', 
                unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 Panel de Control Ejecutivo
    Analiza el presupuesto por organismo, período y plan de cuenta con filtros interactivos.
    """)

    # ========================================================================
    # FILTROS PRINCIPALES
    # ========================================================================

    st.markdown("---")
    st.markdown("### 🔍 Filtros de Análisis")

    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        organismos_disponibles = sorted(df_raw['Organismo'].unique())
        organismos_sel = st.multiselect(
            "🏛️ Organismos",
            options=organismos_disponibles,
            default=organismos_disponibles[:5] if len(organismos_disponibles) >= 5 else organismos_disponibles,
            help="Selecciona uno o más organismos"
        )

    with col_f2:
        periodos_disponibles = sorted(df_raw['Periodo'].unique(), reverse=True)
        periodos_sel = st.multiselect(
            "📅 Períodos",
            options=periodos_disponibles,
            default=periodos_disponibles[:3] if len(periodos_disponibles) >= 3 else periodos_disponibles,
            help="Selecciona uno o más períodos"
        )

    with col_f3:
        planes_disponibles = sorted(df_raw['PlanDeCuenta'].unique())
        planes_sel = st.multiselect(
            "📋 Planes de Cuenta",
            options=planes_disponibles,
            default=None,
            help="Opcional: filtra por planes específicos"
        )

    df_filtrado = df_raw.copy()

    if organismos_sel:
        df_filtrado = df_filtrado[df_filtrado['Organismo'].isin(organismos_sel)]

    if periodos_sel:
        df_filtrado = df_filtrado[df_filtrado['Periodo'].isin(periodos_sel)]

    if planes_sel:
        df_filtrado = df_filtrado[df_filtrado['PlanDeCuenta'].isin(planes_sel)]

    if len(df_filtrado) == 0:
        st.warning("⚠️ No hay datos para los filtros seleccionados. Ajusta los filtros.")
        st.stop()

    st.markdown(f"**📊 Registros filtrados:** {len(df_filtrado):,} de {len(df_raw):,}")

    # ========================================================================
    # MÉTRICAS
    # ========================================================================

    st.markdown("---")
    st.markdown("### 📈 Indicadores Clave")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        total_presupuesto = df_filtrado['TotalPresupuesto'].sum()
        st.metric("💰 Presupuesto Total", f"${total_presupuesto/1e6:.1f}M")

    with col_m2:
        total_gasto = df_filtrado['TotalGastado'].sum()
        st.metric("💸 Gasto Total", f"${total_gasto/1e6:.1f}M")

    with col_m3:
        st.metric("📈 Aumentos Totales", f"${df_filtrado['Aumento'].sum()/1e6:.1f}M")

    with col_m4:
        total_disminuciones = df_filtrado['Disminucion'].sum()
        st.metric("📉 Disminuciones Totales", 
                  f"${total_disminuciones/1e6:.1f}M",
                  delta=f"-{(total_disminuciones/total_presupuesto*100):.1f}%")

    # ========================================================================
    # 1️⃣ GRÁFICO: PRESUPUESTO POR PERÍODO (SOLO BARRAS)
    # ========================================================================

    with st.expander("📊 1. Presupuesto Total por Período"):
        presup_periodo = df_filtrado.groupby('Periodo').agg({
            'Aumento': 'sum',
            'Disminucion': 'sum'
        }).reset_index()

        presup_periodo['Aumento_M'] = presup_periodo['Aumento'] / 1e6
        presup_periodo['Dismin_M'] = presup_periodo['Disminucion'] / 1e6
        presup_periodo['Periodo_str'] = presup_periodo['Periodo'].astype(str)

        chart1 = (
            alt.Chart(presup_periodo)
            .transform_fold(['Aumento_M', 'Dismin_M'], as_=['Tipo', 'Monto'])
            .mark_bar(opacity=0.7)
            .encode(
                x=alt.X('Periodo_str:N', title='Período'),
                y=alt.Y('Monto:Q', title='Monto (Millones $)'),
                color=alt.Color('Tipo:N',
                                scale=alt.Scale(domain=['Aumento_M','Dismin_M'],
                                range=['#27ae60','#e74c3c'])),
                tooltip=[
                    alt.Tooltip('Periodo_str:N', title='Período'),
                    alt.Tooltip('Monto:Q', title='Monto (M$)', format=',.1f')
                ]
            )
            .properties(height=400)
        )

        st.altair_chart(chart1, use_container_width=True)

    # ========================================================================
    # 2️⃣ GRÁFICO: PRESUPUESTO POR ORGANISMO (SIN TOP 10)
    # ========================================================================

    with st.expander("🏛️ 2. Presupuesto por Organismo"):
        presup_org = df_filtrado.groupby('Organismo').agg({
            'TotalPresupuesto': 'sum'
        }).reset_index()

        presup_org['Presup_M'] = presup_org['TotalPresupuesto'] / 1e6
        presup_org['Org_str'] = presup_org['Organismo'].astype(str)

        chart2 = (
            alt.Chart(presup_org)
            .mark_bar(color='steelblue', opacity=0.8)
            .encode(
                x=alt.X('Presup_M:Q', title='Presupuesto (M$)'),
                y=alt.Y('Org_str:N', title='Organismo', sort='-x'),
                tooltip=[
                    alt.Tooltip('Organismo:N'),
                    alt.Tooltip('Presup_M:Q', title='Presupuesto (M$)', format=',.1f')
                ]
            )
            .properties(height=400)
        )

        st.altair_chart(chart2, use_container_width=True)

    # ========================================================================
    # 3️⃣ GRÁFICO: DESGLOSE POR PLAN DIVIDIDO EN 2
    # ========================================================================

    with st.expander("📋 3. Desglose por Plan de Cuenta"):
        presup_plan = df_filtrado.groupby('PlanDeCuenta').agg({
            'TotalPresupuesto': 'sum'
        }).reset_index()

        presup_plan['Pres_M'] = presup_plan['TotalPresupuesto'] / 1e6
        presup_plan['%'] = (presup_plan['TotalPresupuesto'] / presup_plan['TotalPresupuesto'].sum()) * 100
        presup_plan['%_Acum'] = presup_plan['%'].cumsum()
        presup_plan['Plan_str'] = presup_plan['PlanDeCuenta'].astype(str)

        # --- GRÁFICO A (BARRAS) ---
        st.markdown("### 🔹 Presupuesto por Plan de Cuenta")

        chartA = (
            alt.Chart(presup_plan)
            .mark_bar(color='#e67e22', opacity=0.85)
            .encode(
                x=alt.X('Plan_str:N', title='Plan de Cuenta'),
                y=alt.Y('Pres_M:Q', title='Presupuesto (M$)'),
                tooltip=[alt.Tooltip('Pres_M:Q', title='Presupuesto (M$)', format=',.1f')]
            )
            .properties(height=350)
        )

        st.altair_chart(chartA, use_container_width=True)

        # --- GRÁFICO B (ACUMULADO) ---
        st.markdown("### 🔹 Porcentaje Acumulado")

        chartB = (
            alt.Chart(presup_plan)
            .mark_line(point=True, color='#c0392b', strokeWidth=3)
            .encode(
                x=alt.X('Plan_str:N', title='Plan de Cuenta'),
                y=alt.Y('%_Acum:Q', title='% Acumulado'),
                tooltip=[alt.Tooltip('%_Acum:Q', title='% Acumulado', format='.1f')]
            )
            .properties(height=350)
        )

        st.altair_chart(chartB, use_container_width=True)
# ========================================================================
# 4️⃣ VISUALIZACIÓN: AUMENTOS VS DISMINUCIONES (DESPLEGABLE)
# ========================================================================

with st.expander("⚖️ 4. Análisis de Aumentos y Disminuciones"):
    
    col_a1, col_a2 = st.columns(2)

    with col_a1:
        # Por período
        ajustes_periodo = df_filtrado.groupby('Periodo').agg({
            'Aumento': 'sum',
            'Disminucion': 'sum'
        }).reset_index()

        ajustes_periodo['Aumento_M'] = ajustes_periodo['Aumento'] / 1e6
        ajustes_periodo['Dismin_M'] = ajustes_periodo['Disminucion'] / 1e6
        ajustes_periodo['Neto_M'] = ajustes_periodo['Aumento_M'] - ajustes_periodo['Dismin_M']
        ajustes_periodo['Periodo_str'] = ajustes_periodo['Periodo'].astype(str)

        # Gráfico de barras agrupadas
        ajustes_long = ajustes_periodo.melt(
            id_vars='Periodo_str',
            value_vars=['Aumento_M', 'Dismin_M'],
            var_name='Tipo',
            value_name='Monto'
        )

        chart4a = alt.Chart(ajustes_long).mark_bar().encode(
            x=alt.X('Periodo_str:N', title='Período', sort=None),
            y=alt.Y('Monto:Q', title='Monto (Millones $)'),
            color=alt.Color('Tipo:N',
                            scale=alt.Scale(domain=['Aumento_M', 'Dismin_M'],
                                            range=['#27ae60', '#e74c3c']),
                            legend=alt.Legend(title='Tipo')),
            xOffset='Tipo:N',
            tooltip=[
                alt.Tooltip('Periodo_str:N', title='Período'),
                alt.Tooltip('Tipo:N', title='Tipo'),
                alt.Tooltip('Monto:Q', title='Monto (M$)', format=',.1f')
            ]
        ).properties(
            height=350,
            title="Aumentos y Disminuciones por Período"
        )

        st.altair_chart(chart4a, use_container_width=True)

    with col_a2:
        # Comparación presupuesto original vs ajustado
        comp_data = pd.DataFrame({
            'Concepto': ['Presupuesto Original', 'Aumentos', 'Disminuciones', 'Presupuesto Final'],
            'Monto': [
                df_filtrado['TotalPresupuesto'].sum() / 1e6,
                df_filtrado['Aumento'].sum() / 1e6,
                -df_filtrado['Disminucion'].sum() / 1e6,
                (df_filtrado['TotalPresupuesto'].sum() + df_filtrado['Aumento'].sum() - df_filtrado['Disminucion'].sum()) / 1e6
            ],
            'Color': ['blue', 'green', 'red', 'purple']
        })

        chart4b = alt.Chart(comp_data).mark_bar().encode(
            x=alt.X('Concepto:N', title='', sort=None),
            y=alt.Y('Monto:Q', title='Monto (Millones $)'),
            color=alt.Color('Color:N', scale=None, legend=None),
            tooltip=[
                alt.Tooltip('Concepto:N'),
                alt.Tooltip('Monto:Q', title='Monto (M$)', format=',.1f')
            ]
        ).properties(
            height=350,
            title="Flujo Presupuestario: Original → Final"
        )

        st.altair_chart(chart4b, use_container_width=True)
# ============================================================================
# 📈 PÁGINA: EXPLORACIÓN DE DATOS
# ============================================================================

elif pagina == "📈 Exploración de Datos":
    st.markdown('<h1 class="main-header">📈 Exploración de Datos</h1>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Vista General", "🔍 Filtros Avanzados", "📉 Estadísticas"])

    with tab1:
        st.markdown("### 📊 Primeras Filas del Dataset")
        st.dataframe(df_raw.head(100), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📋 Información del Dataset")
            st.write(f"**Forma:** {df_raw.shape[0]:,} filas × {df_raw.shape[1]} columnas")
            st.write(f"**Período:** {df_raw['Periodo'].min()} - {df_raw['Periodo'].max()}")
            st.write(f"**Memoria:** {df_raw.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        with col2:
            st.markdown("### 📊 Columnas")
            st.write(df_raw.dtypes.to_frame('Tipo'))

    with tab2:
        st.markdown("### 🔍 Filtrar Datos")

        col1, col2, col3 = st.columns(3)

        with col1:
            organismos_sel = st.multiselect(
                "Organismos",
                options=sorted(df_raw['Organismo'].unique()),
                default=None
            )

        with col2:
            periodos_sel = st.multiselect(
                "Períodos",
                options=sorted(df_raw['Periodo'].unique()),
                default=None
            )

        with col3:
            planes_sel = st.multiselect(
                "Planes de Cuenta",
                options=sorted(df_raw['PlanDeCuenta'].unique()),
                default=None
            )

        df_filtrado = df_raw.copy()
        if organismos_sel:
            df_filtrado = df_filtrado[df_filtrado['Organismo'].isin(organismos_sel)]
        if periodos_sel:
            df_filtrado = df_filtrado[df_filtrado['Periodo'].isin(periodos_sel)]
        if planes_sel:
            df_filtrado = df_filtrado[df_filtrado['PlanDeCuenta'].isin(planes_sel)]

        st.markdown(f"### 📊 Datos Filtrados ({len(df_filtrado):,} registros)")
        st.dataframe(df_filtrado, use_container_width=True)

        csv = df_filtrado.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar datos filtrados (CSV)",
            data=csv,
            file_name=f"datos_filtrados_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    with tab3:
        st.markdown("### 📉 Estadísticas Descriptivas")

        st.markdown("#### Variables Numéricas")
        st.dataframe(df_raw.describe(), use_container_width=True)

        st.markdown("#### Resumen por Columna")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Presupuesto Total:**")
            st.write(f"- Suma: ${df_raw['TotalPresupuesto'].sum():,.0f}")
            st.write(f"- Promedio: ${df_raw['TotalPresupuesto'].mean():,.0f}")
            st.write(f"- Mediana: ${df_raw['TotalPresupuesto'].median():,.0f}")

        with col2:
            st.write("**Gasto Total:**")
            st.write(f"- Suma: ${df_raw['TotalGastado'].sum():,.0f}")
            st.write(f"- Promedio: ${df_raw['TotalGastado'].mean():,.0f}")
            st.write(f"- Mediana: ${df_raw['TotalGastado'].median():,.0f}")


# ============================================================================
# 📉 PÁGINA: ANÁLISIS EXPLORATORIO
# ============================================================================

elif pagina == "📉 Análisis Exploratorio":
    st.markdown('<h1 class="main-header">📊 Análisis Exploratorio</h1>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📈 Evolución Temporal", "🎨 Concentración"])

    with tab1:
        st.markdown("### 📈 Evolución del Gasto: Top 5 Organismos (2021-2024)")

        df_evol = df_raw[(df_raw['Periodo'] >= 2021) & (df_raw['Periodo'] <= 2024)].copy()
        gasto_org = df_evol.groupby(['Organismo', 'Periodo'])['TotalGastado'].sum().reset_index()

        top5 = gasto_org.groupby('Organismo')['TotalGastado'].sum().nlargest(5).index.tolist()
        df_top5 = gasto_org[gasto_org['Organismo'].isin(top5)].copy()

        df_top5['Gasto_M'] = df_top5['TotalGastado'] / 1e6
        df_top5['Org_str'] = df_top5['Organismo'].astype(str)

        base = alt.Chart(df_top5).encode(
            x=alt.X('Periodo:O', title='Período'),
            y=alt.Y('Gasto_M:Q', title='Gasto (M$)', scale=alt.Scale(zero=False)),
            color=alt.Color('Org_str:N', title='Organismo'),
            tooltip=[
                alt.Tooltip('Organismo:N'),
                alt.Tooltip('Periodo:O', title='Año'),
                alt.Tooltip('Gasto_M:Q', title='Gasto (M$)', format=',.1f')
            ]
        )

        lines = base.mark_line(point=True, strokeWidth=3)
        sel = alt.selection_point(fields=['Org_str'], bind='legend')

        chart = (
            lines.add_params(sel)
            .encode(opacity=alt.condition(sel, alt.value(1), alt.value(0.2)))
            .properties(width=700, height=400)
        )

        st.altair_chart(chart, use_container_width=True)

    with tab2:
        st.markdown("### 🎨 Concentración del Presupuesto: Top 15 Planes")

        pres_plan = df_raw.groupby('PlanDeCuenta').agg({'TotalPresupuesto': 'sum'}).reset_index()
        pres_plan['Pres_M'] = pres_plan['TotalPresupuesto'] / 1e6
        pres_plan = pres_plan.sort_values('Pres_M', ascending=False)
        pres_plan['%'] = (pres_plan['Pres_M'] / pres_plan['Pres_M'].sum() * 100)
        pres_plan['%_Acum'] = pres_plan['%'].cumsum()
        top15 = pres_plan.head(15).copy()

        top15['Plan_str'] = top15['PlanDeCuenta'].astype(str)

        bars = (
            alt.Chart(top15)
            .mark_bar(color='steelblue', opacity=0.8)
            .encode(
                x=alt.X('Pres_M:Q', title='Presupuesto (M$)'),
                y=alt.Y('Plan_str:N', title='Plan de Cuenta', sort='-x'),
                tooltip=[
                    alt.Tooltip('PlanDeCuenta:N', title='Plan'),
                    alt.Tooltip('Pres_M:Q', title='Presupuesto (M$)', format=',.1f'),
                    alt.Tooltip('%:Q', title='% Total', format='.1f'),
                    alt.Tooltip('%_Acum:Q', title='% Acumulado', format='.1f')
                ]
            )
        )

        line = (
            alt.Chart(top15)
            .mark_line(color='red', strokeWidth=3, point=True)
            .encode(
                x='Pres_M:Q',
                y=alt.Y('%_Acum:Q', title='% Acumulado', axis=alt.Axis(orient='right'))
            )
        )

        chart = alt.layer(bars, line).resolve_scale(y='independent').properties(
            width=700, height=450
        )

        st.altair_chart(chart, use_container_width=True)


# ============================================================================
# 🤖 PÁGINA: MODELO PREDICTIVO
# ============================================================================

elif pagina == "🤖 Modelo Predictivo":
    st.markdown('<h1 class="main-header">🤖 Modelo Predictivo</h1>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Métricas", "📈 Predicciones 2025", "🎯 Feature Importance"])

    with tab1:
        st.markdown("### 📊 Métricas del Modelo Random Forest")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("R² Test", f"{metrics['test']['r2']:.4f}")

        with col2:
            st.metric("RMSE Test", f"${metrics['test']['rmse']/1e6:.1f}M")

        with col3:
            st.metric("MAE Test", f"${metrics['test']['mae']/1e6:.1f}M")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📈 Train")
            st.write(f"**R²:** {metrics['train']['r2']:.4f}")
            st.write(f"**RMSE:** ${metrics['train']['rmse']/1e6:.1f}M")
            st.write(f"**MAE:** ${metrics['train']['mae']/1e6:.1f}M")

        with col2:
            st.markdown("#### 📉 Test")
            st.write(f"**R²:** {metrics['test']['r2']:.4f}")
            st.write(f"**RMSE:** ${metrics['test']['rmse']/1e6:.1f}M")
            st.write(f"**MAE:** ${metrics['test']['mae']/1e6:.1f}M")

        gap = metrics['train']['r2'] - metrics['test']['r2']
        st.markdown(f"#### 🎯 Gap R² (Train - Test): {gap:.4f}")

    with tab2:
        st.markdown("### 📈 Predicciones 2025 vs Valores Reales")

        df_pred = pd.DataFrame({
            'Real': y_test.values,
            'Pred': y_test_pred,
            'Real_M': y_test.values / 1e6,
            'Pred_M': y_test_pred / 1e6
        })

        mask = df_pred['Real'] > 0
        df_pred['Error_%'] = 0
        df_pred.loc[mask, 'Error_%'] = (
            (df_pred.loc[mask, 'Real'] - df_pred.loc[mask, 'Pred']).abs()
            / df_pred.loc[mask, 'Real'] * 100
        )

        scatter = (
            alt.Chart(df_pred)
            .mark_circle(size=60, opacity=0.6)
            .encode(
                x=alt.X('Real_M:Q', title='Real (M$)', scale=alt.Scale(type='log')),
                y=alt.Y('Pred_M:Q', title='Predicho (M$)', scale=alt.Scale(type='log')),
                color=alt.Color('Error_%:Q', title='Error %',
                                scale=alt.Scale(scheme='redyellowgreen', domain=[50, 0])),
                tooltip=[
                    alt.Tooltip('Real_M:Q', title='Real (M$)', format=',.2f'),
                    alt.Tooltip('Pred_M:Q', title='Pred (M$)', format=',.2f'),
                    alt.Tooltip('Error_%:Q', title='Error %', format='.1f')
                ]
            )
        )

        min_val = min(df_pred['Real_M'].min(), df_pred['Pred_M'].min())
        max_val = max(df_pred['Real_M'].max(), df_pred['Pred_M'].max())

        line = (
            alt.Chart(pd.DataFrame({'x': [min_val, max_val]}))
            .mark_line(strokeDash=[5, 5], color='gray', strokeWidth=2)
            .encode(x='x:Q', y='x:Q')
        )

        chart = (scatter + line).properties(width=700, height=500)
        st.altair_chart(chart, use_container_width=True)

    with tab3:
        st.markdown("### 🎯 Importancia de Variables")

        rf_model = modelo.named_steps['regressor']
        importancias = rf_model.feature_importances_
        features = X_train.columns.tolist()

        df_imp = pd.DataFrame({
            'Feature': features,
            'Importancia': importancias,
            'Porcentaje': importancias * 100
        }).sort_values('Importancia', ascending=False).head(15)

        chart = (
            alt.Chart(df_imp)
            .mark_bar(color='darkgreen', opacity=0.8)
            .encode(
                x=alt.X('Porcentaje:Q', title='Importancia (%)'),
                y=alt.Y('Feature:N', title='Variable', sort='-x'),
                tooltip=[
                    alt.Tooltip('Feature:N', title='Variable'),
                    alt.Tooltip('Porcentaje:Q', title='Importancia %', format='.2f')
                ]
            )
            .properties(width=700, height=450)
        )

        st.altair_chart(chart, use_container_width=True)


# ============================================================================
# 🎯 PÁGINA: HACER PREDICCIONES
# ============================================================================

elif pagina == "🎯 Hacer Predicciones":
    st.markdown('<h1 class="main-header">🎯 Hacer Predicciones Interactivas</h1>', unsafe_allow_html=True)

    st.markdown("""
    ### 📝 Ingresa los datos para predecir el Presupuesto 2026
    Completa los campos para obtener una predicción del presupuesto usando el modelo Random Forest.
    """)

    col1, col2 = st.columns(2)

    with col1:
        organismo = st.selectbox(
            "Organismo",
            options=sorted(df_modelo['Organismo'].unique())
        )

        plan_cuenta = st.selectbox(
            "Plan de Cuenta",
            options=sorted(df_modelo['PlanDeCuenta'].unique())
        )

        periodo = st.number_input(
            "Período",
            min_value=2026,
            max_value=2030,
            value=2026
        )

    with col2:
        presupuesto_lag1 = st.number_input(
            "Presupuesto Año Anterior ($)",
            min_value=0.0,
            value=1000000.0,
            step=10000.0
        )

        presupuesto_lag2 = st.number_input(
            "Presupuesto 2 Años Atrás ($)",
            min_value=0.0,
            value=900000.0,
            step=10000.0
        )

        presup_promedio = st.number_input(
            "Promedio 3 Años ($)",
            min_value=0.0,
            value=950000.0,
            step=10000.0
        )

    col3, col4 = st.columns(2)

    with col3:
        aumento = st.number_input("Aumento ($)", min_value=0.0, value=0.0, step=1000.0)
        disminucion = st.number_input("Disminución ($)", min_value=0.0, value=0.0, step=1000.0)
        gasto_total = st.number_input("Gasto Total ($)", min_value=0.0, value=850000.0, step=10000.0)

    with col4:
        cantidad_empleados = st.number_input("Cantidad de Empleados", min_value=0, value=10)
        cantidad_cargos = st.number_input("Cantidad de Cargos", min_value=0, value=10)

        crecimiento_anterior = st.slider(
            "Crecimiento Anterior (%)",
            min_value=-100.0,
            max_value=100.0,
            value=5.0
        ) / 100

    st.markdown("---")

    if st.button("🎯 Predecir Presupuesto", type="primary", use_container_width=True):
        with st.spinner("Calculando predicción..."):
            try:
                ajuste_neto = aumento - disminucion
                ratio_ajuste = (aumento / (aumento + disminucion)) if (aumento + disminucion) > 0 else 0.5
                gasto_por_empleado = gasto_total / cantidad_empleados if cantidad_empleados > 0 else 0

                presupuesto_final = presupuesto_lag1 + ajuste_neto
                ratio_ejecucion = (gasto_total / presupuesto_final) if presupuesto_final > 0 else 0

                tiene_empleados = 1 if cantidad_empleados > 0 else 0
                tiene_gasto = 1 if gasto_total > 0 else 0
                indicador_ampliacion = 1 if aumento > disminucion else 0

                input_data = pd.DataFrame({
                    'Organismo': [organismo],
                    'Periodo': [periodo],
                    'PlanDeCuenta': [plan_cuenta],
                    'Aumento': [aumento],
                    'Disminucion': [disminucion],
                    'TotalGastado': [gasto_total],
                    'CantidadEmpleados': [cantidad_empleados],
                    'CantidadCargos': [cantidad_cargos],
                    'Presupuesto_Lag1': [presupuesto_lag1],
                    'Presupuesto_Lag2': [presupuesto_lag2],
                    'PresupuestoPromedio3Anios': [presup_promedio],
                    'CrecimientoAnterior': [crecimiento_anterior],
                    'AjusteNeto': [ajuste_neto],
                    'RatioAjuste': [ratio_ajuste],
                    'GastoPorEmpleado': [gasto_por_empleado],
                    'RatioEjecucion': [ratio_ejecucion],
                    'TieneEmpleados': [tiene_empleados],
                    'TieneGasto': [tiene_gasto],
                    'IndicadorAmpliacion': [indicador_ampliacion]
                })

                input_data = input_data[X_train.columns]

                prediccion = modelo.predict(input_data)[0]

                st.success("✅ Predicción completada!")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("💰 Presupuesto Predicho", f"${prediccion:,.0f}")

                with col2:
                    st.metric("📊 En Millones", f"${prediccion/1e6:.2f}M")

                with col3:
                    cambio_pct = ((prediccion - presupuesto_lag1) / presupuesto_lag1 * 100) if presupuesto_lag1 > 0 else 0
                    st.metric(
                        "📈 Cambio vs Año Anterior",
                        f"{cambio_pct:+.1f}%",
                        delta=f"${prediccion - presupuesto_lag1:,.0f}"
                    )

                st.markdown("### 📋 Resumen de Inputs")
                resumen_df = pd.DataFrame({
                    'Variable': [
                        'Organismo', 'Plan de Cuenta', 'Período',
                        'Presupuesto Año Anterior', 'Aumento', 'Disminución',
                        'Empleados', 'Cargos', 'Gasto Total'
                    ],
                    'Valor': [
                        organismo, plan_cuenta, periodo,
                        f"${presupuesto_lag1:,.0f}", f"${aumento:,.0f}", f"${disminucion:,.0f}",
                        cantidad_empleados, cantidad_cargos, f"${gasto_total:,.0f}"
                    ]
                })

                st.dataframe(resumen_df, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error al hacer la predicción: {str(e)}")


# ============================================================================
# 📚 PÁGINA: DOCUMENTACIÓN
# ============================================================================

elif pagina == "📚 Documentación":
    st.markdown('<h1 class="main-header">📚 Documentación</h1>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📖 Metodología", "🔍 Hallazgos", "💻 Tecnologías"])

    with tab1:
        st.markdown("""
        ### 📖 Metodología del Proyecto  
        (Contenido igual al original)
        """)

    with tab2:
        st.markdown("""
        ### 🔍 Hallazgos Principales  
        (Contenido igual al original)
        """)

    with tab3:
        st.markdown("""
        ### 💻 Stack Tecnológico  
        (Contenido igual al original)
        """)


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p><strong>Análisis de Presupuesto y Gasto de Organismos Públicos</strong></p>
    <p>Desarrollado con ❤️ usando Streamlit | Dataset: 2015-2025 | Modelo: Random Forest</p>
</div>
""", unsafe_allow_html=True)
