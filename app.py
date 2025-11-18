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

# CSS personalizado para mejorar la apariencia
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
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNCIONES DE CARGA Y PREPROCESAMIENTO
# ============================================================================

@st.cache_data
def cargar_datos():
    """Carga y limpia el dataset."""
    try:
        df = pd.read_csv('presupuesto_gasto_output.csv')
        
        # Filtrar años relevantes para análisis más reciente (ej. desde 2021)
        # y eliminar presupuestos 0 para evitar ruido
        df = df[(df['Periodo'] >= 2021) & (df['TotalPresupuesto'] > 0)]
        
        return df
    except FileNotFoundError:
        st.error("No se encontró el archivo 'presupuesto_gasto_output.csv'. Por favor cárgalo.")
        return pd.DataFrame()

def preparar_datos_ml(df):
    """Prepara los datos para el modelo de Machine Learning."""
    # Agrupar por Organismo y Periodo
    df_grouped = df.groupby(['Organismo', 'Periodo']).agg({
        'TotalPresupuesto': 'sum',
        'TotalGastado': 'sum',
        'CantidadEmpleados': 'mean',
        'CantidadCargos': 'mean',
        'Aumento': 'sum',
        'Disminucion': 'sum'
    }).reset_index()
    
    # Feature Engineering
    df_grouped.sort_values(['Organismo', 'Periodo'], inplace=True)
    
    # Lags (Valores anteriores)
    df_grouped['Presupuesto_Lag1'] = df_grouped.groupby('Organismo')['TotalPresupuesto'].shift(1)
    df_grouped['Presupuesto_Lag2'] = df_grouped.groupby('Organismo')['TotalPresupuesto'].shift(2)
    
    # Promedio móvil
    df_grouped['PresupuestoPromedio3Anios'] = df_grouped.groupby('Organismo')['TotalPresupuesto'].transform(lambda x: x.rolling(window=3).mean())
    
    # Ratios
    df_grouped['Ratio_Ejecucion'] = df_grouped['TotalGastado'] / df_grouped['TotalPresupuesto']
    df_grouped['Gasto_Por_Empleado'] = df_grouped['TotalGastado'] / df_grouped['CantidadEmpleados'].replace(0, 1)
    
    # Llenar NaNs generados por los lags
    df_grouped.fillna(0, inplace=True)
    
    return df_grouped

def entrenar_modelo(df_ml):
    """Entrena el modelo Random Forest."""
    features = ['Periodo', 'CantidadEmpleados', 'CantidadCargos', 
                'Aumento', 'Disminucion', 'Presupuesto_Lag1', 
                'Presupuesto_Lag2', 'PresupuestoPromedio3Anios']
    target = 'TotalPresupuesto'
    
    # Split train/test (Usamos 2025 como validación si existe, o split aleatorio)
    train_data = df_ml[df_ml['Periodo'] < 2025]
    test_data = df_ml[df_ml['Periodo'] == 2025]
    
    if test_data.empty:
        # Si no hay 2025, usamos 2024 como test
        train_data = df_ml[df_ml['Periodo'] < 2024]
        test_data = df_ml[df_ml['Periodo'] == 2024]
    
    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]
    
    # Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    
    return pipeline, metrics, X_test, y_test, y_pred

# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

# Cargar datos
df = cargar_datos()

if not df.empty:
    # Sidebar
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2666/2666505.png", width=100)
    st.sidebar.title("Navegación")
    
    page = st.sidebar.radio("Ir a:", ["Dashboard Principal", "Modelo Predictivo", "Hacer Predicciones"])
    
    # Filtros globales en sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("Filtros Globales")
    
    organismos = sorted(df['Organismo'].unique())
    selected_organismo = st.sidebar.multiselect("Seleccionar Organismo", organismos, default=organismos[:1])
    
    periodos = sorted(df['Periodo'].unique())
    selected_periodo = st.sidebar.slider("Rango de Períodos", min(periodos), max(periodos), (min(periodos), max(periodos)))
    
    # Filtrar DF
    if selected_organismo:
        df_filtered = df[df['Organismo'].isin(selected_organismo)]
    else:
        df_filtered = df
        
    df_filtered = df_filtered[(df_filtered['Periodo'] >= selected_periodo[0]) & (df_filtered['Periodo'] <= selected_periodo[1])]

    # ========================================================================
    # PÁGINA 1: DASHBOARD PRINCIPAL
    # ========================================================================
    if page == "Dashboard Principal":
        st.markdown('<h1 class="main-header">Tablero de Control Financiero</h1>', unsafe_allow_html=True)
        
        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        presupuesto_total = df_filtered['TotalPresupuesto'].sum()
        gasto_total = df_filtered['TotalGastado'].sum()
        ejecucion = (gasto_total / presupuesto_total * 100) if presupuesto_total > 0 else 0
        empleados_prom = df_filtered['CantidadEmpleados'].mean()
        
        col1.metric("Presupuesto Total", f"${presupuesto_total:,.0f}")
        col2.metric("Gasto Total", f"${gasto_total:,.0f}")
        col3.metric("% Ejecución", f"{ejecucion:.2f}%")
        col4.metric("Promedio Empleados", f"{empleados_prom:.0f}")
        
        st.markdown("---")
        
        # --------------------------------------------------------------------
        # GRÁFICO 1: Evolución Temporal (SOLO BARRAS)
        # --------------------------------------------------------------------
        with st.expander("📅 1. Presupuesto Total por Período", expanded=True):
            presupuesto_anual = df_filtered.groupby('Periodo')['TotalPresupuesto'].sum().reset_index()
            
            chart_presupuesto = alt.Chart(presupuesto_anual).mark_bar(color='#1f77b4').encode(
                x=alt.X('Periodo:O', title='Año'),
                y=alt.Y('TotalPresupuesto:Q', title='Presupuesto ($)'),
                tooltip=['Periodo', alt.Tooltip('TotalPresupuesto', format='$,.0f')]
            ).properties(
                title='Evolución del Presupuesto'
            ).interactive()
            
            st.altair_chart(chart_presupuesto, use_container_width=True)

        # --------------------------------------------------------------------
        # GRÁFICO 2: Por Organismo (SIN LIMITAR A TOP 10)
        # --------------------------------------------------------------------
        with st.expander("🏢 2. Presupuesto por Organismo", expanded=False):
            presupuesto_org = df_filtered.groupby('Organismo')['TotalPresupuesto'].sum().reset_index()
            presupuesto_org = presupuesto_org.sort_values('TotalPresupuesto', ascending=False)
            # NOTA: Si seleccionas muchos organismos, este gráfico puede ser muy largo.
            
            chart_org = alt.Chart(presupuesto_org).mark_bar().encode(
                x=alt.X('TotalPresupuesto:Q', title='Presupuesto Total'),
                y=alt.Y('Organismo:O', sort='-x', title='Organismo'),
                color=alt.Color('TotalPresupuesto', scale=alt.Scale(scheme='blues')),
                tooltip=['Organismo', alt.Tooltip('TotalPresupuesto', format='$,.0f')]
            ).properties(
                title='Distribución de Presupuesto por Organismo'
            )
            
            st.altair_chart(chart_org, use_container_width=True)

        # --------------------------------------------------------------------
        # GRÁFICO 3: Desglose por Plan de Cuenta (DIVIDIDO EN DOS)
        # --------------------------------------------------------------------
        
        # Datos agrupados para los gráficos 3A y 3B
        plan_cuenta_data = df_filtered.groupby('PlanDeCuenta')[['TotalPresupuesto', 'TotalGastado']].sum().reset_index()

        # Gráfico 3A: Presupuesto por Plan de Cuenta
        with st.expander("💰 3. Presupuesto por Plan de Cuenta", expanded=False):
            chart_plan_presu = alt.Chart(plan_cuenta_data).mark_bar(color='#2ca02c').encode(
                x=alt.X('PlanDeCuenta:O', title='Plan de Cuenta'),
                y=alt.Y('TotalPresupuesto:Q', title='Presupuesto Asignado ($)'),
                tooltip=['PlanDeCuenta', alt.Tooltip('TotalPresupuesto', format='$,.0f')]
            ).properties(
                title='Presupuesto Asignado por Cuenta Contable'
            ).interactive()
            
            st.altair_chart(chart_plan_presu, use_container_width=True)

        # Gráfico 3B: Acumulado (Gastado) por Plan de Cuenta
        with st.expander("📉 4. Acumulado (Gasto) por Plan de Cuenta", expanded=False):
            chart_plan_gasto = alt.Chart(plan_cuenta_data).mark_bar(color='#d62728').encode(
                x=alt.X('PlanDeCuenta:O', title='Plan de Cuenta'),
                y=alt.Y('TotalGastado:Q', title='Total Acumulado/Gastado ($)'),
                tooltip=['PlanDeCuenta', alt.Tooltip('TotalGastado', format='$,.0f')]
            ).properties(
                title='Gasto Acumulado por Cuenta Contable'
            ).interactive()
            
            st.altair_chart(chart_plan_gasto, use_container_width=True)

    # ========================================================================
    # PÁGINA 2: MODELO PREDICTIVO
    # ========================================================================
    elif page == "Modelo Predictivo":
        st.markdown('<h1 class="main-header">Evaluación del Modelo Predictivo</h1>', unsafe_allow_html=True)
        
        # Preparar datos y entrenar
        df_ml = preparar_datos_ml(df)
        model, metrics, X_test, y_test, y_pred = entrenar_modelo(df_ml)
        
        # Mostrar métricas
        col1, col2, col3 = st.columns(3)
        col1.metric("R² Score (Precisión)", f"{metrics['r2']:.4f}")
        col2.metric("RMSE (Error Cuadrático)", f"${metrics['rmse']:,.0f}")
        col3.metric("MAE (Error Absoluto)", f"${metrics['mae']:,.0f}")
        
        st.info("El modelo Random Forest ha sido entrenado con datos históricos para predecir el presupuesto futuro basándose en variables como empleados, cargos y ejecuciones pasadas.")
        
        # Gráfico de Predicción vs Realidad
        comparison_df = pd.DataFrame({'Real': y_test, 'Predicho': y_pred})
        
        chart_pred = alt.Chart(comparison_df.reset_index()).mark_circle(size=60).encode(
            x=alt.X('Real', title='Presupuesto Real'),
            y=alt.Y('Predicho', title='Presupuesto Predicho'),
            tooltip=['Real', 'Predicho']
        ).properties(
            title='Predicción vs Realidad (Set de Prueba)'
        ).interactive()
        
        line = alt.Chart(pd.DataFrame({'x': [0, comparison_df.max().max()], 'y': [0, comparison_df.max().max()]})).mark_line(color='red', strokeDash=[5,5]).encode(
            x='x',
            y='y'
        )
        
        st.altair_chart(chart_pred + line, use_container_width=True)
        
        # Importancia de Variables
        importances = model.named_steps['regressor'].feature_importances_
        feature_names = X_test.columns
        feat_df = pd.DataFrame({'Variable': feature_names, 'Importancia': importances}).sort_values('Importancia', ascending=False)
        
        chart_feat = alt.Chart(feat_df).mark_bar().encode(
            x='Importancia',
            y=alt.Y('Variable', sort='-x'),
            color=alt.value('#ff7f0e')
        ).properties(
            title='Importancia de las Variables en el Modelo'
        )
        
        st.altair_chart(chart_feat, use_container_width=True)

    # ========================================================================
    # PÁGINA 3: HACER PREDICCIONES
    # ========================================================================
    elif page == "Hacer Predicciones":
        st.markdown('<h1 class="main-header">Simulador de Presupuesto Futuro</h1>', unsafe_allow_html=True)
        
        st.write("Ajusta los parámetros para simular el presupuesto estimado para un organismo.")
        
        # Entrenar modelo en background
        df_ml = preparar_datos_ml(df)
        model, _, _, _, _ = entrenar_modelo(df_ml)
        
        col1, col2 = st.columns(2)
        
        with col1:
            input_empleados = st.number_input("Cantidad de Empleados", min_value=0, value=int(df_ml['CantidadEmpleados'].mean()))
            input_cargos = st.number_input("Cantidad de Cargos", min_value=0, value=int(df_ml['CantidadCargos'].mean()))
            input_aumento = st.number_input("Aumento Proyectado ($)", value=0.0)
            
        with col2:
            input_periodo = st.selectbox("Año a Proyectar", [2026, 2027, 2028, 2029, 2030])
            input_disminucion = st.number_input("Disminución Proyectada ($)", value=0.0)
            # Usamos promedios históricos para los lags si es una simulación nueva
            avg_budget = df_ml['TotalPresupuesto'].mean()
            
        # Botón de predicción
        if st.button("Calcular Presupuesto Estimado"):
            # Crear DF de input
            input_data = pd.DataFrame({
                'Periodo': [input_periodo],
                'CantidadEmpleados': [input_empleados],
                'CantidadCargos': [input_cargos],
                'Aumento': [input_aumento],
                'Disminucion': [input_disminucion],
                'Presupuesto_Lag1': [avg_budget], # Simplificación para la demo
                'Presupuesto_Lag2': [avg_budget],
                'PresupuestoPromedio3Anios': [avg_budget]
            })
            
            prediction = model.predict(input_data)[0]
            
            st.success(f"💰 Presupuesto Estimado para {input_periodo}: **${prediction:,.2f}**")
            
            st.markdown("### Análisis de Escenario")
            delta = prediction - avg_budget
            if delta > 0:
                st.write(f"📈 Esta proyección es un **{ (delta/avg_budget)*100 :.1f}% mayor** que el promedio histórico.")
            else:
                st.write(f"📉 Esta proyección es un **{ abs(delta/avg_budget)*100 :.1f}% menor** que el promedio histórico.")

else:
    st.warning("Esperando datos...")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p><strong>Análisis de Presupuesto y Gasto de Organismos Públicos</strong></p>
    <p>Desarrollado con ❤️ usando Streamlit</p>
</div>
""", unsafe_allow_html=True)
