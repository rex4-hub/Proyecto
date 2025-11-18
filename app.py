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
    # Filtrar datos
    df = df[(df['TotalPresupuesto'] > 0) & (df['Periodo'] >= 2021)].copy()
    
    # Ordenar
    df = df.sort_values(['Organismo', 'PlanDeCuenta', 'Periodo']).copy()
    
    # Lag features
    df['Presupuesto_Lag1'] = df.groupby(['Organismo', 'PlanDeCuenta'])['TotalPresupuesto'].shift(1)
    df['Presupuesto_Lag2'] = df.groupby(['Organismo', 'PlanDeCuenta'])['TotalPresupuesto'].shift(2)
    df['PresupuestoPromedio3Anios'] = df.groupby(['Organismo', 'PlanDeCuenta'])['TotalPresupuesto'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    df['CrecimientoAnterior'] = df.groupby(['Organismo', 'PlanDeCuenta'])['TotalPresupuesto'].pct_change()
    df['AjusteNeto'] = df['Aumento'] - df['Disminucion']
    df['RatioAjuste'] = np.where((df['Aumento'] + df['Disminucion']) > 0, 
                                  df['Aumento'] / (df['Aumento'] + df['Disminucion']), 0.5)
    df['GastoPorEmpleado'] = np.where(df['CantidadEmpleados'] > 0, 
                                       df['TotalGastado'] / df['CantidadEmpleados'], 0)
    df['PresupuestoFinal'] = df['TotalPresupuesto'] + df['AjusteNeto']
    df['RatioEjecucion'] = np.where(df['PresupuestoFinal'] > 0, 
                                     df['TotalGastado'] / df['PresupuestoFinal'], 0)
    df['TieneEmpleados'] = (df['CantidadEmpleados'] > 0).astype(int)
    df['TieneGasto'] = (df['TotalGastado'] > 0).astype(int)
    df['IndicadorAmpliacion'] = (df['Aumento'] > df['Disminucion']).astype(int)
    
    # Eliminar NaN
    df = df.dropna(subset=['Presupuesto_Lag1'])
    
    return df

@st.cache_resource
def train_model(df):
    """Entrenar modelo Random Forest"""
    # Partición
    df_train = df[df['Periodo'].between(2022, 2024)].copy()
    df_test = df[df['Periodo'] == 2025].copy()
    
    X_train = df_train.drop(columns=['TotalPresupuesto', 'PresupuestoFinal'])
    y_train = df_train['TotalPresupuesto']
    X_test = df_test.drop(columns=['TotalPresupuesto', 'PresupuestoFinal'])
    y_test = df_test['TotalPresupuesto']
    
    # Entrenar modelo
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Predicciones
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    
    # Métricas
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
    """Calcular métricas de predicción"""
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
# SIDEBAR - NAVEGACIÓN
# ============================================================================

st.sidebar.markdown("# 📊 Navegación")
st.sidebar.markdown("---")

pagina = st.sidebar.radio(
    "Selecciona una página:",
    [
        "📊 Dashboard Principal",
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
**Modelo:** Random Forest  
**R² Test:** {metrics['test']['r2']:.3f}
""")

# ============================================================================
# PÁGINA: DASHBOARD PRINCIPAL
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
        # Filtro de Organismo (multiselect)
        organismos_disponibles = sorted(df_raw['Organismo'].unique())
        organismos_sel = st.multiselect(
            "🏛️ Organismos",
            options=organismos_disponibles,
            default=organismos_disponibles[:5] if len(organismos_disponibles) >= 5 else organismos_disponibles,
            help="Selecciona uno o más organismos"
        )
    
    with col_f2:
        # Filtro de Período (multiselect)
        periodos_disponibles = sorted(df_raw['Periodo'].unique(), reverse=True)
        periodos_sel = st.multiselect(
            "📅 Períodos",
            options=periodos_disponibles,
            default=periodos_disponibles[:3] if len(periodos_disponibles) >= 3 else periodos_disponibles,
            help="Selecciona uno o más períodos"
        )
    
    with col_f3:
        # Filtro de Plan de Cuenta (multiselect)
        planes_disponibles = sorted(df_raw['PlanDeCuenta'].unique())
        planes_sel = st.multiselect(
            "📋 Planes de Cuenta",
            options=planes_disponibles,
            default=None,
            help="Opcional: filtra por planes específicos"
        )
    
    # Aplicar filtros
    df_filtrado = df_raw.copy()
    
    if organismos_sel:
        df_filtrado = df_filtrado[df_filtrado['Organismo'].isin(organismos_sel)]
    
    if periodos_sel:
        df_filtrado = df_filtrado[df_filtrado['Periodo'].isin(periodos_sel)]
    
    if planes_sel:
        df_filtrado = df_filtrado[df_filtrado['PlanDeCuenta'].isin(planes_sel)]
    
    # Validar que hay datos
    if len(df_filtrado) == 0:
        st.warning("⚠️ No hay datos para los filtros seleccionados. Ajusta los filtros.")
        st.stop()
    
    st.markdown(f"**📊 Registros filtrados:** {len(df_filtrado):,} de {len(df_raw):,}")
    
    # ========================================================================
    # MÉTRICAS PRINCIPALES
    # ========================================================================
    
    st.markdown("---")
    st.markdown("### 📈 Indicadores Clave")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        total_presupuesto = df_filtrado['TotalPresupuesto'].sum()
        st.metric(
            "💰 Presupuesto Total",
            f"${total_presupuesto/1e6:.1f}M",
            help="Suma total del presupuesto"
        )
    
    with col_m2:
        total_gasto = df_filtrado['TotalGastado'].sum()
        st.metric(
            "💸 Gasto Total",
            f"${total_gasto/1e6:.1f}M",
            help="Suma total del gasto ejecutado"
        )
    
    with col_m3:
        total_aumentos = df_filtrado['Aumento'].sum()
        st.metric(
            "📈 Aumentos Totales",
            f"${total_aumentos/1e6:.1f}M",
            help="Suma de todos los aumentos"
        )
    
    with col_m4:
        total_disminuciones = df_filtrado['Disminucion'].sum()
        st.metric(
            "📉 Disminuciones Totales",
            f"${total_disminuciones/1e6:.1f}M",
            delta=f"-{(total_disminuciones/total_presupuesto*100):.1f}%",
            delta_color="inverse",
            help="Suma de todas las disminuciones"
        )
    
    # Segunda fila de métricas
    col_m5, col_m6, col_m7, col_m8 = st.columns(4)
    
    with col_m5:
        ratio_ejecucion = (total_gasto / total_presupuesto * 100) if total_presupuesto > 0 else 0
        st.metric(
            "🎯 Ejecución Presupuestaria",
            f"{ratio_ejecucion:.1f}%",
            help="Porcentaje ejecutado del presupuesto"
        )
    
    with col_m6:
        ajuste_neto = total_aumentos - total_disminuciones
        st.metric(
            "⚖️ Ajuste Neto",
            f"${ajuste_neto/1e6:.1f}M",
            delta=f"{(ajuste_neto/total_presupuesto*100):+.1f}%",
            help="Aumentos - Disminuciones"
        )
    
    with col_m7:
        num_organismos = df_filtrado['Organismo'].nunique()
        st.metric(
            "🏛️ Organismos",
            f"{num_organismos}",
            help="Cantidad de organismos en el filtro"
        )
    
    with col_m8:
        num_planes = df_filtrado['PlanDeCuenta'].nunique()
        st.metric(
            "📋 Planes de Cuenta",
            f"{num_planes}",
            help="Cantidad de planes en el filtro"
        )
    
    # ========================================================================
    # VISUALIZACIÓN 1: PRESUPUESTO TOTAL POR PERÍODO
    # ========================================================================
    
    st.markdown("---")
    
    with st.expander("📊 1. Presupuesto Total por Período", expanded=False):
        # Preparar datos
        presup_periodo = df_filtrado.groupby('Periodo').agg({
            'TotalPresupuesto': 'sum',
            'Aumento': 'sum',
            'Disminucion': 'sum',
            'TotalGastado': 'sum'
        }).reset_index()
        
        presup_periodo['Presup_M'] = presup_periodo['TotalPresupuesto'] / 1e6
        presup_periodo['Aumento_M'] = presup_periodo['Aumento'] / 1e6
        presup_periodo['Dismin_M'] = presup_periodo['Disminucion'] / 1e6
        presup_periodo['Gasto_M'] = presup_periodo['TotalGastado'] / 1e6
        presup_periodo['Periodo_str'] = presup_periodo['Periodo'].astype(str)
        
        # Gráfico de barras apiladas
        base = alt.Chart(presup_periodo).transform_fold(
            ['Aumento_M', 'Dismin_M'],
            as_=['Tipo', 'Monto']
        ).encode(
            x=alt.X('Periodo_str:N', title='Período', sort=None),
            y=alt.Y('Monto:Q', title='Monto (Millones $)'),
            color=alt.Color('Tipo:N', 
                           scale=alt.Scale(domain=['Aumento_M', 'Dismin_M'], 
                                          range=['#2ecc71', '#e74c3c']),
                           legend=alt.Legend(title='Tipo de Ajuste')),
            tooltip=[
                alt.Tooltip('Periodo:O', title='Período'),
                alt.Tooltip('Presup_M:Q', title='Presupuesto (M$)', format=',.1f'),
                alt.Tooltip('Aumento_M:Q', title='Aumentos (M$)', format=',.1f'),
                alt.Tooltip('Dismin_M:Q', title='Disminuciones (M$)', format=',.1f'),
                alt.Tooltip('Gasto_M:Q', title='Gasto (M$)', format=',.1f')
            ]
        )
        
        bars = base.mark_bar(opacity=0.7)
        
        # Línea del presupuesto total
        line = alt.Chart(presup_periodo).mark_line(
            point=True, 
            strokeWidth=3, 
            color='#3498db'
        ).encode(
            x=alt.X('Periodo_str:N', sort=None),
            y=alt.Y('Presup_M:Q'),
            tooltip=[
                alt.Tooltip('Periodo:O', title='Período'),
                alt.Tooltip('Presup_M:Q', title='Presupuesto Total (M$)', format=',.1f')
            ]
        )
        
        chart1 = (bars + line).properties(
            width=700,
            height=400,
            title="Evolución del Presupuesto y Ajustes por Período"
        )
        
        st.altair_chart(chart1, use_container_width=True)
        
        # Tabla resumen
        with st.expander("📋 Ver tabla de datos"):
            st.dataframe(
                presup_periodo[['Periodo', 'Presup_M', 'Aumento_M', 'Dismin_M', 'Gasto_M']].rename(columns={
                    'Presup_M': 'Presupuesto (M$)',
                    'Aumento_M': 'Aumentos (M$)',
                    'Dismin_M': 'Disminuciones (M$)',
                    'Gasto_M': 'Gasto (M$)'
                }),
                use_container_width=True,
                hide_index=True
            )
    
    # ========================================================================
    # VISUALIZACIÓN 2: PRESUPUESTO POR ORGANISMO
    # ========================================================================
    
    st.markdown("---")
    
    with st.expander("🏛️ 2. Presupuesto por Organismo", expanded=False):
        # Preparar datos
        presup_org = df_filtrado.groupby('Organismo').agg({
            'TotalPresupuesto': 'sum',
            'TotalGastado': 'sum',
            'Aumento': 'sum',
            'Disminucion': 'sum'
        }).reset_index()
        
        presup_org = presup_org.sort_values('TotalPresupuesto', ascending=False).head(10)
        presup_org['Presup_M'] = presup_org['TotalPresupuesto'] / 1e6
        presup_org['Gasto_M'] = presup_org['TotalGastado'] / 1e6
        presup_org['Org_str'] = presup_org['Organismo'].astype(str)
        presup_org['RatioEjec'] = (presup_org['TotalGastado'] / presup_org['TotalPresupuesto'] * 100)
        
        # Gráfico horizontal
        chart2 = alt.Chart(presup_org).mark_bar(color='steelblue', opacity=0.8).encode(
            x=alt.X('Presup_M:Q', title='Presupuesto (Millones $)'),
            y=alt.Y('Org_str:N', title='Organismo', sort='-x'),
            tooltip=[
                alt.Tooltip('Organismo:N'),
                alt.Tooltip('Presup_M:Q', title='Presupuesto (M$)', format=',.1f'),
                alt.Tooltip('Gasto_M:Q', title='Gasto (M$)', format=',.1f'),
                alt.Tooltip('RatioEjec:Q', title='Ejecución %', format='.1f')
            ]
        ).properties(
            width=700,
            height=400,
            title="Organismos por Presupuesto Total"
        )
        
        st.altair_chart(chart2, use_container_width=True)
        
        with st.expander("📋 Ver detalles por organismo"):
            st.dataframe(
                presup_org[['Org_str', 'Presup_M', 'Gasto_M', 'RatioEjec']].rename(columns={
                    'Org_str': 'Organismo',
                    'Presup_M': 'Presupuesto (M$)',
                    'Gasto_M': 'Gasto (M$)',
                    'RatioEjec': 'Ejecución %'
                }),
                use_container_width=True,
                hide_index=True
            )
    
    # ========================================================================
    # VISUALIZACIÓN 3: DESGLOSE POR PLAN DE CUENTA
    # ========================================================================
    
    st.markdown("---")
    
    with st.expander("📋 3. Desglose por Plan de Cuenta", expanded=False):
        # Opción de desglose
        col_d1, col_d2 = st.columns([1, 3])
        
        with col_d1:
            desglose_tipo = st.radio(
                "Tipo de desglose:",
                ["Top 10 Planes", "Todos los Planes", "Por Organismo Seleccionado"],
                help="Elige cómo visualizar el desglose"
            )
        
        if desglose_tipo == "Por Organismo Seleccionado":
            with col_d2:
                org_detalle = st.selectbox(
                    "Selecciona un organismo:",
                    options=sorted(df_filtrado['Organismo'].unique()),
                    help="Ver desglose de planes para este organismo"
                )
                df_desglose = df_filtrado[df_filtrado['Organismo'] == org_detalle]
        else:
            df_desglose = df_filtrado.copy()
        
        # Preparar datos de desglose
        presup_plan = df_desglose.groupby('PlanDeCuenta').agg({
            'TotalPresupuesto': 'sum',
            'TotalGastado': 'sum',
            'Aumento': 'sum',
            'Disminucion': 'sum'
        }).reset_index()
        
        presup_plan['Presup_M'] = presup_plan['TotalPresupuesto'] / 1e6
        presup_plan['%'] = (presup_plan['TotalPresupuesto'] / presup_plan['TotalPresupuesto'].sum() * 100)
        presup_plan = presup_plan.sort_values('TotalPresupuesto', ascending=False)
        
        if desglose_tipo == "Top 10 Planes":
            presup_plan = presup_plan.head(10)
        
        presup_plan['Plan_str'] = presup_plan['PlanDeCuenta'].astype(str)
        presup_plan['%_Acum'] = presup_plan['%'].cumsum()
        
        # Dos columnas para los dos gráficos
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.markdown("#### 📊 Presupuesto por Plan")
            # Gráfico 1: Plan de Cuenta (eje X) vs Presupuesto (eje Y)
            chart3a = alt.Chart(presup_plan).mark_bar(color='#e67e22', opacity=0.8).encode(
                x=alt.X('Plan_str:N', title='Plan de Cuenta', sort='-y'),
                y=alt.Y('Presup_M:Q', title='Presupuesto (Millones $)'),
                tooltip=[
                    alt.Tooltip('PlanDeCuenta:N', title='Plan'),
                    alt.Tooltip('Presup_M:Q', title='Presupuesto (M$)', format=',.1f'),
                    alt.Tooltip('%:Q', title='% del Total', format='.1f')
                ]
            ).properties(
                width=350,
                height=400,
                title="Presupuesto por Plan de Cuenta"
            )
            
            st.altair_chart(chart3a, use_container_width=True)
        
        with col_g2:
            st.markdown("#### 📈 Porcentaje Acumulado")
            # Gráfico 2: Plan de Cuenta (eje X) vs % Acumulado (eje Y)
            chart3b = alt.Chart(presup_plan).mark_line(
                point=True,
                strokeWidth=3,
                color='#c0392b'
            ).encode(
                x=alt.X('Plan_str:N', title='Plan de Cuenta', sort='-y'),
                y=alt.Y('%_Acum:Q', title='% Acumulado', scale=alt.Scale(domain=[0, 100])),
                tooltip=[
                    alt.Tooltip('PlanDeCuenta:N', title='Plan'),
                    alt.Tooltip('%:Q', title='% Individual', format='.1f'),
                    alt.Tooltip('%_Acum:Q', title='% Acumulado', format='.1f')
                ]
            ).properties(
                width=350,
                height=400,
                title="Porcentaje Acumulado (Pareto)"
            )
            
            st.altair_chart(chart3b, use_container_width=True)
        
        # Tabla completa
        with st.expander("📋 Ver tabla completa de planes"):
            st.dataframe(
                presup_plan[['Plan_str', 'Presup_M', '%', '%_Acum']].rename(columns={
                    'Plan_str': 'Plan de Cuenta',
                    'Presup_M': 'Presupuesto (M$)',
                    '%': '% del Total',
                    '%_Acum': '% Acumulado'
                }),
                use_container_width=True,
                hide_index=True
            )
    
    # ========================================================================
    # VISUALIZACIÓN 4: AUMENTOS VS DISMINUCIONES
    # ========================================================================
    
    st.markdown("---")
    
    with st.expander("⚖️ 4. Análisis de Aumentos y Disminuciones", expanded=False):
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
                width=350,
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
                width=350,
                height=350,
                title="Flujo Presupuestario: Original → Final"
            )
            
            st.altair_chart(chart4b, use_container_width=True)
        
        # Resumen de ajustes
        st.info(f"""
        💡 **Resumen de Ajustes:**
        - **Aumentos totales:** ${total_aumentos/1e6:.1f}M ({(total_aumentos/total_presupuesto*100):.1f}% del presupuesto)
        - **Disminuciones totales:** ${total_disminuciones/1e6:.1f}M ({(total_disminuciones/total_presupuesto*100):.1f}% del presupuesto)
        - **Ajuste neto:** ${ajuste_neto/1e6:+.1f}M ({(ajuste_neto/total_presupuesto*100):+.1f}%)
        - **Presupuesto final:** ${(total_presupuesto + ajuste_neto)/1e6:.1f}M
        """)
    
    # ========================================================================
    # DESCARGA DE DATOS
    # ========================================================================
    
    st.markdown("---")
    st.markdown("### 📥 Exportar Datos Filtrados")
    
    col_d1, col_d2, col_d3 = st.columns(3)
    
    with col_d1:
        csv_filtrado = df_filtrado.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar datos completos (CSV)",
            data=csv_filtrado,
            file_name=f"datos_filtrados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col_d2:
        csv_resumen = presup_periodo.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar resumen por período (CSV)",
            data=csv_resumen,
            file_name=f"resumen_periodo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col_d3:
        csv_planes = presup_plan.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar desglose por plan (CSV)",
            data=csv_planes,
            file_name=f"desglose_planes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# ============================================================================
# PÁGINA: INICIO
# ============================================================================

# ============================================================================
# PÁGINA: MODELO PREDICTIVO
# ============================================================================

elif pagina == "🤖 Modelo Predictivo":
    st.markdown('<h1 class="main-header">🤖 Modelo Predictivo</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Métricas", "📈 Predicciones 2025", "🎯 Feature Importance"])
    
    with tab1:
        st.markdown("### 📊 Métricas del Modelo Random Forest")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("R² Test", f"{metrics['test']['r2']:.4f}", 
                     delta="Excelente" if metrics['test']['r2'] > 0.9 else "Bueno")
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
        
        if gap < 0.05:
            st.success("✅ Excelente balance - Sin overfitting")
        elif gap < 0.15:
            st.info("✅ Buen balance")
        else:
            st.warning("⚠️ Posible overfitting")
    
    with tab2:
        st.markdown("### 📈 Predicciones 2025 vs Valores Reales")
        
        # Preparar datos
        df_pred = pd.DataFrame({
            'Real': y_test.values,
            'Pred': y_test_pred,
            'Real_M': y_test.values / 1e6,
            'Pred_M': y_test_pred / 1e6
        })
        
        mask = df_pred['Real'] > 0
        df_pred['Error_%'] = 0.0
        df_pred.loc[mask, 'Error_%'] = np.abs(
            (df_pred.loc[mask, 'Real'] - df_pred.loc[mask, 'Pred']) / df_pred.loc[mask, 'Real'] * 100
        )
        
        # Scatter con escala lineal
        scatter = alt.Chart(df_pred).mark_circle(size=60, opacity=0.6).encode(
            x=alt.X('Real_M:Q', title='Real (M$)', scale=alt.Scale(type='linear')),
            y=alt.Y('Pred_M:Q', title='Predicho (M$)', scale=alt.Scale(type='linear')),
            color=alt.Color('Error_%:Q', title='Error %', 
                           scale=alt.Scale(scheme='redyellowgreen', domain=[0, 50])),
            tooltip=[
                alt.Tooltip('Real_M:Q', title='Real (M$)', format=',.2f'),
                alt.Tooltip('Pred_M:Q', title='Pred (M$)', format=',.2f'),
                alt.Tooltip('Error_%:Q', title='Error %', format='.1f')
            ]
        )
        
        min_val = min(df_pred['Real_M'].min(), df_pred['Pred_M'].min())
        max_val = max(df_pred['Real_M'].max(), df_pred['Pred_M'].max())
        line = alt.Chart(pd.DataFrame({'x': [min_val, max_val]})).mark_line(
            strokeDash=[5, 5], color='gray', strokeWidth=2
        ).encode(x='x:Q', y='x:Q')
        
        chart = (scatter + line).properties(width=700, height=500)
        st.altair_chart(chart, use_container_width=True)
        
        pct_buenas = (df_pred['Error_%'] <= 25).sum() / len(df_pred) * 100
        st.success(f"✅ {pct_buenas:.1f}% de predicciones con error ≤ 25%")
    
    with tab3:
        st.markdown("### 🎯 Importancia de Variables")
        
        # Extraer importancias
        rf_model = modelo.named_steps['regressor']
        importancias = rf_model.feature_importances_
        features = X_train.columns.tolist()
        
        df_imp = pd.DataFrame({
            'Feature': features,
            'Importancia': importancias,
            'Porcentaje': importancias * 100
        }).sort_values('Importancia', ascending=False).head(15)
        
        # Gráfico
        chart = alt.Chart(df_imp).mark_bar(color='darkgreen', opacity=0.8).encode(
            x=alt.X('Porcentaje:Q', title='Importancia (%)'),
            y=alt.Y('Feature:N', title='Variable', sort='-x'),
            tooltip=[
                alt.Tooltip('Feature:N', title='Variable'),
                alt.Tooltip('Porcentaje:Q', title='Importancia %', format='.2f')
            ]
        ).properties(width=700, height=450)
        
        st.altair_chart(chart, use_container_width=True)
        
        st.info("💡 **Hallazgo:** Las variables lag (históricas) dominan el modelo. El presupuesto anterior explica el 48% de la varianza.")

# ============================================================================
# PÁGINA: HACER PREDICCIONES
# ============================================================================

elif pagina == "🎯 Hacer Predicciones":
    st.markdown('<h1 class="main-header">🎯 Hacer Predicciones Interactivas</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 📝 Ingresa los datos para predecir el Presupuesto 2026
    
    Completa los campos a continuación para obtener una predicción del presupuesto usando el modelo Random Forest.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏛️ Información del Organismo")
        
        organismo = st.selectbox(
            "Organismo",
            options=sorted(df_modelo['Organismo'].unique()),
            help="Selecciona el organismo"
        )
        
        plan_cuenta = st.selectbox(
            "Plan de Cuenta",
            options=sorted(df_modelo['PlanDeCuenta'].unique()),
            help="Selecciona el plan de cuenta"
        )
        
        periodo = st.number_input(
            "Período",
            min_value=2026,
            max_value=2030,
            value=2026,
            help="Año para el que se predice"
        )
    
    with col2:
        st.markdown("#### 💰 Datos Históricos")
        
        presupuesto_lag1 = st.number_input(
            "Presupuesto Año Anterior ($)",
            min_value=0.0,
            value=1000000.0,
            step=10000.0,
            help="Presupuesto del año anterior"
        )
        
        presupuesto_lag2 = st.number_input(
            "Presupuesto 2 Años Atrás ($)",
            min_value=0.0,
            value=900000.0,
            step=10000.0,
            help="Presupuesto de hace 2 años"
        )
        
        presup_promedio = st.number_input(
            "Promedio 3 Años ($)",
            min_value=0.0,
            value=950000.0,
            step=10000.0,
            help="Promedio de los últimos 3 años"
        )
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### 📊 Ajustes y Ejecución")
        
        aumento = st.number_input(
            "Aumento ($)",
            min_value=0.0,
            value=0.0,
            step=1000.0
        )
        
        disminucion = st.number_input(
            "Disminución ($)",
            min_value=0.0,
            value=0.0,
            step=1000.0
        )
        
        gasto_total = st.number_input(
            "Gasto Total ($)",
            min_value=0.0,
            value=850000.0,
            step=10000.0
        )
    
    with col4:
        st.markdown("#### 👥 Recursos Humanos")
        
        cantidad_empleados = st.number_input(
            "Cantidad de Empleados",
            min_value=0,
            value=10,
            step=1
        )
        
        cantidad_cargos = st.number_input(
            "Cantidad de Cargos",
            min_value=0,
            value=10,
            step=1
        )
        
        crecimiento_anterior = st.slider(
            "Crecimiento Anterior (%)",
            min_value=-100.0,
            max_value=100.0,
            value=5.0,
            step=0.1
        ) / 100
    
    st.markdown("---")
    
    if st.button("🎯 Predecir Presupuesto", type="primary", use_container_width=True):
        with st.spinner("Calculando predicción..."):
            try:
                # Calcular features derivadas
                ajuste_neto = aumento - disminucion
                ratio_ajuste = aumento / (aumento + disminucion) if (aumento + disminucion) > 0 else 0.5
                gasto_por_empleado = gasto_total / cantidad_empleados if cantidad_empleados > 0 else 0
                presupuesto_final = presupuesto_lag1 + ajuste_neto
                ratio_ejecucion = gasto_total / presupuesto_final if presupuesto_final > 0 else 0
                tiene_empleados = 1 if cantidad_empleados > 0 else 0
                tiene_gasto = 1 if gasto_total > 0 else 0
                indicador_ampliacion = 1 if aumento > disminucion else 0
                
                # Crear DataFrame con el mismo orden de columnas que X_train
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
                
                # Reordenar columnas para que coincidan con X_train
                input_data = input_data[X_train.columns]
                
                # Hacer predicción
                prediccion = modelo.predict(input_data)[0]
                
                # Mostrar resultado
                st.success("✅ Predicción completada!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "💰 Presupuesto Predicho",
                        f"${prediccion:,.0f}",
                        help="Predicción del modelo Random Forest"
                    )
                
                with col2:
                    st.metric(
                        "📊 En Millones",
                        f"${prediccion/1e6:.2f}M"
                    )
                
                with col3:
                    cambio_pct = ((prediccion - presupuesto_lag1) / presupuesto_lag1 * 100) if presupuesto_lag1 > 0 else 0
                    st.metric(
                        "📈 Cambio vs Año Anterior",
                        f"{cambio_pct:+.1f}%",
                        delta=f"${prediccion - presupuesto_lag1:,.0f}"
                    )
                
                st.markdown("---")
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
                
                st.dataframe(resumen_df, use_container_width=True, hide_index=True)
                
                st.info(f"""
                💡 **Interpretación:**  
                El modelo predice un presupuesto de **${prediccion:,.0f}** para el {periodo}.  
                Esto representa un cambio de **{cambio_pct:+.1f}%** respecto al año anterior.  
                La predicción se basa en {X_train.shape[1]} variables con un R² de {metrics['test']['r2']:.3f}.
                """)
                
            except Exception as e:
                st.error(f"❌ Error al hacer la predicción: {str(e)}")
                st.write("Detalles del error:", e)

# ============================================================================
# PÁGINA: DOCUMENTACIÓN
# ============================================================================

elif pagina == "📚 Documentación":
    st.markdown('<h1 class="main-header">📚 Documentación</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📖 Metodología", "🔍 Hallazgos", "💻 Tecnologías"])
    
    with tab1:
        st.markdown("""
        ### 📖 Metodología del Proyecto
        
        #### 1️⃣ Análisis Exploratorio de Datos (EDA)
        - Comprensión del dataset (24,604 registros, 2015-2025)
        - Identificación de patrones y tendencias
        - Formulación y validación de hipótesis
        
        #### 2️⃣ Feature Engineering
        - **Variables Lag:** Presupuesto años anteriores
        - **Promedios Móviles:** Tendencias de 3 años
        - **Ratios:** Ejecución, ajustes, recursos
        - **Indicadores:** Binarios de presencia
        
        #### 3️⃣ Modelado Predictivo
        - **Algoritmo:** Random Forest Regressor
        - **Partición:** Train (2022-2024), Test (2025)
        - **Métricas:** R², RMSE, MAE, MAPE
        - **Validación:** Sin overfitting (Gap < 0.05)
        
        #### 4️⃣ Visualización
        - **Herramienta:** Altair (gramática de gráficos)
        - **Principios:** Expresividad, comparabilidad, interactividad
        - **Tipos:** Líneas, barras, scatter plots
        """)
    
    with tab2:
        st.markdown("""
        ### 🔍 Hallazgos Principales
        
        #### ✅ Hipótesis Validadas (4/5)
        
        | # | Hipótesis | Estado | Evidencia |
        |---|-----------|--------|-----------|
        | 1 | Crecimiento exponencial organismos top | ✅ **CONFIRMADA** | Org 9: +1,362% |
        | 2 | Correlación positiva P-G | ✅ **CONFIRMADA** | R² = 0.95 |
        | 3 | Concentración en pocos planes | ✅ **CONFIRMADA** | Top 5 = 85% |
        | 4 | Más empleados = más gasto | ❌ **RECHAZADA** | Sin correlación |
        | 5 | Patrones temporales | ✅ **CONFIRMADA** | Análisis previo |
        
        #### 💡 Insights Clave
        
        **1. Tendencia Inflacionaria Dominante**
        - Crecimiento desde 2023 vinculado a inflación >100% anual
        - No refleja expansión real de servicios
        
        **2. Predictibilidad Alta**
        - R² = 0.95 permite planificación informada
        - Variables históricas son los mejores predictores
        
        **3. Concentración Estructural**
        - Plan 34 (Salarios) = 60% del presupuesto
        - Masa salarial es el driver principal
        
        **4. Eficiencia Variable**
        - Tasa ejecución: 0-150%
        - Indica gestión heterogénea entre organismos
        """)
    
    with tab3:
        st.markdown("""
        ### 💻 Stack Tecnológico
        
        #### 📊 Análisis de Datos
        - **Python 3.x**: Lenguaje principal
        - **Pandas**: Manipulación de datos
        - **NumPy**: Cálculos numéricos
        
        #### 📈 Visualización
        - **Altair**: Visualizaciones declarativas
        - **Streamlit**: Aplicación web interactiva
        
        #### 🤖 Machine Learning
        - **scikit-learn**: Random Forest, pipelines
        - **StandardScaler**: Normalización
        
        #### 🚀 Despliegue
        - **Streamlit Cloud**: Hosting de la app
        - **GitHub**: Control de versiones
        
        #### 📦 Estructura del Repositorio
        ```
        proyecto/
        ├── app.py                          # Aplicación Streamlit
        ├── presupuesto_gasto_output.csv   # Dataset
        ├── requirements.txt                # Dependencias
        ├── README.md                       # Documentación
        └── notebooks/
            ├── Análisis_Exploratorio.ipynb
            ├── Modelado.ipynb
            └── Visualizaciones_Altair.ipynb
        ```
        
        #### 📋 requirements.txt
        ```
        streamlit>=1.28.0
        pandas>=2.0.0
        numpy>=1.24.0
        altair>=5.0.0
        scikit-learn>=1.3.0
        ```
        """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p><strong>Análisis de Presupuesto y Gasto de Organismos Públicos</strong></p>
    <p>Desarrollado con ❤️ usando Streamlit | Dataset: 2015-2025 | Modelo: Random Forest (R²=0.95)</p>
    <p>© 2025 - Proyecto de Visualización de Datos</p>
</div>
""", unsafe_allow_html=True)
