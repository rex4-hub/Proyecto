"""
Aplicación Streamlit: Análisis y Predicción de Presupuesto Público
Análisis de Presupuesto y Gasto de Organismos Públicos
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
**Dataset:** Presupuesto Público 
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
    
    with st.expander("📊 1. Evolución del Gasto por Organismo", expanded=False):
        # Sub-filtro para seleccionar organismos específicos
        organismos_disponibles_grafico = sorted(df_filtrado['Organismo'].unique())
        
        # Por defecto mostrar top 5 organismos por gasto total
        top_organismos = df_filtrado.groupby('Organismo')['TotalGastado'].sum().nlargest(5).index.tolist()
        
        organismos_sel_grafico = st.multiselect(
            "Selecciona organismos para visualizar:",
            options=organismos_disponibles_grafico,
            default=top_organismos if len(top_organismos) > 0 else organismos_disponibles_grafico[:5],
            help="Selecciona uno o más organismos para ver su evolución"
        )
        
        if not organismos_sel_grafico:
            st.warning("⚠️ Selecciona al menos un organismo para visualizar.")
        else:
            # Filtrar por los organismos seleccionados
            df_evol = df_filtrado[df_filtrado['Organismo'].isin(organismos_sel_grafico)].copy()
            
            # Agrupar por Organismo y Periodo
            gasto_org_periodo = df_evol.groupby(['Organismo', 'Periodo'])['TotalGastado'].sum().reset_index()
            gasto_org_periodo['Gasto_M'] = gasto_org_periodo['TotalGastado'] / 1e6
            
            # Crear gráfico de líneas
            chart1 = alt.Chart(gasto_org_periodo).mark_line(
                point=True, 
                strokeWidth=3
            ).encode(
                x=alt.X('Periodo:O', title='Año'),
                y=alt.Y('Gasto_M:Q', title='Gasto Total (Millones $)'),
                color=alt.Color('Organismo:N', 
                               legend=alt.Legend(title="Organismo", orient='right')),
                tooltip=[
                    alt.Tooltip('Organismo:N', title='Organismo'),
                    alt.Tooltip('Periodo:O', title='Año'),
                    alt.Tooltip('Gasto_M:Q', title='Gasto (M$)', format=',.1f')
                ]
            ).properties(
                width=700,
                height=400,
                title=f"Evolución del Gasto por Organismo ({gasto_org_periodo['Periodo'].min()}-{gasto_org_periodo['Periodo'].max()})"
            )
            
            st.altair_chart(chart1, use_container_width=True)
            
            # Mostrar estadísticas resumidas
            col_e1, col_e2, col_e3 = st.columns(3)
            
            with col_e1:
                org_mayor_gasto = gasto_org_periodo.loc[gasto_org_periodo['Gasto_M'].idxmax(), 'Organismo']
                mayor_gasto = gasto_org_periodo['Gasto_M'].max()
                st.metric(
                    "🏆 Mayor Gasto Individual",
                    f"${mayor_gasto:.1f}M",
                    delta=org_mayor_gasto
                )
            
            with col_e2:
                # Calcular crecimiento promedio
                crecimientos = []
                for org in organismos_sel_grafico:
                    df_org = gasto_org_periodo[gasto_org_periodo['Organismo'] == org].sort_values('Periodo')
                    if len(df_org) > 1:
                        primer_valor = df_org['Gasto_M'].iloc[0]
                        ultimo_valor = df_org['Gasto_M'].iloc[-1]
                        if primer_valor > 0:
                            crec = ((ultimo_valor - primer_valor) / primer_valor) * 100
                            crecimientos.append(crec)
                
                crec_promedio = np.mean(crecimientos) if crecimientos else 0
                st.metric(
                    "📈 Crecimiento Promedio",
                    f"{crec_promedio:+.1f}%",
                    help="Crecimiento promedio entre primer y último período"
                )
            
            with col_e3:
                total_gasto_periodo = gasto_org_periodo['Gasto_M'].sum()
                st.metric(
                    "💰 Gasto Total Acumulado",
                    f"${total_gasto_periodo:.1f}M"
                )
        
        # Tabla resumen
        with st.expander("📋 Ver tabla de datos"):
            tabla_evol = gasto_org_periodo.pivot(
                index='Periodo', 
                columns='Organismo', 
                values='Gasto_M'
            ).reset_index()
            st.dataframe(
                tabla_evol,
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
        
        # Gráfico de presupuesto por plan
        st.markdown("#### 📊 Presupuesto por Plan de Cuenta")
        
        chart3 = alt.Chart(presup_plan).mark_bar(color='#e67e22', opacity=0.8).encode(
            x=alt.X('Plan_str:N', title='Plan de Cuenta', sort='-y'),
            y=alt.Y('Presup_M:Q', title='Presupuesto (Millones $)'),
            tooltip=[
                alt.Tooltip('PlanDeCuenta:N', title='Plan'),
                alt.Tooltip('Presup_M:Q', title='Presupuesto (M$)', format=',.1f'),
                alt.Tooltip('%:Q', title='% del Total', format='.1f')
            ]
        ).properties(
            width=700,
            height=400,
            title="Presupuesto por Plan de Cuenta"
        )
        
        st.altair_chart(chart3, use_container_width=True)
        
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
    
    # Preparar datos de resumen por período para exportación
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
                           scale=alt.Scale(scheme='redyellowgreen', domain=[50, 0])),
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
    ### 📝 Predice el Presupuesto 2025 o 2026
    
    Selecciona el organismo, plan de cuenta y período. Los datos históricos se calcularán automáticamente 
    basándose en los valores reales del dataset.
    """)
    
    # Filtros principales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        organismo = st.selectbox(
            "🏛️ Organismo",
            options=sorted(df_raw['Organismo'].unique()),
            help="Selecciona el organismo"
        )
    
    with col2:
        plan_cuenta = st.selectbox(
            "📋 Plan de Cuenta",
            options=sorted(df_raw['PlanDeCuenta'].unique()),
            help="Selecciona el plan de cuenta"
        )
    
    with col3:
        periodo = st.selectbox(
            "📅 Período a Predecir",
            options=[2025, 2026],
            help="Año para el que se predice"
        )
    
    st.markdown("---")
    
    # Botón de predicción
    if st.button("🎯 Calcular Predicción", type="primary", use_container_width=True):
        with st.spinner("Calculando predicción..."):
            try:
                # Filtrar datos históricos para el organismo y plan seleccionados
                df_historico = df_raw[
                    (df_raw['Organismo'] == organismo) & 
                    (df_raw['PlanDeCuenta'] == plan_cuenta)
                ].sort_values('Periodo')
                
                if len(df_historico) == 0:
                    st.error("❌ No hay datos históricos para esta combinación de Organismo y Plan de Cuenta.")
                    st.stop()
                
                # Obtener datos del año anterior y 2 años atrás
                if periodo == 2025:
                    # Para 2025, necesitamos datos de 2024 y 2023
                    df_2024 = df_historico[df_historico['Periodo'] == 2024]
                    df_2023 = df_historico[df_historico['Periodo'] == 2023]
                    df_2022 = df_historico[df_historico['Periodo'] == 2022]
                    
                    if len(df_2024) == 0:
                        st.error("❌ No hay datos de 2024 para calcular la predicción de 2025.")
                        st.stop()
                    
                    # Datos del año anterior (2024)
                    presupuesto_lag1 = df_2024['TotalPresupuesto'].iloc[0]
                    aumento = df_2024['Aumento'].iloc[0]
                    disminucion = df_2024['Disminucion'].iloc[0]
                    gasto_total = df_2024['TotalGastado'].iloc[0]
                    cantidad_empleados = df_2024['CantidadEmpleados'].iloc[0]
                    cantidad_cargos = df_2024['CantidadCargos'].iloc[0]
                    
                    # Datos de 2 años atrás (2023)
                    presupuesto_lag2 = df_2023['TotalPresupuesto'].iloc[0] if len(df_2023) > 0 else presupuesto_lag1
                    
                    # Promedio de 3 años (2022, 2023, 2024)
                    presup_promedio = df_historico[
                        df_historico['Periodo'].isin([2022, 2023, 2024])
                    ]['TotalPresupuesto'].mean()
                    
                    # Crecimiento anterior (2023 a 2024)
                    if len(df_2023) > 0 and df_2023['TotalPresupuesto'].iloc[0] > 0:
                        crecimiento_anterior = (presupuesto_lag1 - df_2023['TotalPresupuesto'].iloc[0]) / df_2023['TotalPresupuesto'].iloc[0]
                    else:
                        crecimiento_anterior = 0.0
                    
                elif periodo == 2026:
                    # Para 2026, primero necesitamos predecir 2025
                    # Verificamos que tengamos datos de 2024
                    df_2024 = df_historico[df_historico['Periodo'] == 2024]
                    df_2023 = df_historico[df_historico['Periodo'] == 2023]
                    df_2022 = df_historico[df_historico['Periodo'] == 2022]
                    
                    if len(df_2024) == 0:
                        st.error("❌ No hay datos de 2024 para calcular la predicción de 2026.")
                        st.stop()
                    
                    # === PASO 1: Predecir 2025 ===
                    st.info("📊 Calculando predicción intermedia para 2025...")
                    
                    # Datos para predecir 2025
                    presup_2024 = df_2024['TotalPresupuesto'].iloc[0]
                    presup_2023 = df_2023['TotalPresupuesto'].iloc[0] if len(df_2023) > 0 else presup_2024
                    presup_prom_2025 = df_historico[
                        df_historico['Periodo'].isin([2022, 2023, 2024])
                    ]['TotalPresupuesto'].mean()
                    
                    crec_2024 = (presup_2024 - presup_2023) / presup_2023 if (len(df_2023) > 0 and presup_2023 > 0) else 0.0
                    
                    # Features para 2025
                    ajuste_neto_2024 = df_2024['Aumento'].iloc[0] - df_2024['Disminucion'].iloc[0]
                    ratio_ajuste_2024 = df_2024['Aumento'].iloc[0] / (df_2024['Aumento'].iloc[0] + df_2024['Disminucion'].iloc[0]) if (df_2024['Aumento'].iloc[0] + df_2024['Disminucion'].iloc[0]) > 0 else 0.5
                    gasto_por_emp_2024 = df_2024['TotalGastado'].iloc[0] / df_2024['CantidadEmpleados'].iloc[0] if df_2024['CantidadEmpleados'].iloc[0] > 0 else 0
                    presup_final_2024 = presup_2024 + ajuste_neto_2024
                    ratio_ejec_2024 = df_2024['TotalGastado'].iloc[0] / presup_final_2024 if presup_final_2024 > 0 else 0
                    
                    input_2025 = pd.DataFrame({
                        'Organismo': [organismo],
                        'Periodo': [2025],
                        'PlanDeCuenta': [plan_cuenta],
                        'Aumento': [df_2024['Aumento'].iloc[0]],
                        'Disminucion': [df_2024['Disminucion'].iloc[0]],
                        'TotalGastado': [df_2024['TotalGastado'].iloc[0]],
                        'CantidadEmpleados': [df_2024['CantidadEmpleados'].iloc[0]],
                        'CantidadCargos': [df_2024['CantidadCargos'].iloc[0]],
                        'Presupuesto_Lag1': [presup_2024],
                        'Presupuesto_Lag2': [presup_2023],
                        'PresupuestoPromedio3Anios': [presup_prom_2025],
                        'CrecimientoAnterior': [crec_2024],
                        'AjusteNeto': [ajuste_neto_2024],
                        'RatioAjuste': [ratio_ajuste_2024],
                        'GastoPorEmpleado': [gasto_por_emp_2024],
                        'RatioEjecucion': [ratio_ejec_2024],
                        'TieneEmpleados': [1 if df_2024['CantidadEmpleados'].iloc[0] > 0 else 0],
                        'TieneGasto': [1 if df_2024['TotalGastado'].iloc[0] > 0 else 0],
                        'IndicadorAmpliacion': [1 if df_2024['Aumento'].iloc[0] > df_2024['Disminucion'].iloc[0] else 0]
                    })
                    
                    input_2025 = input_2025[X_train.columns]
                    prediccion_2025 = modelo.predict(input_2025)[0]
                    
                    st.success(f"✅ Predicción 2025: ${prediccion_2025:,.0f}")
                    
                    # === PASO 2: Usar predicción de 2025 para predecir 2026 ===
                    st.info("📊 Calculando predicción final para 2026...")
                    
                    presupuesto_lag1 = prediccion_2025  # Usar predicción de 2025
                    presupuesto_lag2 = presup_2024  # Dato real de 2024
                    presup_promedio = (prediccion_2025 + presup_2024 + presup_2023) / 3  # Promedio con predicción 2025
                    
                    # Usar datos de 2024 como base para los demás valores
                    aumento = df_2024['Aumento'].iloc[0]
                    disminucion = df_2024['Disminucion'].iloc[0]
                    gasto_total = df_2024['TotalGastado'].iloc[0]
                    cantidad_empleados = df_2024['CantidadEmpleados'].iloc[0]
                    cantidad_cargos = df_2024['CantidadCargos'].iloc[0]
                    
                    # Crecimiento anterior (2024 a 2025 predicho)
                    crecimiento_anterior = (prediccion_2025 - presup_2024) / presup_2024 if presup_2024 > 0 else 0.0
                
                # Calcular features derivadas
                ajuste_neto = aumento - disminucion
                ratio_ajuste = aumento / (aumento + disminucion) if (aumento + disminucion) > 0 else 0.5
                gasto_por_empleado = gasto_total / cantidad_empleados if cantidad_empleados > 0 else 0
                presupuesto_final = presupuesto_lag1 + ajuste_neto
                ratio_ejecucion = gasto_total / presupuesto_final if presupuesto_final > 0 else 0
                tiene_empleados = 1 if cantidad_empleados > 0 else 0
                tiene_gasto = 1 if gasto_total > 0 else 0
                indicador_ampliacion = 1 if aumento > disminucion else 0
                
                # Crear DataFrame con los inputs
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
                
                # Reordenar columnas
                input_data = input_data[X_train.columns]
                
                # Hacer predicción final
                prediccion = modelo.predict(input_data)[0]
                
                # Mostrar resultado
                st.markdown("---")
                st.success(f"✅ Predicción completada para {periodo}!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        f"💰 Presupuesto Predicho {periodo}",
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
                
                # Mostrar datos históricos utilizados
                st.markdown("---")
                st.markdown("### 📊 Datos Históricos Utilizados")
                
                col_h1, col_h2 = st.columns(2)
                
                with col_h1:
                    st.markdown("#### 💰 Presupuestos Históricos")
                    hist_presup = pd.DataFrame({
                        'Concepto': [
                            f'Presupuesto {periodo-1}' + (' (predicho)' if periodo == 2026 else ''),
                            f'Presupuesto {periodo-2}',
                            'Promedio 3 años'
                        ],
                        'Valor': [
                            f"${presupuesto_lag1:,.0f}",
                            f"${presupuesto_lag2:,.0f}",
                            f"${presup_promedio:,.0f}"
                        ]
                    })
                    st.dataframe(hist_presup, use_container_width=True, hide_index=True)
                
                with col_h2:
                    st.markdown("#### 📊 Otros Indicadores")
                    otros_ind = pd.DataFrame({
                        'Concepto': [
                            'Aumentos',
                            'Disminuciones',
                            'Gasto Total',
                            'Empleados'
                        ],
                        'Valor': [
                            f"${aumento:,.0f}",
                            f"${disminucion:,.0f}",
                            f"${gasto_total:,.0f}",
                            f"{cantidad_empleados}"
                        ]
                    })
                    st.dataframe(otros_ind, use_container_width=True, hide_index=True)
                
                # Interpretación
                st.markdown("---")
                if periodo == 2026:
                    st.info(f"""
                    💡 **Interpretación:**  
                    El modelo predice un presupuesto de **${prediccion:,.0f}** para el {periodo}.  
                    
                    **Proceso de predicción:**
                    1. Primero se predijo 2025: **${prediccion_2025:,.0f}**
                    2. Luego se usó esa predicción para calcular 2026: **${prediccion:,.0f}**
                    
                    Esto representa un cambio de **{cambio_pct:+.1f}%** respecto a la predicción de 2025.  
                    La predicción se basa en {X_train.shape[1]} variables con un R² de {metrics['test']['r2']:.3f}.
                    """)
                else:
                    st.info(f"""
                    💡 **Interpretación:**  
                    El modelo predice un presupuesto de **${prediccion:,.0f}** para el {periodo}.  
                    Esto representa un cambio de **{cambio_pct:+.1f}%** respecto al año anterior (2024).  
                    La predicción se basa en datos históricos reales y {X_train.shape[1]} variables con un R² de {metrics['test']['r2']:.3f}.
                    """)
                
            except Exception as e:
                st.error(f"❌ Error al hacer la predicción: {str(e)}")
                with st.expander("Ver detalles del error"):
                    st.write("Detalles técnicos:")
                    st.exception(e)

# ============================================================================
# PÁGINA: DOCUMENTACIÓN
# ============================================================================

elif pagina == "📚 Documentación":
    st.markdown('<h1 class="main-header">📚 Documentación</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📖 Metodología", "🔍 Hallazgos", "💻 Tecnologías", "📘 Glosario"])
    
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
    
    with tab4:
        st.markdown("""
        ### 📘 Glosario de Términos Presupuestarios
        
        Esta sección explica los conceptos clave utilizados en el análisis de presupuesto público.
        Ideal para usuarios sin conocimientos previos del área.
        
        ---
        
        #### 💰 CONCEPTOS PRESUPUESTARIOS BÁSICOS
        
        **🏛️ Organismo**
        > Entidad o institución del sector público que recibe y ejecuta presupuesto.
        
        *Ejemplo:* Ministerio de Educación, Ministerio de Salud, Dirección de Transporte.
        
        ---
        
        **📋 Plan de Cuenta**
        > Código o categoría que clasifica en qué se gasta el dinero del presupuesto.
        
        *Tipos comunes:*
        - **Plan 34**: Salarios y cargas sociales (personal)
        - **Plan 12**: Bienes y servicios (materiales, contratos)
        - **Plan 45**: Transferencias (subsidios, ayudas)
        - **Plan 56**: Inversión física (infraestructura, equipamiento)
        
        ---
        
        **📅 Período**
        > Año fiscal o ejercicio presupuestario analizado (2015-2025 en este dataset).
        
        ---
        
        #### 💵 MONTOS Y AJUSTES
        
        **💰 Total Presupuesto**
        > Monto total de dinero asignado inicialmente a un organismo para un año.
        
        *También llamado:* Presupuesto Inicial, Crédito Inicial
        
        ---
        
        **📈 Aumento (Ampliación)**
        > Dinero adicional que se suma al presupuesto inicial durante el año.
        
        *Razones comunes:* Emergencias, nuevos proyectos, inflación no prevista.
        
        *Ejemplo:* Si el presupuesto inicial era $1,000,000 y se aprueban aumentos por $200,000, 
        el nuevo presupuesto es $1,200,000.
        
        ---
        
        **📉 Disminución (Reducción)**
        > Dinero que se resta del presupuesto inicial durante el año.
        
        *Razones comunes:* Reasignación a otras áreas, recortes presupuestarios, ahorro forzoso.
        
        *Ejemplo:* Si el presupuesto inicial era $1,000,000 y hay reducciones por $150,000, 
        el nuevo presupuesto es $850,000.
        
        ---
        
        **🎯 Presupuesto Final**
        > Presupuesto después de aplicar todos los aumentos y disminuciones.
        
        *Fórmula:* `Presupuesto Final = Presupuesto Inicial + Aumentos - Disminuciones`
        
        ---
        
        **💸 Total Gastado (Ejecutado)**
        > Dinero realmente utilizado o gastado por el organismo durante el año.
        
        *También llamado:* Presupuesto Ejecutado, Devengado
        
        ---
        
        **⚖️ Ajuste Neto**
        > Diferencia entre los aumentos y las disminuciones del presupuesto.
        
        *Fórmula:* `Ajuste Neto = Aumentos - Disminuciones`
        
        - Si es **positivo**: El presupuesto creció
        - Si es **negativo**: El presupuesto se redujo
        
        ---
        
        #### 👥 RECURSOS HUMANOS
        
        **👤 Cantidad de Empleados**
        > Número total de personas que trabajan en el organismo.
        
        *Incluye:* Personal de planta permanente, contratados, temporarios.
        
        ---
        
        **📊 Cantidad de Cargos**
        > Número de posiciones o puestos de trabajo existentes en el organismo.
        
        *Nota:* Puede ser diferente a la cantidad de empleados si hay cargos vacantes.
        
        ---
        
        #### 📊 INDICADORES Y RATIOS
        
        **📈 Ratio de Ejecución**
        > Porcentaje del presupuesto final que realmente se gastó.
        
        *Fórmula:* `Ratio = (Total Gastado / Presupuesto Final) × 100`
        
        *Interpretación:*
        - **< 100%**: Se gastó menos de lo asignado (sub-ejecución)
        - **= 100%**: Se gastó exactamente lo asignado (ejecución perfecta)
        - **> 100%**: Se gastó más de lo asignado (sobre-ejecución, puede indicar problemas)
        
        ---
        
        **💵 Gasto por Empleado**
        > Promedio de cuánto dinero se gasta por cada empleado.
        
        *Fórmula:* `Gasto por Empleado = Total Gastado / Cantidad de Empleados`
        
        *Uso:* Comparar eficiencia operativa entre organismos similares.
        
        ---
        
        **🔄 Crecimiento Anterior**
        > Variación porcentual del presupuesto respecto al año anterior.
        
        *Fórmula:* `Crecimiento = ((Presupuesto Año N - Presupuesto Año N-1) / Presupuesto Año N-1) × 100`
        
        *Ejemplo:* Si 2023 tuvo $1,000,000 y 2024 tiene $1,200,000, el crecimiento es +20%.
        
        ---
        
        #### 🤖 TÉRMINOS DEL MODELO PREDICTIVO
        
        **📊 Presupuesto Lag1**
        > Presupuesto del año inmediatamente anterior (año N-1).
        
        *Ejemplo:* Para predecir 2025, Lag1 es el presupuesto de 2024.
        
        ---
        
        **📊 Presupuesto Lag2**
        > Presupuesto de hace dos años (año N-2).
        
        *Ejemplo:* Para predecir 2025, Lag2 es el presupuesto de 2023.
        
        ---
        
        **📊 Presupuesto Promedio 3 Años**
        > Promedio del presupuesto de los últimos 3 años.
        
        *Uso:* Suavizar variaciones extremas y captar tendencia general.
        
        ---
        
        **🎯 R² (Coeficiente de Determinación)**
        > Métrica que indica qué tan bien el modelo predice los datos (0 a 1).
        
        *Interpretación:*
        - **R² = 0.95**: El modelo explica el 95% de la variabilidad (excelente)
        - **R² = 0.50**: El modelo explica el 50% de la variabilidad (regular)
        - **R² = 0.20**: El modelo explica el 20% de la variabilidad (pobre)
        
        ---
        
        #### 💡 CONCEPTOS ADICIONALES
        
        **📈 Tendencia Inflacionaria**
        > Aumento generalizado de precios que obliga a incrementar presupuestos sin expandir servicios.
        
        *Importante:* Un aumento de 100% en presupuesto con 100% de inflación = mismo poder adquisitivo.
        
        ---
        
        **🎯 Concentración Presupuestaria**
        > Situación donde pocos planes de cuenta o organismos representan la mayoría del presupuesto.
        
        *En este dataset:* Top 5 planes = 85% del presupuesto total.
        
        ---
        
        **⚠️ Sub-ejecución**
        > Cuando un organismo gasta menos dinero del que tenía asignado.
        
        *Posibles causas:* Planificación deficiente, trabas administrativas, proyectos cancelados.
        
        ---
        
        **⚠️ Sobre-ejecución**
        > Cuando un organismo gasta más dinero del que tenía asignado.
        
        *Posibles causas:* Emergencias, mala estimación inicial, gastos imprevistos.
        
        ---
        
        ### 📚 ¿Necesitas más información?
        
        Si algún término no está claro o quieres más detalles, puedes:
        1. Explorar los gráficos interactivos del Dashboard
        2. Revisar la sección de Hallazgos para ver cómo se aplican estos conceptos
        3. Consultar la Metodología para entender cómo se procesan los datos
        
        💡 **Tip**: Usa los filtros del Dashboard para ver ejemplos reales de estos conceptos con datos específicos.
        """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p><strong>Análisis de Presupuesto y Gasto de Organismos Públicos</strong></p>
    <p>Desarrollado usando Streamlit | Dataset: 2015-2025 
    <p>© 2025 - Proyecto de Visualización de Datos</p>
</div>
""", unsafe_allow_html=True)
