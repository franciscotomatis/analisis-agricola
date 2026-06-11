# app.py - Plataforma de Análisis Agrícola (SIN IA)
# Versión optimizada para despliegue en Streamlit Cloud

import streamlit as st
import streamlit.components.v1 as components
import geopandas as gpd
import pandas as pd
import numpy as np
import tempfile
import os
import zipfile
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
from io import BytesIO

warnings.filterwarnings('ignore')

# ================= GOOGLE EARTH ENGINE =================
try:
    import ee
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False
    st.warning("⚠️ earthengine-api no instalado")

# ================= FUNCIONES GEE (simplificadas) =================
def inicializar_gee():
    """Inicializa GEE usando secrets o proyecto por defecto"""
    if not GEE_AVAILABLE:
        return False
    
    # Intentar autenticar con cuenta de servicio desde secrets
    if 'gee_service_account' in st.secrets:
        try:
            creds = st.secrets["gee_service_account"]
            # Si es string, parsear como JSON
            if isinstance(creds, str):
                import json
                creds = json.loads(creds)
            
            credentials = ee.ServiceAccountCredentials(
                creds['client_email'],
                key_data=creds['private_key']
            )
            ee.Initialize(credentials, project=creds.get('project_id', 'gee-streamlit'))
            return True
        except Exception as e:
            st.error(f"❌ Error con cuenta de servicio: {e}")
    
    # Fallback: intentar autenticación normal (solo funciona localmente)
    try:
        ee.Initialize(project='applied-oxygen-459415-e2')
        return True
    except Exception as e:
        st.error(f"❌ Error autenticando GEE: {e}")
        return False

# ================= FUNCIONES DE CARGA DE PARCELA =================
def validar_crs(gdf):
    if gdf is None or len(gdf) == 0:
        return gdf
    try:
        if gdf.crs is None:
            gdf = gdf.set_crs('EPSG:4326', inplace=False)
        elif str(gdf.crs).upper() != 'EPSG:4326':
            gdf = gdf.to_crs('EPSG:4326')
        return gdf
    except:
        return gdf

def calcular_superficie(gdf):
    try:
        gdf_proj = gdf.to_crs('EPSG:3857')
        area_m2 = gdf_proj.geometry.area.sum()
        return area_m2 / 10000
    except:
        return 0.0

def cargar_shapefile_desde_zip(zip_file):
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(tmp_dir)
            shp_files = [f for f in os.listdir(tmp_dir) if f.endswith('.shp')]
            if shp_files:
                shp_path = os.path.join(tmp_dir, shp_files[0])
                gdf = gpd.read_file(shp_path)
                gdf = validar_crs(gdf)
                return gdf
            else:
                st.error("❌ No se encontró archivo .shp en el ZIP")
                return None
    except Exception as e:
        st.error(f"❌ Error cargando ZIP: {e}")
        return None

def parsear_kml_manual(contenido_kml):
    try:
        root = ET.fromstring(contenido_kml)
        namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}
        polygons = []
        for polygon_elem in root.findall('.//kml:Polygon', namespaces):
            coords_elem = polygon_elem.find('.//kml:coordinates', namespaces)
            if coords_elem is not None and coords_elem.text:
                coords = []
                for coord_pair in coords_elem.text.strip().split():
                    parts = coord_pair.split(',')
                    if len(parts) >= 2:
                        coords.append((float(parts[0]), float(parts[1])))
                if len(coords) >= 3:
                    polygons.append(Polygon(coords))
        if polygons:
            return gpd.GeoDataFrame({'geometry': polygons}, crs='EPSG:4326')
        return None
    except:
        return None

def cargar_kml(kml_file):
    try:
        if kml_file.name.endswith('.kmz'):
            with tempfile.TemporaryDirectory() as tmp_dir:
                with zipfile.ZipFile(kml_file, 'r') as zip_ref:
                    zip_ref.extractall(tmp_dir)
                kml_files = [f for f in os.listdir(tmp_dir) if f.endswith('.kml')]
                if kml_files:
                    kml_path = os.path.join(tmp_dir, kml_files[0])
                    with open(kml_path, 'r', encoding='utf-8') as f:
                        contenido = f.read()
                    gdf = parsear_kml_manual(contenido)
                    if gdf is not None:
                        return gdf
        else:
            contenido = kml_file.read().decode('utf-8')
            gdf = parsear_kml_manual(contenido)
            if gdf is not None:
                return gdf
        kml_file.seek(0)
        gdf = gpd.read_file(kml_file)
        gdf = validar_crs(gdf)
        return gdf
    except Exception as e:
        st.error(f"❌ Error cargando KML/KMZ: {e}")
        return None

def cargar_archivo_parcela(uploaded_file):
    try:
        if uploaded_file.name.endswith('.zip'):
            gdf = cargar_shapefile_desde_zip(uploaded_file)
        elif uploaded_file.name.endswith(('.kml', '.kmz')):
            gdf = cargar_kml(uploaded_file)
        elif uploaded_file.name.endswith('.geojson'):
            gdf = gpd.read_file(uploaded_file)
            gdf = validar_crs(gdf)
        else:
            st.error("Formato no soportado. Use ZIP, KML, KMZ o GeoJSON.")
            return None
        if gdf is not None:
            gdf = validar_crs(gdf)
            gdf = gdf.explode(ignore_index=True)
            gdf = gdf[gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])]
            if len(gdf) == 0:
                st.error("No se encontraron polígonos.")
                return None
            geom_unida = gdf.unary_union
            gdf_unido = gpd.GeoDataFrame({'geometry': [geom_unida]}, crs='EPSG:4326')
            st.info(f"✅ Se unieron {len(gdf)} polígonos.")
            return gdf_unido
        return None
    except Exception as e:
        st.error(f"❌ Error cargando archivo: {e}")
        return None

# ================= FUNCIONES GEE PARA OBTENER DATOS =================
def obtener_serie_temporal_ndvi(gdf, fecha_inicio, fecha_fin):
    """Retorna DataFrame con columnas ['date', 'ndvi']"""
    try:
        bounds = gdf.total_bounds
        region = ee.Geometry.Rectangle([bounds[0], bounds[1], bounds[2], bounds[3]])
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                      .filterBounds(region)
                      .filterDate(fecha_inicio, fecha_fin)
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                      .limit(200))
        
        def to_feature(image):
            ndvi = image.normalizedDifference(['B8', 'B4'])
            mean = ndvi.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=region, scale=30, maxPixels=1e9
            ).get('nd')
            return ee.Feature(None, {
                'date_ms': image.date().millis(),
                'ndvi': mean
            })
        
        fc = collection.map(to_feature)
        info = fc.getInfo()
        
        records = []
        for f in info.get('features', []):
            props = f.get('properties', {})
            val = props.get('ndvi')
            date_ms = props.get('date_ms')
            if val is not None and date_ms is not None:
                records.append({'date': datetime.utcfromtimestamp(date_ms / 1000), 'ndvi': val})
        
        df = pd.DataFrame(records).sort_values('date') if records else pd.DataFrame()
        return df
    except Exception as e:
        st.warning(f"Error NDVI: {e}")
        return pd.DataFrame()

def obtener_valor_actual(gdf, indice):
    """Obtiene valor actual de un índice (NDVI, temperatura, precipitación)"""
    try:
        fecha_fin = datetime.now().strftime('%Y-%m-%d')
        fecha_inicio = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        if indice == 'ndvi':
            df = obtener_serie_temporal_ndvi(gdf, fecha_inicio, fecha_fin)
            if not df.empty:
                return float(df['ndvi'].iloc[-1])
        
        # Placeholder para temperatura y precipitación
        return 0.5 if indice == 'ndvi' else 20.0
    except:
        return 0.5

# ================= INTERFAZ PRINCIPAL =================
st.set_page_config(page_title="Análisis Agrícola - Gestión de Riesgos Climáticos", layout="wide")
st.title("🌾 Plataforma de Análisis Agrícola")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    uploaded_file = st.file_uploader("Subir parcela (GeoJSON, KML, KMZ, ZIP Shapefile)", 
                                      type=['geojson','kml','kmz','zip'])
    fecha_fin = st.date_input("Fecha fin", datetime.now())
    fecha_inicio = st.date_input("Fecha inicio", datetime.now() - timedelta(days=90))
    usar_gee = st.checkbox("Usar GEE (requiere autenticación)", value=True)
    
    st.markdown("---")
    st.caption("📊 Datos satelitales: Sentinel-2, CHIRPS, ERA5-Land")
    
    if usar_gee:
        if st.button("🔌 Autenticar GEE"):
            with st.spinner("Autenticando..."):
                if inicializar_gee():
                    st.success("✅ GEE autenticado")
                    st.session_state.gee_ok = True
                else:
                    st.error("❌ Error de autenticación")
                    st.session_state.gee_ok = False
        
        gee_ok = st.session_state.get('gee_ok', False)
        st.caption(f"GEE: {'✅ Autenticado' if gee_ok else '❌ No autenticado'}")

if not uploaded_file:
    st.info("👈 Sube un archivo de parcela para comenzar el análisis.")
    st.stop()

# Cargar parcela
with st.spinner("Cargando parcela..."):
    gdf = cargar_archivo_parcela(uploaded_file)
    if gdf is None:
        st.error("No se pudo cargar la parcela.")
        st.stop()
    area_ha = calcular_superficie(gdf)
    st.success(f"✅ Parcela cargada: {area_ha:.2f} ha")

# Obtener datos si GEE está autenticado
ndvi_val = 0.5
temp_val = 20.0
precip_val = 0.0

if usar_gee and st.session_state.get('gee_ok', False):
    with st.spinner("Obteniendo datos satelitales..."):
        try:
            ndvi_val = obtener_valor_actual(gdf, 'ndvi')
            st.success(f"✅ NDVI obtenido: {ndvi_val:.2f}")
        except Exception as e:
            st.warning(f"⚠️ Error obteniendo NDVI: {e}")

# ================= DASHBOARD =================
st.header("📊 Dashboard de Indicadores")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🌱 NDVI", f"{ndvi_val:.2f}")
with col2:
    st.metric("🌡️ Temperatura", f"{temp_val:.1f} °C")
with col3:
    st.metric("💧 Precipitación", f"{precip_val:.1f} mm")
with col4:
    st.metric("📐 Área", f"{area_ha:.2f} ha")

# ================= MAPA SIMPLE =================
st.header("🗺️ Vista de la Parcela")
try:
    import folium
    from folium.plugins import Fullscreen
    
    bounds = gdf.total_bounds
    centro_lat = (bounds[1] + bounds[3]) / 2
    centro_lon = (bounds[0] + bounds[2]) / 2
    
    mapa = folium.Map(location=[centro_lat, centro_lon], zoom_start=14)
    folium.GeoJson(gdf.__geo_interface__, name='Parcela').add_to(mapa)
    folium.LayerControl().add_to(mapa)
    
    # Mostrar mapa
    map_html = mapa.get_root().render()
    components.html(map_html, height=500)
except Exception as e:
    st.warning(f"No se pudo generar el mapa: {e}")

# ================= SERIES TEMPORALES =================
st.header("📈 Series Temporales")
if usar_gee and st.session_state.get('gee_ok', False):
    with st.spinner("Cargando series temporales..."):
        df_ndvi = obtener_serie_temporal_ndvi(gdf, fecha_inicio.strftime('%Y-%m-%d'), fecha_fin.strftime('%Y-%m-%d'))
        
        if not df_ndvi.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df_ndvi['date'], df_ndvi['ndvi'], 'g-', linewidth=2)
            ax.set_ylabel('NDVI')
            ax.set_xlabel('Fecha')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.info("No hay datos históricos disponibles para esta parcela.")
else:
    st.info("Activa GEE y autentícate para ver series temporales.")

st.caption("Plataforma de análisis agrícola - Datos satelitales de Google Earth Engine")
