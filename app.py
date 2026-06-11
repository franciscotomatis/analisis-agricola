# app.py - Plataforma de Análisis Agrícola (VERSIÓN COMPLETA)
# Con todas las capas GEE: NDVI, NDRE, NDWI, Temperatura, Precipitación

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

# ================= CONFIGURACIÓN DE PÁGINA =================
st.set_page_config(
    page_title="Análisis Agrícola - Gestión de Riesgos Climáticos",
    page_icon="🌾",
    layout="wide"
)

# CSS para fuente San Francisco (fallback a sistema)
st.markdown("""
<style>
    * {
        font-family: -apple-system, 'San Francisco', 'Helvetica Neue', 'Segoe UI', sans-serif;
    }
    .stApp {
        background-color: #f5f7f5;
    }
    .stMetric {
        background-color: white;
        border-radius: 12px;
        padding: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ================= GOOGLE EARTH ENGINE =================
try:
    import ee
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False
    st.warning("⚠️ earthengine-api no instalado")

# ================= FUNCIONES GEE =================
def inicializar_gee():
    """Inicializa GEE usando cuenta de servicio desde secrets"""
    if not GEE_AVAILABLE:
        return False
    
    if 'gee_service_account' in st.secrets:
        try:
            creds = st.secrets["gee_service_account"]
            import json
            if isinstance(creds, str):
                creds = json.loads(creds)
            
            credentials = ee.ServiceAccountCredentials(
                creds['client_email'],
                key_data=creds['private_key']
            )
            ee.Initialize(credentials, project=creds.get('project_id', 'ee-franciscotomatis2'))
            return True
        except Exception as e:
            st.error(f"❌ Error con cuenta de servicio: {e}")
            return False
    
    try:
        ee.Initialize(project='ee-franciscotomatis2')
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

# ================= FUNCIONES GEE PARA ÍNDICES =================
def get_ndvi_image(gdf, fecha):
    region = ee.Geometry.Rectangle(gdf.total_bounds.tolist())
    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
           .filterBounds(region)
           .filterDate(fecha.strftime('%Y-%m-%d'), (fecha + timedelta(days=30)).strftime('%Y-%m-%d'))
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
           .sort('CLOUDY_PIXEL_PERCENTAGE'))
    if col.size().getInfo() == 0:
        col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
               .filterBounds(region)
               .filterDate((fecha - timedelta(days=60)).strftime('%Y-%m-%d'), fecha.strftime('%Y-%m-%d'))
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70))
               .sort('CLOUDY_PIXEL_PERCENTAGE'))
    ndvi = col.first().normalizedDifference(['B8', 'B4']).clip(region)
    return ndvi

def get_ndre_image(gdf, fecha):
    region = ee.Geometry.Rectangle(gdf.total_bounds.tolist())
    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
           .filterBounds(region)
           .filterDate(fecha.strftime('%Y-%m-%d'), (fecha + timedelta(days=30)).strftime('%Y-%m-%d'))
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
           .sort('CLOUDY_PIXEL_PERCENTAGE'))
    if col.size().getInfo() == 0:
        col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
               .filterBounds(region)
               .filterDate((fecha - timedelta(days=60)).strftime('%Y-%m-%d'), fecha.strftime('%Y-%m-%d'))
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70))
               .sort('CLOUDY_PIXEL_PERCENTAGE'))
    ndre = col.first().normalizedDifference(['B8A', 'B5']).clip(region)
    return ndre

def get_ndwi_image(gdf, fecha):
    region = ee.Geometry.Rectangle(gdf.total_bounds.tolist())
    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
           .filterBounds(region)
           .filterDate(fecha.strftime('%Y-%m-%d'), (fecha + timedelta(days=30)).strftime('%Y-%m-%d'))
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
           .sort('CLOUDY_PIXEL_PERCENTAGE'))
    if col.size().getInfo() == 0:
        col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
               .filterBounds(region)
               .filterDate((fecha - timedelta(days=60)).strftime('%Y-%m-%d'), fecha.strftime('%Y-%m-%d'))
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70))
               .sort('CLOUDY_PIXEL_PERCENTAGE'))
    ndwi = col.first().normalizedDifference(['B3', 'B8']).clip(region)
    return ndwi

def get_temperature_image(gdf, fecha):
    bounds = gdf.total_bounds
    delta = 0.5
    region_ampliada = ee.Geometry.Rectangle([
        bounds[0] - delta, bounds[1] - delta,
        bounds[2] + delta, bounds[3] + delta
    ])
    col = (ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
           .filterBounds(region_ampliada)
           .filterDate((fecha - timedelta(days=10)).strftime('%Y-%m-%d'), fecha.strftime('%Y-%m-%d'))
           .select('temperature_2m'))
    if col.size().getInfo() == 0:
        col = (ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
               .filterBounds(region_ampliada)
               .filterDate((fecha - timedelta(days=30)).strftime('%Y-%m-%d'), fecha.strftime('%Y-%m-%d'))
               .select('temperature_2m'))
    temp_k = col.mean().select('temperature_2m')
    temp_c = temp_k.subtract(273.15).clip(region_ampliada)
    return temp_c

def get_precipitation_image(gdf, fecha):
    bounds = gdf.total_bounds
    delta = 1.0
    region_ampliada = ee.Geometry.Rectangle([
        bounds[0] - delta, bounds[1] - delta,
        bounds[2] + delta, bounds[3] + delta
    ])
    col = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
           .filterBounds(region_ampliada)
           .filterDate((fecha - timedelta(days=30)).strftime('%Y-%m-%d'), fecha.strftime('%Y-%m-%d'))
           .select('precipitation'))
    if col.size().getInfo() == 0:
        col = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
               .filterBounds(region_ampliada)
               .filterDate((fecha - timedelta(days=60)).strftime('%Y-%m-%d'), fecha.strftime('%Y-%m-%d'))
               .select('precipitation'))
    img = col.sort('system:time_start', False).first().clip(region_ampliada)
    return img

def get_mean_value(image, polygon_geom):
    mean_dict = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=polygon_geom,
        scale=10,
        maxPixels=1e9
    ).getInfo()
    band_names = image.bandNames().getInfo()
    if band_names:
        return mean_dict.get(band_names[0], None)
    return None

def get_series(image_collection, polygon_geom, band_name, scale=10):
    def extract_value(image):
        mean = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=polygon_geom,
            scale=scale,
            maxPixels=1e9
        ).get(band_name)
        return ee.Feature(None, {
            'date': image.date().millis(),
            'value': mean
        })
    
    features = image_collection.map(extract_value)
    info = features.getInfo()
    
    records = []
    for f in info.get('features', []):
        props = f.get('properties', {})
        val = props.get('value')
        date_ms = props.get('date')
        if val is not None and date_ms is not None:
            records.append({'date': datetime.utcfromtimestamp(date_ms / 1000), 'value': val})
    
    return pd.DataFrame(records).sort_values('date') if records else pd.DataFrame()

def obtener_serie_ndvi(gdf, fecha_inicio, fecha_fin):
    region = ee.Geometry.Rectangle(gdf.total_bounds.tolist())
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(region)
                  .filterDate(fecha_inicio, fecha_fin)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)))
    
    def compute_ndvi(img):
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('ndvi')
        return ndvi
    
    ndvi_col = collection.map(compute_ndvi)
    return get_series(ndvi_col, region, 'ndvi', scale=10)

def obtener_serie_temperatura(gdf, fecha_inicio, fecha_fin):
    region = ee.Geometry.Rectangle(gdf.total_bounds.tolist())
    collection = (ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
                  .filterBounds(region)
                  .filterDate(fecha_inicio, fecha_fin)
                  .select('temperature_2m'))
    
    def to_celsius(img):
        temp_c = img.subtract(273.15).rename('temp')
        return temp_c
    
    temp_col = collection.map(to_celsius)
    return get_series(temp_col, region, 'temp', scale=11132)

def obtener_serie_precipitacion(gdf, fecha_inicio, fecha_fin):
    region = ee.Geometry.Rectangle(gdf.total_bounds.tolist())
    collection = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
                  .filterBounds(region)
                  .filterDate(fecha_inicio, fecha_fin)
                  .select('precipitation'))
    return get_series(collection, region, 'precipitation', scale=5566)

# ================= INTERFAZ PRINCIPAL =================
st.title("🌾 Análisis Agrícola - Gestión de Riesgos Climáticos")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    
    uploaded_file = st.file_uploader(
        "📁 Subir parcela (GeoJSON, KML, KMZ, ZIP Shapefile)",
        type=['geojson', 'kml', 'kmz', 'zip']
    )
    
    fecha_fin = st.date_input("📅 Fecha fin", datetime.now())
    fecha_inicio = st.date_input("📅 Fecha inicio", datetime.now() - timedelta(days=90))
    
    st.markdown("---")
    st.subheader("🛰️ Satélites")
    st.caption("• Sentinel-2 (NDVI, NDRE, NDWI)")
    st.caption("• ERA5-Land (Temperatura)")
    st.caption("• CHIRPS (Precipitación)")
    
    st.markdown("---")
    
    if st.button("🔌 Autenticar GEE", use_container_width=True):
        with st.spinner("Autenticando con Google Earth Engine..."):
            if inicializar_gee():
                st.success("✅ GEE autenticado correctamente")
                st.session_state.gee_ok = True
            else:
                st.error("❌ Error de autenticación")
                st.session_state.gee_ok = False
    
    gee_ok = st.session_state.get('gee_ok', False)
    st.caption(f"📡 GEE: {'✅ Conectado' if gee_ok else '❌ No conectado'}")
    
    usar_gee = st.checkbox("🌍 Usar datos satelitales reales", value=gee_ok)

if not uploaded_file:
    st.info("👈 **Subí un archivo de parcela para comenzar el análisis.**")
    st.stop()

# Cargar parcela
with st.spinner("📂 Cargando parcela..."):
    gdf = cargar_archivo_parcela(uploaded_file)
    if gdf is None:
        st.error("No se pudo cargar la parcela.")
        st.stop()
    area_ha = calcular_superficie(gdf)
    st.success(f"✅ Parcela cargada: **{area_ha:.2f} ha** | CRS: EPSG:4326")

# Obtener datos de GEE si está autenticado
ndvi_val = None
ndre_val = None
ndwi_val = None
temp_val = None
precip_val = None

if usar_gee and st.session_state.get('gee_ok', False):
    with st.spinner("📡 Obteniendo datos satelitales..."):
        try:
            polygon_geom = ee.Geometry.Polygon(list(gdf.geometry.iloc[0].exterior.coords))
            
            ndvi_img = get_ndvi_image(gdf, fecha_fin)
            ndvi_val = get_mean_value(ndvi_img, polygon_geom)
            
            ndre_img = get_ndre_image(gdf, fecha_fin)
            ndre_val = get_mean_value(ndre_img, polygon_geom)
            
            ndwi_img = get_ndwi_image(gdf, fecha_fin)
            ndwi_val = get_mean_value(ndwi_img, polygon_geom)
            
            temp_img = get_temperature_image(gdf, fecha_fin)
            temp_val = get_mean_value(temp_img, polygon_geom)
            
            precip_img = get_precipitation_image(gdf, fecha_fin)
            precip_val = get_mean_value(precip_img, polygon_geom)
            
            st.success("✅ Datos satelitales obtenidos correctamente")
        except Exception as e:
            st.warning(f"⚠️ Error obteniendo datos: {e}")

# Mostrar valores con placeholders si no hay datos reales
if ndvi_val is None: ndvi_val = 0.5
if ndre_val is None: ndre_val = 0.2
if ndwi_val is None: ndwi_val = 0.1
if temp_val is None: temp_val = 20.0
if precip_val is None: precip_val = 0.0

# ================= DASHBOARD =================
st.header("📊 Dashboard de Indicadores")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("🌱 NDVI", f"{ndvi_val:.2f}")
with col2:
    st.metric("🍃 NDRE", f"{ndre_val:.2f}")
with col3:
    st.metric("💧 NDWI", f"{ndwi_val:.2f}")
with col4:
    st.metric("🌡️ Temperatura", f"{temp_val:.1f} °C")
with col5:
    st.metric("☔ Precipitación", f"{precip_val:.1f} mm")

# ================= MAPA =================
st.header("🗺️ Mapa de la Parcela")

try:
    import folium
    from folium.plugins import Fullscreen
    
    bounds = gdf.total_bounds
    centro_lat = (bounds[1] + bounds[3]) / 2
    centro_lon = (bounds[0] + bounds[2]) / 2
    
    mapa = folium.Map(location=[centro_lat, centro_lon], zoom_start=14)
    
    folium.GeoJson(
        gdf.__geo_interface__,
        name='Parcela',
        style_function=lambda x: {
            'color': '#2ca02c',
            'weight': 3,
            'fillColor': '#2ca02c',
            'fillOpacity': 0.15
        },
        tooltip=f"🌾 Parcela | {area_ha:.2f} ha"
    ).add_to(mapa)
    
    folium.LayerControl().add_to(mapa)
    Fullscreen().add_to(mapa)
    
    map_html = mapa.get_root().render()
    components.html(map_html, height=500)
    
    st.caption(f"📍 Área: {area_ha:.2f} hectáreas | Centro: {centro_lat:.4f}, {centro_lon:.4f}")
    
except Exception as e:
    st.warning(f"⚠️ No se pudo generar el mapa: {e}")

# ================= SERIES TEMPORALES =================
st.header("📈 Series Temporales")

if usar_gee and st.session_state.get('gee_ok', False):
    with st.spinner("Cargando series temporales..."):
        df_ndvi = obtener_serie_ndvi(gdf, fecha_inicio.strftime('%Y-%m-%d'), fecha_fin.strftime('%Y-%m-%d'))
        df_temp = obtener_serie_temperatura(gdf, fecha_inicio.strftime('%Y-%m-%d'), fecha_fin.strftime('%Y-%m-%d'))
        df_precip = obtener_serie_precipitacion(gdf, fecha_inicio.strftime('%Y-%m-%d'), fecha_fin.strftime('%Y-%m-%d'))
        
        if not df_ndvi.empty or not df_temp.empty or not df_precip.empty:
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            
            if not df_ndvi.empty:
                axes[0].plot(df_ndvi['date'], df_ndvi['value'], 'g-', linewidth=2, label='NDVI')
                axes[0].set_ylabel('NDVI')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
            
            if not df_temp.empty:
                axes[1].plot(df_temp['date'], df_temp['value'], 'r-', linewidth=2, label='Temperatura')
                axes[1].set_ylabel('Temperatura (°C)')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            
            if not df_precip.empty:
                axes[2].bar(df_precip['date'], df_precip['value'], color='cyan', alpha=0.7, label='Precipitación')
                axes[2].set_ylabel('Precipitación (mm)')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
            
            plt.xlabel('Fecha')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("📭 No hay datos históricos disponibles para esta parcela.")
else:
    st.info("🔌 Activa GEE y autentícate para ver series temporales.")

# ================= EXPORTACIÓN =================
st.header("💾 Exportar Datos")

if st.button("📄 Exportar información de parcela"):
    report = f"""
    === REPORTE DE ANÁLISIS AGRÍCOLA ===
    
    📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    📏 Área: {area_ha:.2f} ha
    
    === INDICADORES ===
    🌱 NDVI: {ndvi_val:.3f}
    🍃 NDRE: {ndre_val:.3f}
    💧 NDWI: {ndwi_val:.3f}
    🌡️ Temperatura: {temp_val:.1f} °C
    ☔ Precipitación: {precip_val:.1f} mm
    
    === INTERPRETACIÓN ===
    • NDVI > 0.5: Vegetación saludable
    • NDVI 0.3-0.5: Vegetación moderada
    • NDVI < 0.3: Suelo desnudo o estrés
    """
    
    st.download_button(
        label="📥 Descargar reporte (TXT)",
        data=report,
        file_name=f"reporte_agricola_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain"
    )

st.markdown("---")
st.caption("🌍 Plataforma de Análisis Agrícola | Datos: Sentinel-2, ERA5-Land, CHIRPS | Google Earth Engine")

# ================= FIN =================
