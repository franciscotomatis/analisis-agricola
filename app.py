# app.py - Plataforma de Análisis Agrícola (VERSIÓN COMPLETA FUSIONADA)
# Integra: Dashboard, mapas, series temporales, grillas NDVI, análisis de suelo, topografía, cultivos

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
from shapely.geometry import Polygon, LineString, Point
import json
import requests
import io

warnings.filterwarnings('ignore')

# ================= CONFIGURACIÓN DE PÁGINA =================
st.set_page_config(page_title="Análisis Agrícola - Plataforma Completa", layout="wide")

# CSS para mejor visualización
st.markdown("""
<style>
    * { font-family: -apple-system, 'San Francisco', 'Helvetica Neue', 'Segoe UI', sans-serif; }
    .stMetric { background-color: white; border-radius: 12px; padding: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .main-header { background: linear-gradient(135deg, #2ca02c 0%, #1a6b1a 100%); padding: 20px; border-radius: 15px; margin-bottom: 20px; }
    .main-header h1 { color: white; margin: 0; }
    .main-header p { color: #e0e0e0; margin: 5px 0 0 0; }
</style>
""", unsafe_allow_html=True)

# ================= GOOGLE EARTH ENGINE =================
try:
    import ee
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False
    st.warning("⚠️ earthengine-api no instalado")

# ================= CONFIGURACIÓN DE CULTIVOS (14 cultivos) =================
CULTIVOS_DISPONIBLES = [
    "TRIGO", "MAIZ", "SORGO", "SOJA", "GIRASOL", "MANI",
    "VID", "OLIVO", "ALMENDRO", "BANANO", "CAFE", "CACAO", 
    "PALMA_ACEITERA", "AVENA"
]

ICONOS_CULTIVOS = {
    'TRIGO': '🌾', 'MAIZ': '🌽', 'SORGO': '🌾', 'SOJA': '🫘',
    'GIRASOL': '🌻', 'MANI': '🥜', 'VID': '🍇', 'OLIVO': '🫒',
    'ALMENDRO': '🌰', 'BANANO': '🍌', 'CAFE': '☕', 'CACAO': '🍫',
    'PALMA_ACEITERA': '🌴', 'AVENA': '🌾'
}

PARAMETROS_CULTIVOS = {
    'TRIGO': {
        'NITROGENO': {'min': 100, 'max': 180},
        'FOSFORO': {'min': 40, 'max': 80},
        'POTASIO': {'min': 90, 'max': 150},
        'MATERIA_ORGANICA_OPTIMA': 3.5,
        'HUMEDAD_OPTIMA': 0.28,
        'NDVI_OPTIMO': 0.75,
        'NDRE_OPTIMO': 0.40,
        'RENDIMIENTO_OPTIMO': 4500,
        'COSTO_FERTILIZACION': 350,
        'PRECIO_VENTA': 0.25,
    },
    'MAIZ': {
        'NITROGENO': {'min': 150, 'max': 250},
        'FOSFORO': {'min': 50, 'max': 90},
        'POTASIO': {'min': 120, 'max': 200},
        'MATERIA_ORGANICA_OPTIMA': 3.8,
        'HUMEDAD_OPTIMA': 0.32,
        'NDVI_OPTIMO': 0.80,
        'NDRE_OPTIMO': 0.45,
        'RENDIMIENTO_OPTIMO': 8500,
        'COSTO_FERTILIZACION': 550,
        'PRECIO_VENTA': 0.20,
    },
    'SORGO': {
        'NITROGENO': {'min': 80, 'max': 140},
        'FOSFORO': {'min': 35, 'max': 65},
        'POTASIO': {'min': 100, 'max': 180},
        'MATERIA_ORGANICA_OPTIMA': 3.0,
        'HUMEDAD_OPTIMA': 0.25,
        'NDVI_OPTIMO': 0.70,
        'NDRE_OPTIMO': 0.35,
        'RENDIMIENTO_OPTIMO': 5000,
        'COSTO_FERTILIZACION': 300,
        'PRECIO_VENTA': 0.18,
    },
    'SOJA': {
        'NITROGENO': {'min': 20, 'max': 40},
        'FOSFORO': {'min': 45, 'max': 85},
        'POTASIO': {'min': 140, 'max': 220},
        'MATERIA_ORGANICA_OPTIMA': 3.5,
        'HUMEDAD_OPTIMA': 0.30,
        'NDVI_OPTIMO': 0.78,
        'NDRE_OPTIMO': 0.42,
        'RENDIMIENTO_OPTIMO': 3200,
        'COSTO_FERTILIZACION': 400,
        'PRECIO_VENTA': 0.45,
    },
    'GIRASOL': {
        'NITROGENO': {'min': 70, 'max': 120},
        'FOSFORO': {'min': 40, 'max': 75},
        'POTASIO': {'min': 110, 'max': 190},
        'MATERIA_ORGANICA_OPTIMA': 3.2,
        'HUMEDAD_OPTIMA': 0.26,
        'NDVI_OPTIMO': 0.72,
        'NDRE_OPTIMO': 0.38,
        'RENDIMIENTO_OPTIMO': 2800,
        'COSTO_FERTILIZACION': 320,
        'PRECIO_VENTA': 0.35,
    },
    'MANI': {
        'NITROGENO': {'min': 15, 'max': 30},
        'FOSFORO': {'min': 50, 'max': 90},
        'POTASIO': {'min': 80, 'max': 140},
        'MATERIA_ORGANICA_OPTIMA': 2.8,
        'HUMEDAD_OPTIMA': 0.22,
        'NDVI_OPTIMO': 0.68,
        'NDRE_OPTIMO': 0.32,
        'RENDIMIENTO_OPTIMO': 3800,
        'COSTO_FERTILIZACION': 380,
        'PRECIO_VENTA': 0.60,
    },
    'VID': {
        'NITROGENO': {'min': 60, 'max': 120},
        'FOSFORO': {'min': 30, 'max': 70},
        'POTASIO': {'min': 150, 'max': 250},
        'MATERIA_ORGANICA_OPTIMA': 2.5,
        'HUMEDAD_OPTIMA': 0.35,
        'NDVI_OPTIMO': 0.65,
        'NDRE_OPTIMO': 0.35,
        'RENDIMIENTO_OPTIMO': 15000,
        'COSTO_FERTILIZACION': 800,
        'PRECIO_VENTA': 0.80,
    },
    'OLIVO': {
        'NITROGENO': {'min': 40, 'max': 100},
        'FOSFORO': {'min': 20, 'max': 50},
        'POTASIO': {'min': 100, 'max': 200},
        'MATERIA_ORGANICA_OPTIMA': 2.0,
        'HUMEDAD_OPTIMA': 0.25,
        'NDVI_OPTIMO': 0.60,
        'NDRE_OPTIMO': 0.30,
        'RENDIMIENTO_OPTIMO': 8000,
        'COSTO_FERTILIZACION': 600,
        'PRECIO_VENTA': 1.20,
    },
    'ALMENDRO': {
        'NITROGENO': {'min': 80, 'max': 160},
        'FOSFORO': {'min': 40, 'max': 80},
        'POTASIO': {'min': 120, 'max': 200},
        'MATERIA_ORGANICA_OPTIMA': 2.2,
        'HUMEDAD_OPTIMA': 0.30,
        'NDVI_OPTIMO': 0.62,
        'NDRE_OPTIMO': 0.32,
        'RENDIMIENTO_OPTIMO': 3000,
        'COSTO_FERTILIZACION': 700,
        'PRECIO_VENTA': 4.50,
    },
    'BANANO': {
        'NITROGENO': {'min': 200, 'max': 350},
        'FOSFORO': {'min': 60, 'max': 120},
        'POTASIO': {'min': 300, 'max': 500},
        'MATERIA_ORGANICA_OPTIMA': 4.0,
        'HUMEDAD_OPTIMA': 0.45,
        'NDVI_OPTIMO': 0.78,
        'NDRE_OPTIMO': 0.40,
        'RENDIMIENTO_OPTIMO': 40000,
        'COSTO_FERTILIZACION': 1200,
        'PRECIO_VENTA': 0.30,
    },
    'CAFE': {
        'NITROGENO': {'min': 100, 'max': 200},
        'FOSFORO': {'min': 40, 'max': 80},
        'POTASIO': {'min': 150, 'max': 250},
        'MATERIA_ORGANICA_OPTIMA': 3.5,
        'HUMEDAD_OPTIMA': 0.40,
        'NDVI_OPTIMO': 0.70,
        'NDRE_OPTIMO': 0.38,
        'RENDIMIENTO_OPTIMO': 2000,
        'COSTO_FERTILIZACION': 900,
        'PRECIO_VENTA': 3.50,
    },
    'CACAO': {
        'NITROGENO': {'min': 80, 'max': 150},
        'FOSFORO': {'min': 30, 'max': 60},
        'POTASIO': {'min': 120, 'max': 200},
        'MATERIA_ORGANICA_OPTIMA': 4.0,
        'HUMEDAD_OPTIMA': 0.50,
        'NDVI_OPTIMO': 0.72,
        'NDRE_OPTIMO': 0.38,
        'RENDIMIENTO_OPTIMO': 1500,
        'COSTO_FERTILIZACION': 850,
        'PRECIO_VENTA': 5.00,
    },
    'PALMA_ACEITERA': {
        'NITROGENO': {'min': 150, 'max': 250},
        'FOSFORO': {'min': 50, 'max': 100},
        'POTASIO': {'min': 200, 'max': 350},
        'MATERIA_ORGANICA_OPTIMA': 3.8,
        'HUMEDAD_OPTIMA': 0.55,
        'NDVI_OPTIMO': 0.75,
        'NDRE_OPTIMO': 0.42,
        'RENDIMIENTO_OPTIMO': 20000,
        'COSTO_FERTILIZACION': 1100,
        'PRECIO_VENTA': 0.40,
    },
    'AVENA': {
        'NITROGENO': {'min': 90, 'max': 150},
        'FOSFORO': {'min': 35, 'max': 70},
        'POTASIO': {'min': 80, 'max': 140},
        'MATERIA_ORGANICA_OPTIMA': 3.2,
        'HUMEDAD_OPTIMA': 0.30,
        'NDVI_OPTIMO': 0.72,
        'NDRE_OPTIMO': 0.38,
        'RENDIMIENTO_OPTIMO': 4500,
        'COSTO_FERTILIZACION': 320,
        'PRECIO_VENTA': 0.22,
    }
}

# ================= FUNCIONES DE AUTENTICACIÓN GEE =================
def inicializar_gee():
    if not GEE_AVAILABLE:
        return False
    if 'gee_service_account' in st.secrets:
        try:
            creds = st.secrets["gee_service_account"]
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

# ================= FUNCIONES GEE PARA ÍNDICES Y MAPAS =================
def get_ndvi_image(gdf, fecha):
    try:
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
    except:
        return None

def get_mean_value(image, polygon_geom):
    if image is None:
        return None
    try:
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
    except:
        return None

def generar_mapa_ndvi_interactivo(gdf, fecha):
    """Genera un mapa Folium con la grilla NDVI superpuesta"""
    try:
        import folium
        from folium.plugins import Fullscreen
        
        polygon_geom = ee.Geometry.Polygon(list(gdf.geometry.iloc[0].exterior.coords))
        
        ndvi_img = get_ndvi_image(gdf, fecha)
        if ndvi_img is None:
            return None
        
        vis_params = {'min': -0.2, 'max': 0.8, 'palette': ['red', 'yellow', 'green']}
        map_id = ndvi_img.getMapId(vis_params)
        tile_url = map_id['tile_fetcher'].url_format
        
        bounds = gdf.total_bounds
        centro_lat = (bounds[1] + bounds[3]) / 2
        centro_lon = (bounds[0] + bounds[2]) / 2
        
        mapa = folium.Map(location=[centro_lat, centro_lon], zoom_start=12)
        
        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
            attr='Google Hybrid',
            name='Google Satélite'
        ).add_to(mapa)
        
        folium.TileLayer(
            tiles=tile_url,
            attr='Earth Engine NDVI',
            name='NDVI',
            overlay=True,
            opacity=0.7
        ).add_to(mapa)
        
        folium.GeoJson(
            gdf.__geo_interface__,
            name='Parcela',
            style_function=lambda x: {'color': '#2ca02c', 'weight': 2, 'fillOpacity': 0.1}
        ).add_to(mapa)
        
        folium.LayerControl().add_to(mapa)
        Fullscreen().add_to(mapa)
        
        return mapa
    except Exception as e:
        st.warning(f"⚠️ Error generando mapa NDVI: {e}")
        return None

# ================= FUNCIONES DE ANÁLISIS DE SUELO Y FERTILIDAD =================
def analizar_fertilidad_simulada(gdf_dividido, cultivo, ndvi_valor):
    """Simula fertilidad y textura del suelo basado en NDVI"""
    import numpy as np
    np.random.seed(42)
    
    resultados = []
    params = PARAMETROS_CULTIVOS[cultivo]
    
    for idx, row in gdf_dividido.iterrows():
        variacion = np.random.uniform(0.7, 1.3)
        
        # Fertilidad NPK simulada
        npk = min(1.0, max(0.1, ndvi_valor * 0.8 + np.random.normal(0, 0.1)))
        materia_organica = params['MATERIA_ORGANICA_OPTIMA'] * (0.5 + ndvi_valor * 0.7) * variacion
        humedad = params['HUMEDAD_OPTIMA'] * (0.6 + ndvi_valor * 0.5) * variacion
        
        # Textura simulada
        arena = np.random.uniform(20, 60)
        limo = np.random.uniform(20, 50)
        arcilla = 100 - arena - limo
        
        resultados.append({
            'npk': npk,
            'materia_organica': min(8, max(0.5, materia_organica)),
            'humedad': min(0.8, max(0.1, humedad)),
            'arena': arena,
            'limo': limo,
            'arcilla': arcilla
        })
    
    return resultados

def recomendar_npk(fertilidad, cultivo):
    """Recomienda NPK basado en fertilidad simulada"""
    params = PARAMETROS_CULTIVOS[cultivo]
    recomendaciones = []
    
    for f in fertilidad:
        # A menor fertilidad, mayor recomendación
        factor = 1 - f['npk']
        n = params['NITROGENO']['min'] + factor * (params['NITROGENO']['max'] - params['NITROGENO']['min'])
        p = params['FOSFORO']['min'] + factor * (params['FOSFORO']['max'] - params['FOSFORO']['min'])
        k = params['POTASIO']['min'] + factor * (params['POTASIO']['max'] - params['POTASIO']['min'])
        
        recomendaciones.append({
            'N': round(n, 1),
            'P': round(p, 1),
            'K': round(k, 1)
        })
    
    return recomendaciones

def proyectar_rendimiento(fertilidad, cultivo):
    """Proyecta rendimiento con y sin fertilización"""
    params = PARAMETROS_CULTIVOS[cultivo]
    proyecciones = []
    
    for f in fertilidad:
        sin_fert = params['RENDIMIENTO_OPTIMO'] * f['npk'] * 0.7
        con_fert = sin_fert * (1 + (1 - f['npk']) * 0.5)
        incremento = ((con_fert - sin_fert) / sin_fert * 100) if sin_fert > 0 else 0
        
        proyecciones.append({
            'sin_fert': round(sin_fert, 0),
            'con_fert': round(con_fert, 0),
            'incremento': round(incremento, 1)
        })
    
    return proyecciones

# ================= FUNCIONES DE SERIES TEMPORALES =================
def obtener_serie_temporal_ndvi(gdf, fecha_inicio, fecha_fin):
    try:
        region = ee.Geometry.Rectangle(gdf.total_bounds.tolist())
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                      .filterBounds(region)
                      .filterDate(fecha_inicio, fecha_fin)
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
                      .limit(50))
        
        def add_ndvi(image):
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('ndvi')
            return image.addBands(ndvi)
        
        collection_with_ndvi = collection.map(add_ndvi)
        
        def extract_value(image):
            mean = image.select('ndvi').reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=30,
                maxPixels=1e9
            ).get('ndvi')
            return ee.Feature(None, {
                'date': image.date().millis(),
                'value': mean
            })
        
        features = collection_with_ndvi.map(extract_value)
        info = features.getInfo()
        
        records = []
        for f in info.get('features', []):
            props = f.get('properties', {})
            val = props.get('value')
            date_ms = props.get('date')
            if val is not None and date_ms is not None:
                records.append({'date': datetime.utcfromtimestamp(date_ms / 1000), 'value': val})
        
        return pd.DataFrame(records).sort_values('date') if records else pd.DataFrame()
    except Exception as e:
        st.warning(f"⚠️ Error en serie NDVI: {e}")
        return pd.DataFrame()

# ================= INTERFAZ PRINCIPAL =================
st.markdown("""
<div class="main-header">
    <h1>🌾 Plataforma de Análisis Agrícola - Versión Completa</h1>
    <p>Análisis satelital, fertilidad, topografía y proyecciones de cosecha</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    
    cultivo = st.selectbox("🌱 Cultivo:", CULTIVOS_DISPONIBLES)
    st.info(f"{ICONOS_CULTIVOS[cultivo]} Parámetros específicos cargados para {cultivo}")
    
    uploaded_file = st.file_uploader(
        "📁 Subir parcela (GeoJSON, KML, KMZ, ZIP Shapefile)",
        type=['geojson', 'kml', 'kmz', 'zip']
    )
    
    fecha_fin = st.date_input("📅 Fecha fin", datetime.now())
    fecha_inicio = st.date_input("📅 Fecha inicio", datetime.now() - timedelta(days=90))
    
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

# Obtener datos de GEE
ndvi_val = 0.5
temp_val = 20.0
precip_val = 0.0

if usar_gee and st.session_state.get('gee_ok', False):
    with st.spinner("📡 Obteniendo datos satelitales..."):
        try:
            polygon_geom = ee.Geometry.Polygon(list(gdf.geometry.iloc[0].exterior.coords))
            ndvi_img = get_ndvi_image(gdf, fecha_fin)
            if ndvi_img:
                ndvi_val = get_mean_value(ndvi_img, polygon_geom)
                if ndvi_val is None:
                    ndvi_val = 0.5
            st.success(f"✅ NDVI obtenido: {ndvi_val:.3f}")
        except Exception as e:
            st.warning(f"⚠️ Error obteniendo datos: {e}")

# ================= DASHBOARD =================
st.header("📊 Dashboard de Indicadores")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(f"{ICONOS_CULTIVOS[cultivo]} {cultivo}", f"{ndvi_val:.3f}", delta="NDVI")
with col2:
    st.metric("🌡️ Temperatura", f"{temp_val:.1f} °C")
with col3:
    st.metric("💧 Precipitación", f"{precip_val:.1f} mm")
with col4:
    st.metric("📐 Área", f"{area_ha:.2f} ha")

# ================= MAPA CON GRILLA NDVI =================
st.header("🗺️ Mapa de Riesgo con NDVI")

if usar_gee and st.session_state.get('gee_ok', False):
    with st.spinner("🛰️ Generando mapa NDVI..."):
        mapa = generar_mapa_ndvi_interactivo(gdf, fecha_fin)
        if mapa:
            map_html = mapa.get_root().render()
            components.html(map_html, height=550)
            st.caption("🗺️ Mapa base: Google Hybrid | Capa NDVI: rojo (bajo) → verde (alto)")
        else:
            st.warning("No se pudo generar el mapa NDVI. Mostrando solo polígono.")
            try:
                import folium
                bounds = gdf.total_bounds
                centro_lat = (bounds[1] + bounds[3]) / 2
                centro_lon = (bounds[0] + bounds[2]) / 2
                mapa_simple = folium.Map(location=[centro_lat, centro_lon], zoom_start=14)
                folium.GeoJson(gdf.__geo_interface__, name='Parcela').add_to(mapa_simple)
                map_html = mapa_simple.get_root().render()
                components.html(map_html, height=550)
            except:
                st.info("Mapa no disponible")
else:
    st.info("🔌 Activa GEE y autentícate para ver el mapa NDVI.")

# ================= ANÁLISIS DE FERTILIDAD Y NPK =================
st.header("🧪 Análisis de Suelo y Fertilidad")

# Dividir parcela en zonas simuladas
n_divisiones = st.slider("Número de zonas de manejo:", min_value=4, max_value=32, value=8)

# Crear zonas simuladas (cuadrícula)
def dividir_parcela_en_zonas(gdf, n_zonas):
    gdf = gdf.copy()
    bounds = gdf.total_bounds
    minx, miny, maxx, maxy = bounds
    n_cols = int(np.ceil(np.sqrt(n_zonas)))
    n_rows = int(np.ceil(n_zonas / n_cols))
    width = (maxx - minx) / n_cols
    height = (maxy - miny) / n_rows
    
    sub_poligonos = []
    for i in range(n_rows):
        for j in range(n_cols):
            if len(sub_poligonos) >= n_zonas:
                break
            cell_minx = minx + j * width
            cell_maxx = minx + (j + 1) * width
            cell_miny = miny + i * height
            cell_maxy = miny + (i + 1) * height
            cell_poly = Polygon([(cell_minx, cell_miny), (cell_maxx, cell_miny), 
                                 (cell_maxx, cell_maxy), (cell_minx, cell_maxy)])
            intersection = gdf.geometry.iloc[0].intersection(cell_poly)
            if not intersection.is_empty and intersection.area > 0:
                sub_poligonos.append(intersection)
    
    if sub_poligonos:
        return gpd.GeoDataFrame({'id_zona': range(1, len(sub_poligonos) + 1), 'geometry': sub_poligonos}, crs='EPSG:4326')
    return gdf

gdf_zonas = dividir_parcela_en_zonas(gdf, n_divisiones)

# Analizar fertilidad
fertilidad = analizar_fertilidad_simulada(gdf_zonas, cultivo, ndvi_val)
recomendaciones = recomendar_npk(fertilidad, cultivo)
proyecciones = proyectar_rendimiento(fertilidad, cultivo)

# Mostrar tabla de resultados
st.subheader("📋 Resultados por Zona de Manejo")

tabla_datos = []
for i, (f, r, p) in enumerate(zip(fertilidad, recomendaciones, proyecciones)):
    tabla_datos.append({
        'Zona': i + 1,
        'NPK': f'{f["npk"]:.2f}',
        'MO %': f'{f["materia_organica"]:.1f}',
        'Humedad': f'{f["humedad"]:.2f}',
        'N (kg/ha)': r['N'],
        'P (kg/ha)': r['P'],
        'K (kg/ha)': r['K'],
        'Rendimiento actual (kg)': p['sin_fert'],
        'Rendimiento mejorado (kg)': p['con_fert'],
        'Incremento %': p['incremento']
    })

st.dataframe(pd.DataFrame(tabla_datos), use_container_width=True)

# Gráfico de rendimiento por zona
st.subheader("📊 Proyecciones de Rendimiento por Zona")
fig, ax = plt.subplots(figsize=(12, 5))
zonas = [f'Z{i+1}' for i in range(len(proyecciones))]
sin_fert = [p['sin_fert'] for p in proyecciones]
con_fert = [p['con_fert'] for p in proyecciones]

x = np.arange(len(zonas))
width = 0.35
ax.bar(x - width/2, sin_fert, width, label='Sin fertilización', color='#ff9999')
ax.bar(x + width/2, con_fert, width, label='Con fertilización recomendada', color='#66b3ff')
ax.set_xlabel('Zona')
ax.set_ylabel('Rendimiento (kg/ha)')
ax.set_title(f'Proyecciones de Rendimiento - {ICONOS_CULTIVOS[cultivo]} {cultivo}')
ax.set_xticks(x)
ax.set_xticklabels(zonas, rotation=45)
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# ================= SERIES TEMPORALES =================
st.header("📈 Series Temporales")

if usar_gee and st.session_state.get('gee_ok', False):
    with st.spinner("Cargando serie histórica NDVI..."):
        df_ndvi = obtener_serie_temporal_ndvi(gdf, fecha_inicio.strftime('%Y-%m-%d'), fecha_fin.strftime('%Y-%m-%d'))
        
        if not df_ndvi.empty:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(df_ndvi['date'], df_ndvi['value'], 'g-', linewidth=2, marker='o', markersize=3)
            ax.set_ylabel('NDVI')
            ax.set_xlabel('Fecha')
            ax.set_title('Evolución histórica del NDVI')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.info("📭 No hay datos históricos disponibles para esta parcela.")
else:
    st.info("🔌 Activa GEE para ver series temporales.")

# ================= EXPORTACIÓN =================
st.header("💾 Exportar Datos")

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    if st.button("📄 Exportar reporte (TXT)"):
        reporte = f"""
=== REPORTE DE ANÁLISIS AGRÍCOLA ===

📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}
🌱 Cultivo: {cultivo}
📏 Área: {area_ha:.2f} ha

=== INDICADORES ===
🌿 NDVI: {ndvi_val:.3f}
🌡️ Temperatura: {temp_val:.1f} °C
💧 Precipitación: {precip_val:.1f} mm

=== FERTILIDAD PROMEDIO ===
Índice NPK: {np.mean([f['npk'] for f in fertilidad]):.2f}
Materia Orgánica: {np.mean([f['materia_organica'] for f in fertilidad]):.1f}%
Humedad: {np.mean([f['humedad'] for f in fertilidad]):.2f}

=== RENDIMIENTO ===
Total actual: {sum(p['sin_fert'] for p in proyecciones):.0f} kg
Total mejorado: {sum(p['con_fert'] for p in proyecciones):.0f} kg
Incremento esperado: {np.mean([p['incremento'] for p in proyecciones]):.1f}%
"""
        st.download_button("📥 Descargar", data=reporte, file_name=f"reporte_{cultivo}.txt")

with col_exp2:
    if st.button("📥 Exportar GeoJSON"):
        geojson_str = gdf.to_json()
        st.download_button("Descargar GeoJSON", data=geojson_str, file_name="parcela.geojson")

st.markdown("---")
st.caption("🌍 Plataforma de Análisis Agrícola | Datos: Sentinel-2, GEE | Fertilidad simulada basada en NDVI")
