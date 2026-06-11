# agroia_gee.py - Funciones para GEE (Sin IA)
import geopandas as gpd
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Google Earth Engine
try:
    import ee
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False

# ================= INICIALIZACIÓN =================
def inicializar_gee(project='applied-oxygen-459415-e2'):
    """Inicializa GEE con un proyecto por defecto"""
    if not GEE_AVAILABLE:
        return False
    try:
        ee.Initialize(project=project)
        return True
    except Exception as e:
        print(f"Error inicializando GEE: {e}")
        return False

# ================= FUNCIONES DE SERIES TEMPORALES =================
def _fc_to_dataframe(fc_info, date_key, value_key):
    """Convierte FeatureCollection a DataFrame"""
    records = []
    for f in fc_info.get('features', []):
        props = f.get('properties', {})
        val = props.get(value_key)
        date_ms = props.get(date_key)
        if val is not None and date_ms is not None:
            records.append({'date': datetime.utcfromtimestamp(date_ms / 1000), value_key: val})
    return pd.DataFrame(records).sort_values('date') if records else pd.DataFrame()

def obtener_serie_temporal_ndvi(gdf, fecha_inicio, fecha_fin):
    """Retorna DataFrame con ['date', 'ndvi']"""
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
        return _fc_to_dataframe(info, 'date_ms', 'ndvi')
    except Exception as e:
        print(f"Error NDVI: {e}")
        return pd.DataFrame()

def obtener_serie_temporal_temperatura(gdf, fecha_inicio, fecha_fin):
    """Retorna DataFrame con ['date', 'temp'] en °C"""
    try:
        bounds = gdf.total_bounds
        region = ee.Geometry.Rectangle([bounds[0], bounds[1], bounds[2], bounds[3]])
        collection = (ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
                      .filterBounds(region)
                      .filterDate(fecha_inicio, fecha_fin)
                      .select('temperature_2m')
                      .limit(400))
        
        def to_feature(image):
            mean = (image.subtract(273.15)
                    .reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=11132, maxPixels=1e9)
                    .get('temperature_2m'))
            return ee.Feature(None, {
                'date_ms': image.date().millis(),
                'temp': mean
            })
        
        fc = collection.map(to_feature)
        info = fc.getInfo()
        return _fc_to_dataframe(info, 'date_ms', 'temp')
    except Exception as e:
        print(f"Error Temp: {e}")
        return pd.DataFrame()

def obtener_serie_temporal_precipitacion(gdf, fecha_inicio, fecha_fin):
    """Retorna DataFrame con ['date', 'precip'] en mm"""
    try:
        bounds = gdf.total_bounds
        region = ee.Geometry.Rectangle([bounds[0], bounds[1], bounds[2], bounds[3]])
        collection = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
                      .filterBounds(region)
                      .filterDate(fecha_inicio, fecha_fin)
                      .select('precipitation')
                      .limit(400))
        
        def to_feature(image):
            mean = image.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=region, scale=5566, maxPixels=1e9
            ).get('precipitation')
            return ee.Feature(None, {
                'date_ms': image.date().millis(),
                'precip': mean
            })
        
        fc = collection.map(to_feature)
        info = fc.getInfo()
        return _fc_to_dataframe(info, 'date_ms', 'precip')
    except Exception as e:
        print(f"Error Precip: {e}")
        return pd.DataFrame()

def obtener_ndvi_actual(gdf):
    """NDVI más reciente"""
    try:
        fecha_fin = datetime.now().strftime('%Y-%m-%d')
        fecha_inicio = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        df = obtener_serie_temporal_ndvi(gdf, fecha_inicio, fecha_fin)
        if not df.empty:
            return float(df['ndvi'].iloc[-1])
    except:
        pass
    return 0.5

def obtener_temperatura_actual(gdf):
    """Temperatura más reciente"""
    try:
        fecha_fin = datetime.now().strftime('%Y-%m-%d')
        fecha_inicio = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
        df = obtener_serie_temporal_temperatura(gdf, fecha_inicio, fecha_fin)
        if not df.empty:
            return float(df['temp'].iloc[-1])
    except:
        pass
    return 20.0

def obtener_precipitacion_actual(gdf):
    """Precipitación más reciente"""
    try:
        fecha_fin = datetime.now().strftime('%Y-%m-%d')
        fecha_inicio = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
        df = obtener_serie_temporal_precipitacion(gdf, fecha_inicio, fecha_fin)
        if not df.empty:
            return float(df['precip'].iloc[-1])
    except:
        pass
    return 0.0
