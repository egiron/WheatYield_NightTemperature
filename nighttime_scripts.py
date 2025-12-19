#******************************************************************************
#
# Estimating Wheat Yield at nighttime temperatures
# 
# Copyright: (c) December 2025
# Author: Ernesto Giron (e.giron.e@gmail.com)
#
#
# This source is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# This code is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# A copy of the GNU General Public License is available on the World Wide Web
# at <http://www.gnu.org/copyleft/gpl.html>. You can also obtain it by writing
# to the Free Software Foundation, Inc., 59 Temple Place - Suite 330, Boston,
# MA 02111-1307, USA.
#
#******************************************************************************


from __future__ import absolute_import, division, print_function

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

import os, time, gc
import datetime as dt
import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm import tqdm
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.ticker import StrMethodFormatter
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import MultipleLocator
from sklearn import metrics
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score
from scipy import stats

HOY = dt.datetime.now().strftime('%Y%m%d')
working_dir = os.path.dirname(os.path.abspath(__file__))

# --- Configuration for 180mm width and 300 dpi ---
# Ex for NC papers: 
# 1-Column: 88 x 130, 88 x 180, 88 x 220mm
# 2-Column: 180 x 185, 180 x 210, 180 x 225
# Otros
# 1-Column: 58 x 130
# 2-Column: 121 x 130mm
# 3-Column: 185 x 130, 185 x 185, 185x 210
dpi_value = 300
# Optional: Set global rcParams for consistent style
#plt.rcParams['figure.figsize'] = [fig_width_inches, fig_height_inches]
plt.rcParams['figure.dpi'] = dpi_value
plt.rcParams['savefig.dpi'] = dpi_value # Ensures saved file uses the high DPI

def loadPheno_dataset():
    # Load curated IWIN (ESWYT) dataset for paper
    df = pd.read_parquet(os.path.join(working_dir, 'data', 'curatedESWYTdataset_20251215.parquet'))
    return df

def loadPhenoWeather40yrs_dataset():
    # Corrected Pheno-Weather dataset for 42yrs
    df = pd.read_parquet(os.path.join(working_dir, 'data', 'pheno_weather_ESWYT_40yrs.parquet'))
    return df

def loadFigures_dataset(df):
    # Data for Figure 1
    cols_sel_to_maps = [ 'Loc_no', 'Loc_desc', 'Country', 'BLUE_YLD_t_ha', #'BLUE_YLD_t_ha', 
                        'avg_TMax_GrainFill', 'avg_TMin_GrainFill', 'avg_SolRad_GrainFill', 'Days_GFill_Obs',
                        'Lat', 'Long',
    ]
    data_for_Fig1 = df[cols_sel_to_maps]
    gdf_for_Fig1 = gpd.GeoDataFrame(
        data_for_Fig1, geometry=gpd.points_from_xy(data_for_Fig1.Long, data_for_Fig1.Lat), crs="EPSG:4326"
    )
    del data_for_Fig1
    _ = gc.collect()
    return gdf_for_Fig1

def loadVariablesDescription():
    df = pd.read_csv(os.path.join(working_dir, 'data', 'variablesDescription.csv'))
    df.fillna('', inplace=True)
    return df

def loadTablaofImportantVariables():
    # Load curated IWIN ESWYT dataset for paper
    df = pd.read_csv(os.path.join(working_dir, 'data', 'S1_Importance_variables.csv'), header=2)
    df.drop(columns=['Unnamed: 5'], inplace=True)
    return df

# ***********************************
# Short Function to display Figures 
# ***********************************

# ------------------------
# Grain Yield (t/ha)
# ------------------------
def dispGrainYieldMap(gpdf, width_mm=180, height_mm=100, column='BLUE_YLD_t_ha', label="Grain Yield (t ha$^{-1}$)", 
                      legendVert=False, showFig=True, saveFig=False, 
                      fname=f'Fig_1a_Map_GrainYield_distribution', fmt='jpg', figures_path='./'
                     ):
    data = gpdf[column].reset_index(drop=True)
    sldfile = os.path.join(working_dir, 'data', 'legends', 'GrainYield_v1.sld')
    basemap = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    if (legendVert is False):
        plotMap_Histo_v5(
            data, gpdf, column, label, #xlim=[-130.5, 178.5], ylim=[-70.5, 65.5],
            markersize=2, basemap=basemap, marker_alphaMap=0.85,
            subfig_pos=None, sldfile=sldfile, bins=10, wbar=0.8, cmap=None, #'RdBu_r',
            precision=1, fontsizetickslabels=3.5, fontsizetickslabels1=2.5,
            fontsizebarLabel=3, labelsizeCoords=6, colorbars='skyblue', secondary_xyaxis_tick_pad=2.5, 
            secondary_xyaxis_location=-0.35, vlw=0.5, width_mm=width_mm, height_mm=height_mm, 
            sp=0.02, bxw=0.02,  widths=0.35, whis=1.5, markersizeBoxplotMean=3,  dispBoxPlot=True,
            dispHisto=True, dispBaseMap=True, dispMainMap=True,
            dispXYlabel=True, dispGridlines=False, dispStatslines=True, dispTopBarsCounts=True,
            vert=False, dispBoxplotSpines=False,
            showFig=showFig, saveFig=saveFig, fname=f'{fname}_Vert', fmt=fmt, figures_path=figures_path
        )
    else:
        plotMap_Histo_v5(
            data, gpdf, column, label, #xlim=[-130.5, 178.5], ylim=[-70.5, 65.5],
            markersize=2, basemap=basemap, marker_alphaMap=0.85,
            subfig_pos=None, sldfile=sldfile, bins=10, wbar=0.6, cmap=None, #'RdBu_r',
            precision=1, fontsizetickslabels=3.5, fontsizetickslabels1=2.5,
            fontsizebarLabel=3, labelsizeCoords=6, colorbars='skyblue', secondary_xyaxis_tick_pad=2.5, 
            secondary_xyaxis_location=-0.35, vlw=0.5, width_mm=width_mm, height_mm=height_mm, 
            sp=0.008, bxw=0.025, widths=0.35, whis=1.5, markersizeBoxplotMean=3,  dispBoxPlot=True,
            dispHisto=True, dispBaseMap=True, dispMainMap=True,
            dispXYlabel=True, dispGridlines=False, dispStatslines=True, dispTopBarsCounts=True,
            vert=True, dispBoxplotSpines=False,
            showFig=showFig, saveFig=saveFig, fname=f'{fname}_Vert', fmt=fmt, figures_path=figures_path
        )


# ------------------------
# Solar Radiation MJ/m2/day
# ------------------------
def dispSolRadMap(gpdf, width_mm=180, height_mm=100, column='avg_SolRad_GrainFill', 
                      label="Avg. Solar Radiation (MJ m$^{-2}$ day$^{-1}$)", 
                      legendVert=False, showFig=True, saveFig=False, 
                      fname=f'Fig_1b_Map_SolarRadiation_distribution', fmt='jpg', figures_path='./'
                     ):
    data = gpdf[column].reset_index(drop=True)
    sldfile = os.path.join(working_dir, 'data', 'legends', 'SolRad_v1.sld')
    basemap = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    if (legendVert is False):
        plotMap_Histo_v5(
                data, gpdf, column, label, #xlim=[-130.5, 178.5], ylim=[-70.5, 65.5],
                markersize=0.4, basemap=basemap, marker_alphaMap=0.85,
                subfig_pos=None, sldfile=sldfile, bins=10, wbar=0.8,cmap=None, #'RdBu_r',
                precision=1, fontsizetickslabels=3.5, fontsizetickslabels1=2.5,
                fontsizebarLabel=3, labelsizeCoords=6, colorbars='skyblue', secondary_xyaxis_tick_pad=2.5, 
                secondary_xyaxis_location=-0.35, vlw=0.5, width_mm=width_mm, height_mm=height_mm, 
                sp=0.02, bxw=0.02, widths=0.35, whis=1.5, markersizeBoxplotMean=3,  dispBoxPlot=True,
                dispHisto=True, dispBaseMap=True, dispMainMap=True,
                dispXYlabel=True, dispGridlines=False, dispStatslines=True, dispTopBarsCounts=True,
                vert=False, dispBoxplotSpines=False,
                showFig=showFig, saveFig=saveFig, fname=f'{fname}_Vert', fmt=fmt, figures_path=figures_path
            )
    else:
        plotMap_Histo_v5(
                data, gpdf, column, label, #xlim=[-130.5, 178.5], ylim=[-70.5, 65.5],
                markersize=0.4, basemap=basemap, marker_alphaMap=0.85,
                subfig_pos=None, sldfile=sldfile, bins=10, wbar=0.8,cmap=None, #'RdBu_r',
                precision=1, fontsizetickslabels=3.5, fontsizetickslabels1=2.5,
                fontsizebarLabel=3, labelsizeCoords=6, colorbars='skyblue', secondary_xyaxis_tick_pad=2.5, 
                secondary_xyaxis_location=-0.35, vlw=0.5, width_mm=width_mm, height_mm=height_mm, 
                sp=0.008, bxw=0.025, widths=0.35, whis=1.5, markersizeBoxplotMean=3,  dispBoxPlot=True,
                dispHisto=True, dispBaseMap=True, dispMainMap=True,
                dispXYlabel=True, dispGridlines=False, dispStatslines=True, dispTopBarsCounts=True,
                vert=True, dispBoxplotSpines=False,
                showFig=showFig, saveFig=saveFig, fname=f'{fname}_Vert', fmt=fmt, figures_path=figures_path
            )

# ----------------------------------
# Avg. Maximum Temperature (ºC)
# ----------------------------------
def dispAvgMaxTemperatureMap(gpdf, width_mm=180, height_mm=100, column='avg_TMax_GrainFill', 
                      label="Avg. Maximum Temperature (ºC)", 
                      legendVert=False, showFig=True, saveFig=False, 
                      fname=f'Fig_1c_Map_AvgMaxTemperature_distribution', fmt='jpg', figures_path='./'
                     ):
    data = gpdf[column].reset_index(drop=True)
    sldfile = os.path.join(working_dir, 'data', 'legends', 'TMax_v1.sld')
    basemap = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    if (legendVert is False):
        plotMap_Histo_v5(
                data, gpdf, column, label, #xlim=[-130.5, 178.5], ylim=[-70.5, 65.5],
                markersize=0.3, basemap=basemap, marker_alphaMap=0.85,
                subfig_pos=None, sldfile=sldfile, bins=10, wbar=2.2,cmap=None, #'RdBu_r',
                precision=1, fontsizetickslabels=3.5, fontsizetickslabels1=2.5,
                fontsizebarLabel=3, labelsizeCoords=6, colorbars='skyblue', secondary_xyaxis_tick_pad=2.5, 
                secondary_xyaxis_location=-0.35, vlw=0.5, width_mm=width_mm, height_mm=height_mm, 
                sp=0.02, bxw=0.02, widths=0.35, whis=1.5, markersizeBoxplotMean=3,  dispBoxPlot=True,
                dispHisto=True, dispBaseMap=True, dispMainMap=True,
                dispXYlabel=True, dispGridlines=False, dispStatslines=True, dispTopBarsCounts=True,
                vert=False, dispBoxplotSpines=False,
                showFig=showFig, saveFig=saveFig, fname=f'{fname}_Vert', fmt=fmt, figures_path=figures_path
            )
    else:
        plotMap_Histo_v5(
                data, gpdf, column, label, #xlim=[-130.5, 178.5], ylim=[-70.5, 65.5],
                markersize=0.3, basemap=basemap, marker_alphaMap=0.85,
                subfig_pos=None, sldfile=sldfile, bins=10, wbar=2.2, cmap=None, #'RdBu_r',
                precision=1, fontsizetickslabels=3.5, fontsizetickslabels1=2.5,
                fontsizebarLabel=3, labelsizeCoords=6, colorbars='skyblue', secondary_xyaxis_tick_pad=2.5, 
                secondary_xyaxis_location=-0.35, vlw=0.5, width_mm=width_mm, height_mm=height_mm, 
                sp=0.008, bxw=0.025, widths=0.35, whis=1.5, markersizeBoxplotMean=3,  dispBoxPlot=True,
                dispHisto=True, dispBaseMap=True, dispMainMap=True,
                dispXYlabel=True, dispGridlines=False, dispStatslines=True, dispTopBarsCounts=True,
                vert=True, dispBoxplotSpines=False,
                showFig=showFig, saveFig=saveFig, fname=f'{fname}_Vert', fmt=fmt, figures_path=figures_path
            )


# ------------------------
# Avg. Minimum Temperature (ºC)
# ------------------------
def dispAvgMinTemperatureMap(gpdf, width_mm=180, height_mm=100, column='avg_TMin_GrainFill', 
                      label="Avg. Minimum Temperature (ºC)", 
                      legendVert=False, showFig=True, saveFig=False, 
                      fname=f'Fig_1d_Map_AvgMinTemperature_distribution', fmt='jpg', figures_path='./'
                     ):
    data = gpdf[column].reset_index(drop=True)
    sldfile = os.path.join(working_dir, 'data', 'legends', 'TMin_v1.sld')
    basemap = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    if (legendVert is False):
        plotMap_Histo_v5(
                data, gpdf, column, label, #xlim=[-130.5, 178.5], ylim=[-70.5, 65.5],
                markersize=0.5, basemap=basemap, marker_alphaMap=0.85,
                subfig_pos=None, sldfile=sldfile, bins=10, wbar=2.2,cmap=None, #'RdBu_r',
                precision=1, fontsizetickslabels=3.5, fontsizetickslabels1=2.5,
                fontsizebarLabel=3, labelsizeCoords=6, colorbars='skyblue', secondary_xyaxis_tick_pad=2.5, 
                secondary_xyaxis_location=-0.35, vlw=0.5, width_mm=width_mm, height_mm=height_mm, 
                sp=0.02, bxw=0.02, widths=0.35, whis=1.5, markersizeBoxplotMean=3,  dispBoxPlot=True,
                dispHisto=True, dispBaseMap=True, dispMainMap=True,
                dispXYlabel=True, dispGridlines=False, dispStatslines=True, dispTopBarsCounts=True,
                vert=False, dispBoxplotSpines=False,
                showFig=showFig, saveFig=saveFig, fname=f'{fname}_Vert', fmt=fmt, figures_path=figures_path
            )
    else:
        plotMap_Histo_v5(
                data, gpdf, column, label, #xlim=[-130.5, 178.5], ylim=[-70.5, 65.5],
                markersize=0.5, basemap=basemap, marker_alphaMap=0.85,
                subfig_pos=None, sldfile=sldfile, bins=10, wbar=2.2, cmap=None, #'RdBu_r',
                precision=1, fontsizetickslabels=3.5, fontsizetickslabels1=2.5,
                fontsizebarLabel=3, labelsizeCoords=6, colorbars='skyblue', secondary_xyaxis_tick_pad=2.5, 
                secondary_xyaxis_location=-0.35, vlw=0.5, width_mm=width_mm, height_mm=height_mm, 
                sp=0.008, bxw=0.025, widths=0.35, whis=1.5, markersizeBoxplotMean=3,  dispBoxPlot=True,
                dispHisto=True, dispBaseMap=True, dispMainMap=True,
                dispXYlabel=True, dispGridlines=False, dispStatslines=True, dispTopBarsCounts=True,
                vert=True, dispBoxplotSpines=False,
                showFig=showFig, saveFig=saveFig, fname=f'{fname}_Vert', fmt=fmt, figures_path=figures_path
            )


# ------------------------
# Changes in Avg. Minimum Temperature (ºC)
# ------------------------
def dispChangesInTMinMap(gpdf, width_mm=180, height_mm=100, column='TMinChangeGFill', 
                      label="Changes in Avg. Minimum Temperature (ºC)", 
                      legendVert=False, showFig=True, saveFig=False, 
                      fname=f'Fig_3_Map_ChangesInTMin_distribution', fmt='jpg', figures_path='./'
                     ):
    data = gpdf[column].reset_index(drop=True)
    sldfile = os.path.join(working_dir, 'data', 'legends', 'ChangesInTMin_v2.sld')
    basemap = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    if (legendVert is False):
        plotMap_Histo_v5(
            data, gpdf, column, label, #xlim=[-130.5, 178.5], ylim=[-70.5, 65.5],
            markersize=6, basemap=basemap, marker_alphaMap=0.75,
            subfig_pos=None, sldfile=sldfile, bins=10, wbar=0.4, cmap=None, #'Spectral_r',
            precision=1, fontsizetickslabels=3.5, fontsizetickslabels1=2.5,
            fontsizebarLabel=3, labelsizeCoords=6, colorbars='skyblue', secondary_xyaxis_tick_pad=2.5, 
            secondary_xyaxis_location=-0.28, vlw=0.5, width_mm=width_mm, height_mm=height_mm, 
            sp=0.02, bxw=0.02, widths=0.35, whis=1.5, markersizeBoxplotMean=3,  dispBoxPlot=True,
            dispHisto=True, dispBaseMap=True, dispMainMap=True,
            dispXYlabel=True, dispGridlines=False, dispStatslines=True, dispTopBarsCounts=True,
            vert=False, dispBoxplotSpines=False,
            showFig=showFig, saveFig=saveFig, fname=f'{fname}_Vert', fmt=fmt, figures_path=figures_path
        )
    else:
        plotMap_Histo_v5(
            data, gpdf, column, label, #xlim=[-130.5, 178.5], ylim=[-70.5, 65.5],
            markersize=6, basemap=basemap, marker_alphaMap=0.75,
            subfig_pos=None, sldfile=sldfile, bins=10, wbar=0.4, cmap=None, #'Spectral_r',
            precision=1, fontsizetickslabels=3.5, fontsizetickslabels1=2.5,
            fontsizebarLabel=3, labelsizeCoords=6, colorbars='skyblue', secondary_xyaxis_tick_pad=2.5, 
            secondary_xyaxis_location=-0.28, vlw=0.5, width_mm=width_mm, height_mm=height_mm, 
            sp=0.008, bxw=0.025, widths=0.35, whis=1.5, markersizeBoxplotMean=3,  dispBoxPlot=True,
            dispHisto=True, dispBaseMap=True, dispMainMap=True,
            dispXYlabel=True, dispGridlines=False, dispStatslines=True, dispTopBarsCounts=True,
            vert=True, dispBoxplotSpines=False,
            showFig=showFig, saveFig=saveFig, fname=f'{fname}_Vert', fmt=fmt, figures_path=figures_path
        )


# ------------------------
# Yield Loss (%)
# ------------------------
def dispYldLossMap(gpdf, width_mm=180, height_mm=100, column='YldLossPc', 
                      label="Yield Loss (%)", 
                      legendVert=False, showFig=True, saveFig=False, 
                      fname=f'Fig_3_Map_YldLossPc_distribution', fmt='jpg', figures_path='./'
                     ):
    data = gpdf[column].reset_index(drop=True)
    sldfile = os.path.join(working_dir, 'data', 'legends', 'YieldLoss_v2.sld')
    basemap = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    if (legendVert is False):
        plotMap_Histo_v5(
            data, gpdf, column, label, #xlim=[-130.5, 178.5], ylim=[-70.5, 65.5],
            markersize=1.2, basemap=basemap, marker_alphaMap=0.65,
            subfig_pos=None, sldfile=sldfile, bins=9, wbar=0.3, cmap=None, #'inferno_r',
            precision=1, fontsizetickslabels=3.5, fontsizetickslabels1=2.5,
            fontsizebarLabel=3, labelsizeCoords=6,  colorbars='skyblue', secondary_xyaxis_tick_pad=2.5, 
            secondary_xyaxis_location=-0.28, vlw=0.5, width_mm=width_mm, height_mm=height_mm, 
            sp=0.02, bxw=0.02, widths=0.35, whis=1.5, markersizeBoxplotMean=3,  dispBoxPlot=True,
            dispHisto=True, dispBaseMap=True, dispMainMap=True,
            dispXYlabel=True, dispGridlines=False, dispStatslines=True, dispTopBarsCounts=True,
            vert=False, dispBoxplotSpines=False,
            showFig=showFig, saveFig=saveFig, fname=f'{fname}_Vert', fmt=fmt, figures_path=figures_path
        )
    else:
        plotMap_Histo_v5(
            data, gpdf, column, label, #xlim=[-130.5, 178.5], ylim=[-70.5, 65.5],
            markersize=1.2, basemap=basemap, marker_alphaMap=0.65,
            subfig_pos=None, sldfile=sldfile, bins=9, wbar=0.3, cmap=None, #'inferno_r',
            precision=1, fontsizetickslabels=3.5, fontsizetickslabels1=2.5,
            fontsizebarLabel=3, labelsizeCoords=6,  colorbars='skyblue', secondary_xyaxis_tick_pad=2.5, 
            secondary_xyaxis_location=-0.28, vlw=0.5, width_mm=width_mm, height_mm=height_mm, 
            sp=0.008, bxw=0.025, widths=0.35, whis=1.5, markersizeBoxplotMean=3,  dispBoxPlot=True,
            dispHisto=True, dispBaseMap=True, dispMainMap=True,
            dispXYlabel=True, dispGridlines=False, dispStatslines=True, dispTopBarsCounts=True,
            vert=True, dispBoxplotSpines=False,
            showFig=showFig, saveFig=saveFig, fname=f'{fname}_Vert', fmt=fmt, figures_path=figures_path
        )


# ------------------------
# Observed yield at 255 locations (t/ha)
# ------------------------
def dispLocAveYldMap(gpdf, width_mm=180, height_mm=100, column='BLUE_YLD_t_ha', 
                      label="Observed yield at 255 locations (t ha$^{-1}$)", 
                      legendVert=False, showFig=True, saveFig=False, 
                      fname=f'Fig_3_Map_LocAveYld_distribution', fmt='jpg', figures_path='./'
                     ):
    data = gpdf[column].reset_index(drop=True)
    sldfile = os.path.join(working_dir, 'data', 'legends', 'LocAveYld_v2.sld')
    basemap = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    if (legendVert is False):
        plotMap_Histo_v5(
            data, gpdf, column, label, #xlim=[-130.5, 178.5], ylim=[-70.5, 65.5],
            markersize=1.8, basemap=basemap, marker_alphaMap=0.75,
            subfig_pos=None, sldfile=sldfile, bins=10, wbar=2, cmap=None, #'Spectral_r',
            precision=1, fontsizetickslabels=3.5, fontsizetickslabels1=2.5,
            fontsizebarLabel=3, labelsizeCoords=6,  colorbars='skyblue', secondary_xyaxis_tick_pad=2.5, 
            secondary_xyaxis_location=-0.28, vlw=0.5, width_mm=width_mm, height_mm=height_mm, 
            sp=0.02, bxw=0.02, widths=0.35, whis=1.5, markersizeBoxplotMean=3,  dispBoxPlot=True,
            dispHisto=True, dispBaseMap=True, dispMainMap=True,
            dispXYlabel=True, dispGridlines=False, dispStatslines=True, dispTopBarsCounts=True,
            vert=False, dispBoxplotSpines=False,
            showFig=showFig, saveFig=saveFig, fname=f'{fname}_Vert', fmt=fmt, figures_path=figures_path
        )
    else:
        plotMap_Histo_v5(
            data, gpdf, column, label, #xlim=[-130.5, 178.5], ylim=[-70.5, 65.5],
            markersize=1.8, basemap=basemap, marker_alphaMap=0.75,
            subfig_pos=None, sldfile=sldfile, bins=10, wbar=2, cmap=None, #'Spectral_r',
            precision=1, fontsizetickslabels=3.5, fontsizetickslabels1=2.5,
            fontsizebarLabel=3, labelsizeCoords=6,  colorbars='skyblue', secondary_xyaxis_tick_pad=2.5, 
            secondary_xyaxis_location=-0.28, vlw=0.5, width_mm=width_mm, height_mm=height_mm, 
            sp=0.008, bxw=0.025, widths=0.35, whis=1.5, markersizeBoxplotMean=3,  dispBoxPlot=True,
            dispHisto=True, dispBaseMap=True, dispMainMap=True,
            dispXYlabel=True, dispGridlines=False, dispStatslines=True, dispTopBarsCounts=True,
            vert=True, dispBoxplotSpines=False,
            showFig=showFig, saveFig=saveFig, fname=f'{fname}_Vert', fmt=fmt, figures_path=figures_path
        )
# ------------------------------
# Function to support Figures
# ------------------------------

def CCC(y_true, y_pred):
    '''Lin's Concordance correlation coefficient'''
    # covariance between y_true and y_pred
    s_xy = np.cov([y_true, y_pred])[0,1]
    # means
    x_m = np.mean(y_true)
    y_m = np.mean(y_pred)
    # variances
    s_x_sq = np.var(y_true)
    s_y_sq = np.var(y_pred)
    # condordance correlation coefficient
    ccc = (2.0*s_xy) / (s_x_sq + s_y_sq + (x_m-y_m)**2)
    
    return ccc

def Cb(x,y):
    '''
        Variable C.b is a bias correction factor that measures how far the best-fit line deviates 
        from a line at 45 degrees (a measure of accuracy). 
        
        No deviation from the 45 degree line occurs when C.b = 1. See Lin (1989 page 258).
    '''
    k = len(y)
    yb = np.mean(y)
    sy2 = np.var(y) * (k - 1) / k
    sd1 = np.std(y)
    #print(k, yb, sy2, sd1)
    xb = np.mean(x)
    sx2 = np.var(x) * (k - 1) / k
    sd2 = np.std(x)
    r = np.corrcoef(x, y)[0,1] ## same as pearson CC
    sl = r * sd1 / sd2
    sxy = r * np.sqrt(sx2 * sy2)
    p = 2 * sxy / (sx2 + sy2 + (yb - xb)**2)
    # The following taken from the Stata code for function "concord" (changed 290408):
    bcf = p / r
    return bcf
    
def getScores(df, fld1=None, fld2=None):
    ''' Get stats for model results '''
    if (df is None):
        print("Input data not valid")
        return
    if (fld1 is None or fld2 is None):
        print("Variable are not valid")
        return
    df_notnull = df[[fld1, fld2]].dropna()
    y_test = df_notnull[fld1].astype('double') #float16
    y_predicted = df_notnull[fld2].astype('double') #float16
    accuracy = round(getAccuracy(y_test, y_predicted),2)
    r2score = round((np.corrcoef(y_test.values,y_predicted.values)[0, 1])**2, 2)
    # Calculate Mean Squared Error (MSE)
    mse = round(mean_squared_error(y_test.values, y_predicted.values),2)
    rmse = round(mean_squared_error(np.float64(y_test.values), np.float64(y_predicted.values)) ,2) #squared=False
    n_rmse = round((rmse / y_test.values.mean()), 3)
    mape = round(np.mean(np.abs((y_test.values - y_predicted.values)/y_test.values))*100, 2)
    d1 = ((y_test.values - y_predicted.values).astype('double') ** 2).sum()
    d2 = ((np.abs(y_predicted.values - y_test.values.mean()) + np.abs(y_test.values - y_test.values.mean())).astype('double') ** 2).sum()
    d_index = round(1 - (d1 / d2) ,3)
    ef = round(1 - ( np.sum((y_test.values - y_predicted.values)**2) / np.sum((y_test.values - np.mean(y_test.values))**2) ), 2)
    ccc = round(CCC(y_test.values, y_predicted.values),2)
    cb = round(Cb(y_test.values, y_predicted.values),2)
    return r2score, mape, mse, rmse, n_rmse, d_index, ef, ccc, cb, accuracy,

# Calculate accuracy and precision for each year in a nursery site
def getAccuracy(y_true, y_predicted):
    '''
        Calculate accuracy and precision for each year in a nursery site
    '''
    mape = np.mean(np.abs((y_true - y_predicted)/y_true))*100
    if (mape<=100):
        accuracy = np.round((100 - mape), 2)
    else:
        mape = np.mean(np.abs((y_predicted - y_true)/ y_predicted))*100
        accuracy = np.round((100 - mape), 2)
    return accuracy

# Esta función tiene líneas abajo unas 4 versiones casí iguales que deben ser agrupadas y mejoradas
def estimateLR(df, fld1, fld2, verbose=False):
    x = df[fld1].to_numpy()
    y = df[fld2].to_numpy()
    n = y.size # Number of observations
    # Fit a linear model (degree=1) and get coefficients and covariance matrix
    # setting cov=True returns a tuple (coefficients, covariance_matrix)
    p, cov_matrix = np.polyfit(x, y, deg=1, cov=True) #, full=True
    slope, intercept = p[0], p[1] # slope and intercept
    #par = np.polyfit(x, y, 1, full=True)
    #pend=par[0][0]
    #intercept=par[0][1]
    slope, intercept = p[0], p[1] # slope and intercept
    y_pred = [slope*i + intercept  for i in x]
    # Calculate the R-squared score
    r2 = r2_score(y, y_pred)
    # Calculate the standard error of the coefficients
    # The standard errors are the square roots of the diagonal of the covariance matrix
    std_errors = np.sqrt(np.diag(cov_matrix))
    m_se, b_se = std_errors[0], std_errors[1] # slope and intercept
    # Calculate the t-statistics
    t_m = slope / m_se # t-statistic for the slope
    t_b = intercept / b_se # t-statistic for the intercept
    # Calculate the p-values using the t-distribution
    # For a linear regression with n observations and 2 parameters (slope and intercept),
    # the degrees of freedom (dof) is n - 2.
    dof = n - 2
    # The p-value is calculated as the two-sided probability of the t-statistic
    p_m = 2 * (1 - stats.t.cdf(np.abs(t_m), dof))
    p_b = 2 * (1 - stats.t.cdf(np.abs(t_b), dof))
    if (verbose is True):
        print(f"Slope: {slope:.4f}, t-value: {t_m:.4f}, p-value: {p_m:.4f}")
        print(f"Intercept: {intercept:.4f}, t-value: {t_b:.4f}, p-value: {p_b:.4f}")
        print(r'$y$ = {:.3f}$X$ + {:.3f}'.format(slope, intercept) + r' - R$^{{2}}$: {:.2f}'.format(r2))
    return slope, intercept, y_pred, p_m, p_b

# Function to get metrics and draw linear regression
def addLinearReg(df, ax, fld1, fld2, clr='black', dispNumObs=False, label='Linear Regression',
                text_x=.95, text_y=.95, fontsizeEq=10):
    df_cleaned = df.dropna(subset=[fld1, fld2]).reset_index(drop=True)
    x = df_cleaned[fld1].to_numpy()
    y = df_cleaned[fld2].to_numpy()
    # determine best fit line
    par = np.polyfit(x, y, 1, full=True)
    pend=par[0][0]
    intercept=par[0][1]
    y_predicted = [pend*i + intercept  for i in x]
    l1 = sns.lineplot(x=x,y=y_predicted, color=clr, ax=ax, ls='-', lw=1.85, label=label)
    r2score, mape, mse, rmse, n_rmse, d_index, ef, ccc, cb, accuracy = getScores(df, fld1, fld2)
    if (dispNumObs is True):
        ax.text(text_x, text_y,r'$y$ = {:.3f}$X$ + {:.3f}'.format(pend, intercept)+ ' - R$^2$: {:.2f}'.format(r2score)+ '\nObservations: {}'.format(len(df_cleaned)), 
                fontsize=fontsizeEq, ha='right', va='top', transform=ax.transAxes)
    else:
        ax.text(0.95, 0.95,r'$y$ = {:.3f}$X$ + {:.3f}'.format(pend, intercept)+ ' - R$^2$: {:.2f}'.format(r2score)+'\n',
            fontsize=fontsizeEq, ha='right', va='top', transform=ax.transAxes)
    ax.get_legend().remove()


def getLinearParameters(df, fld1, fld2):
    df_cleaned = df.dropna(subset=[fld1, fld2]).reset_index(drop=True)
    x = df_cleaned[fld1].to_numpy()
    y = df_cleaned[fld2].to_numpy()
    # determine best fit line
    par = np.polyfit(x, y, 1, full=True)
    slope=par[0][0]
    intercept=par[0][1]
    y_predicted = [slope*i + intercept  for i in x]
    return slope, intercept, y_predicted
    
def predExpression_v2(tmin, srad, avg_tmin, avg_srad):
    r = 7.6236613 + (-0.499618 * tmin) + (0.2432738 * srad) + (-0.008957 * (srad - avg_srad) * (tmin - avg_tmin))
    return r

# Function to get metrics and draw linear regression
def addLinearReg3(df, ax, fld1, fld2, s=10, alpha=0.75, clr='black', ls='-', lw=1.5, label='Linear Regression', 
                  eq_x=0.02, eq_y=0.14, text_x=.95, text_y=.95, fontsizeEq=8, dispFunc=True):
    df_cleaned = df.dropna(subset=[fld1, fld2]).reset_index(drop=True)
    x = df_cleaned[fld1].to_numpy()
    y = df_cleaned[fld2].to_numpy()
    par = np.polyfit(x, y, 1, full=True)
    pend=par[0][0]
    intercept=par[0][1]
    y_predicted = [pend*i + intercept  for i in x]
    l1 = sns.lineplot(x=x,y=y_predicted, color=clr, ax=ax, ls=ls, lw=lw, label=label)
    r2score, mape, mse, rmse, n_rmse, d_index, ef, ccc, cb, accuracy = getScores(df, fld1, fld2)
    if (dispFunc is True):
        ax.text(eq_x, eq_y,r'{}: '.format(label)+r'$y$ = {:.3f}$X$ + {:.3f}'.format(pend, intercept),
                fontsize=fontsizeEq, ha='left', va='top', color=clr, transform=ax.transAxes)
    ax.get_legend().remove()
    
def addLinearReg_v4(df, ax, fld1, fld2, clr='black', dispNumObs=False, 
                text_x=.95, text_y=.95, fontsizeEq=10):
    # Add linear regression for GY
    df_cleaned = df.dropna(subset=[fld1, fld2]).reset_index(drop=True)
    x = df_cleaned[fld1].to_numpy()
    y = df_cleaned[fld2].to_numpy()
    # determine best fit line
    par = np.polyfit(x, y, 1, full=True)
    pend=par[0][0]
    intercept=par[0][1]
    y_predicted = [pend*i + intercept  for i in x]
    l1 = sns.lineplot(x=x,y=y_predicted, color=clr, ax=ax, ls='-', lw=1.85) #, label='Linear Regression')
    r2score, mape, mse, rmse, n_rmse, d_index, ef, ccc, cb, accuracy = getScores(df, fld1, fld2)
    if (dispNumObs is True):
        ax.text(text_x, text_y,r'$y$ = {:.3f}$X$ + {:.3f}'.format(pend, intercept)+ ' - R$^2$: {:.2f}'.format(0.52)+ '\nObservations: {}'.format(850), 
                fontsize=fontsizeEq, ha='right', va='top', transform=ax.transAxes)
    else:
        ax.text(0.95, 0.95,r'$y$ = {:.3f}$X$ + {:.3f}'.format(pend, intercept)+ ' - R$^2$: {:.2f}'.format(0.52)+'\n',
            fontsize=fontsizeEq, ha='right', va='top', transform=ax.transAxes)
    
        
    ax.get_legend().remove()

def addSolRad_linearReg3(df, ax, fld1='BLUE_YLD_t_ha', fldTmin='avg_TMin_GrainFill', fldSrad='avg_SolRad_GrainFill', 
                        srad=14.7, s=10, alpha=0.10, clr='gray', ls='-', lw=1.5, mkr='o', 
                        eq_x=0.02, eq_y=0.1, text_x=.95, text_y=.95, fontsizeEq=9):
    _df = df.copy()
    slope_tmin, inter_tmin, y_predicted_tmin = getLinearParameters(_df, fldTmin, fld1)
    avg_tmin = _df[fldTmin].mean() 
    slope_srad, inter_srad, y_predicted_srad = getLinearParameters(_df, fldSrad, fld1)
    avg_srad = _df[fldSrad].mean() 
    #_df[f'SR_{srad}_YLD'] = _df.apply(lambda row:  7.62366132515163 + (-0.499618425515709 * row[fldTmin]) + (0.243273806618153 * srad) + (-0.00895713609452329 * ((row[fldTmin] - avg_tmin) * (srad - avg_srad)) ), axis=1)
    _df[f'SR_{srad}_YLD'] = _df.apply(lambda row:  predExpression_v2(row['avg_TMin_GrainFill'], srad, avg_tmin, avg_srad), axis=1)
    #print(f'SR_{srad}_YLD -->',_df[f'SR_{srad}_YLD'].min(), _df[f'SR_{srad}_YLD'].max())
    srad_idx_mn = _df[f'SR_{srad}_YLD'].idxmin()
    srad_idx_mx = _df[f'SR_{srad}_YLD'].idxmax()
    addLinearReg3(_df, ax, fldTmin, f'SR_{srad}_YLD', s=s, alpha=1, clr=clr, ls=ls, lw=lw, label=f'SR {srad}', 
                  eq_x=eq_x, eq_y=eq_y, text_x=text_x, text_y=text_y, fontsizeEq=fontsizeEq)



def getParametersForHisto(data, sldfile=None, bins=10, cmap=None, precision=1 ):
    # ------------------------------
    # Pre-process data
    # ------------------------------
    def _getLegendFromQGIS_SLD(xmlfile):
        legendColors = []
        legendLabels = []
        bin_edges = []
        with open(xmlfile, 'r') as file:
            for line in file:
                l = line.strip()
                if ('<se:Title>' in l):
                    label = l.replace('<se:Title>',"'").replace('</se:Title>',"'")
                    label = label.replace('\'', '').replace('&lt;', '<').replace('&gt;', '>')
                    legendLabels.append(label)
                if ('<ogc:Literal>' in l):
                    literal = l.replace('<ogc:Literal>',"").replace('</ogc:Literal>',"")
                    bin_edges.append(float(literal)) #round(float(literal), 3))
                if ('<se:SvgParameter name="fill">' in l):
                    color = l.replace('<se:SvgParameter name="fill">',"'").replace('</se:SvgParameter>',"'")
                    legendColors.append(color.replace('\'', ''))
            # end for
            bin_edges = list(set(sorted(bin_edges)))
            bin_edges = sorted(bin_edges)
        return legendLabels, legendColors, bin_edges
    # 
    # Get Histogram
    #hist, bin_edges = np.histogram(data, bins=bins )
    # The rightmost edge is included by default.
    counts = pd.cut(data, bins=bins, include_lowest=False, precision=precision) #, labels=rangos)
    rangos = None
    yield_palette = None
    hist, bin_edges = None, None
    if (sldfile is not None):
        rangos, yield_palette, bins_ud = _getLegendFromQGIS_SLD(sldfile)
        cmap = mcolors.ListedColormap(yield_palette) # Force to give colors in geopandas plot
        assert len(rangos)==len(yield_palette)
        hist, bin_edges = np.histogram(data, bins=bins_ud) 
    else:
        bins_ud = bins
        counts_rounded = counts.round(precision)
        rangos = [str(i).replace(',',' -').replace('(','').replace(')','').replace('[','').replace(']','') for i in counts_rounded.value_counts().sort_index().keys()]
        hist, bin_edges = np.histogram(data, bins=len(rangos) )
    if (rangos is None):
        print("Ranges or labels are not valid!")
        return
    if (cmap is not None and yield_palette is None):
        yield_palette = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(hist)))
    
    return hist, bin_edges, rangos, yield_palette, bins_ud

def plotHistogram_v5(
        data, fig=None, ax=None, subfig_pos=None, sldfile=None, xylabel='', bins=10, wbar=0.8, cmap=None,
        precision=1,  fontsizetickslabels=6.5, fontsizetickslabels1=5.5,
        fontsizebarLabel=6, colorbars='skyblue', primary_xyaxis_tick_pad=1, secondary_xyaxis_tick_pad=2.5, 
        secondary_xyaxis_location=-0.09, vlw=0.5, width_mm=180, height_mm=100, 
        x=0.35, y=0.19, w=0.45, h=0.08, sp=0.01, bxw=0.025, widths=0.5, whis=1.5, markersizeBoxplotMean=6,  
        dispBoxPlot=True, dispXYlabel=True, dispGridlines=False, dispStatslines=True, dispTopBarsCounts=True,
        vert=False, dispBoxplotSpines=True,
        showFig=True, saveFig=False, fname = 'Fig_X', fmt='jpg', figures_path='./'
    ):
    
    inches_per_mm = 1 / 25.4
    fig_width_inches = width_mm * inches_per_mm
    fig_height_inches = height_mm * inches_per_mm

    def getBoxPlotPosition(fig, subfig_pos=None, vert=False, x=None, y=None, w=None, h=None, 
                           sp=None, bxw=None ):
        if (subfig_pos is None):
            subfig_pos=[0.35, 0.19, 0.45, 0.08] # Default Hztal legend
        if (w is None):
            w = subfig_pos[2]
        if (h is None):
            h = subfig_pos[3]
        if (vert is True):
            # x=0.35, y=0.19, w=0.45, h=0.08, sp=0.015, bxw=0.025
            if (x is None):
                x = subfig_pos[0] #0.11 
            if (y is None):
                y = subfig_pos[1] #0.18
            hx_pos = [x, y, w, h]
        else:
            # x=0.05, y=0.15, w=0.35, h=0.13, sp=0.025, bxw=0.025
            if (x is None):
                x = subfig_pos[0] #0.35 
            if (y is None):
                y = subfig_pos[1] #0.22
            hx_pos = [x, y, w, h]
        
        #hx = fig.add_axes(hx_pos)
        # boxplot
        if (sp is None):
            sp=0.015
        if (bxw is None):
            bxw=0.025
        if (vert is True):
            x = x + w + sp
            bx_pos = [x, y, bxw, h]
        else:
            y = y + h + sp
            bx_pos = [x, y, w, bxw]

        return hx_pos, bx_pos

    # ------------------------------
    # Pre-process data
    # ------------------------------
    hist, bin_edges, rangos, yield_palette, bins_ud = getParametersForHisto(data, sldfile, bins, cmap, precision )
    if (cmap is None):
        cmap = mcolors.ListedColormap('CustomYieldCmap',yield_palette)
    # -----------------------------
    # VERTICAL
    # -----------------------------
    hx = None
    bx = None
    if (fig is None):
        fig = plt.figure(figsize=(fig_width_inches, fig_height_inches),
                         facecolor='white', linewidth=0.25, 
                         edgecolor='black', #gridspec_kw={'width_ratios': (0.9, 0.1)}
                        )
        if (vert is True):
            gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[0.9, 0.1])
            hx = fig.add_subplot(gs[0, 0])
            bx = fig.add_subplot(gs[0, 1])
            bx.sharey(hx)
        else:
            gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[0.1, 0.9])
            bx = fig.add_subplot(gs[0])
            hx = fig.add_subplot(gs[1])
            bx.sharex(hx)
    else:
        if (ax is not None):
            hx = ax
            if (subfig_pos is not None and len(subfig_pos)>3):
                hx_pos, bx_pos = getBoxPlotPosition(fig, subfig_pos, vert, sp=sp, bxw=bxw ) 
                bx = fig.add_axes(bx_pos)
                bx.set_facecolor('white')
                if (vert is True):
                    bx.sharey(hx)
                else:
                    bx.sharex(hx)
        else:
            if (subfig_pos is not None and len(subfig_pos)>3):
                hx_pos, bx_pos = getBoxPlotPosition(fig, subfig_pos, vert, sp=sp, bxw=bxw)
                hx = fig.add_axes(hx_pos)
                bx = fig.add_axes(bx_pos)
                bx.set_facecolor('white')
                if (vert is True):
                    bx.sharey(hx)
                else:
                    bx.sharex(hx)
    
    #
    if (vert is True):
        # Hide bars. Only used to display labels in continuous numeric format
        hx.hist(data, bins=bin_edges, color='white', edgecolor='white', align='mid', 
                lw=0.01, alpha=0.0, orientation='horizontal')
    else:
        hx.hist(data, bins=bin_edges, color='white', edgecolor='white', align='mid', 
                lw=0.01, alpha=0.0, orientation='vertical')
    
    # 2. Get histogram data (counts and bins) using np.histogram or plt.hist
    #    using density=True normalizes the histogram area to 1 (probability density)
    counts, bins = np.histogram(data, bins=bin_edges) #, density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:]) # Calculate center of each bin
    
    if (vert is True):
        bars = hx.barh(bin_centers, counts, height=(bins[1]-bins[0]) * wbar, color=colorbars, 
                      edgecolor='gray', alpha=0.95, lw=0.15) #label='Observed Data',
    else:
        bars = hx.bar(bin_centers, counts, width=(bins[1]-bins[0]) * wbar, color=colorbars, 
                      edgecolor='gray', alpha=0.95, lw=0.15) #label='Observed Data',

    if (vert is True):
        # First Y-axis
        hx.set_yticks(bin_centers, labels=bin_centers.round(1), fontsize=fontsizetickslabels1, color='gray')
        hx.set_yticks(bin_edges, labels=bin_edges.round(1), fontsize=fontsizetickslabels1, color='gray')
        hx.tick_params(reset=True)
        hx.tick_params(axis='y', which='both', color='black', pad=primary_xyaxis_tick_pad,
                       length=2, width=0.3, right=False, labelright=False, direction='inout',
                       labelsize=fontsizetickslabels1, labelcolor='black', #labelrotation=90,
                      ) 
        # Secondary Y-axis
        sec_ax2 = hx.secondary_yaxis(location=secondary_xyaxis_location)
        sec_ax2.tick_params(reset=True)
        sec_ax2.tick_params(axis='y', which='both', color='black', pad=secondary_xyaxis_tick_pad, 
                       length=1.5, width=0.3, right=False, labelright=False, direction='in',
                       labelsize=fontsizetickslabels, labelcolor='#444', #labelrotation=90,
                      ) 
        sec_ax2.set_yticks(bin_centers, labels=rangos, fontsize=fontsizetickslabels, color='black', 
                           ha='right', va='center')
        sec_ax2.set_yticklabels(rangos, fontsize=fontsizetickslabels, color='black', ha='right', va='center')
        sec_ax2.spines['left'].set_visible(True)
        sec_ax2.spines['left'].set_edgecolor('#000000') 
        sec_ax2.spines['left'].set_linewidth(0.2) 
        hx.tick_params(axis='x', which='both', color='black', pad=1,
                       length=2, width=0.3, bottom=False, labelbottom=False, 
                       right=False, labelright=False, top=False, labeltop=False,
                      ) 
        hx.set_xticklabels(hx.get_xticks(), fontsize=fontsizetickslabels, color='black') #, va='top')
    else:
        # First X-axis
        hx.set_xticks(bin_centers, labels=bin_centers.round(1), fontsize=fontsizetickslabels1, color='gray')
        hx.set_xticks(bin_edges, labels=bin_edges.round(1), fontsize=fontsizetickslabels1, color='gray')
        hx.tick_params(reset=True)
        hx.tick_params(axis='x', which='both', color='black', pad=primary_xyaxis_tick_pad,
                       length=2, width=0.3, top=False, labeltop=False, direction='inout',
                       labelsize=fontsizetickslabels1, labelcolor='#444', #labelrotation=90,
                      ) 
        # Secondary X-axis
        sec_ax2 = hx.secondary_xaxis(location=secondary_xyaxis_location)
        sec_ax2.tick_params(reset=True)
        sec_ax2.tick_params(axis='x', which='both', color='black', pad=secondary_xyaxis_tick_pad, 
                       length=1.5, width=0.3, top=False, labeltop=False, direction='in',
                       labelsize=fontsizetickslabels, labelcolor='#444', labelrotation=90,
                      ) 
        sec_ax2.set_xticks(bin_centers, labels=rangos, fontsize=fontsizetickslabels, color='black', 
                           ha='center', va='top')
        sec_ax2.set_xticklabels(rangos, rotation=90, fontsize=fontsizetickslabels, color='black', 
                                ha='center', va='top')
        sec_ax2.spines['bottom'].set_visible(True)
        sec_ax2.spines['bottom'].set_edgecolor('#000000') 
        sec_ax2.spines['bottom'].set_linewidth(0.2) 
        hx.tick_params(axis='y', which='both', color='black', pad=1,
                       length=2, width=0.3, left=False, labelleft=False, 
                       right=False, labelright=False, top=False, labeltop=False,
                      ) 
        hx.set_yticklabels(hx.get_yticks(), fontsize=fontsizetickslabels, color='black') 
    
    # Iterate through all spines and apply modifications
    for spine in hx.spines.values():
        spine.set_edgecolor('#000000') 
        spine.set_facecolor('#fff')
        spine.set_linewidth(0.2) 
        spine.set_visible(False)
    
    if (vert is True):
        ## Disp hist vert
        hx.spines['left'].set_visible(True)
    else:
        ## Disp hist hztal
        hx.spines['bottom'].set_visible(True)
    
    # Si labels
    if (dispXYlabel is True):
        if (vert is True):
            sec_ax2.set_ylabel(xylabel, fontsize=fontsizetickslabels+1, fontweight='bold', labelpad=1) 
        else:
            sec_ax2.set_xlabel(xylabel, fontsize=fontsizetickslabels+1, fontweight='bold', labelpad=1) 

    if (dispGridlines is True):
        hx.grid(visible=True, which='major', axis='both', color='#d3d3d3', linestyle='-', linewidth=0.25)
        hx.grid(visible=True, which='minor', axis='both', color='#d3d3d3', linestyle='--', linewidth=0.25, alpha=0.5)
    else:
        hx.grid(visible=True, which='major', axis='both', color='white', linestyle='-', linewidth=0.25)
        hx.grid(visible=True, which='minor', axis='both', color='white', linestyle='--', linewidth=0.25, alpha=0.5)
        # 
    if (dispStatslines is True):
        # A more robust method to get the exact stats used:
        stats = matplotlib.cbook.boxplot_stats(data, whis=1.5)
        if (vert is True):
            hx.axhline(data.mean(), color='purple', ls='--', lw=vlw)
            hx.axhline(data.mean() - data.std(), color='gray', ls='--', lw=vlw)
            hx.axhline(data.mean() + data.std(), color='gray', ls='--', lw=vlw)
            hx.axhline(stats[0]['q1'], color='blue', ls='--', lw=vlw)
            hx.axhline(stats[0]['med'], color='red', ls='--', lw=vlw)
            hx.axhline(stats[0]['q3'], color='blue', ls='--', lw=vlw)
        else:
            hx.axvline(data.mean(), color='purple', ls='--', lw=vlw)
            hx.axvline(data.mean() - data.std(), color='gray', ls='--', lw=vlw)
            hx.axvline(data.mean() + data.std(), color='gray', ls='--', lw=vlw)
            hx.axvline(stats[0]['q1'], color='blue', ls='--', lw=vlw)
            hx.axvline(stats[0]['med'], color='red', ls='--', lw=vlw)
            hx.axvline(stats[0]['q3'], color='blue', ls='--', lw=vlw)
    
    # -----------------------
    # Add labels on top of the bars
    if (dispTopBarsCounts is True):
        hx.bar_label(bars, fmt='{:,.0f}', padding=0.7, fontsize=fontsizebarLabel, color='#444')
    
    # Colors
    if (yield_palette is not None):
        for i, clr in enumerate(yield_palette):
            hx.patches[i].set_facecolor(clr)
            bars.patches[i].set_facecolor(clr)
    
    # ---------------------- 
    # Boxplot
    # ----------------------
    if (dispBoxPlot is True and bx is not None):
        bxplt = bx.boxplot(x=data, widths=widths, whis=whis, #orientation='horizontal', showbox=False,
                      vert=vert, notch=False, showfliers=True, showmeans=True, 
                      boxprops={"color": "gray", "linewidth": 0.25},
                      whiskerprops={"color": "gray", "linewidth": 0.25},
                      flierprops={"color": "red", "markeredgewidth": 0.15, "marker":'o',
                                  "markersize":markersizeBoxplotMean},
                      capprops={"color": "gray", "linewidth": 0.25},
                      medianprops={ "color": "r", "linewidth": 0.5},
                      meanprops={"marker":"D", "markerfacecolor":"none", "markeredgewidth": 0.15,
                                 "markeredgecolor":"black", "markersize":markersizeBoxplotMean},
                     ) 
        if (vert is True):
            bx.tick_params(axis='both', which='both', color='black', direction='in', pad=0, 
                           length=1.5, width=0.3, top=False, labeltop=False, 
                           bottom=False, labelbottom=False, 
                           right=False, labelright=True, left=False, labelleft=False,
                           labelsize=fontsizetickslabels, labelcolor='gray', labelrotation=0,
                          ) 
        else:
            bx.tick_params(axis='both', which='both', color='black', direction='in', pad=0, 
                           length=1.5, width=0.3, top=False, labeltop=True, 
                           bottom=False, labelbottom=False, left=False, labelleft=False,
                           labelsize=fontsizetickslabels, labelcolor='gray', labelrotation=0,
                          ) 
            
        if (dispBoxplotSpines is True):
            for spine in bx.spines.values():
                spine.set_edgecolor('#000000') 
                spine.set_facecolor('#fff')
                spine.set_linewidth(0.2) 
                spine.set_visible(False)
            # hide
            bx.grid(visible=True, which='major', axis='both', color='white', linestyle='-', linewidth=0.25)
            bx.grid(visible=True, which='minor', axis='both', color='white', linestyle='--', 
                    linewidth=0.25, alpha=0.5)
        else:
            bx.axis('off')
    else:
        if (fig is not None and bx is not None):
            fig.delaxes(bx)
    
    fig.tight_layout(pad=0.1)
    if (saveFig is True):
        hoy = dt.datetime.now().strftime('%Y%m%d')
        figures_path = os.path.join(figures_path, '{}_{}'.format('Figures', hoy))
        if not os.path.isdir(figures_path):
            os.makedirs(figures_path)
        if (saveFig is True and fmt=='pdf'):
            fig.savefig(os.path.join(figures_path,"{}_{}.{}".format(fname, hoy, fmt)), 
                        bbox_inches='tight', orientation='portrait', papertype='a4', 
                        edgecolor='none', transparent=False, pad_inches=0.5, dpi=dpi_value)
        
        if (saveFig==True and (fmt=='jpg' or fmt=='png')):
            fig.savefig(os.path.join(figures_path,"{}_{}.{}".format(fname, hoy, fmt)), 
                        bbox_inches='tight',  facecolor=fig.get_facecolor(), edgecolor='none', 
                        transparent=False, dpi=dpi_value)
    if (showFig is True):
        if (saveFig is False):
            plt.tight_layout()
        fig.show()
    else:
        del fig
        plt.close();

    return fig, hx, bx



def plotMap_Histo_v5(
        data, gpdf=None, column='', label='', xlim=None, ylim=None,
        markersize=6, basemap=None, marker_alphaMap=0.75,
        subfig_pos=None, sldfile=None, bins=10, wbar=0.8, cmap=None,
        precision=1, fontsizetickslabels=3.5, fontsizetickslabels1=2.5,
        fontsizebarLabel=3, labelsizeCoords=6, colorbars='skyblue', secondary_xyaxis_tick_pad=2.5, 
        secondary_xyaxis_location=-0.28, vlw=0.5, width_mm=180, height_mm=100, 
        sp=0.02, bxw=0.02, widths=0.35, whis=1.5, markersizeBoxplotMean=3,  dispBoxPlot=True,
        dispHisto=True, dispBaseMap=True, dispMainMap=True,
        dispXYlabel=False, dispGridlines=False, dispStatslines=True, dispTopBarsCounts=True,
        vert=False, dispBoxplotSpines=True,
        showFig=True, saveFig=False, fname = 'Fig_X', fmt='jpg', figures_path='./'
    ):
    
    inches_per_mm = 1 / 25.4
    fig_width_inches = width_mm * inches_per_mm
    fig_height_inches = height_mm * inches_per_mm

    # ------------------------------
    # Pre-process data
    # ------------------------------
    hist, bin_edges, rangos, yield_palette, bins_ud = getParametersForHisto(data, sldfile, bins, cmap, precision )
    if (cmap is None):
        cmap = mcolors.ListedColormap(yield_palette)
    # Set up the plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(fig_width_inches, fig_height_inches))
    ax = fig.add_subplot(1, 1, 1) 
    # -----------------------------
    # Define map config
    # -----------------------------
    if (dispBaseMap is True):
        if (basemap is not None):
            basemap.plot(ax=ax, color='white', edgecolor='darkgrey', linewidth=0.25)
        if (vert is True): # small map
            #xlim=[-130.5, 178.5], ylim=[-70.5, 65.5]
            if (xlim is None): 
                xlim=[-130.5, 178.5]
            if (ylim is None): 
                ylim=[-70.5, 65.5]
        else:
            #xlim=[-130.5, 178.5], ylim=[-87.5, 65.5],
            if (xlim is None): 
                xlim=[-130.5, 178.5]
            if (ylim is None):
                ylim=[-87.5, 65.5]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    # -----------
    # Plot main Map 
    # -----------
    if (dispMainMap is True):
        if(gpdf is not None):
            gpdf.plot(
                column=column,  
                cmap=cmap, 
                legend=False,
                legend_kwds={'label': label, 'orientation': "horizontal"},
                classification_kwds={'bins': bins_ud}, 
                scheme='user_defined', 
                ax=ax,
                edgecolor='#444', 
                linewidth=0.1,
                alpha=marker_alphaMap,
                markersize=data * markersize, 
                zorder=10
            )
        # Define a function to format the labels using LaTeX
        def degree_formatter_lat(value, tick_number):
            t = r"${:.0f}^\circ$".format(value)
            if (float(value)<0):
                t = r"${:.0f}^\circ$S".format(value)
            elif (float(value)>0):
                    t = r"${:.0f}^\circ$N".format(value)
            return t
        def degree_formatter_lon(value, tick_number):
            t = r"${:.0f}^\circ$".format(value)
            if (float(value)<0):
                t = r"${:.0f}^\circ$W".format(value)
            elif (float(value)>0):
                    t = r"${:.0f}^\circ$E".format(value)
            return t
        
        # Apply the FuncFormatter to both axes
        ax.xaxis.set_major_formatter(StrMethodFormatter(u"{x:.0f}°"))
        ax.yaxis.set_major_formatter(StrMethodFormatter(u"{x:.0f}°"))
        ax.tick_params(axis='both', which='both', labelsize=labelsizeCoords)
        ax.set_xlabel("Longitude", fontsize=7, va='top')
        ax.set_ylabel("Latitude", fontsize=7, ha='right')
        ax.grid(visible=True, which='major', axis='both', color='#d3d3d3', linestyle='-', linewidth=0.15, alpha=0.35)
        ax.grid(visible=True, which='minor', axis='both', color='#d3d3d3', linestyle='-', linewidth=0.15, alpha=0.35)
        for spine in ax.spines.values():
            spine.set_edgecolor('#000000') 
            spine.set_linewidth(0.3)           
    # 
    if (dispHisto is True):
        # -----------------------
        # For Hztal Legend
        # -----------------------
        if (vert is False):
            #x=0.35, y=0.19, w=0.45, h=0.08
            if (dispXYlabel is True):
                if (subfig_pos is None):
                    #subfig_pos = [0.35, 0.235, 0.335, 0.05]
                    subfig_pos = [0.35, 0.22, 0.5, 0.105]
            else:
                if (subfig_pos is None):
                    #subfig_pos = [0.35, 0.22, 0.335, 0.05]
                    subfig_pos = [0.35, 0.21, 0.5, 0.105]
            fig2, hx, bx = plotHistogram_v5(
                data, fig, ax=None, subfig_pos=subfig_pos, sldfile=sldfile, xylabel=label, bins=bins, 
                wbar=wbar, cmap=cmap, 
                precision=precision, fontsizetickslabels=fontsizetickslabels, 
                fontsizetickslabels1=fontsizetickslabels1,
                fontsizebarLabel=fontsizebarLabel, colorbars=colorbars, 
                secondary_xyaxis_tick_pad=secondary_xyaxis_tick_pad, 
                secondary_xyaxis_location=secondary_xyaxis_location,
                vlw=vlw, width_mm=width_mm, height_mm=height_mm, 
                sp=sp, bxw=bxw, widths=widths, whis=whis, 
                markersizeBoxplotMean=markersizeBoxplotMean, dispBoxPlot=dispBoxPlot,
                dispXYlabel=dispXYlabel, dispGridlines=dispGridlines, 
                dispStatslines=dispStatslines, dispTopBarsCounts=dispTopBarsCounts,
                vert=vert, dispBoxplotSpines=dispBoxplotSpines,
                showFig=showFig, saveFig=saveFig, fname=fname, fmt=fmt, figures_path=figures_path
            )
        else:
            # -----------------------
            # For vertical  Legend
            # -----------------------
            #x=0.05, y=0.15, w=0.35, h=0.13, sp=0.025, bxw=0.025
            if (dispXYlabel is True):
                if (subfig_pos is None):
                    subfig_pos = [0.14, 0.17, 0.05, 0.38]
            else:
                if (subfig_pos is None):
                    subfig_pos = [0.13, 0.18, 0.05, 0.38]
                    #subfig_pos = [0.05, 0.15, 0.35, 0.13]
            fig3, hx, bx = plotHistogram_v5(
                data, fig, ax=None, subfig_pos=subfig_pos, sldfile=sldfile, xylabel=label, bins=bins, 
                wbar=wbar, cmap=cmap, 
                precision=precision, fontsizetickslabels=fontsizetickslabels, 
                fontsizetickslabels1=fontsizetickslabels1,
                fontsizebarLabel=fontsizebarLabel, colorbars=colorbars, 
                secondary_xyaxis_tick_pad=secondary_xyaxis_tick_pad,
                secondary_xyaxis_location=secondary_xyaxis_location,
                vlw=vlw, width_mm=width_mm, height_mm=height_mm, 
                sp=sp, bxw=bxw, widths=widths, whis=whis, markersizeBoxplotMean=markersizeBoxplotMean,
                dispBoxPlot=dispBoxPlot,
                dispXYlabel=dispXYlabel, dispGridlines=dispGridlines, dispStatslines=dispStatslines,
                dispTopBarsCounts=dispTopBarsCounts,
                vert=vert, dispBoxplotSpines=dispBoxplotSpines,
                showFig=showFig, saveFig=saveFig, fname=fname, fmt=fmt, figures_path=figures_path
            )

    fig.tight_layout(pad=0.1)
    if (saveFig is True):
        hoy = dt.datetime.now().strftime('%Y%m%d')
        figures_path = os.path.join(figures_path, '{}_{}'.format('Figures', hoy))
        if not os.path.isdir(figures_path):
            os.makedirs(figures_path)
        if (saveFig is True and fmt=='pdf'):
            fig.savefig(os.path.join(figures_path,"{}_{}.{}".format(fname, hoy, fmt)), 
                        bbox_inches='tight', orientation='portrait', papertype='a4', 
                        edgecolor='none', transparent=False, pad_inches=0.5, dpi=dpi_value)
        
        if (saveFig==True and (fmt=='jpg' or fmt=='png')):
            fig.savefig(os.path.join(figures_path,"{}_{}.{}".format(fname, hoy, fmt)), 
                        bbox_inches='tight',  facecolor=fig.get_facecolor(), edgecolor='none', transparent=False, dpi=dpi_value)
    if (showFig is True):
        if (saveFig is False):
            plt.tight_layout()
        fig.show()
    else:
        del fig
        plt.close();


def dispFig2_SolRad_avgTmin(df_tmp, fld1='avg_TMin_GrainFill', fld2 = 'BLUE_YLD_t_ha',
             fld3='avg_SolRad_GrainFill', solRadPnts=[14.7, 18.00, 21.3, 24.6, 27.8, 31.1],
             width_mm=121, height_mm=100, 
             showFig=True, saveFig=False, fname = 'Fig_2_SolRad_avgTmin_grainfilling', 
             fmt='jpg', figures_path='./'
            ):
    # Convert width and height from mm to inches
    inches_per_mm = 1 / 25.4
    fig_width_inches = width_mm * inches_per_mm
    fig_height_inches = height_mm * inches_per_mm
    colores = ['#d73027','#fc8d59','#fec44f','#d8b365','#cc4c02','#91bfdb','#4575b4']
    mrks = ['o', '^', '+', 's', '*', 'o', '+']
    sp = 0.03
    df = df_tmp.copy()
    df = df.dropna(subset=[fld1, fld2]).reset_index(drop=True)
    # ---------------------
    # Estimate Grain Yield using Tmin and Solar Radiation during grain-filling period
    avg_tmin = df[fld1].mean()
    avg_srad = df[fld3].mean()
    df['pred_YLD'] = df.apply(lambda row:  predExpression_v2(row[fld1], row[fld3], avg_tmin, avg_srad), axis=1)
    # ---------------------

    fig, ax1 = plt.subplots(1,1, figsize=(fig_width_inches,fig_height_inches), facecolor='white')
    #fig.suptitle('Grain Yield vs Avg. Minimum Temperature', fontweight='bold', fontsize=18)
    ax1.minorticks_on()
    
    g1 = sns.scatterplot(data=df, x=fld1, y=fld2, alpha=0.35, s=12, color='gray') #hue='Country')
    ax1.tick_params(reset=True)
    ax1.tick_params(axis='both', which='minor', length=4, width=1, color='gray', labelsize=7, pad=0)
    ax1.tick_params(axis='both', which='major', labelsize=7, pad=0)
    ax1.tick_params(axis='both', which='major', color='black', pad=1,
                    length=3, width=0.5, right=False, labelright=False,  labelleft=True, left=True,
                    top=False, labeltop=False, bottom=True, labelbottom=True,
                   ) 
    ax1.grid(visible=True, which='major', axis='both', color='#d3d3d3', linestyle='-', linewidth=0.25)
    ax1.grid(visible=True, which='minor', axis='both', color='#d3d3d3', linestyle='--', linewidth=0.25, alpha=0.5)
    ax1.set_xlabel('Avg. Minimum Temperature (ºC)', fontsize=10)
    ax1.set_ylabel('Grain Yield (t ha$^{-1}$)', fontsize=10)
    # Mean before threshold
    ax1.axvline(df[fld1].mean(), ls='--', c='red', linewidth=0.5, label="Mean Grain Yield")
    ax1.axhline(df[fld2].mean(), ls='--', c='red', linewidth=0.5, label="Mean Minimum Temperature")
    addLinearReg_v4(df, ax1, fld1, fld2, clr='black', dispNumObs=True, text_x=.95, text_y=.96, fontsizeEq=7)
    for i, srad in enumerate(solRadPnts):
        addSolRad_linearReg3(df, ax1, fld1='pred_YLD', fldTmin=fld1, fldSrad=fld3, 
                            srad=srad, s=20, alpha=0.90, clr=colores[i], ls='-', lw=1.5, mkr=mrks[i], 
                            #eq_x=0.02, eq_y=0.14 + i*sp, text_x=.95, text_y=.95)
                            eq_x=0.71, eq_y=0.7 + i*sp, text_x=.5, text_y=.5, fontsizeEq=5.5)
    
    legend = ax1.legend( loc='lower left', ncol=2, borderaxespad=0.5, 
                        fontsize=6., framealpha=1, frameon=True,  facecolor='white', edgecolor='black')
    frame = legend.get_frame()
    frame.set_linewidth(0.25)
    for spine in ax1.spines.values():
        spine.set_edgecolor('#000000') 
        spine.set_facecolor('#fff')
        spine.set_linewidth(0.2) 
        
    hoy = dt.datetime.now().strftime('%Y%m%d')
    fig.tight_layout(pad=0.1)
    if (saveFig is True):
        hoy = dt.datetime.now().strftime('%Y%m%d')
        figures_path = os.path.join(figures_path, '{}_{}'.format('Figures', hoy))
        if not os.path.isdir(figures_path):
            os.makedirs(figures_path)
        if (saveFig is True and fmt=='pdf'):
            fig.savefig(os.path.join(figures_path,"{}_{}.{}".format(fname, hoy, fmt)), 
                        bbox_inches='tight', orientation='portrait', papertype='a4', 
                        edgecolor='none', transparent=False, pad_inches=0.5, dpi=dpi_value)
        
        if (saveFig==True and (fmt=='jpg' or fmt=='png')):
            fig.savefig(os.path.join(figures_path,"{}_{}.{}".format(fname, hoy, fmt)), 
                        bbox_inches='tight',  facecolor=fig.get_facecolor(), edgecolor='none', 
                        transparent=False, dpi=dpi_value)
    if (showFig is True):
        if (saveFig is False):
            plt.tight_layout()
        fig.show()
    else:
        del fig
        plt.close();


def dispFigS4(df_tmp, fld1='avg_TMin_GrainFill', fld2='Days_GFill_Obs', width_mm=121, height_mm=100,
              showFig=True, saveFig=False, fname = 'Fig_S4_bySiteOccYear', 
              fmt='jpg', figures_path='./'
             ):
    sns.set_theme(style="whitegrid")
    df = df_tmp.copy()
    # Remove outliers # Loc_no 20349
    df = df[ (df[fld2] > 10.5)].reset_index(drop=True)
    # Convert width and height from mm to inches
    inches_per_mm = 1 / 25.4
    fig_width_inches = width_mm * inches_per_mm
    fig_height_inches = height_mm * inches_per_mm
    fig, ax1 = plt.subplots(1,1, figsize=(fig_width_inches,fig_height_inches), facecolor='white')
    sns.set_theme(style="whitegrid")
    g1 = sns.scatterplot(data=df, x=fld1, y=fld2, alpha=0.85, s=20)
    addLinearReg(df, ax1, fld1, fld2, clr='black', dispNumObs=True, 
                 text_x=.95, text_y=.95, fontsizeEq=8)
    #
    #fig.suptitle('Wheat duration of Grain-filling vs Avg. Minimum Temperature (ESWYT)', fontweight='bold', fontsize=18) 
    #plt.title('Grain Yield vs Avg. Minimum Temperature\nGrain-filling period', fontweight='bold')
    # Add hztal and vertical lines
    ax1.axvline(df[fld1].mean(), ls='--', c='red', linewidth=0.5, label="Mean Duration of Grain-filling")
    ax1.axhline(df[fld2].mean(), ls='--', c='red', linewidth=0.5, label="Mean Minimum Temperature")
    #ax1.set_title('Grain Yield vs Avg. Minimum Temperature\nGrain-filling period', fontweight='bold', fontsize=16)
    ax1.set_xlabel('Avg. Minimum Temperature (ºC)', fontsize=10)
    ax1.set_ylabel('Duration of Grain-filling (days)', fontsize=10)
    ax1.tick_params(reset=True)
    ax1.tick_params(axis='both', which='minor', length=4, width=1, color='gray', labelsize=7, pad=0)
    ax1.tick_params(axis='both', which='major', labelsize=7, pad=0)
    ax1.tick_params(axis='both', which='major', color='black', pad=1,
                    length=3, width=0.5, right=False, labelright=False,  labelleft=True, left=True,
                    top=False, labeltop=False, bottom=True, labelbottom=True,
                   ) 
    ax1.grid(visible=True, which='major', axis='both', color='#d3d3d3', linestyle='-', linewidth=0.25)
    ax1.grid(visible=True, which='minor', axis='both', color='#d3d3d3', linestyle='--', linewidth=0.25, alpha=0.5)
    ax1.set_axisbelow(True)
    for spine in ax1.spines.values():
        spine.set_edgecolor('#000000') 
        spine.set_facecolor('#fff')
        spine.set_linewidth(0.2) 
    #
    legend = ax1.legend( loc='lower left', ncol=1, borderaxespad=0.5, 
                        fontsize=5.5, framealpha=1, frameon=True,  facecolor='white', edgecolor='black')
    frame = legend.get_frame()
    frame.set_linewidth(0.25)
    fig.tight_layout(pad=0)
    hoy = dt.datetime.now().strftime('%Y%m%d')
    if (saveFig is True):
        hoy = dt.datetime.now().strftime('%Y%m%d')
        figures_path = os.path.join(figures_path, '{}_{}'.format('Figures', hoy))
        if not os.path.isdir(figures_path):
            os.makedirs(figures_path)
        if (saveFig is True and fmt=='pdf'):
            fig.savefig(os.path.join(figures_path,"{}_{}.{}".format(fname, hoy, fmt)), 
                        bbox_inches='tight', orientation='portrait', papertype='a4', 
                        edgecolor='none', transparent=False, pad_inches=0.5, dpi=dpi_value)
        
        if (saveFig==True and (fmt=='jpg' or fmt=='png')):
            fig.savefig(os.path.join(figures_path,"{}_{}.{}".format(fname, hoy, fmt)), 
                        bbox_inches='tight',  facecolor=fig.get_facecolor(), edgecolor='none', 
                        transparent=False, dpi=dpi_value)
    if (showFig is True):
        if (saveFig is False):
            plt.tight_layout()
        fig.show()
    else:
        del fig
        plt.close();

def dispFigS2(df_tmp, fld2='BLUE_YLD_t_ha', width_mm=180, height_mm=65,
              showFig=True, saveFig=False, fname = 'Fig_S2_bySiteOccYear', 
              fmt='jpg', figures_path='./'
             ):
    sns.set_theme(style="whitegrid")
    df = df_tmp.copy()
    # Convert width and height from mm to inches
    inches_per_mm = 1 / 25.4
    fig_width_inches = width_mm * inches_per_mm
    fig_height_inches = height_mm * inches_per_mm
    fig, axis = plt.subplots(1, 3, figsize=(fig_width_inches,fig_height_inches), facecolor='white', 
                             constrained_layout=True, sharex=False, sharey=True)
    #fig.suptitle('Grain Yield (Site-year-Occ) vs Tmax, Tmin, SolRad\nduring ESWYT Grain-filling period', fontweight='bold', fontsize=18) 
    #fig.suptitle('Grain Yield vs Tmax, Tmin, SolRad\nduring Grain-filling period', fontweight='bold', fontsize=18) 
    def drawFig(df, ax, fld1='', fld2='', subtitle='', xlbl='', ylbl='', clr='red', fl='', fontsize=7):
        g1 = sns.scatterplot(data=df, x=fld1, y=fld2, alpha=0.5, s=6, #label='SiteYrOcc',
                             #hue='Country', 
                             ax=ax)
        ax.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
        ax.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
        ax.set_axisbelow(True)
        addLinearReg(df, ax, fld1, fld2, clr='black', dispNumObs=True, text_x=.98, text_y=.97, fontsizeEq=5.5)
        # Add figure letter
        ax.text(.01, 1.12,'{}'.format(fl), fontweight='bold', fontsize=12, 
                ha='left', va='top', transform=ax.transAxes)
        # Add hztal and vertical lines
        ax.axvline(df[fld1].mean(), ls='--', c=clr, linewidth=0.5, label="Mean Grain Yield")
        ax.axhline(df[fld2].mean(), ls='--', c=clr, linewidth=0.5, label="Mean X axis")
        ax.set_xlabel(f'{xlbl}', fontsize=fontsize)
        ax.set_ylabel(f'{ylbl}', fontsize=fontsize)
    #
    # ------------------------------
    # Chart 1
    # ------------------------------
    ax1 = axis[0]
    drawFig(df, ax1, 'avg_TMax_GrainFill' , fld2, subtitle='', 
            xlbl='Avg. Maximum Temperature (ºC)', 
            ylbl='Grain Yield (t ha$^{-1}$)', fl='a')
    
    ax2 = axis[1]
    drawFig(df, ax2, 'avg_TMin_GrainFill' , fld2, subtitle='', 
            xlbl='Avg. Minimum Temperature (ºC)', 
            ylbl='Grain Yield (t ha$^{-1}$)', fl='b')
    
    ax3 = axis[2]
    drawFig(df, ax3, 'avg_SolRad_GrainFill', fld2, subtitle='', 
            xlbl='Avg. Solar Radiation (MJ/m$^{2}$/day)', #J m⁻² day⁻¹
            ylbl='Grain Yield (t ha$^{-1}$)', fl='c')
    
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(reset=True)
        ax.tick_params(axis='both', which='minor', length=1.5, width=1, color='gray', labelsize=6, pad=0)
        ax.tick_params(axis='both', which='major', labelsize=6, pad=0)
        ax.tick_params(axis='both', which='major', color='black', pad=1,
                        length=2, width=0.35, right=False, labelright=False,  labelleft=True, left=True,
                        top=False, labeltop=False, bottom=True, labelbottom=True,
                       ) 
        ax.grid(visible=True, which='major', axis='both', color='#d3d3d3', linestyle='-', linewidth=0.25)
        ax.grid(visible=True, which='minor', axis='both', color='#d3d3d3', linestyle='--', linewidth=0.25, alpha=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor('#000000') 
            spine.set_facecolor('#fff')
            spine.set_linewidth(0.2) 
    for ax in [ax2, ax3]:
        ax.tick_params(axis='both', which='major', color='black', pad=0,
                        length=2, width=0.35, right=False, labelright=False,  labelleft=False, left=False,
                        top=False, labeltop=False, bottom=True, labelbottom=True,
                       ) 
    # ------------------------------
    # Add legend()
    def getLegend_HandlesLabels(ax, handout, lablout):
        handles, labels = ax.get_legend_handles_labels()
        for h,l in zip(handles,labels):
            if l not in lablout:
                lablout.append(l)
                handout.append(h)
        return handout, lablout
    
    handout=[]
    lablout=[]
    handout, lablout = getLegend_HandlesLabels(ax1, handout, lablout)
    handout, lablout = getLegend_HandlesLabels(ax2, handout, lablout)
    handout, lablout = getLegend_HandlesLabels(ax3, handout, lablout)
    legend = fig.legend(handout,lablout, bbox_to_anchor=(0.5, -0.05), loc="center", ncol=8, 
               borderaxespad=0,fontsize=8) 
    legend.set_visible(False)
    fig.tight_layout(pad=0.5)
    fig.tight_layout(pad=0)
    hoy = dt.datetime.now().strftime('%Y%m%d')
    if (saveFig is True):
        hoy = dt.datetime.now().strftime('%Y%m%d')
        figures_path = os.path.join(figures_path, '{}_{}'.format('Figures', hoy))
        if not os.path.isdir(figures_path):
            os.makedirs(figures_path)
        if (saveFig is True and fmt=='pdf'):
            fig.savefig(os.path.join(figures_path,"{}_{}.{}".format(fname, hoy, fmt)), 
                        bbox_inches='tight', orientation='portrait', papertype='a4', 
                        edgecolor='none', transparent=False, pad_inches=0.5, dpi=dpi_value)
        
        if (saveFig==True and (fmt=='jpg' or fmt=='png')):
            fig.savefig(os.path.join(figures_path,"{}_{}.{}".format(fname, hoy, fmt)), 
                        bbox_inches='tight',  facecolor=fig.get_facecolor(), edgecolor='none', 
                        transparent=False, dpi=dpi_value)
    if (showFig is True):
        if (saveFig is False):
            plt.tight_layout()
        fig.show()
    else:
        del fig
        plt.close();


# Estimate Change in Tmin and Yield loss
def estimateChangeInTmin_YieldLoss(df_pw, df_ESWYT):
    df_ChgTMin = pd.DataFrame()
    for _loc in tqdm(df_pw['Loc_no'].unique()): #[:2]
        df = df_pw[df_pw['Loc_no']==_loc]
        df.sort_values(by=['Year'], inplace=True)
        try:
            slope, intercept, y_pred, p_m, p_b = estimateLR(df, fld1='Year', fld2='avg_TMin_Hplus10dM')
            _df = pd.DataFrame({
                'Loc_no': [_loc],
                'Loc_desc': df['Loc_desc'].values[0], 
                'Country': df['Country'].values[0], 
                'Lat':df['Lat'].values[0], 'Long':df['Long'].values[0], 
                'SlopeTMinGFill': [slope], 
                'p_value': [p_m],
                'TMinChangeGFill': [42 * slope], 
                'YldLossTon': [42 * slope * 0.494], #from Fig 2. 1ºC decreases yield by 0.5 t/ha x 42 years 
                #'YldLossPc'
            })
            if len(df_ChgTMin)>0:
                df_ChgTMin = pd.concat([df_ChgTMin, _df], axis=0)
            else:
                df_ChgTMin = _df
        except:
            pass
    #
    df_ChgTMin.reset_index(drop=True, inplace=True)
    # Merge with Avg TMin and Avg Grain Yield for each location
    df_LocAveYld = df_ESWYT[['Loc_no','Country', 'avg_TMin_GrainFill', 'BLUE_YLD_t_ha',
                             ]].groupby(['Loc_no'], as_index=False).agg({
        'avg_TMin_GrainFill':'mean', 'BLUE_YLD_t_ha':'mean', 
                  }).round(3).reset_index(drop=True)
    df_ChgTMin = pd.merge(df_ChgTMin, df_LocAveYld, how='left', on=['Loc_no'])
    # Estimate Yield loss in percentage
    df_ChgTMin['YldLossPc'] = df_ChgTMin[['BLUE_YLD_t_ha','YldLossTon']]\
    .apply(lambda row: row['YldLossTon'] / row['BLUE_YLD_t_ha'] * 100, axis=1)
    # Export to GIS format
    gdf_ChgTMin = gpd.GeoDataFrame(
        df_ChgTMin, geometry=gpd.points_from_xy(df_ChgTMin.Long, 
                                                df_ChgTMin.Lat), crs="EPSG:4326"
    )
    del df_LocAveYld, df_ChgTMin
    return gdf_ChgTMin

def loadTableS1(df):
    # Supplementary Table 1
    df_TableS1 = df[['Country', 'SlopeTMinGFill', #'p_value', 
            'avg_TMin_GrainFill', 'BLUE_YLD_t_ha', 
            'TMinChangeGFill', 'YldLossTon', 'YldLossPc' ]].groupby(['Country'], as_index=False).mean().round(3)
    sel_col = [ 'Country', 'TMinChangeGFill', 'BLUE_YLD_t_ha', 'YldLossTon', 'YldLossPc' ]
    df_TableS1 = df_TableS1[sel_col].groupby(['Country'], as_index=False).agg({
        'TMinChangeGFill': 'mean', 'BLUE_YLD_t_ha': 'mean',
        'YldLossTon': 'mean', 'YldLossPc': 'mean'
    }).round(3)
    df_TableS1['Country'] = df_TableS1['Country'].apply(lambda x: str(x).title())
    df_TableS1.rename(columns={
        'TMinChangeGFill':'Change in TMin grain filling (℃)',
        'BLUE_YLD_t_ha':'Yield (t/ha)', 
        'YldLossTon':'Yield loss (t/ha)', 
        'YldLossPc': 'Yield loss (%)'
    }, inplace=True)
    return df_TableS1

# --------------------------
# Panel
# --------------------------
def subplotMap(data,
        gpdf=None, ax=None, column='', label='', xlim=None, ylim=None,
        basemap=None, markersizeMap=6, marker_alphaMap=0.85,
        subfig_pos=None, sldfile=None, bins=10, wbar=0.8, cmap=None, #'RdBu_r',
        precision=1, fontsizetickslabels=3.5, fontsizetickslabels1=2.5,
        fontsizebarLabel=3, labelsizeCoords=6, colorbars='skyblue', secondary_xyaxis_tick_pad=2.5, 
        secondary_xyaxis_location=-0.28, vlw=0.5, width_mm=180, height_mm=100, 
        widths=0.5, whis=1.5, markersizeBoxplotMean=3,  dispBoxPlot=True,
        dispHisto=True, dispBaseMap=True, dispMainMap=True,
        dispXYlabel=False, dispGridlines=False, dispStatslines=True, dispTopBarsCounts=True,
        vert=False, dispBoxplotSpines=True
    ):
    inches_per_mm = 1 / 25.4
    fig_width_inches = width_mm * inches_per_mm
    fig_height_inches = height_mm * inches_per_mm
    # ------------------------------
    # Pre-process data
    # ------------------------------
    hist, bin_edges, rangos, yield_palette, bins_ud = getParametersForHisto(data, sldfile, bins, cmap, precision )
    if (cmap is None):
        cmap = mcolors.ListedColormap(yield_palette)
    plt.style.use('seaborn-v0_8-whitegrid')
    # 1. Create the main figure and axes
    if (ax is None):
        fig = plt.figure(figsize=(fig_width_inches, fig_height_inches), facecolor='white', linewidth=0.25)
        ax = fig.add_subplot(1, 1, 1)
    # -----------------------------
    # Define map config
    # -----------------------------
    if (dispBaseMap is True):
        if (basemap is not None):
            basemap.plot(ax=ax, color='white', edgecolor='#444', linewidth=0.18)
        if (vert is True): # small map
            if (xlim is None): 
                xlim=[-165.5, 178.5]
            if (ylim is None): 
                ylim=[-77.5, 65.5] #ylim=[-70.5, 65.5]
        else:
            if (xlim is None): 
                xlim=[-130.5, 178.5]
            if (ylim is None):
                ylim=[-87.5, 65.5]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    if(gpdf is not None):
        gpdf.plot(
            column=column, 
            cmap=cmap, 
            legend=False,
            legend_kwds={'label': label, 'orientation': "horizontal"},
            classification_kwds={'bins': bins_ud}, 
            scheme='user_defined', #scheme='fisher_jenks',
            ax=ax,
            edgecolor='#444',
            linewidth=0.2,
            alpha=marker_alphaMap,
            markersize=gpdf[column] * markersizeMap, # Vary size by column
            #marker='o', color='red', 
            zorder=10
        )
        #
    #ax.set_xlabel("Longitude (°)")
    #ax.set_ylabel("Latitude (°)")
    ax.xaxis.set_major_formatter(StrMethodFormatter(u"{x:.0f}°"))
    ax.yaxis.set_major_formatter(StrMethodFormatter(u"{x:.0f}°"))
    ax.tick_params(axis='both', which='both', labelsize=labelsizeCoords)
    
    return ax, hist, bin_edges, rangos, yield_palette, bins_ud

def top_letter_for_figure(ax, xoffset=0.01, yoffset=1.03, letter=''):
    ax.text(xoffset, yoffset, letter, transform=ax.transAxes, size=12, weight='bold')

def getDataforMap(df, column, xmlfile):
    data = df[column].reset_index(drop=True)
    #sldfile = os.path.join(DATASET_IWIN_PATH, 'DataforNCCpaper20251205/DataForFigures/LegendSLDs', xmlfile)
    sldfile = os.path.join('./data/legends', xmlfile)
    if not os.path.isfile(sldfile):
        sldfile = None
    return data, sldfile

# -----------------
# Add Map Legend
# -----------------
def addMapLegend(sfig, data, label='', vert=False, subfig_pos=None, x=None, y=None, w=None, h=None, 
                 sp=None, bxw=None, labelsize=4, 
                 hist=None, bin_edges=None, rangos=None, yield_palette=None, bins_ud=None,
                 colorbars='skyblue', wbar=0.8, 
                 fontsizetickslabels=3.5, fontsizetickslabels1=3.5, fontsizebarLabel=3,
                 secondary_xyaxis_tick_pad=2.5, secondary_xyaxis_location=-0.09, vlw=0.25,
                 widths=0.5, whis=1.5, markersizeBoxplotMean=3,  dispBoxPlot=True,
                 dispXYlabel=True, dispGridlines=False, dispStatslines=True, dispTopBarsCounts=True,
                 dispBoxplotSpines=True
                ):
    # [0.65, 0.65, 0.2, 0.2]) # Located at (0.65, 0.65) with 20% width/height
    if (subfig_pos is None):
        subfig_pos=[0.35, 0.19, 0.45, 0.08] # Default Hztal legend
    if (w is None):
        w = subfig_pos[2]
    if (h is None):
        h = subfig_pos[3]
    if (vert is True):
        if (x is None):
            x = subfig_pos[0] #0.11 
        if (y is None):
            y = subfig_pos[1] #0.18
        hx_pos = [x, y, w, h]
    else:
        if (x is None):
            x = subfig_pos[0] #0.35 
        if (y is None):
            y = subfig_pos[1] #0.22
        hx_pos = [x, y, w, h]
    # boxplot
    if (sp is None):
        sp=0.015
    if (bxw is None):
        bxw=0.025
    if (vert is True):
        x = x + w + sp
        bx_pos = [x, y, bxw, h]
    else:
        y = y + h + sp
        bx_pos = [x, y, w, bxw]
    hx = sfig.add_axes(hx_pos)
    bx = sfig.add_axes(bx_pos)
    bx.set_facecolor('white')
    hx.set_facecolor('white')
    if (vert is True):
        bx.sharey(hx)
    else:
        bx.sharex(hx)
    #hx.set_title('Legend')
    for ax in [hx, bx]:
        ax.tick_params(reset=True)
        ax.tick_params(axis='both', which='minor', length=0.8, width=0.35, color='gray', labelsize=labelsize, pad=0)
        ax.tick_params(axis='both', which='major', labelsize=labelsize, pad=0)
        ax.tick_params(axis='both', which='major', color='black', length=0.8, width=0.35, pad=1,
                       #right=False, labelright=False,  labelleft=True, left=True,
                       #top=False, labeltop=False, bottom=True, labelbottom=True,
                       ) 
        ax.grid(visible=True, which='major', axis='both', color='#d3d3d3', linestyle='-', linewidth=0.25)
        ax.grid(visible=True, which='minor', axis='both', color='#d3d3d3', linestyle='--', linewidth=0.25, alpha=0.5)
        # Iterate through all spines and apply modifications
        for spine in ax.spines.values():
            spine.set_edgecolor('#000000') 
            spine.set_facecolor('#fff')
            spine.set_linewidth(0.15) 
            #spine.set_visible(False)
    #
    # ------------------------------
    # Pre-process data
    # ------------------------------
    #hist, bin_edges, rangos, yield_palette, bins_ud = getParametersForHisto(data, sldfile, bins, cmap, precision )
    #if (cmap is None):
    #    cmap = mcolors.ListedColormap('CustomYieldCmap',yield_palette)
    if (vert is True):
        # Hide bars. Only used to display labels in continuous numeric format
        hx.hist(data, bins=bin_edges, color='white', edgecolor='white', align='mid', 
                lw=0.01, alpha=0.0, orientation='horizontal')
    else:
        hx.hist(data, bins=bin_edges, color='white', edgecolor='white', align='mid', 
                lw=0.01, alpha=0.0, orientation='vertical')
    #
    # Get histogram data (counts and bins) using np.histogram or plt.hist
    #    using density=True normalizes the histogram area to 1 (probability density)
    counts, bins = np.histogram(data, bins=bin_edges) #, density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:]) # Calculate center of each bin
    # Plot the histogram using
    if (vert is True):
        bars = hx.barh(bin_centers, counts, height=(bins[1]-bins[0]) * wbar, color=colorbars, 
                      edgecolor='gray', alpha=0.95, lw=0.15) #label='Observed Data',
        # First Y-axis
        hx.set_yticks(bin_centers, labels=bin_centers.round(1), fontsize=fontsizetickslabels1, color='gray')
        hx.set_yticks(bin_edges, labels=bin_edges.round(1), fontsize=fontsizetickslabels1, color='gray')
        hx.tick_params(reset=True)
        hx.tick_params(axis='y', which='both', color='black', pad=1,
                       length=2, width=0.3, right=False, labelright=False, direction='inout',
                       labelsize=fontsizetickslabels1, labelcolor='black', #labelrotation=90,
                      ) 
        # Secondary Y-axis
        sec_ax2 = hx.secondary_yaxis(location=secondary_xyaxis_location)
        sec_ax2.tick_params(reset=True)
        sec_ax2.tick_params(axis='y', which='both', color='black', pad=secondary_xyaxis_tick_pad, 
                       length=1.5, width=0.3, right=False, labelright=False, direction='in',
                       labelsize=fontsizetickslabels, labelcolor='#444', #labelrotation=90,
                      ) 
        sec_ax2.set_yticks(bin_centers, labels=rangos, fontsize=fontsizetickslabels, color='black', ha='right', va='center')
        sec_ax2.set_yticklabels(rangos, fontsize=fontsizetickslabels, color='black', ha='right', va='center')
        sec_ax2.spines['left'].set_visible(True)
        sec_ax2.spines['left'].set_edgecolor('#000000') 
        sec_ax2.spines['left'].set_linewidth(0.2) 
        hx.tick_params(axis='x', which='both', color='black', pad=1,
                       length=2, width=0.3, bottom=False, labelbottom=False, 
                       right=False, labelright=False, top=False, labeltop=False,
                      ) 
        hx.set_xticklabels(hx.get_xticks(), fontsize=fontsizetickslabels, color='black') #, va='top')
    else:
        bars = hx.bar(bin_centers, counts, width=(bins[1]-bins[0]) * wbar, color=colorbars, 
                      edgecolor='gray', alpha=0.95, lw=0.15) #label='Observed Data',
        # First X-axis
        hx.set_xticks(bin_centers, labels=bin_centers.round(1), fontsize=fontsizetickslabels1, color='gray')
        hx.set_xticks(bin_edges, labels=bin_edges.round(1), fontsize=fontsizetickslabels1, color='gray')
        hx.tick_params(reset=True)
        hx.tick_params(axis='x', which='both', color='black', pad=1,
                       length=1, width=0.23, top=False, labeltop=False, direction='inout',
                       labelsize=fontsizetickslabels1, labelcolor='#444', #labelrotation=90,
                      ) 
        # Secondary X-axis
        sec_ax2 = hx.secondary_xaxis(location=secondary_xyaxis_location)
        sec_ax2.tick_params(reset=True)
        sec_ax2.tick_params(axis='x', which='both', color='black', pad=secondary_xyaxis_tick_pad, 
                       length=1, width=0.23, top=False, labeltop=False, direction='in',
                       labelsize=fontsizetickslabels, labelcolor='#444', labelrotation=90,
                      ) 
        sec_ax2.set_xticks(bin_centers, labels=rangos, fontsize=fontsizetickslabels, color='black', ha='center', va='top')
        sec_ax2.set_xticklabels(rangos, rotation=90, fontsize=fontsizetickslabels, color='black', ha='center', va='top')
        sec_ax2.spines['bottom'].set_visible(True)
        sec_ax2.spines['bottom'].set_edgecolor('#000000') 
        sec_ax2.spines['bottom'].set_linewidth(0.2) 
        hx.tick_params(axis='y', which='both', color='black', pad=1,
                       length=1, width=0.23, left=False, labelleft=False, 
                       right=False, labelright=False, top=False, labeltop=False,
                      ) 
        hx.set_yticklabels(hx.get_yticks(), fontsize=fontsizetickslabels, color='black') 
    
    # Iterate through all spines and apply modifications
    for spine in hx.spines.values():
        #spine.set_edgecolor('#d3d3d3')      # Change the color of the spine
        spine.set_edgecolor('#000000') 
        spine.set_facecolor('#fff')
        spine.set_linewidth(0.2) 
        spine.set_visible(False)
    
    if (vert is True):
        ## Disp hist vert
        hx.spines['left'].set_visible(True)
    else:
        ## Disp hist hztal
        hx.spines['bottom'].set_visible(True)

    # Si labels
    if (dispXYlabel is True):
        if (vert is True):
            #hx.set_xlabel('Grain Yield', fontsize=fontsizetickslabels+1, fontweight='bold', labelpad=1) 
            sec_ax2.set_ylabel(label, fontsize=fontsizetickslabels+1, fontweight='bold', labelpad=1) 
        else:
            sec_ax2.set_xlabel(label, fontsize=fontsizetickslabels+1, fontweight='bold', labelpad=1) 

    if (dispGridlines is True):
        hx.grid(visible=True, which='major', axis='both', color='#d3d3d3', linestyle='-', linewidth=0.25)
        hx.grid(visible=True, which='minor', axis='both', color='#d3d3d3', linestyle='--', linewidth=0.25, alpha=0.5)
    else:
        hx.grid(visible=True, which='major', axis='both', color='white', linestyle='-', linewidth=0.25)
        hx.grid(visible=True, which='minor', axis='both', color='white', linestyle='--', linewidth=0.25, alpha=0.5)
        # 
    if (dispStatslines is True):
        # A more robust method to get the exact stats used:
        stats = matplotlib.cbook.boxplot_stats(data, whis=1.5)
        if (vert is True):
            hx.axhline(data.mean(), color='purple', ls='--', lw=vlw)
            hx.axhline(data.mean() - data.std(), color='gray', ls='--', lw=vlw)
            hx.axhline(data.mean() + data.std(), color='gray', ls='--', lw=vlw)
            hx.axhline(stats[0]['q1'], color='blue', ls='--', lw=vlw)
            hx.axhline(stats[0]['med'], color='red', ls='--', lw=vlw)
            hx.axhline(stats[0]['q3'], color='blue', ls='--', lw=vlw)
        else:
            hx.axvline(data.mean(), color='purple', ls='--', lw=vlw)
            hx.axvline(data.mean() - data.std(), color='gray', ls='--', lw=vlw)
            hx.axvline(data.mean() + data.std(), color='gray', ls='--', lw=vlw)
            hx.axvline(stats[0]['q1'], color='blue', ls='--', lw=vlw)
            hx.axvline(stats[0]['med'], color='red', ls='--', lw=vlw)
            hx.axvline(stats[0]['q3'], color='blue', ls='--', lw=vlw)
    
    # -----------------------
    # Add labels on top of the bars
    if (dispTopBarsCounts is True):
        hx.bar_label(bars, fmt='{:,.0f}', padding=0.7, fontsize=fontsizebarLabel, color='#444')
    # Colors
    if (yield_palette is not None):
        for i, clr in enumerate(yield_palette):
            hx.patches[i].set_facecolor(clr)
            bars.patches[i].set_facecolor(clr)
    # ---------------------- 
    # Boxplot
    # ----------------------
    if (dispBoxPlot is True and bx is not None):
        bxplt = bx.boxplot(x=data, widths=widths, whis=whis, #orientation='horizontal', showbox=False,
                      vert=vert, notch=False, showfliers=True, showmeans=True, 
                      boxprops={"color": "gray", "linewidth": 0.25},
                      whiskerprops={"color": "gray", "linewidth": 0.25},
                      flierprops={"color": "red", "markeredgewidth": 0.15, "marker":'o', "markersize":markersizeBoxplotMean},
                      capprops={"color": "gray", "linewidth": 0.25},
                      medianprops={ "color": "r", "linewidth": 0.25},
                      meanprops={"marker":"D", "markerfacecolor":"none", "markeredgewidth": 0.15,
                                 "markeredgecolor":"black", "markersize":markersizeBoxplotMean},
                     ) 
        if (vert is True):
            bx.tick_params(axis='both', which='both', color='black', direction='in', pad=0, 
                           length=1.5, width=0.3, top=False, labeltop=False, 
                           bottom=False, labelbottom=False, 
                           right=False, labelright=True, left=False, labelleft=False,
                           labelsize=fontsizetickslabels, labelcolor='gray', labelrotation=0,
                          ) 
        else:
            bx.tick_params(axis='both', which='both', color='black', direction='in', pad=0, 
                           length=1.5, width=0.3, top=False, labeltop=True, 
                           bottom=False, labelbottom=False, left=False, labelleft=False,
                           labelsize=fontsizetickslabels, labelcolor='gray', labelrotation=0,
                          ) 
            
        if (dispBoxplotSpines is True):
            for spine in bx.spines.values():
                spine.set_edgecolor('#000000') 
                spine.set_facecolor('#fff')
                spine.set_linewidth(0.2) 
                spine.set_visible(False)
            # hide
            bx.grid(visible=True, which='major', axis='both', color='white', linestyle='-', linewidth=0.25)
            bx.grid(visible=True, which='minor', axis='both', color='white', linestyle='--', linewidth=0.25, alpha=0.5)
        else:
            bx.axis('off')
    
    return hx, bx


#for sf in subfigs.flat:
def setup_legend(sf, data, label='', vert=False, subfig_pos=None, hist=None, bin_edges=None, 
                 rangos=None, yield_palette=None, bins_ud=None,
                 colorbars='skyblue', wbar=0.8, 
                 fontsizetickslabels=3.5, fontsizetickslabels1=3.5, fontsizebarLabel=3,
                 secondary_xyaxis_tick_pad=2.5, secondary_xyaxis_location=-0.09, vlw=0.5,
                 widths=0.5, whis=1.5, markersizeBoxplotMean=3,  dispBoxPlot=True,
                 dispXYlabel=True, dispGridlines=False, dispStatslines=True, dispTopBarsCounts=True,
                 dispBoxplotSpines=True
                ):
    sf.set_facecolor('white')
    #sf.set_edgecolor('#444')
    #sf.set_linewidth(0.15)
    if (vert is False):
        if (dispXYlabel is True):
            if (subfig_pos is None):
                subfig_pos = [0.35, 0.23, 0.5, 0.1]
        else:
            if (subfig_pos is None):
                subfig_pos = [0.35, 0.21, 0.5, 0.105]
        hx, bx = addMapLegend(sf, data, label, vert,  subfig_pos, #x=0.35, y=0.19, w=0.45, h=0.08, sp=0.015, bxw=0.025,  
                              labelsize=4,
                              hist=hist, bin_edges=bin_edges, rangos=rangos, 
                              yield_palette=yield_palette, bins_ud=bins_ud,
                              colorbars=colorbars, wbar=wbar, 
                             fontsizetickslabels=fontsizetickslabels, fontsizetickslabels1=fontsizetickslabels1, 
                              fontsizebarLabel=fontsizebarLabel,
                             secondary_xyaxis_tick_pad=secondary_xyaxis_tick_pad, 
                              secondary_xyaxis_location=secondary_xyaxis_location, vlw=vlw,
                             widths=widths, whis=whis, markersizeBoxplotMean=markersizeBoxplotMean,  dispBoxPlot=dispBoxPlot,
                             dispXYlabel=dispXYlabel, dispGridlines=dispGridlines, dispStatslines=dispStatslines, 
                              dispTopBarsCounts=dispTopBarsCounts, dispBoxplotSpines=dispBoxplotSpines
                             )
    else:
        if (dispXYlabel is True):
            if (subfig_pos is None):
                subfig_pos = [0.12, 0.10, 0.025, 0.52]
        else:
            if (subfig_pos is None):
                subfig_pos = [0.09, 0.10, 0.025, 0.52]
        hx, bx = addMapLegend(sf, data, label, vert, subfig_pos, #x=0.05, y=0.15, w=0.35, h=0.13, sp=0.025, bxw=0.025, 
                              labelsize=4,
                              hist=hist, bin_edges=bin_edges, rangos=rangos, 
                              yield_palette=yield_palette, bins_ud=bins_ud,
                              colorbars=colorbars, wbar=wbar, 
                             fontsizetickslabels=fontsizetickslabels, fontsizetickslabels1=fontsizetickslabels1, 
                              fontsizebarLabel=fontsizebarLabel,
                             secondary_xyaxis_tick_pad=secondary_xyaxis_tick_pad, 
                              secondary_xyaxis_location=secondary_xyaxis_location, vlw=vlw,
                             widths=widths, whis=whis, markersizeBoxplotMean=markersizeBoxplotMean,  dispBoxPlot=dispBoxPlot,
                             dispXYlabel=dispXYlabel, dispGridlines=dispGridlines, dispStatslines=dispStatslines, 
                              dispTopBarsCounts=dispTopBarsCounts, dispBoxplotSpines=dispBoxplotSpines
                             )

    return hx, bx

# ======================================
# 
def plotFig1Panels(gpdf, vert=False, width_mm=185, height_mm=105,fontsizeSubTitle=2,
                   colorbars='skyblue', fontsizetickslabels=4, fontsizetickslabels1=2.5, 
                   fontsizebarLabel=3.1, secondary_xyaxis_tick_pad=1.2, secondary_xyaxis_location=-0.3,
                   vlw=0.15, widths=0.35, whis=1.5,  markersizeBoxplotMean=1.8, 
                   dispBoxPlot=True, dispXYlabel=False, dispGridlines=False, dispStatslines=True,
                   dispTopBarsCounts=True, dispBoxplotSpines=False,
                   cmp = 'v1',showFig=True, saveFig=False, fname = 'Fig_1_Map_Panels', fmt='jpg', figures_path='./'
                  ):
    #
    inches_per_mm = 1 / 25.4
    fig_width_inches = width_mm * inches_per_mm
    fig_height_inches = height_mm * inches_per_mm
    basemap = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    #
    fig = plt.figure(figsize=(fig_width_inches, fig_height_inches),
                     facecolor='white', linewidth=0.25, edgecolor='black', layout="constrained"
                    ) 
    # Create a top-level figure and split it into two subfigures
    subfigs = fig.subfigures(2, 2, wspace=0.01, hspace=0.01)
    # Access the left and right subfigures
    subfig_topleft = subfigs[0,0]
    subfig_topright = subfigs[0,1]
    subfig_bottomleft = subfigs[1,0]
    subfig_bottomright = subfigs[1,1]
    # Add subplots
    ax_TL = subfig_topleft.subplots(1, 1)
    ax_TR = subfig_topright.subplots(1, 1)
    ax_BL = subfig_bottomleft.subplots(1, 1)
    ax_BR = subfig_bottomright.subplots(1, 1)
    # Annotate plotots with letters
    top_letter_for_figure(ax_TL, letter='a')
    top_letter_for_figure(ax_TR, letter='b')
    top_letter_for_figure(ax_BL, letter='c')
    top_letter_for_figure(ax_BR, letter='d')
    #
    for ax in [ax_TL, ax_TR, ax_BL, ax_BR]:
        ax.tick_params(reset=True)
        ax.tick_params(axis='both', which='minor', length=1.5, width=1, color='gray', labelsize=6, pad=0)
        ax.tick_params(axis='both', which='major', labelsize=6, pad=0)
        ax.tick_params(axis='both', which='major', color='black', length=2, width=0.35, pad=1,
                       right=False, labelright=False,  labelleft=True, left=True,
                       ) 
        ax.grid(visible=True, which='major', axis='both', color='#d3d3d3', linestyle='-', linewidth=0.25)
        ax.grid(visible=True, which='minor', axis='both', color='#d3d3d3', linestyle='--', 
                linewidth=0.25, alpha=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor('#000000') 
            spine.set_facecolor('#fff')
            spine.set_linewidth(0.2) 
    for ax in [ax_TL, ax_TR]:
        ax.tick_params(axis='x', which='major', bottom=False, labelbottom=False,
                       top=False, labeltop=False,
                      )
    for ax in [ax_BL, ax_BR]:
        ax.tick_params(axis='x', which='major', top=False, labeltop=False,
                      )
    for ax in [ax_TR, ax_BR]:
        ax.tick_params(axis='y', which='major', bottom=True, labelleft=False, left=False,
                       top=False, labeltop=False
                      )
    #
    # ------------------------
    # Link to Setup legend
    # ------------------------
    def _setup_leg(sf, data, label='', vert=False, hist=None, bin_edges=None, 
                 rangos=None, yield_palette=None, bins_ud=None,
                 subfig_pos=None, bins=10, wbar=2, cmap=None, #'Spectral_r',
                precision=1, fontsizetickslabels=3.5, fontsizetickslabels1=2.5,
                fontsizebarLabel=3, labelsizeCoords=6,  colorbars='skyblue', secondary_xyaxis_tick_pad=2.5, 
                secondary_xyaxis_location=-0.28, vlw=0.5, width_mm=width_mm, height_mm=height_mm, 
                sp=0.008, bxw=0.025, widths=0.35, whis=1.5, markersizeBoxplotMean=3,  dispBoxPlot=True,
                dispHisto=True, dispBaseMap=True, dispMainMap=True,
                dispXYlabel=True, dispGridlines=False, dispStatslines=True, dispTopBarsCounts=True,
                dispBoxplotSpines=False
                ):
        hx, bx = setup_legend(sf, data, label, vert, hist=hist, bin_edges=bin_edges, rangos=rangos, 
                      yield_palette=yield_palette, bins_ud=bins_ud, colorbars=colorbars, wbar=wbar, 
                      fontsizetickslabels=fontsizetickslabels, fontsizetickslabels1=fontsizetickslabels1, 
                      fontsizebarLabel=fontsizebarLabel,
                      secondary_xyaxis_tick_pad=secondary_xyaxis_tick_pad, 
                      secondary_xyaxis_location=secondary_xyaxis_location, vlw=vlw,
                      widths=widths, whis=whis, markersizeBoxplotMean=markersizeBoxplotMean, 
                      dispBoxPlot=dispBoxPlot, dispXYlabel=dispXYlabel, dispGridlines=dispGridlines, 
                      dispStatslines=dispStatslines, dispTopBarsCounts=dispTopBarsCounts, 
                      dispBoxplotSpines=dispBoxplotSpines
                     )
    # ------------------------
    # Map 1 - Grain Yield
    # ------------------------
    #ax_TL = subfig_topleft.subplots(1, 1) #, sharey=True)
    column='BLUE_YLD_t_ha'
    data, sldfile = getDataforMap(gpdf, column=column, xmlfile=f'GrainYield_{cmp}.sld')
    label="Grain Yield (t ha$^{-1}$)"
    #subfig_topleft.suptitle(label, fontsize=fontsizeSubTitle)
    ax_TL, hist, bin_edges, rangos, yield_palette, bins_ud = subplotMap(data, gpdf, ax=ax_TL, 
                                                                        column=column, label=label, 
                       basemap=basemap, sldfile=sldfile, markersizeMap=1.1, labelsizeCoords=4,
                      ) #, cmap='RdBu_r')
    _setup_leg(subfig_topleft, data, label, vert, hist=hist, bin_edges=bin_edges, rangos=rangos, 
               yield_palette=yield_palette, bins_ud=bins_ud, wbar=0.8,
               fontsizetickslabels=fontsizetickslabels, fontsizetickslabels1=fontsizetickslabels1, 
                      fontsizebarLabel=fontsizebarLabel,
                      secondary_xyaxis_tick_pad=secondary_xyaxis_tick_pad, 
                      secondary_xyaxis_location=secondary_xyaxis_location, vlw=vlw,
                      widths=widths, whis=whis, markersizeBoxplotMean=markersizeBoxplotMean, 
               dispBoxPlot=dispBoxPlot, dispXYlabel=dispXYlabel, dispGridlines=dispGridlines, 
               dispStatslines=dispStatslines, dispTopBarsCounts=dispTopBarsCounts, 
               dispBoxplotSpines=dispBoxplotSpines
              )
    
    # ------------------------
    # Map 2 - Avg. Solar Radiation (MJ m$^{-2}$ day$^{-1}$)
    # ------------------------
    #ax_TR = subfig_topright.subplots(1, 1)
    column='avg_SolRad_GrainFill'
    data, sldfile = getDataforMap(gpdf, column=column, xmlfile=f'SolRad_{cmp}.sld')
    label='Avg. Solar Radiation (MJ m$^{-2}$ day$^{-1}$)'
    #subfig_topright.suptitle(label, fontsize=fontsizeSubTitle)
    ax_TR, hist, bin_edges, rangos, yield_palette, bins_ud = subplotMap(data, gpdf, ax=ax_TR, 
                                                                        column=column, label=label, 
                       basemap=basemap, sldfile=sldfile, markersizeMap=0.25, labelsizeCoords=4
                      )
    if (cmp=='v1'):
        wbar=0.8
    else:
        wbar=1.8
    _setup_leg(subfig_topright, data, label, vert,  hist=hist, bin_edges=bin_edges, rangos=rangos, 
               yield_palette=yield_palette, bins_ud=bins_ud, wbar=wbar,
               fontsizetickslabels=fontsizetickslabels, fontsizetickslabels1=fontsizetickslabels1, 
                      fontsizebarLabel=fontsizebarLabel,
                      secondary_xyaxis_tick_pad=secondary_xyaxis_tick_pad, 
                      secondary_xyaxis_location=secondary_xyaxis_location, vlw=vlw,
                      widths=widths, whis=whis, markersizeBoxplotMean=markersizeBoxplotMean, 
               dispBoxPlot=dispBoxPlot, dispXYlabel=dispXYlabel, dispGridlines=dispGridlines, 
               dispStatslines=dispStatslines, dispTopBarsCounts=dispTopBarsCounts, 
               dispBoxplotSpines=dispBoxplotSpines
              )
    #
    # ------------------------
    # Map 3 - Avg. Maximum Temperature (ºC)
    # ------------------------
    #ax_BL = subfig_bottomleft.subplots(1, 1)
    column='avg_TMax_GrainFill'
    data, sldfile = getDataforMap(gpdf, column=column, xmlfile=f'TMax_{cmp}.sld')
    label='Avg. Maximum Temperature (ºC)'
    #subfig_bottomleft.suptitle(label, fontsize=fontsizeSubTitle)
    ax_BL, hist, bin_edges, rangos, yield_palette, bins_ud = subplotMap(data, gpdf=gpdf, ax=ax_BL, 
                                                                        column=column, label=label, 
                       basemap=basemap, sldfile=sldfile, markersizeMap=0.25, labelsizeCoords=4
                      )
    _setup_leg(subfig_bottomleft, data, label, vert, hist=hist, bin_edges=bin_edges, rangos=rangos, 
               yield_palette=yield_palette, bins_ud=bins_ud, wbar=1.8, 
               fontsizetickslabels=fontsizetickslabels, fontsizetickslabels1=fontsizetickslabels1, 
                      fontsizebarLabel=fontsizebarLabel,
                      secondary_xyaxis_tick_pad=secondary_xyaxis_tick_pad, 
                      secondary_xyaxis_location=secondary_xyaxis_location, vlw=vlw,
                      widths=widths, whis=whis, markersizeBoxplotMean=markersizeBoxplotMean, 
               dispBoxPlot=dispBoxPlot, dispXYlabel=dispXYlabel, dispGridlines=dispGridlines, 
               dispStatslines=dispStatslines, dispTopBarsCounts=dispTopBarsCounts, 
               dispBoxplotSpines=dispBoxplotSpines
              )
    #
    # ------------------------
    # Map 4 - Avg. Minimum Temperature (ºC)
    # ------------------------
    #ax_BR = subfig_bottomright.subplots(1, 1)
    column='avg_TMin_GrainFill'
    data, sldfile = getDataforMap(gpdf, column=column, xmlfile=f'TMin_{cmp}.sld')
    label='Avg. Minimum Temperature (ºC)'
    #subfig_bottomright.suptitle(label, fontsize=fontsizeSubTitle)
    ax_BR, hist, bin_edges, rangos, yield_palette, bins_ud = subplotMap(data, gpdf, ax=ax_BR, 
                                                                        column=column, label=label, 
                       basemap=basemap, sldfile=sldfile, markersizeMap=0.28, labelsizeCoords=4
                      )
    _setup_leg(subfig_bottomright, data, label, vert, hist=hist, bin_edges=bin_edges, rangos=rangos, 
               yield_palette=yield_palette, bins_ud=bins_ud, wbar=1.8,
               fontsizetickslabels=fontsizetickslabels, fontsizetickslabels1=fontsizetickslabels1, 
                      fontsizebarLabel=fontsizebarLabel,
                      secondary_xyaxis_tick_pad=secondary_xyaxis_tick_pad, 
                      secondary_xyaxis_location=secondary_xyaxis_location, vlw=vlw,
                      widths=widths, whis=whis, markersizeBoxplotMean=markersizeBoxplotMean, 
               dispBoxPlot=dispBoxPlot, dispXYlabel=dispXYlabel, dispGridlines=dispGridlines, 
               dispStatslines=dispStatslines, dispTopBarsCounts=dispTopBarsCounts, 
               dispBoxplotSpines=dispBoxplotSpines
              )
    #
    #
    plt.subplots_adjust( left=0.03, right=0.97, bottom=0.03, top=0.97, wspace=0.01, hspace=0.01 )
    fig.tight_layout(pad=0.1) #pad=0.01)
    if (saveFig is True):
        hoy = dt.datetime.now().strftime('%Y%m%d')
        figures_path = os.path.join(figures_path, '{}_{}'.format('Figures', hoy))
        if not os.path.isdir(figures_path):
            os.makedirs(figures_path)
        if (saveFig is True and fmt=='pdf'):
            fig.savefig(os.path.join(figures_path,"{}_{}.{}".format(fname, hoy, fmt)), 
                        bbox_inches='tight', orientation='portrait', papertype='a4', 
                        edgecolor='none', transparent=False, pad_inches=0.5, dpi=dpi_value)
        
        if (saveFig==True and (fmt=='jpg' or fmt=='png')):
            fig.savefig(os.path.join(figures_path,"{}_{}.{}".format(fname, hoy, fmt)), 
                        bbox_inches='tight',  facecolor=fig.get_facecolor(), edgecolor='none', transparent=False, dpi=dpi_value)
    if (showFig is True):
        if (saveFig is False):
            plt.tight_layout()
        fig.show()
    else:
        del fig
        plt.close();







