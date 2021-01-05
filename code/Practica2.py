#!/usr/bin/env python
# coding: utf-8

# # Pràctica 2: Neteja i anàlisis de les dades

# In[1]:


import numpy as np
import pandas as pd
import warnings
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# # Descripció del dataset
# 
# El dataset utilitzat per realitzar aquesta practica tracta sobre el canvi climàtic en les temperatures de l'aire a la superficie de la Terra, es pot trobar a partir del següent enllaç: [**climate-change-earth-surface-temperature-data**](https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data).
# Aquest dataset d'ús public a traves de la plataforma Kaggle, consta de la licencia *CC BY-NC-SA 4.0*. 
# 
# És tracta d'un dataset que conté registres de dades des de l'any 1750 fins al 2015 sobre la temperatura de l'aire a la superficie mesurada en diferents punts de la Terra.
# 
# En aquesta pràctica es vol plantejar l'estudi de l'evolució de la temperatura en la superficíe terrestre, per compendre si es cert que hi ha hagut un augment de les temperatures en els ultims anys, i consequentment confirmar que el canvi climatic referent a la temperatura terrestre es real. 
# 
# El dataset constà de 4 archius de dades en format *csv*:
# 
# 
# - GlobalTemperatures.csv
# - GlobalLandTemperaturesByCountry.csv
# - GlobalLandTemperaturesByState.csv
# - GlobalLandTemperaturesByMajorCity.csv
# - GlobalLandTemperaturesByCity.csv
# 
# Per al cas d'estudi plantejat en aquesta pràctica, utilitzarem les dades dels fitxers *GlobalTemperatures*, *GlobalLandTemperaturesByCountry* i *GlobalLandTemperaturesByCity*.
# 
# A continuació es detalla la informació que contenen cadascún d'aquests datasets, però primer, es llegirant aquests archius per poder obtindre un millor resum.

# In[2]:


global_temp=pd.read_csv('../data/GlobalTemperatures.csv')
countries_temp=pd.read_csv('../data/GlobalLandTemperaturesByCountry.csv')
cities_temp=pd.read_csv('../data/GlobalLandTemperaturesByCity.csv')


# ## GlobalTemperatures
# 
# Dataset info:

# In[3]:


global_temp.info()


# ### Variables
# 
# El dataset *GlobalTemperatures* conté 3192 registres i 9 columnes (no totes les columnes contenen informació en tots els registres i per tant més endavant s'hauràn de tractar aquests valors nuls), les quals es corresponen a cadascuna de les següents variables:
# 
# - **Date**: data del registre, començant des de l'any 1750 on es registraba la temperatura mitjana en la terra, i a partira del 1850, es registraba també els maxims i minims de les temperatures a la superficie terrestre i la dels oceans.
# 
# - **LandAverageTemperature**: promig global de la temperatura a la terra en graus celsius.
# 
# - **LandAverageTemperatureUncertainty**:  valor del 95% de l'interval de confiança sobre la variable de la mitjana.
# 
# - **LandMaxTemperature**:  promig global de la temperatura maxima en la terra en graus celsius.
# 
# - **LandMaxTemperatureUncertainty**: valor del 95% de l'interval de confiança sobre la variable de la mitjana de la temperatura máxima.
# 
# - **LandMinTemperature**: promig globla de la temperatura minima en la terra en graus celsius.
# 
# - **LandMinTemperatureUncertainty**: valor del 95% de l'interval de confiança sobre la variable de la mitjana de la temperatura minima.
# 
# - **LandAndOceanAverageTemperature**: promig global de la temperatura als oceans i a la terra en celsius.
# 
# - **LandAndOceanAverageTemperatureUncertainty**: valor del 95% de l'interval de confiança sobre la variable de la mitjana de la temperatura als oceans i a la terra.

# ## GlobalLandTemperaturesByCountry
# 
# Dataset info:

# In[4]:


countries_temp.info()


# ### Variables
# 
# El dataset *GlobalLandTemperaturesByCountry* conté 577462 registres i 4 columnes, que es corresponen a les següents variables:
# 
# - **dt**: data en la qual es va mesura la informació. 
# - **AverageTemperature**: promig de la temperatura terrestre en celsius.
# - **AverateTemperatureUncertainty**: valor del 95% de l'interval de confiança de la mitjana.
# - **Country**: Pais on es va obtindre el valor de la temperatura.
# 

# ## GlobalLandTemperaturesByCity
# 
# Dataset info:

# In[5]:


cities_temp.info()


# ### Variables
# 
# El dataset *GlobalLantTemperaturesByCity* conté 8588212 registre i 7 columnes que es corresponen a les següents variables:
# 
# - **dt**: data en la qual es va mesura la informació. 
# - **AverageTemperature**: promig de la temperatura terrestre en celsius.
# - **AverateTemperatureUncertainty**: valor del 95% de l'interval de confiança de la mitjana.
# - **City**: Ciutat on es va realitzar la mesura de la temperatura registrada.
# - **Country**: Pais on pertany la ciutat on es va realitzar la mesura.
# - **Latitude**: Valor de la latitud  de la localització de la ciutat en graus
# - **Longitud**: Valor de la longitud de la localització de la ciutat en graus.

# # Integració i selecció de les dades d'interes

# Primerament, observarem les dades per aclarir quines dades ens poden ser d'interès i quines no per a l'estudi plantejat en la pràctica.

# In[6]:


global_temp.describe()


# In[7]:


global_temp.head()


# In[8]:


countries_temp.describe()


# In[9]:


cities_temp.describe()


# ### Dades d'interes
# 
# A partir de l'observació anterior, es pot determinar:
# 
# - Els tres datasets contenent dades que s'hauran de netejar previament a l'estudi a realitzar.
# - El dataset *GlobalTemperatures* conte la variable *LandAverageTemperature*, la qual es d'interes per l'estudi.
# - Els datasets *GlobalLandTemperaturesByCountry* i *GlobalLandTemperaturesByCity* contenen també informació d'interès per l'estudi en les variables *AverageTemperature*.

# # Neteja de dades
# 
# ## Les dades contenen zeros o elements buits? Com gestionaries aquests casos?

# Les dades contenen elements NaN (nulls). Aquests NaN en la seva majoria es corresponent a les dates entre 1750 i 1850 ja que com s'ha descrit abans en el dataset *GlobalTemperatures*, durant aquell periode nomes registrava la temperatura mitjana en terra i per altra banda, es comprensible que tractantse d'un registre de dades tant antic, hi haguin casos de dades perdudes. 
# 
# Tot i això, per al cas d'estudi no afecta ja que, com sabem, el canvi climàtic i l'augment de temperatures es un desastre humà que es va començar a esdevenir durant l'última meitat del segle XX, i per tant, el fet de no tindre alguns registres del segle XVIII, a priori, no a d'afectar. 
# 
# Aleshores, s'obtarà per eliminar els registres de dades amb valors nulls dels datasets. 

# In[10]:


global_temp.dropna(inplace=True)
countries_temp.dropna(inplace=True)
cities_temp.dropna(inplace=True)


# ## Identificació i tractament de valors extrems.

# A continuació eliminarem els valors **outliers** dels tres datasets càrregats:

# In[12]:


global_temp[(np.abs(stats.zscore(global_temp['LandAverageTemperature'])) < 3)]


# In[13]:


countries_temp[(np.abs(stats.zscore(countries_temp['AverageTemperature'])) < 3)]


# In[14]:


cities_temp[(np.abs(stats.zscore(cities_temp['AverageTemperature'])) < 3)]


# En el cas del dataset de temperatures per pais, aprofitem per corretgir alguns del noms utilitzats per a registra el pais:

# In[15]:


countries_temp['Country'].replace({'Denmark (Europe)':'Denmark','France (Europe)':'France','Netherlands (Europe)':'Netherlands','United Kingdom (Europe)':'United Kingdom'},inplace=True)
temp_country1=countries_temp.groupby(['Country'])['AverageTemperature'].mean().reset_index()


# # Anàlisis de les dades

# ## Selecció dels grups de dades que es volen analitzar.
# 
# Com s'ha descrit anteriorment, les dades d'interes que es volen analitzar són: 
# 
# - AverageTemperature del dataset GlobalLandTemperaturesByCity, es carregarà en la variable *cities_average_temp*.
# - AverageTemperature del dataset GlobalLandTemperaturesByCountry, es carregarà en la variable *country_average_temp*.
# - LandAverageTemperature del dataset GlobalTemperature, es carregarà en la variable *global_land_average*.

# In[17]:


country_average_temp=countries_temp.groupby(['dt'])['AverageTemperature'].mean().reset_index()
country_average_temp=countries_temp[['AverageTemperature']]
country_average_temp.describe()


# In[16]:


cities_temp['year']=cities_temp['dt'].apply(lambda x: x[:4])
cities_temp['month']=cities_temp['dt'].apply(lambda x: x[5:7])
cities_temp.drop('dt',axis=1,inplace=True)
cities_temp=cities_temp[['year','month','AverageTemperature','City','Country','Latitude','Longitude']]
cities_temp['Latitude']=cities_temp['Latitude'].str.strip('N')
cities_temp['Longitude']=cities_temp['Longitude'].str.strip('E')


# In[18]:


cities_average_temp=cities_temp.groupby(['year'])['AverageTemperature'].mean().reset_index()
cities_average_temp=cities_temp[['AverageTemperature']]
cities_average_temp.describe()


# In[11]:


global_temp['dt']=pd.to_datetime(global_temp.dt).dt.strftime('%d/%m/%Y')
global_temp['dt']=global_temp['dt'].apply(lambda x:x[6:])


# In[19]:


global_land_average=global_temp.groupby(['dt'])['LandAverageTemperature'].mean().reset_index()
global_land_average=global_temp[['LandAverageTemperature']]
global_land_average.describe()


# ## Comprovació de la normalitat i homogeneïtat de la variància.

# ### Shapiro
# A continuació es realitzarà el test de Shapiro per comprovar la normalitat de les variables a estudiar

# In[20]:


stats.shapiro(global_land_average)


# In[21]:


stats.shapiro(country_average_temp)


# In[22]:


stats.shapiro(cities_average_temp)


# ### QQplots

# In[40]:


ax1 = plt.subplot(221).set_title('Global Average Temperature')
res = stats.probplot(global_temp['LandAverageTemperature'], plot=plt)
ax2 = plt.subplot(222)
ax2.set_title('Countries Average Temperature')
res = stats.probplot(countries_temp['AverageTemperature'], plot=plt)
ax3 = plt.subplot(223)
ax3.set_title('Cities Average Temperature')
res = stats.probplot(cities_temp['AverageTemperature'], plot=plt)
plt.show()


# ## Aplicació de proves estadístiques per comparar els grups de dades. 
# En funció de les  dades  i  de  l’objectiu  de  l’estudi,  aplicar  proves  de  contrast  d’hipòtesis, correlacions, regressions, etc. Aplicar almenys tres mètodes d’anàlisi diferents.

# ### Regressió lineal de les dades globals

# In[23]:


glm_binom = sm.GLM(global_land_average.astype(float), global_temp.astype(float), family=sm.families.Binomial())
res = glm_binom.fit()
print(res.summary())


# ### Correlació de les dades globals
# 
# Aquesta correlació ens permetra observar les relacions entre les variables del dataset de dades globals.

# In[24]:


sns.heatmap(global_temp.corr())


# Observant el *heatmap* anterior es pot observar, com era d'esperar, que la variable *LandAverageTemperature*, te una forta correlació amb les variables *LandMinAverageTemperature*, *LandMaxAverageTemperature* i *LandAndOceanAverageTemperature*

# # Representació dels resultats a partir de taules i gràfiques.

# ## Evolució de la temperatura segons l'estació de l'any

# In[25]:


global_temp = pd.read_csv('../data/GlobalTemperatures.csv')
global_temp = global_temp[['dt', 'LandAverageTemperature']]

global_temp['dt'] = pd.to_datetime(global_temp['dt'])
global_temp['year'] = global_temp['dt'].map(lambda x: x.year)
global_temp['month'] = global_temp['dt'].map(lambda x: x.month)

def get_season(month):
    if month >= 3 and month <= 5:
        return 'spring'
    elif month >= 6 and month <= 8:
        return 'summer'
    elif month >= 9 and month <= 11:
        return 'autumn'
    else:
        return 'winter'
    
min_year = global_temp['year'].min()
max_year = global_temp['year'].max()
years = range(min_year, max_year + 1)

global_temp['season'] = global_temp['month'].apply(get_season)

spring_temps = []
summer_temps = []
autumn_temps = []
winter_temps = []

for year in years:
    curr_years_data = global_temp[global_temp['year'] == year]
    spring_temps.append(curr_years_data[curr_years_data['season'] == 'spring']['LandAverageTemperature'].mean())
    summer_temps.append(curr_years_data[curr_years_data['season'] == 'summer']['LandAverageTemperature'].mean())
    autumn_temps.append(curr_years_data[curr_years_data['season'] == 'autumn']['LandAverageTemperature'].mean())
    winter_temps.append(curr_years_data[curr_years_data['season'] == 'winter']['LandAverageTemperature'].mean())
sns.set(style="whitegrid")
sns.set_color_codes("pastel")
f, ax = plt.subplots(figsize=(10, 6))

plt.plot(years, summer_temps, label='Estiu', color='orange')
plt.plot(years, autumn_temps, label='Tardor', color='r')
plt.plot(years, spring_temps, label='Primavera', color='g')
plt.plot(years, winter_temps, label='Hivern', color='b')

plt.xlim(min_year, max_year)

ax.set_ylabel('Temperatura mitjana')
ax.set_xlabel('Any')
ax.set_title('Mitjana de la temperatura per estació')
legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, borderpad=1, borderaxespad=1)


# ### Top 10 Paisos mes càlids i més freds

# In[26]:


hot=temp_country1.sort_values(by='AverageTemperature',ascending=False)[:10]
cold=temp_country1.sort_values(by='AverageTemperature',ascending=True)[:10]
top_countries=pd.concat([hot,cold])
top_countries.sort_values('AverageTemperature',ascending=False,inplace=True)
f,ax=plt.subplots(figsize=(12,8))
sns.barplot(y='Country',x='AverageTemperature',data=top_countries,palette='cubehelix',ax=ax).set_title('Top Hottest And Coldest Countries')
plt.xlabel('Mean Temperture')
plt.ylabel('Country')


# ### Evolució de les temperatures en diferents països

# In[27]:


countries=countries_temp.copy()
countries['dt']=pd.to_datetime(countries.dt).dt.strftime('%d/%m/%Y')
countries['dt']=countries['dt'].apply(lambda x: x[6:])
countries=countries[countries['AverageTemperature']!=0]
countries.drop('AverageTemperatureUncertainty',axis=1,inplace=True)
li=['United States','France','Japan','Germany','United Kingdom', 'Spain', 'South Africa']
countries=countries[countries['Country'].isin(li)]
countries=countries.groupby(['Country','dt'])['AverageTemperature'].mean().reset_index()
countries=countries[countries['dt'].astype(int)>1850]
abc=countries.pivot('dt','Country','AverageTemperature')
f,ax=plt.subplots(figsize=(20,10))
abc.plot(ax=ax)


# ## Evolució de les temperatures a Espanya:

# ### Durant els ultims 50 anys

# In[28]:


spanish_cities=cities_temp[cities_temp['Country']=='Spain']
spanish_cities=spanish_cities[spanish_cities['year']>'1970']
major_cities=spanish_cities[spanish_cities['City'].isin(['Barcelona','Madrid','Sevilla','Malaga','Bilbao', 'Valencia'])]
graph=major_cities[major_cities['year']>'1970']
graph=graph.groupby(['City','year'])['AverageTemperature'].mean().reset_index()
graph=graph.pivot('year','City','AverageTemperature').fillna(0)
graph.plot()
fig=plt.gcf()
fig.set_size_inches(18,8)


# ### Durant els ultims 20 anys

# In[29]:


spanish_cities=cities_temp[cities_temp['Country']=='Spain']
spanish_cities=spanish_cities[spanish_cities['year']>'1995']
major_cities=spanish_cities[spanish_cities['City'].isin(['Barcelona','Madrid','Sevilla','Malaga','Bilbao', 'Valencia'])]
graph=major_cities[major_cities['year']>'1995']
graph=graph.groupby(['City','year'])['AverageTemperature'].mean().reset_index()
graph=graph.pivot('year','City','AverageTemperature').fillna(0)
graph.plot()
fig=plt.gcf()
fig.set_size_inches(18,8)


# # Resolució del problema. 
# 
# A partir dels resultats obtinguts, quines són les conclusions? Els resultats permeten respondre al problema?

# ## Conclusions
# 
# A partir de l'estudi realitzat sobre els diferents datasets que conformen el conjunt de dades sobre les temperatures a l'aire de la superficie de la terra des del 1750 fins al 2015, es pot concloure que en els ultims 50 anys, es troba una tendència **global** d'augment de les temperatures, en aproximadament 2-3 graus globalment, tenint en compte diferents factors que poden fer variar aquest augment, com l'estació de l'any, el pais o la ciutat on s'ha mesurat. 
# 
# Per tant els resultat obtinguts han permès respondre al problema plantejat al inici de la pràctica, tot i que no són uns resultats positius per la salut del planeta.
