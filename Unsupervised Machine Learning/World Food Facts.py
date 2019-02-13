
# coding: utf-8

# # World Food Facts

# * Original Data Source:  https://www.kaggle.com/openfoodfacts/world-food-facts
# * Modified Source:  https://www.kaggle.com/lwodarzek/nutrition-table-clustering/output

# ## Ingest

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
init_notebook_mode(connected = True)
import plotly.graph_objs as go
get_ipython().run_line_magic('matplotlib', 'inline')

warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import gensim
from gensim import corpora

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer


# In[2]:


df = pd.read_csv(
    "https://raw.githubusercontent.com/noahgift/food/master/data/features.en.openfoodfacts.org.products.csv")
df.drop(["Unnamed: 0", "exceeded", "g_sum", "energy_100g"], axis=1, inplace=True) #drop two rows we don't need
df = df.drop(df.index[[1,11877]]) #drop outlier
df.rename(index=str, columns={"reconstructed_energy": "energy_100g"}, inplace=True)
df.head()


# ## EDA

# In[3]:


df.columns


# ### Simple Histgram and Distribution of Single Variable

# In[4]:


warnings.filterwarnings("ignore")
sns.set(style='white', palette='muted', color_codes=True)

f, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True)
sns.despine(left=True)

sns.distplot(df.fat_100g, color='b', ax=axes[0, 0])
sns.distplot(df.carbohydrates_100g, color='g', ax=axes[0, 1])
sns.distplot(df.sugars_100g, color='r', ax=axes[1, 0])
sns.distplot(df.proteins_100g, color='m', ax=axes[1, 1])
plt.tight_layout()


# ### Wordcloud of Different Kinds of Food
# This part I will use wordcloud to dig out some features of specific kind of food. For example, for high protein food, what kinds of products may play an important part in it?

# #### High Fat Food

# In[5]:


high_fat_df = df[df.fat_100g > df.fat_100g.quantile(.98)]
high_fat_text = high_fat_df['product'].values

wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(high_fat_text))

fig = plt.figure(
    figsize = (10, 7),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)


# #### High Carbohydrates Food

# In[6]:


high_carbohydrate_df = df[df.carbohydrates_100g > df.carbohydrates_100g.quantile(.98)]
high_carbohydrate_text = high_carbohydrate_df['product'].values

wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(high_carbohydrate_text))

fig = plt.figure(
    figsize = (10, 7),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)


# #### High Sugar Food

# In[7]:


high_sugar_df = df[df.sugars_100g > df.sugars_100g.quantile(.98)]
high_sugar_text = high_sugar_df['product'].values

wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(high_sugar_text))

fig = plt.figure(
    figsize = (10, 7),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)


# #### High Protein Food

# In[8]:


high_protein_df = df[df.proteins_100g > df.proteins_100g.quantile(.98)]
high_protein_text = high_protein_df['product'].values

wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(high_protein_text))

fig = plt.figure(
    figsize = (10, 7),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)


# #### High Salt Food

# In[9]:


high_salt_df = df[df.salt_100g > df.salt_100g.quantile(.98)]
high_salt_text = high_salt_df['product'].values

wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(high_salt_text))

fig = plt.figure(
    figsize = (10, 7),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)


# #### High Energy Food

# In[10]:


high_energy_df = df[df.energy_100g > df.energy_100g.quantile(.98)]
high_energy_text = high_energy_df['product'].values

wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(high_energy_text))

fig = plt.figure(
    figsize = (10, 7),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)


# ## Clustering Models

# ### Gensim example

# Preprocess Text

# In[11]:


dataset = df['product'].fillna("").values
raw_text_data = [d.split() for d in dataset]


# In[12]:


stop = stopwords.words('english')


# Remove stop words
# 

# In[13]:


text_data = [item for item in raw_text_data if item not in stop]


# In[14]:


from gensim import corpora
dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]


# In[15]:


import gensim
NUM_TOPICS = 5
ldamodel = gensim.models.ldamodel.LdaModel(
    corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)

topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)


# ## Clustering

# ### Scale the data

# In[16]:


df_cluster_features = df.drop("product", axis=1)
scaler = MinMaxScaler()
scaler.fit(df_cluster_features)
scaler.transform(df_cluster_features)


# ### Cluster Diagnostics
# Find best number of clusters

# #### Yellowbrick Visualizer Elbow Method

# In[17]:


model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,11))

visualizer.fit(df_cluster_features)
visualizer.poof() 


# Three clusters seems better

# #### Yellowbrick Silhouette Visualizer
# 

# In[18]:


model = MiniBatchKMeans(3)
visualizer = SilhouetteVisualizer(model)

visualizer.fit(df_cluster_features) 
visualizer.poof()


# ### Add Cluster Labels

# In[19]:


k_means = KMeans(n_clusters=3)
kmeans = k_means.fit(scaler.transform(df_cluster_features))
df['cluster'] = kmeans.labels_
df.head()


# ### 3D ClusterPlot

# #### Fat-Carb-Sugar 3D Plot

# In[20]:


trace = go.Scatter3d(
    x=df['fat_100g'],
    y=df['carbohydrates_100g'],
    z=df['sugars_100g'],
    mode='markers',
    text=df['product'],
    marker=dict(
        size=12,
        color=df['cluster'],                
        colorscale='Viridis',
        opacity=0.8
    )
)

data = [trace]
layout = go.Layout(
    showlegend=False,
    title='Fat-Carb-Sugar:  Food Energy Types',
    scene = dict(
        xaxis = dict(title='X: Fat Content-100g'),
        yaxis = dict(title="Y:  Carbohydrate Content-100g"),
        zaxis = dict(title="Z:  Sugar Content-100g"),
    ),
    width=900,
    height=900,
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# #### Fat-Carb-Protein 3D Plot

# In[21]:


trace = go.Scatter3d(
    x=df['fat_100g'],
    y=df['carbohydrates_100g'],
    z=df['proteins_100g'],
    mode='markers',
    text=df['product'],
    marker=dict(
        size=12,
        color=df['cluster'],                
        colorscale='Viridis',
        opacity=0.8
    )
)

data = [trace]
layout = go.Layout(
    showlegend=False,
    title='Fat-Carb-Protein:  Food Energy Types',
    scene = dict(
        xaxis = dict(title='X: Fat Content-100g'),
        yaxis = dict(title="Y:  Carbohydrate Content-100g"),
        zaxis = dict(title="Z:  Protein Content-100g"),
    ),
    width=900,
    height=900,
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# #### Fat-Carb-Salt-3D Plot

# In[22]:


trace = go.Scatter3d(
    x=df['fat_100g'],
    y=df['carbohydrates_100g'],
    z=df['salt_100g'],
    mode='markers',
    text=df['product'],
    marker=dict(
        size=12,
        color=df['cluster'],                
        colorscale='Viridis',
        opacity=0.8
    )
)

data = [trace]
layout = go.Layout(
    showlegend=False,
    title='Fat-Carb-Salt:  Food Energy Types',
    scene = dict(
        xaxis = dict(title='X: Fat Content-100g'),
        yaxis = dict(title="Y:  Carbohydrate Content-100g"),
        zaxis = dict(title="Z:  Salt Content-100g"),
    ),
    width=900,
    height=900,
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# #### Fat-Carb-Energy-3D Plot

# In[23]:


trace = go.Scatter3d(
    x=df['fat_100g'],
    y=df['carbohydrates_100g'],
    z=df['energy_100g'],
    mode='markers',
    text=df['product'],
    marker=dict(
        size=12,
        color=df['cluster'],                
        colorscale='Viridis',
        opacity=0.8
    )
)

data = [trace]
layout = go.Layout(
    showlegend=False,
    title='Fat-Carb-Energy:  Food Energy Types',
    scene = dict(
        xaxis = dict(title='X: Fat Content-100g'),
        yaxis = dict(title="Y:  Carbohydrate Content-100g"),
        zaxis = dict(title="Z:  n=Energy Content-100g"),
    ),
    width=900,
    height=900,
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# #### Protein-Salt-Carb-3D Plot

# In[24]:


trace = go.Scatter3d(
    x=df['proteins_100g'],
    y=df['carbohydrates_100g'],
    z=df['salt_100g'],
    mode='markers',
    text=df['product'],
    marker=dict(
        size=12,
        color=df['cluster'],                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)

data = [trace]
layout = go.Layout(
    showlegend=False,
    title='Protein, Carb, Salt:  Food Energy Types',
    scene = dict(
        xaxis = dict(title='X: Protein Content-100g'),
        yaxis = dict(title='Y: Carbohydrate Content-100g'),
        zaxis = dict(title='Z: Salt Content-100g'),
    ),
    width=1000,
    height=900,
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# From the above 3D plot, we can find that actually these clusters are not so good that they can seperate different kinds of 
#food(0,1,2). All kinds of food are mixed with each other. Therefore, we need to find new clusters

# ### Create New Clusters

# In[25]:


df_new = df.copy()
df_new.columns


# In[26]:


df_new['calorie_100g'] = df['fat_100g'] + df['carbohydrates_100g'] + df['sugars_100g']
df_new_features = df_new.drop(['fat_100g', 'carbohydrates_100g', 'sugars_100g', 'cluster', 'product'], axis=1)
df_new_features.head()


# #### Cluster Diagnostics

# In[27]:


scaler.fit(df_new_features)
scaler.transform(df_new_features)

model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,11))

visualizer.fit(df_new_features)
visualizer.poof() 


# The better number of clusters is three

# #### Add Cluster Labels

# In[28]:


k_means = KMeans(n_clusters=3)
kmeans = k_means.fit(scaler.transform(df_new_features))
df_new_features['cluster'] = kmeans.labels_
df_new_features.head()


# #### Protein-Salt-Energy-3D Plot

# In[29]:


trace = go.Scatter3d(
    x=df_new_features['proteins_100g'],
    y=df_new_features['salt_100g'],
    z=df_new_features['energy_100g'],
    mode='markers',
    marker=dict(
        size=12,
        color=df_new_features['cluster'],                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)

data = [trace]
layout = go.Layout(
    showlegend=False,
    title='Protein, Salt, Energy:  Food Types',
    scene = dict(
        xaxis = dict(title='X: Protein Content-100g'),
        yaxis = dict(title='Y: Salt Content-100g'),
        zaxis = dict(title='Z: Energy Content-100g'),
    ),
    width=1000,
    height=900,
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# #### Protein-Salt-Calorie-3D Plot

# In[30]:


df_features_calorie = df_new_features.copy()
df_features_calorie = df_features_calorie.join(df['product'])
df_features_calorie['text'] = df_features_calorie['cluster'].astype(str) + ' ' + df_features_calorie['product']
df_features_calorie.head()


# In[31]:


trace = go.Scatter3d(
    x=df_features_calorie['proteins_100g'],
    y=df_features_calorie['salt_100g'],
    z=df_features_calorie['calorie_100g'],
    text=df_features_calorie['text'],
    mode='markers',
    marker=dict(
        size=12,
        color=df_features_calorie['cluster'],               
        colorscale='Viridis',   
        opacity=0.8
    )
)

data = [trace]
layout = go.Layout(
    showlegend=False,
    title='Protein, Salt, Calorie:  Food Types',
    scene = dict(
        xaxis = dict(title='X: Protein Content-100g'),
        yaxis = dict(title='Y: Salt Content-100g'),
        zaxis = dict(title='Z: Calorie Content-100g'),
    ),
    width=1000,
    height=900,
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# The last 3D plot shows that 'calorie_100g' is a very powerful cluster to seperate various kinds of food. We can easily call them high-calorie, average-calorie and low-calorie food.

# In[32]:


df_final = df_features_calorie.drop(['energy_100g'], axis=1)
df_final = df_final.replace({'cluster':{0:'high calorie', 1:'low calorie', 2:'average calorie'}})
df_final.head()


# ### PCA 
# In this part, I want to use principle component analysis (PCA) to check whether PCA can help us find better clustering variables. 

# #### Eigenvalues and Eigenvertors
# By checking eigenvalues, I want to decide how many components should we keep in the final clustering. Also, eigenvectors can help us determine meanings of different components.

# In[33]:


df_cluster_features.head()


# In[34]:


cf_std = StandardScaler().fit_transform(df_cluster_features)


# In[35]:


cov_matrix = np.cov(cf_std.T)
cov_matrix


# In[36]:


eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)


# In[37]:


total = sum(eig_vals)
var_exp = [(i / total)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

trace1 = dict(
    type='bar',
    x=['PC %s' %i for i in range(1,7)],
    y=var_exp,
    name='Individual'
)

trace2 = dict(
    type='scatter',
    x=['PC %s' %i for i in range(1,7)], 
    y=cum_var_exp,
    name='Cumulative'
)

data = [trace1, trace2]

layout=dict(
    title='Explained variance by different principal components',
    yaxis=dict(
        title='Explained variance in percent'
    ),
    annotations=list([
        dict(
            x=1.16,
            y=1.05,
            xref='paper',
            yref='paper',
            text='Explained Variance',
            showarrow=False,
        )
    ])
)

fig = dict(data=data, layout=layout)
iplot(fig)


# The plot above clearly shows that most of the variance (83.95% of the variance to be precise) can be explained by the first three principal components. The forth principal component still bears some information (10.59%) while the fifth and sixth principal components can safely be dropped without losing to much information. Together, the first four principal components contain 94.53% of the information.

# #### Dimension Reduction

# In[38]:


pca = PCA(n_components=4)
principal_components = pca.fit_transform(df_cluster_features)
# Principles' names come from eigenvectors. Those eigenvectors stands for different weights of original variables
pc_df = pd.DataFrame(data = principal_components
             , columns = ['chemical element orinted', 'high energy', 'diabetes friendly', 'hypertension friendly'])
pc_df.head()


# #### Clustering Diagnostic

# In[39]:


scaler.fit(pc_df)
scaler.transform(pc_df)

model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,11))

visualizer.fit(pc_df)
visualizer.poof() 


# Three clusters can work well

# #### Add Cluster Labels

# In[40]:


k_means = KMeans(n_clusters=3)
kmeans = k_means.fit(scaler.transform(pc_df))
pc_df['cluster'] = kmeans.labels_
pc_df.head()


# #### Element-Energy-Diabetes-3D Plot

# In[41]:


trace = go.Scatter3d(
    x=pc_df['chemical element orinted'],
    y=pc_df['high energy'],
    z=pc_df['diabetes friendly'],
    text=pc_df['cluster'],
    mode='markers',
    marker=dict(
        size=12,
        color=pc_df['cluster'],              
        colorscale='Viridis',   
        opacity=0.8
    )
)

data = [trace]
layout = go.Layout(
    showlegend=False,
    title='Different Kinds of Food',
    scene = dict(
        xaxis = dict(title='X: Chemical element orinted'),
        yaxis = dict(title='Y: High energy'),
        zaxis = dict(title='Z: Diabetes friendly'),
    ),
    width=1000,
    height=900,
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# #### Element-Energy-Hypertension-3D Plot

# In[42]:


trace = go.Scatter3d(
    x=pc_df['chemical element orinted'],
    y=pc_df['high energy'],
    z=pc_df['hypertension friendly'],
    text=pc_df['cluster'],
    mode='markers',
    marker=dict(
        size=12,
        color=pc_df['cluster'],                
        colorscale='Viridis', 
        opacity=0.8
    )
)

data = [trace]
layout = go.Layout(
    showlegend=False,
    title='Different Kinds of Food',
    scene = dict(
        xaxis = dict(title='X: Chemical element orinted'),
        yaxis = dict(title='Y: High energy'),
        zaxis = dict(title='Z: Hypertension friendly'),
    ),
    width=1000,
    height=900,
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# ## Conclusion

# 1.For people who want to lower their weight, they should eat less roasted and organic food. Also, they should consume fewer 
   #nuts, tea, candy chocolate, and ice cream. Certainly, they can find some kinds of food with lower calorie just as above 
   #clustering plot shows. For people dreaming of increasing muscles, they should ingest much more beef and various cheese. 
   #And for some office workers who want to get more energies, they can choose peanut, chocolate and organic food. These 
   #foods will help them perform better in their work.
# 
# 2.For patients who get diabetes, they should eat the food with normal levels of chemical element and energies but the lower 
#level of fat, sugars, and carbohydrates, which is combined as ‘diabetes friendly’ in the plot. The suitable range of it can 
#be [-19, 19] (to get ‘diabetes friendly’, you should multiply respective coefficients (eigenvectors) with original data). It
#means that anytime you get one kind of food, if the product of its contents in 100g and coefficients is in [-19, 19], then you
#can eat it. Also, for patients who get hypertension, [-2, 21] is the safe range of ‘hypertension friendly’. And the way of 
#calculating it is the same as ‘diabetes friendly’.
# 
