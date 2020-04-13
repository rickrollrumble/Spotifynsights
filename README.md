# What is this project about?

I recently started a Coursera course on Python Visualizations. I wanted to test my skills out. I really liked the visualizations that Spotify had created to summarize our listening history. Inspired by that, I decided to brush up on my Matplotlib and Seaborn skills. I hope you find this useful.

I based these visualizations off a [dataset](https://www.kaggle.com/zaheenhamidani/ultimate-spotify-tracks-db/data) I found on Kaggle. It was available as a CSV file.  


```python
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # needed for waffle Charts
import seaborn as sb
import numpy as np
```


```python
tracks = pd.read_csv('data/SpotifyFeatures.csv')
```


```python
print(tracks['genre'].value_counts())
```

    Comedy              9681
    Soundtrack          9646
    Indie               9543
    Jazz                9441
    Pop                 9386
    Electronic          9377
    Children’s Music    9353
    Folk                9299
    Hip-Hop             9295
    Rock                9272
    Alternative         9263
    Classical           9256
    Rap                 9232
    World               9096
    Soul                9089
    Blues               9023
    R&B                 8992
    Anime               8936
    Reggaeton           8927
    Ska                 8874
    Reggae              8771
    Dance               8701
    Country             8664
    Opera               8280
    Movie               7806
    Children's Music    5403
    A Capella            119
    Name: genre, dtype: int64
    

Here, it has to be noted that there's two records of Children's Music available, so we keep the larger one since it's inclusive of all the records. 


```python
tracks = tracks[tracks['genre']!="Children's Music"]
```

# What are the most popular genres from the dataset?

First, we get a list of all unique genres of the songs.


```python
genres = tracks['genre'].unique()
```

Then we iterate through the genres, find the count of songs belong to each genre, and them store in a dictionary.Then we calculate the average popularity of songs in each decade using the numpy ```mean``` function and store it into a dictionary


```python
pops = {}
for genre in genres:
    pops[genre] = np.mean(tracks[tracks['genre']==genre]['popularity'])
```

This dictionary is converted into dataframe and plotted as a horizontal bar graph.


```python
avg_pop = pd.DataFrame(data = pops.values(),index=pops.keys(),columns=['Average Popularity'])

```


```python
fig = plt.figure(figsize = (16,9))
ax = plt.axes()
plt.barh(y=avg_pop.index,width=avg_pop['Average Popularity'], color='#80eda7')
plt.title('Average Genre Popularity')
for a in pops:
    plt.text(pops[a]+0.1,a,str('%.2f'%(pops[a])),va='center')
fig.savefig('a.png',dpi=300)
```


![png](reports/figures/output_14_0.png)


Hasan Minhaj once said in his episode about hiphop on [Patriot Act](https://youtu.be/MEZV6EE8JMA?t=296) that shorter tracks are streamed a lot more. I wanted to test this statement. So we'll use the popularity measure and compare it to the track length. The track length is in milliseconds, however,and hence are converted to minutes by dividing by 1000 and then by 60.


```python
tracks.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genre</th>
      <th>artist_name</th>
      <th>track_name</th>
      <th>track_id</th>
      <th>popularity</th>
      <th>acousticness</th>
      <th>danceability</th>
      <th>duration_ms</th>
      <th>energy</th>
      <th>instrumentalness</th>
      <th>key</th>
      <th>liveness</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>valence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Movie</td>
      <td>Henri Salvador</td>
      <td>C'est beau de faire un Show</td>
      <td>0BRjO6ga9RKCKjfDqeFgWV</td>
      <td>0</td>
      <td>0.611</td>
      <td>0.389</td>
      <td>99373</td>
      <td>0.910</td>
      <td>0.000</td>
      <td>C#</td>
      <td>0.3460</td>
      <td>-1.828</td>
      <td>Major</td>
      <td>0.0525</td>
      <td>166.969</td>
      <td>4/4</td>
      <td>0.814</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Movie</td>
      <td>Martin &amp; les fées</td>
      <td>Perdu d'avance (par Gad Elmaleh)</td>
      <td>0BjC1NfoEOOusryehmNudP</td>
      <td>1</td>
      <td>0.246</td>
      <td>0.590</td>
      <td>137373</td>
      <td>0.737</td>
      <td>0.000</td>
      <td>F#</td>
      <td>0.1510</td>
      <td>-5.559</td>
      <td>Minor</td>
      <td>0.0868</td>
      <td>174.003</td>
      <td>4/4</td>
      <td>0.816</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Movie</td>
      <td>Joseph Williams</td>
      <td>Don't Let Me Be Lonely Tonight</td>
      <td>0CoSDzoNIKCRs124s9uTVy</td>
      <td>3</td>
      <td>0.952</td>
      <td>0.663</td>
      <td>170267</td>
      <td>0.131</td>
      <td>0.000</td>
      <td>C</td>
      <td>0.1030</td>
      <td>-13.879</td>
      <td>Minor</td>
      <td>0.0362</td>
      <td>99.488</td>
      <td>5/4</td>
      <td>0.368</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Movie</td>
      <td>Henri Salvador</td>
      <td>Dis-moi Monsieur Gordon Cooper</td>
      <td>0Gc6TVm52BwZD07Ki6tIvf</td>
      <td>0</td>
      <td>0.703</td>
      <td>0.240</td>
      <td>152427</td>
      <td>0.326</td>
      <td>0.000</td>
      <td>C#</td>
      <td>0.0985</td>
      <td>-12.178</td>
      <td>Major</td>
      <td>0.0395</td>
      <td>171.758</td>
      <td>4/4</td>
      <td>0.227</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Movie</td>
      <td>Fabien Nataf</td>
      <td>Ouverture</td>
      <td>0IuslXpMROHdEPvSl1fTQK</td>
      <td>4</td>
      <td>0.950</td>
      <td>0.331</td>
      <td>82625</td>
      <td>0.225</td>
      <td>0.123</td>
      <td>F</td>
      <td>0.2020</td>
      <td>-21.150</td>
      <td>Major</td>
      <td>0.0456</td>
      <td>140.576</td>
      <td>4/4</td>
      <td>0.390</td>
    </tr>
  </tbody>
</table>
</div>




```python
tracks['duration_ms'] = tracks['duration_ms'].divide(1000*60)
tracks.rename(columns={'duration_ms':'duration_min'},inplace=True)
```


```python
popdur = tracks[['popularity','duration_min']]
# popdur['duration_ms'] = popdur['duration_ms'].div(1000*60)
```


```python
hists,bin_edges = np.histogram(popdur['duration_min'],10)
```

Let's see how the track length is distributed.


```python
fig2 = plt.figure(figsize=(10,7))
ax = plt.axes()
bxplt = sb.boxplot(popdur['duration_min'],color='#819be6')
plt.title('Box Plot of Track Lengths')
plt.xlabel('Track Duration (minutes)')
```




    Text(0.5, 0, 'Track Duration (minutes)')




![png](reports/figures/output_21_1.png)


There are multiple outliers, which skews the plot a lot. We will hide these outliers and plot them again. Since this plot was generated using Seaborn, we set the ```showfliers``` to ```False```.


```python
sb.set_style('whitegrid')
fig2 = plt.figure(figsize=(10,7))
ax = plt.axes()
bxplt = sb.boxplot(popdur['duration_min'], showfliers=False, color='orange')
plt.title('Box Plot of Track Lengths')
plt.xlabel('Track Duration (minutes)')
```




    Text(0.5, 0, 'Track Duration (minutes)')




![png](reports/figures/output_23_1.png)


It is necessary to drop the outliers to prevent long tails from showing up in the plot, and thus stretching it. We will calculate the Z-Scores for the song durations, drop songs with Z-Scores more than 2 positive standard deviations away from the mean.


```python
from scipy import stats
```


```python
popdur = popdur.assign(z_score=stats.zscore(popdur['duration_min']))
```

The cell above was used to calculate Z-Scores, the cell below used to drop those tracks.


```python
popdur = popdur[popdur['z_score']<2]
```

Now, let's make a scatter plot of the duration vs. popularity.


```python
fig2 = plt.figure(2,figsize=(16,9))
sctr = sb.scatterplot(x=popdur['duration_min'],y=popdur['popularity'],color='black')
```


![png](reports/figures/output_30_0.png)


Well this data is too messy and crowded to make sense of. We need to find a better way to understand it. Interestingly though, we do notice a trend here; that songs closer to 3 minutes are often the most popular. <br>

We will use the previously-used box plots again. We noticed that the songs close to 3 minutes seem to be most popular, so we will classify songs into 4 categories, `Short`,for songs lower than 2 min. `Med`, for songs between 2 and 4 minutes, `Longish`, for songs between 4 to 6 minutes, and `Long` for songs longer than 6 minutes. This category variable column is added to the `popdur` dataframe.


```python
length_cat = []
for i in popdur['duration_min']:
    if i <=2:
        length_cat.append('Short')
    elif i<=4:
        length_cat.append('Med')
    elif i<=6:
        length_cat.append('Longish')
    else:
        length_cat.append('Long')
```


```python
popdur = popdur.assign(length_category = length_cat) 
popdur
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>popularity</th>
      <th>duration_min</th>
      <th>z_score</th>
      <th>length_category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1.656217</td>
      <td>-1.160210</td>
      <td>Short</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2.289550</td>
      <td>-0.840605</td>
      <td>Med</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2.837783</td>
      <td>-0.563946</td>
      <td>Med</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>2.540450</td>
      <td>-0.713992</td>
      <td>Med</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1.377083</td>
      <td>-1.301072</td>
      <td>Short</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>232720</th>
      <td>39</td>
      <td>5.437333</td>
      <td>0.747888</td>
      <td>Longish</td>
    </tr>
    <tr>
      <th>232721</th>
      <td>38</td>
      <td>4.707450</td>
      <td>0.379561</td>
      <td>Longish</td>
    </tr>
    <tr>
      <th>232722</th>
      <td>47</td>
      <td>2.782667</td>
      <td>-0.591760</td>
      <td>Med</td>
    </tr>
    <tr>
      <th>232723</th>
      <td>44</td>
      <td>3.707367</td>
      <td>-0.125120</td>
      <td>Med</td>
    </tr>
    <tr>
      <th>232724</th>
      <td>35</td>
      <td>5.383783</td>
      <td>0.720865</td>
      <td>Longish</td>
    </tr>
  </tbody>
</table>
<p>221620 rows × 4 columns</p>
</div>




```python
fig = plt.figure(figsize=(16,9))
bxplt = sb.boxplot(y=popdur['popularity'],x=popdur['length_category'],palette=sb.color_palette('pastel'))
plt.title('Popularity of songs in different length categories')
plt.xlabel('Length Category')
plt.ylabel('Popularity')
```




    Text(0, 0.5, 'Popularity')




![png](reports/figures/output_34_1.png)


Having tried out the waffle chart to unsatisfactory results, I looked at a much simpler way of visualizing the most popular genre in different song length categories. We add the z-score column to the original `tracks` dataframe so that we can drop the outliers. Then we add another column for the song's `length_category` to the dataframe.


```python
tracks = tracks.assign(z_score = stats.zscore(tracks['duration_min']))
tracks = tracks[tracks['z_score']<2]
tracks = tracks.assign(length_category = length_cat)
```

We create separate dataframes for each of the length categories.


```python
short = tracks[tracks['length_category']=='Short']
medium = tracks[tracks['length_category']=='Med']
longish = tracks[tracks['length_category']=='Longish']
long = tracks[tracks['length_category']=='Long']
```

The `plot_hor_bars` function takes a dataframe, the column based on which the chart is plotted, and a color palette. Because I like colors. We create a dictionary with the genre being used as a key, and the count of that genre the corresponding value. This dictionary is converted to a dataframe, then sorted in descending order for better visibility and then plotted using Seaborn.


```python
def plot_hor_bars(data,column,palettes):
    pops = dict()
    for key in data[column].unique():
        pops[key] = np.mean(data[data[column]==key]['popularity'])
    pops = pd.DataFrame(pops.values(),index=list(pops.keys()))
    pops.columns = ['Average Popularity']
    pops.sort_values(by=['Average Popularity'],inplace=True,ascending=False)
    pl1 = plt.figure(figsize=(16,9))
    sb.barplot(y=pops.index.values,
               x=pops['Average Popularity'],
               palette=sb.color_palette(palettes,
                                        n_colors=len(pops.index.values)))
```

## Short Songs


```python
plot_hor_bars(short,'genre','Reds_r')
title = plt.title('Most Popular Genres of Short Songs')
```


![png](reports/figures/output_42_0.png)


## Medium Songs


```python
plot_hor_bars(medium,'genre','Greens_r')
title = plt.title('Most Popular Genres of Medium Songs')
```


![png](reports/figures/output_44_0.png)


## Longish Songs


```python
plot_hor_bars(longish,'genre','Blues_r')
title = plt.title('Most Popular Genres of Longish Songs')
```


![png](reports/figures/output_46_0.png)


## Long Songs


```python
plot_hor_bars(long,'genre','icefire_r')
title = plt.title('Most Popular Genres of Medium Songs')
```


![png](reports/figures/output_48_0.png)


## Conclusion

Pop music is popular, pop music is mainstream. It gets played the for a variety of song lengths. Acapella songs are often the least-played songs. <br>
The difference between pop and rap music is the least for short songs. That does not necessarily mean Hasan Minhaj was wrong. The lines between genres have blurred significantly in 
