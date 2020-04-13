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
    
tracks = pd.read_csv('SpotifyFeatures.csv')
```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    <ipython-input-2-2bdb7d7534c6> in <module>
    ----> 1 tracks = pd.read_csv('SpotifyFeatures.csv')
          2 
    

    e:\installations\miniconda3\lib\site-packages\pandas\io\parsers.py in parser_f(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, squeeze, prefix, mangle_dupe_cols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, dialect, error_bad_lines, warn_bad_lines, delim_whitespace, low_memory, memory_map, float_precision)
        683         )
        684 
    --> 685         return _read(filepath_or_buffer, kwds)
        686 
        687     parser_f.__name__ = name
    

    e:\installations\miniconda3\lib\site-packages\pandas\io\parsers.py in _read(filepath_or_buffer, kwds)
        455 
        456     # Create the parser.
    --> 457     parser = TextFileReader(fp_or_buf, **kwds)
        458 
        459     if chunksize or iterator:
    

    e:\installations\miniconda3\lib\site-packages\pandas\io\parsers.py in __init__(self, f, engine, **kwds)
        893             self.options["has_index_names"] = kwds["has_index_names"]
        894 
    --> 895         self._make_engine(self.engine)
        896 
        897     def close(self):
    

    e:\installations\miniconda3\lib\site-packages\pandas\io\parsers.py in _make_engine(self, engine)
       1133     def _make_engine(self, engine="c"):
       1134         if engine == "c":
    -> 1135             self._engine = CParserWrapper(self.f, **self.options)
       1136         else:
       1137             if engine == "python":
    

    e:\installations\miniconda3\lib\site-packages\pandas\io\parsers.py in __init__(self, src, **kwds)
       1915         kwds["usecols"] = self.usecols
       1916 
    -> 1917         self._reader = parsers.TextReader(src, **kwds)
       1918         self.unnamed_cols = self._reader.unnamed_cols
       1919 
    

    pandas/_libs/parsers.pyx in pandas._libs.parsers.TextReader.__cinit__()
    

    pandas/_libs/parsers.pyx in pandas._libs.parsers.TextReader._setup_parser_source()
    

    FileNotFoundError: [Errno 2] File b'SpotifyFeatures.csv' does not exist: b'SpotifyFeatures.csv'



```python
tracks['genre'].value_counts()
```

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

Hasan Minhaj once said in his episode about hiphop on [Patriot Act](https://youtu.be/MEZV6EE8JMA?t=296) that shorter tracks are streamed a lot more. I wanted to test this statement. So we'll use the popularity measure and compare it to the track length. The track length is in milliseconds, however,and hence are converted to minutes by dividing by 1000 and then by 60.


```python
tracks.head()
```


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

There are multiple outliers, which skews the plot a lot. We will hide these outliers and plot them again. Since this plot was generated using Seaborn, we set the ```showfliers``` to ```False```.


```python
sb.set_style('whitegrid')
fig2 = plt.figure(figsize=(10,7))
ax = plt.axes()
bxplt = sb.boxplot(popdur['duration_min'], showfliers=False, color='orange')
plt.title('Box Plot of Track Lengths')
plt.xlabel('Track Duration (minutes)')
```

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


```python
fig = plt.figure(figsize=(16,9))
bxplt = sb.boxplot(y=popdur['popularity'],x=popdur['length_category'],palette=sb.color_palette('pastel'))
plt.title('Popularity of songs in different length categories')
plt.xlabel('Length Category')
plt.ylabel('Popularity')
```

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

## Medium Songs


```python
plot_hor_bars(medium,'genre','Greens_r')
title = plt.title('Most Popular Genres of Medium Songs')
```

## Longish Songs


```python
plot_hor_bars(longish,'genre','Blues_r')
title = plt.title('Most Popular Genres of Longish Songs')
```

## Long Songs


```python
plot_hor_bars(long,'genre','icefire_r')
title = plt.title('Most Popular Genres of Medium Songs')
```

## Conclusion

Pop music is popular, pop music is mainstream. It gets played the for a variety of song lengths. Acapella songs are often the least-played songs. <br>
The difference between pop and rap music is the least for short songs. That does not necessarily mean Hasan Minhaj was wrong. The lines between genres have blurred significantly in 
