<head>
  <link rel="stylesheet" href="style.css">
</head>

# Predicting Best Picture Winners 

**Davis Williams, Ian Haessler, Chaitu Konjeti, Saket Shirsath, Sania Setayesh**

## Introduction

The Oscars, also known as the Academy Awards, are a series of awards given to exceptional films. These awards were first started in 1929 and have since become the most prestigious award that a film can receive. The award categories range from Best Actor/Actress to Best Original Song . For our purposes, we will focus on the highest honor: Best Picture. [1]

In this project, we aimed to design an algorithm that could predict the Oscar winners for Best Picture at any point in the future. Given an input of nominees, we wanted our algorithm to classify whether or not each movie would win the Best Picture award. In order to do this, we decided to utilize data such as runtime, genre, month released, and earnings as our predictive features. We believed that this information would show an historical trend in Oscar winners. We utilized supervised learning algorithms such as K nearest neighbors, linear regression, naive bayes, neural networks, and decision trees. We decided to use supervised learning because we have the knowledge of which movies won Best Picture and which movies did not.

We believe that this is an important problem to solve because it could help those in the movie industry determine what makes a movie successful. Investing in a movie is a huge financial risk, so being able to algorithmically determine whether or not a movie is likely to win the Oscar could be helpful to producers and artists.

## Dataset

To solve this problem, we needed a list of all movies that had ever been nominated for Best Picture at the Oscars since 1927\. We used the The Oscar Award, 1927 - 2020 dataset on Kaggle for this [2].

Using pandas, we searched through the dataset and extracted all movies that were tagged under Best Picture (or any of the equivalent categories- the Best Picture award has had five different names since 1927) [1]. Once we had a list of the movies we were using, we needed to collect actual information about these movies. For that we, used the IMDB Movies Dataset on Kaggle, which gave us raw data about the movie's IMDB rating, number of ratings, and runtime [3]. Using The Movie Database API, we collected data regarding the film's budget, earnings, and release date [4]. Finally, we used The Movies Dataset on Kaggle to get information regarding genre [5]. This gave us all the raw data we needed, but we had to process much of it so that we could represent it in a meaningful way.

#### Features

The final features we wanted to include in our dataset are as follows:

-Runtime: Movies with longer runtimes have generally won more Oscars than shorter movies.

-Release Month: Historically, movies that are released closer to the Oscars do better, so release month gives us a gauge as to how far each individual movie was from the ceremony.

-Earnings/Budget Ratio: A movie that is financially successful is going to be more likely to catch the attention and favor of the committee, regardless of whether it was a smaller budget indie film or a big blockbuster.

-Weighted IMDB Score: General reception of a film is likely to be higher in award winning films. However, not all films had the same number of ratings on IMDB. The average rating of a movie with thousands of reviws should be more indicative of success than an average of 20-30 reviews.

-Up to 3 Movie Genres: Certain genres, such as war, historical, and drama have won Best Picture more often than other categories, such as adventure or fantasy [6].

To get these features from our raw data, we had to do some manipulation. Runtime and release month were fairly simple. Runtime could be extracted straight from the raw data. To get release month, we just had to parse through the release date and extract the month. The earnings/budget ratio was a simple calulation of (earnings/budget) * 100. Some budget/earnings data was missing, so in those cases, the movies were assigned a default value of 100. We assigned each genre a unique id and set the features genre_1, genre_2, and genre_3 to the id scores for each movie's top three genres. If a movie fit under less than 3 genres, we assigned it 0 for the extra features. This accounts for most of our features, but we had to come up with a more novel way of making use of the IMDB ratings.

#### Deriving A New Rating Score

One of the biggest issue with a dataset like this is that record keeping and participation increased greatly since the Oscars' inception. This is clearly seen in the ratings for the winners over time.

![picture](https://drive.google.com/uc?id=1-XJeJLa-P_wab0q0QQ4Tw2NaWzNetKsv)

In this figure, we see ratings over time displayed by the blue line. As seen by the number of votes over time (red), participation in rating movies drastically increases as the years get closer to present day. Simply using the rating as a feature in our supervised learning models artifically inflates the importance of the ratig for movies with fewer votes. This is because movies with fewer number of votes are compared to those with more votes on the same level.

To overcome this discrepancy, we took the Z-score of the number of votes by decade, and we multiplied it by the rating for each movie in that decade. The result scores were generated:

![picture](https://drive.google.com/uc?id=1IEshJEuCJnQyd-hZltblOmzV2N4_JWpg)

In general, using this method makes the ratings comparable over the years in a way that would be useful for our models.

#### Labels

Finally, we completed our dataset but adding the labels. The labels were fairly simple: either a movie won the Oscar or was only nominated. If a movie won, we set the label "winner" to 1\. If it lost, we set it to 0.

Below are a few sample data points with features and labels:

<table>

<thead>

<tr>

<th>film</th>

<th>runtime</th>

<th>earnings_ratio</th>

<th>new_rating_score</th>

<th>release_month</th>

<th>genre_1</th>

<th>genre_2</th>

<th>genre3</th>

<th>winner</th>

</tr>

</thead>

<tbody>

<tr>

<td>Parasite</td>

<td>131</td>

<td>2164.2530141687935</td>

<td>0.0</td>

<td>5</td>

<td>27</td>

<td>878</td>

<td>0</td>

<td>1</td>

</tr>

<tr>

<td>Once upon a Time...in Hollywood</td>

<td>162</td>

<td>393.94868105263157</td>

<td>-0.698</td>

<td>7</td>

<td>35</td>

<td>18</td>

<td>0</td>

<td>0</td>

</tr>

<tr>

<td>Joker</td>

<td>122</td>

<td>1953.0023836363637</td>

<td>10.7</td>

<td>10</td>

<td>35</td>

<td>10769</td>

<td>0</td>

<td>0</td>

</tr>

</tbody>

</table>

## Methods

### What's Novel

As discussed, one of the novel things about our method is the feature set. For example, we used a z-score of the rankings instead of the raw rankings itself. We also looked at up to three separate genres, ranked by relevance, for each movie. Finally, we decided to use several different methods of classification so that we could compare them against each other and decide which was the most accurate.

### K Nearest Neighbors

One classification algorithm we ran on our data is the K Nearest Neighbors algorithm. Using various values for K, we achieved the following confusion matrices:

#### K = 1

![picture](https://drive.google.com/uc?id=1fKK0oZ9rUFg89V0oDgcnR1Ll4XvUTNzM)

#### K = 3

![picture](https://drive.google.com/uc?id=1_3HpsSxdddvLnA4lLSb-3565XAqt843r)

#### K = 7

![picture](https://drive.google.com/uc?id=1_GNDntG75a-e-GXaCNA-YeJou3VbhwbO)

#### K = 10

![picture](https://drive.google.com/uc?id=1UzkWOhnOw_JB_v79A710kbsg7v343fTs)

The accuracy for K values below 10 was approximately .6; lower K values yeilded the best results. As K values increased from 10, the accuracy approached .5, and the confusion matrix yeilded a 50-50 split of true negatives and false positives.

Using K = 1, which found our best results, the KNN algorithm correctly labeled 28.57% of the true winners. Of the movies predicted to win by KNN, 83.33% did actually win.

### Logistic Regression

Logistic Regression was another approach we used on our data. After training on 70% of the data and testing on 30%, the confusion matrix below is the result.

![picture](https://drive.google.com/uc?id=1e6CGdkF0k0tRF4NgnoUjmfcw6paAcLTM)

Though overall accuracy was 83%, the model was only able to correctly predict 3 winners out of a total of 27.

### Naive-Bayes

We also classified our data using Naive-Bayes. Using k-fold cross validation, we split the data into 10 folds and ran Naive-Bayes on the data 10 times, using each split as the test data once. Some of the results are displayed in the following 3 confusion matrices. ![picture](https://drive.google.com/uc?id=1pCD6WnLfEcumH_dSHllA_1zxcdv2lprr) ![picture](https://drive.google.com/uc?id=1kk_eh0G2UWYdTj6o-y6KQ4ZlxE9EM6Ew) ![picture](https://drive.google.com/uc?id=1aThpa-eflW9b57jcWEg50ejN_1ie7Vhe)

As you can tell, the accuracy varies between 0.4 and 0.6, so the Naive-Bayes predictor is still essentially a guess. Even in the best case, it returns about as many false positives as true positives. After running on all 10 k-folds, we get an average accuracy of 52.77%. Furthermore, of all the movies our Naive-Bayes classifier predicted to win the Oscars, only 52.87% actually won Best Picture. Conversely, of the movies that actually did win Best Picture, only 43.33% were correctly labeled by the classifier.

### Neural Network

We designed a neural network with the following layers: an input layer with an input vector size of 7, 2 fully-connected hidden layers of size 7 both with the relu activation function, and an output layer of size 1 with the sigmoid activation function. The model also utilized the Adam optimizer, which essentially combines root squared mean error with stochastic gradient descent and momentum. The binary cross-entropy loss function was used because it is especially useful when dealing with yes/no situations, as we have in this problem. The neural network was run for 100 epochs with a batch size of 20.

![picture](https://drive.google.com/uc?id=12dudD2LVZga4SRyPQwJkJEqxlsH7hJAu)

Here, the neural net had a total accuracy of 58.18%. However, the model correctly classified only 16.67% of the true winners as winners. Furthermore, only 27.27% of the movies our neural net declared as winners turned out to actually be winners.

### Decision Trees

We also used the decision tree algorithm for our supervised learning algorithm. We designed our decision tree using various random samples. Each internal node of the tree corresponds to an attribute, and each leaf node corresponds to a class label. ![picture](https://drive.google.com/uc?id=1fy6AgxHh1qrVZkZsgHfRsE5i19KpifDJ)

Certain runs of the decision tree look promising. The confusion matrix below shows 70% accuracy with few false positives and false negatives: ![picture](https://drive.google.com/uc?id=1ET7oHPb3VuI2Ct2IxbUZaKY3mck_2lQR)

But other runs of the decision tree are not so great. Here, we have a total accuracy of 54%. ![picture](https://drive.google.com/uc?id=1xq3iS71DrTE-FkrnBG-FSNK2n_IpDqLd)

So while decision trees are the most promising classifier so far, they are still far from reliable. Over several iterations on random splits of the data, the average accuracy is 65.95%. When it predicts that a movie has won the Oscar, there is a 70.12% chance that it actually has. Finally, if a movie has won Best Picture, there is a 65.66% chance that our decision tree would correctly classify it as a winner.

### Comparison

![picture](https://drive.google.com/uc?id=100IIKVpGeEdz9T4q81DAL1wUZr-wf8vF)

The above chart rates our classifiers on the following accuracy measurements:

(1) Total Accuracy

(2) Of the movies that our classifier labeled as a winner, what percent actually were winners?

(3) Of the movies that were actually winners, what percent did our classifier actually declare as winners?

K-nearest neighbors does fairly well at correctly labeling winners. However, it also ends up labeling a lot of losers as winners, which means this classifier just has a high bias towards labeling winners. Of our algorithms, neural net seems to have done the worst- going as low as ~17% accuracy for correctly labeling winners. Another notable stand out is the 82% overall accuracy in our logistic regression. Still, a closer examination reveals a dismal 10% accuracy regarding correctly labeling winning films. Our best classifier is decision trees. However, even then, we are hovering around 65-70% average accuracy for all measures which is not a very reliable measurement. Ultimately, it seems as if we have failed to develop a classifier that can predict Oscar Best Picture winners.

## Conclusion

Overall, our classifers were not much better than flipping a coin. There are multiple reasons for this outcome, and when you look at the data the picture becomes a lot clearer. For starters, the classification of Oscar winners and losers is most likely a bad candidate for supervised learning. In supervised learning, you have to make the assumption that there is "some" underlying function that maps to the datapoints that are present in your data. The problem with Oscar winnners is that the reasoning behind a winner and a loser is a subjective decision, not an objective one.

On top of this, our dataset contains information on all Oscar winnners and nominees since the beginning of the Academy Awards. The problem with this is that the qualities inherent in a winning film in the 1920s may be very different than the qualities that the Academy looks for now. This means that the actual year you ask the question could have an effect on what a winner "should" be to our classifier.

Other issues become apparent when looking at the distribution of the data. By plotting the means of each feature, we can see that though the difference between the mean features of winners and nominees exists, it is very small.

![picture](https://drive.google.com/uc?id=1zPD-vIT4esftc8ELEHHTL422XWb8pYqx) ![picture](https://drive.google.com/uc?id=1la9UDIL0KmVcEerIVSRS63gdwEnCvx-u)

The obvious exception is our earnings ratio. At first, this would make it seem as though earnings ratio may be a good feature to use in the classifer, but there is a less visible problem inherent with our earnings ratio feature. Earlier movies were missing data on their budget and earnings, so we had to assign them a defualt value. This may have skewed the data and disproportionately brought down the average for movies that did not win, as that category is missing a non-trivial amount of data.

To conclude, because of its subjectivity, there is likely no way to make a classifer that accurately predicts Best Picture winners. However, better accuracy could potentially be improved by using data only from more recent years and a different feature set.

## Works Cited

[1] Academy Award for Best Picture. (2020, April 6). Retrieved March 5, 2020, from [https://en.wikipedia.org/wiki/Academy_Award_for_Best_Picture](https://en.wikipedia.org/wiki/Academy_Award_for_Best_Picture)

[2] Fontes, R. (2020, February 19). The Oscar Award, 1927 - 2020, Version 7\. Retrieved March 8, 2020 from [https://www.kaggle.com/unanimad/the-oscar-award/](https://www.kaggle.com/unanimad/the-oscar-award/).

[3] Leka, O. (2016, November 15). IMDB Movies Dataset, Version 1\. Retrieved March 8, 2020 from [https://www.kaggle.com/orgesleka/imdbmovies/](https://www.kaggle.com/orgesleka/imdbmovies/).

[4] The MovieDB. (n.d.). Retrieved March 8, 2020, from [https://www.themoviedb.org/](https://www.themoviedb.org/)

[5] Banik, R. (2017, November 9). The Movies Dataset, Version 7\. Retrieved March 15, 2020 from [https://www.kaggle.com/rounakbanik/the-movies-dataset/](https://www.kaggle.com/rounakbanik/the-movies-dataset/).

[6] Lee, N. (2020, February 7). There's a formula to winning the Oscars, and it's all in the statistics. Retrieved March 5, 2020, from [https://www.businessinsider.com/oscars-academy-awards-rigged-best-picture-nominations-win-2019-2](https://www.businessinsider.com/oscars-academy-awards-rigged-best-picture-nominations-win-2019-2)

## Contributions

Davis Williams: Processed raw data, Naive Bayes, Dataset paragraphs, Comparison paragraphs, Created graphics, Write-Up Editor

Ian Haessler: Created initial data set, Logistic Regression, Created Webpage, Created graphics

Chaitu Konjeti: Neural network, methods for the project proposal, wrote introduction, Presenter

Saket Shirsath: K Nearest Neighbors, Processed new rating feature, Wrote Rating Score paragraph, Presenter

Sania Setayesh: Decision Trees, Presenter
