---
title: "Week10Assignment"
author: "Bin Lin"
date: "2016-11-6"
output: html_document
---
Assignment Instruction: It can be useful to be able to classify new "test" documents using already classified "training" documents.  A common example is using a corpus of labeled spam and ham (non-spam) e-mails to predict whether or not a new document is spam.  

For this project, you can start with a spam/ham dataset, then predict the class of new documents (either withheld from the training dataset or from another source such as your own spam folder).   One example corpus:  https://spamassassin.apache.org/publiccorpus/



My first step is to load neccessary packages and libraries.
```{r}
#install.packages("tm")
#install.packages("SnowballC")
#install.packages("RTextTools")
library(stringr)
library(tm)
library(SnowballC)
library(RTextTools)
```



I downloaded the files into my local machine. Obtain the directories (ham email), then then extract all the names of files in that directory using list.files function.  
```{r}
ham_dir <- "C:/Users/blin261/Desktop/DATA607/spamham/easy_ham/"
ham_files <- list.files(path = ham_dir)
head(ham_files)
```



My next step is to create corpus(a structured collection of texts). str_c function can join multiple strings into one single string. By setting collapse into "", the single string will represent the entire row for the output string matrix. VectorSource function is just helping wrap up the colllection of text.
```{r}
ham <- c()
for (i in 1: length(ham_files)) {
  tmp <- readLines(str_c(ham_dir, ham_files[i]))
  tmp <- str_c(tmp, collapse = "")
  ham <- c(ham, tmp)
}
ham_corpus <- Corpus(VectorSource(ham))
```



I label or tag the type of email that exist in this file. This is also called meta information. 
```{r}
for (i in 1: length(ham_corpus)) {
  meta(ham_corpus[[i]], "Type") <- "Ham"
}
ham_corpus
```



Repeat everything for the spam folder. In the end, I combined both corpus file. 
```{r}
spam_dir <- "C:/Users/blin261/Desktop/DATA607/spamham/spam_2/"
spam_files <- list.files(path = spam_dir)
head(spam_files)

spam <- c()
for (i in 1: length(spam_files)) {
  tmp <- readLines(str_c(spam_dir, spam_files[i]))
  tmp <- str_c(tmp, collapse = "")
  spam <- c(spam, tmp)
}
spam_corpus <- Corpus(VectorSource(spam))

for (i in 1: length(spam_corpus)) {
  meta(spam_corpus[[i]], "Type") <- "Spam"
}
spam_corpus

total_corpus <- c(ham_corpus, spam_corpus)
```



Next step mainly involve cleaning the data. I removed the numbers punctuation characters, remove the stop words ("i", "me", ""you" etc), and then perform a stemming operation. All the functions that is used following are available in tm package.
```{r}
total_corpus <- tm_map(total_corpus, removeNumbers)
total_corpus <- tm_map(total_corpus, str_replace_all, pattern = "[[:punct:]]", replacement = " ")
total_corpus <- tm_map(total_corpus, removeWords, words = stopwords("en"))
total_corpus <- tm_map(total_corpus, tolower)
total_corpus <- tm_map(total_corpus, stemDocument)
total_corpus <- tm_map(total_corpus, PlainTextDocument)
```


Just want to make sure all the tags are available after all these procedures. 
```{r}
meta_data <- meta(spam_corpus, "Type")
head(meta_data)

meta_data <- meta(ham_corpus, "Type")
head(meta_data)

meta_data <- meta(total_corpus, "Type")
head(meta_data)
```


TermDocumentMatrix function can help me change the text corpus into term-document matrix. Document-term matrix is just another way of outputing the result. After that, I remove the sparse terms, those appear on 100 documents or less.  
```{r}
tdm <- DocumentTermMatrix(total_corpus)
tdm
dtm <- DocumentTermMatrix(total_corpus)
dtm <- removeSparseTerms(dtm, 1-(100/length(total_corpus)))
dtm
```


Afterwards, from the RTextTools package, there is a function called create_container. By using this function, I created a container for all the information and specify the size of training data and test data.  
```{r}
org_labels <- unlist(meta(total_corpus, "Type"))
head(org_labels)
length(org_labels)
length(dtm)
```


```{r}
set.seed(888)
training_index <- sample(1: length(org_labels), size = 0.75 * length(org_labels), replace = FALSE)
test_index <- setdiff(1: length(org_labels), training_index)

container <- create_container(
  dtm,
  labels = org_labels,
  trainSize = training_index,
  testSize = test_index,
  virgin = FALSE
)
```


I am setting up 3 training models 
```{r}
svm_model <- train_model(container, "SVM")
tree_model <- train_model(container, "TREE")
maxent_model <- train_model(container, "MAXENT")
svm_out <- classify_model(container, svm_model)
tree_out <- classify_model(container, tree_model)
maxent_out <- classify_model(container, maxent_model)

head(svm_out)
head(tree_out)
head(maxent_out)
```


I constructed a data frame that contains the correct labels, and the labels that are predicted by three models. Then, I used the prop.table function to print out the proportion of those actuals label match the predicted labels and those do not.  According to the table, SVM and TREE model generate higher accuracy compare to MAXENT model when used to predict if email is a spam or ham. 
```{r}
labels_out <- data.frame(
  correct_label = org_labels[test_index],
  svm = as.character(svm_out[,1]),
  tree = as.character(tree_out[,1]),
  maxent = as.character(maxent_out[,1]),
  stringsAsFactors = F)


prop.table(table(labels_out[,1] == labels_out[,2]))
prop.table(table(labels_out[,1] == labels_out[,3]))
prop.table(table(labels_out[,1] == labels_out[,4]))
```


