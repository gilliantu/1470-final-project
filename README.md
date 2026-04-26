# 1470-final-project

Introduction:

Capstone students: No

Are you implementing an existing paper/project or solving something new?: Implementing an existing paper [1470]

Team name: JAG

Member cs logins: gtu3, jshou, ajacob39

Project category: Transformers + CNN + Generative Model

Project idea:

Our project will take a photo of an outfit and rate how closely it matches a chosen aesthetic like streetwear, minimalist, or casual by giving a score from 1 to 10, where a higher score means it fits the aesthetic better. Using the fashion.json dataset, we will train a model to recognize patterns like clothing type, color, and overall vibe that define each aesthetic, and then compare a new outfit to those learned patterns to produce a similarity score. To evaluate the model, we will split the data into training and validation sets and check if it gives higher scores to outfits that clearly match an aesthetic and lower scores to ones that do not.
https://www.kaggle.com/datasets/pypiahmad/shop-the-look-dataset/data?select=fashion.json

Challenges: What has been the hardest part of the project you've encountered so far?
Figuring out a weighting system for cosine similarity and scaling. At first our model was scoring all 10 aesthetics as 10/10 but once we rescaled the cosine similarity, we were able to assign a score out of 10 for each aesthetic.

Insights: Are there any m concrete results you can show at this point?
We currently have a prototype app that can be locally hosted where you submit an image of an outfit and it will return a ranking of how closely your outfit matches each of the 10 aesthetics.

How is your model performing compared with expectations?
So far, because we are using huggingface transformer model training to test our implementation, we haven’t started our actual model implementation. But, everything else is working as expected.

Plan: Are you on track with your project?
Yes! Although we are currently using the huggingface model, we plan to implement our own model after this, which we anticipate won’t take longer than a few days. 

What do you need to dedicate more time to?
We need to create our own transformer and recommendation system/generative model.

What are you thinking of changing, if anything?
We were thinking of adding a feature where we have an article of clothing and trying to figure out what will go best with it based on an aesthetic of choice.
