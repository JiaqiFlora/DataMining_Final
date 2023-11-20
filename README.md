# Data Mining Final Project

Chelsea Joe, Jiaqi(Flora) Gan, Kewen(Coco) Huang

## Questions
- Main Question: Can we develop a model that can differentiate between AI and human generated answers?
- Should we use a pretrained model or should we use the basic functionality in Scikit learn?
- How can we get sufficient AI and human generated answer to train our model?
- Can we make an immersive VR experience that can instill fear into the player as they fight for humanity?


## Final Deliverable
- scraped data: in the `dataset` folder in this project
- trained model: [model link]()
- game record video: [video link](https://drive.google.com/file/d/1yMtKrH_1vQNTIWOQPleaeZu5j9gJtD7W/view?usp=sharing)
- game repo: [game repo link]()

## Usage
Some important usage of this project are as following:
- scrape the data from stack exchange: ` python stack_exchange_scraper.py`
- get the OpenAI answer: `python get_openai_answer.py`
        
    but you first need to create a `config.txt` file in the root directory, and put your OpenAI api key there. The format in this file is like
    ```
    [openai]
    api_key = 'you api key here'
    ```
- train and test the model: `python BERTModel.py`
- predict using saved model: `python predict_using_bert.py`
