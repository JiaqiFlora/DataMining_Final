import requests
from bs4 import BeautifulSoup
import csv
import re

# all topic choices
topic = "pets"
# topic = "sports"


BASE_URL = f"https://{topic}.stackexchange.com/questions?tab=votes&page="


def extract_question_and_answer(page_num):
    data = []
    response = requests.get(BASE_URL + str(page_num))
    soup = BeautifulSoup(response.content, 'html.parser')

    # get all cell content
    main_divs = soup.select("div.s-post-summary.js-post-summary")
    for cur_div in main_divs:
        # check answer count first
        answer_count_div = cur_div.find('div', title=re.compile(r'answer(s)?$'))
        count_num = answer_count_div.select_one(".s-post-summary--stats-item-number").text
        if int(count_num) <= 0:
            # print("<<<<<<<<<<00000000")
            # print(count_num)
            continue

        # check for question and its link
        question_div = cur_div.select_one(".s-post-summary--content-title")
        question = question_div.select_one("a.s-link")
        if not question:
            continue

        # get current question's title and link
        question_title = question.text.strip()
        link = question['href']
        question_url = 'https://pets.stackexchange.com' + link

        # get answer for this question
        q_response = requests.get(question_url)
        q_soup = BeautifulSoup(q_response.content, 'html.parser')
        answer = q_soup.select_one("div.answercell.post-layout--right")

        if not answer:
            print("!!!!!!!!!!!!!!!!!!!!!")
            print(question_title)
            print(q_response)
            continue

        answer = answer.select_one("div.s-prose.js-post-body[itemprop='text']")
        if answer:
            paragraphs = answer.select('p')
            answer_text = ' '.join([p.text for p in paragraphs])
            data.append([question_title, answer_text])

    return data


def main():
    all_data = []
    page_num = 1
    max_pages = 10

    while True:
        print(f"Scraping page {page_num}...")
        data = extract_question_and_answer(page_num)
        if not data:
            break

        all_data.extend(data)

        if page_num >= max_pages:
            break
        page_num += 1

    print("==============")
    print(len(all_data))

    with open(f"dataset/stackexchange/{topic}_{page_num}.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["question", "answer"])
        writer.writerows(all_data)


if __name__ == "__main__":
    main()
