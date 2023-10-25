import requests
from bs4 import BeautifulSoup
import csv
import re
import time
from fake_useragent import UserAgent

# all topic choices
# topic = "pets"
# topic = "sports"
# topic = "history"

# topics = ["sports", "history", "poker", "astronomy", "ebooks", "genai", "bicycles", "writing", "parenting", "travel", "movies", "politics", "ebooks", "alcohol", "coffee", "economics", "literature"]
topics = ["cooking", "music", "law", "pets", "freelancing", "pm"]
max_pages = 10
# BASE_URL = f"https://{cur_topic}.stackexchange.com/questions?tab=votes&pagesize=50&page="


# avoid banned: use user agent to do the request
ua = UserAgent()


def get_random_headers():
    return {
        'User-Agent': ua.random
    }


def extract_question_and_answer(page_num, headers, cur_topic):
    data = []
    BASE_URL = f"https://{cur_topic}.stackexchange.com/questions?tab=votes&pagesize=50&page="
    response = requests.get(BASE_URL + str(page_num), headers=headers)
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

        time.sleep(3)

        # get current question's title and link
        question_title = question.text.strip()
        link = question['href']
        question_url = f'https://{cur_topic}.stackexchange.com' + link

        # get answer for this question
        q_response = requests.get(question_url, headers=headers)
        q_soup = BeautifulSoup(q_response.content, 'html.parser')
        answer = q_soup.select_one("div.answercell.post-layout--right")

        if not answer:
            print("!!!!!!!!!!!!!!!!!!!!!")
            print(question_title)
            print(q_response)
            # print(q_soup)

            # avoid banned, change header here and sleep
            headers = get_random_headers()
            continue
        else:
            print("====>access to the answer!")

        answer = answer.select_one("div.s-prose.js-post-body[itemprop='text']")
        if answer:
            paragraphs = answer.select('p')
            answer_text = ' '.join([p.text for p in paragraphs])
            data.append([question_title, answer_text])

        # avoid banned: sleeping!
        time.sleep(3)

    return data


def get_answer_from_topic(cur_topic):
    all_data = []
    page_num = 1
    headers = get_random_headers()

    with open(f"dataset/stackexchange/{cur_topic}_{max_pages}.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["question", "answer"])
        # writer.writerows(all_data)

        while True:
            print(f"\n\n==============={cur_topic}===================")
            print(f"Scraping page {page_num}...")
            try:
                data = extract_question_and_answer(page_num, headers, cur_topic)
            except Exception as e:
                print(f"=====> Exception occured: {e}")
                # avoid banned: sleep and change header
                time.sleep(3)
                headers = get_random_headers()
                break

            if not data:
                break

            writer.writerows(data)
            all_data.extend(data)

            if page_num >= max_pages:
                break
            page_num += 1

            # avoid banned: sleep and change header
            time.sleep(3)
            headers = get_random_headers()

        print("\n\n=========================")
        print(f"finish collecting data for topic {cur_topic}")
        print(len(all_data))


if __name__ == "__main__":
    for topic in topics:
        get_answer_from_topic(topic)
